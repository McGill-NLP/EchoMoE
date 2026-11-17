"""The folded student of a larger MoE model on vLLM 0.29 for throughput runs (Qwen3-235B-A22B,
DeepSeek-V3 and Kimi-K2.6 with random weights written by `fixtures.py`, and the trained Kanana-2
and Gemma 4 students), data parallel attention and expert parallel MoE layers over DeepEP v2, and
the plugin that registers it.

Each rank builds only its own experts' readers and factors, stored as the base model stores its
experts (BF16, block-128 FP8 or group-32 INT4). A reference layer sends its rows (with their routes)
through DeepEP to the ranks of their experts, as the base model does, and all-gathers them on a side
stream into every rank's cache while its experts run. The next layer joins that all-gather. A
consuming layer encodes its residual against this rank's own reference rows, dispatches only its
code through DeepEP, and each receiver joins the cached reference of every row it receives. The
combine returns each owner its summed code, which it decodes with `D_c`.

The dispatch carries what the base model's dispatch carries: BF16, or on DeepSeek-V3 (FP8 experts)
FP8 with block-128 scales, quantized by the sender with the experts' own quantization. The CLI's
--fp8 sends FP8 to every model (DeepEP's per-token cast where the experts take BF16 rows, cast back
by the receiver) and --bf16-wire BF16 to DeepSeek-V3 (its experts quantize on the receiver). Every
reference row lives in one cache per capacity bucket (the data-parallel batch rounded up to a power
of two) and carries the epoch of the forward that published it, so a CUDA graph replay reads only
the current forward's rows. Over the all-gather transport (tcp.py) a GatherBuffer stands in for
DeepEP's buffer. With `live_rows` (Gemma 4) the BF16 all-gather sends each origin's rows up to the
data-parallel batch instead of its whole bucket. Every row a consumer reads is the same.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from deep_ep import ElasticBuffer
from torch import nn
from torch.nn import functional
from vllm import ModelRegistry, _custom_ops as ops
from vllm.distributed import get_ep_group
from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe import GateLinear
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEParallelConfig,
    fp8_w8a8_moe_quant_config,
    int4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_v2 import (
    DeepEPV2PrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.router.router_factory import create_fused_moe_router
from vllm.model_executor.layers.fused_moe.utils import _resize_cache, moe_kernel_quantize_input
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
    marlin_moe_permute_scales,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import initialize_model, process_weights_after_loading
from vllm.model_executor.models import gemma4, kimi_k25
from vllm.model_executor.models.config import MODELS_CONFIG_MAP
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2Attention,
    DeepseekV2DecoderLayer,
    DeepseekV2MLAAttention,
    DeepseekV2MLP,
    DeepseekV2Model,
    _get_moe_router_dtype,
)
from vllm.model_executor.models.qwen3_moe import Qwen3MoeModel
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    extract_layer_index,
    init_vllm_registered_model,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.utils.torch_utils import set_default_torch_dtype

from .checkpoint import PortableExport
from .fairness import on_side_stream
from .kernels import assemble, compact_routes, decode_into, fp8_cast, zero_inactive_
from .model import FoldedDecoderLayer, FoldedTritonExperts
from .portable_kernels import (
    assemble_fp8,
    fp8_cast_back_blocks,
    local_residual,
    mask_router_output,
    pack4,
    prepare_router_input,
    prepare_routing,
    prepare_sender,
    publish_received,
    publish_reference,
    publish_rows,
    publish_valid,
    receive_routes,
    reference_inputs,
    reference_rows,
    unpack4,
)
from .tcp import GatherBuffer

LOAD_FORMAT = "echomoe_portable"
# the folded architecture of each base model family (`model_type` of its config)
ARCHITECTURES = {
    "qwen3_moe": "PortableQwen3MoeForCausalLM",
    "deepseek_v3": "PortableDeepseekV3ForCausalLM",
    "kimi_k25": "PortableKimiK25ForConditionalGeneration",
    "gemma4_text": "PortableGemma4ForCausalLM",
}


def register() -> None:
    """The folded architectures and their loader (registered in-process by the CLI)."""
    register_model_loader(LOAD_FORMAT)(PortableLoader)
    for name in ARCHITECTURES.values():
        ModelRegistry.register_model(name, f"{__name__}:{name}")
    # vLLM's Gemma 4 config hook, keyed by architecture name: FlashAttention on every layer, as the
    # head dimensions differ between sliding and global layers
    MODELS_CONFIG_MAP[ARCHITECTURES["gemma4_text"]] = MODELS_CONFIG_MAP["Gemma4ForCausalLM"]


# --- reference caches ----------------------------------------------------------------------------


class ReferenceState:
    """One capacity bucket's cache of the last reference layer's rows with BF16 dispatch, by (origin
    rank, row), with the buffers its background all-gather writes and this rank's own rows."""

    def __init__(self, capacity, hidden, *, device, group, live_rows):
        self.group = group
        self.origins = dist.get_world_size(group)
        self.capacity, self.hidden = capacity, hidden
        # the rows each origin sends: the bucket, or with `live_rows` the data-parallel batch
        self.live_rows, self.rows = live_rows, capacity
        self.values = torch.empty(
            (self.origins, capacity, hidden), device=device, dtype=torch.bfloat16
        )
        self.valid = torch.zeros((self.origins, capacity), device=device, dtype=torch.bool)
        # the epoch whose rows these are, and the reference layer that published them
        self.epoch = torch.full((), -1, device=device, dtype=torch.int64)
        self.layer = torch.full((), -1, device=device, dtype=torch.int64)
        self.received = torch.empty_like(self.values)
        self.counts = torch.empty(self.origins, device=device, dtype=torch.int64)
        self.stream = None
        self.sender = None
        self.pending = None

    def begin(self, epoch):
        self.valid.zero_()
        self.layer.fill_(-1)
        self.epoch.copy_(epoch)

    def gather(self, epoch, layer, payload, count):
        """All-gather every origin's rows and count on a side stream, after the work already queued
        (the reference layer's dispatch), so it overlaps the experts."""
        rows = self.rows
        if self.stream is None:
            self.stream = torch.cuda.Stream(device=self.received.device)
        self.stream.wait_stream(torch.cuda.current_stream())
        received = self.received.view(-1)[: self.origins * rows * self.hidden]
        with torch.cuda.stream(self.stream):
            dist.all_gather_into_tensor(
                received.view(self.origins * rows, self.hidden), payload[:rows], group=self.group
            )
            dist.all_gather_into_tensor(self.counts, count, group=self.group)
            event = torch.cuda.Event()
            event.record(self.stream)
        # the side stream reads the inputs. Keep them until the compute stream has waited for it
        self.pending = (event, layer, epoch, (payload, count), rows)

    def join(self):
        """Wait for the background all-gather and publish its rows as the cache (once)."""
        if self.pending is None:
            return
        event, layer, epoch, _, rows = self.pending
        torch.cuda.current_stream().wait_event(event)
        if self.live_rows:
            publish_rows(self, layer, epoch, self.received, self.counts, rows)
        else:
            publish_reference(self, layer, epoch, self.received, self.counts)
        self.pending = None


class FP8ReferenceState:
    """The same with FP8 dispatch: the cache holds every origin's FP8 rows and their block-128
    scales, gathered in place. This rank's rows are also kept in BF16 for its own residuals."""

    def __init__(self, capacity, hidden, *, device, group):
        self.group = group
        self.origins = dist.get_world_size(group)
        self.origin = dist.get_rank(group)
        self.capacity, self.hidden = capacity, hidden
        self.groups = hidden // 128
        self.values = torch.zeros(
            (self.origins, capacity, hidden), device=device, dtype=torch.float8_e4m3fn
        )
        self.scales = torch.zeros(
            (self.origins, capacity, self.groups), device=device, dtype=torch.float32
        )
        self.valid = torch.zeros((self.origins, capacity), device=device, dtype=torch.bool)
        self.epoch = torch.full((), -1, device=device, dtype=torch.int64)
        self.layer = torch.full((), -1, device=device, dtype=torch.int64)
        self.sender = torch.zeros((capacity, hidden), device=device, dtype=torch.bfloat16)
        self.counts = torch.zeros(self.origins, device=device, dtype=torch.int64)
        self.buffers = None
        self.pending = None

    def begin(self, epoch):
        self.valid.zero_()
        self.layer.fill_(-1)
        self.epoch.copy_(epoch)

    def gather(self, epoch, layer, fp8_rows, fp8_scales, count):
        """Mark this rank's own rows valid now (its residuals read the BF16 sender store), then
        all-gather every origin's FP8 rows and `[count; scales]` on a side stream."""
        if self.buffers is None:
            words = 2 + fp8_scales.numel()
            self.buffers = (
                torch.empty(words, device=fp8_rows.device, dtype=torch.int32),
                torch.empty(self.origins * words, device=fp8_rows.device, dtype=torch.int32),
                torch.cuda.Stream(device=fp8_rows.device),
            )
        packed, gathered, stream = self.buffers
        publish_valid(self, layer, epoch, count, local=True)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            pack4((count, fp8_scales), packed)
            values = self.values.view(self.origins * self.capacity, self.hidden)
            dist.all_gather_into_tensor(values, fp8_rows, group=self.group)
            dist.all_gather_into_tensor(gathered, packed, group=self.group)
            unpack4(gathered, (self.counts, self.scales), self.origins)
            event = torch.cuda.Event()
            event.record(stream)
        self.pending = (event, layer, epoch, (fp8_rows, fp8_scales, count))

    def join(self):
        if self.pending is None:
            return
        event, layer, epoch, _ = self.pending
        torch.cuda.current_stream().wait_event(event)
        publish_valid(self, layer, epoch, self.counts)
        self.pending = None


class Execution:
    """The forward's epoch and live row count, and the reference cache of each capacity bucket."""

    def __init__(self, capacity, hidden, *, device, group, fp8, live_rows):
        self.maximum_capacity, self.hidden = capacity, hidden
        self.device, self.group, self.fp8 = device, group, fp8
        self.live_rows = live_rows and not fp8
        if fp8:
            self.state_type = FP8ReferenceState
        else:
            self.state_type = partial(ReferenceState, live_rows=self.live_rows)
        self.states = {capacity: self.state_type(capacity, hidden, device=device, group=group)}
        self.state = self.states[capacity]
        self.epoch = torch.zeros((), device=device, dtype=torch.int64)
        self.count = torch.zeros(1, device=device, dtype=torch.int64)

    def select_capacity(self, rows):
        """Use the bucket of the data-parallel padded batch. Each keeps its tensors for the CUDA
        graphs captured with it."""
        capacity = min(1 << max(rows - 1, 0).bit_length(), self.maximum_capacity)
        if capacity not in self.states:
            self.states[capacity] = self.state_type(
                capacity, self.hidden, device=self.device, group=self.group
            )
        self.state = self.states[capacity]
        if self.live_rows:
            self.state.rows = max(rows, 1)

    @contextmanager
    def forward(self, count):
        """A new epoch: forget the previous forward's references (replayed with the graph). Every
        background all-gather is joined before the forward ends."""
        self.epoch.add_(1)
        self.state.begin(self.epoch)
        self.count.copy_(count)
        try:
            yield
        finally:
            for state in self.states.values():
                state.join()


# --- transport ------------------------------------------------------------------------------------


class PortablePrepareAndFinalize(DeepEPV2PrepareAndFinalize):
    """What one folded layer sends before its experts run and after: the reference layer's rows or
    the consuming layer's code through DeepEP's dispatch, its experts' codes through DeepEP's
    combine. Then the owner's decode."""

    def __init__(self, block, *, buffer, rank, world, num_sms):
        super().__init__(
            buffer=buffer,
            num_dispatchers=world,
            dp_size=world,
            rank_expert_offset=block.expert_offset,
            num_experts=block.num_experts,
            num_topk=block.top_k,
            use_cudagraph=True,
        )
        self.block = block
        self.world = world
        self.num_sms = num_sms
        self.context = None

    def prepare(
        self,
        a1,
        topk_weights,
        topk_ids,
        num_experts,
        expert_map,
        apply_router_weight_on_input,
        quant_config,
        defer_input_quant=False,
    ):
        if self.block.reference is None:
            sent = self.send_rows(a1, topk_weights, topk_ids, num_experts, quant_config)
        else:
            sent = self.send_code(
                a1, topk_weights, topk_ids, num_experts, expert_map, quant_config
            )
        values, scales, ids, weights, handle, active, owner_active = sent
        if scales is not None and self.block.expert_format != "fp8_block128":
            # FP8 dispatch (--fp8) of experts that take BF16 rows
            total = handle.psum_num_recv_tokens_per_scaleup_rank[-1:]
            values, scales = fp8_cast_back_blocks(values, scales, total), None

        # this rank's routes in the receive slots, in slot order
        local = self.block.local_experts
        if local < ids.shape[1]:
            ids, weights = compact_routes(
                ids,
                weights,
                active,
                offset=self.rank_expert_offset,
                local=local,
                experts=num_experts,
            )
        else:
            ids, weights = receive_routes(
                ids, weights, active, expert_map, offset=self.rank_expert_offset, local=local
            )
        self.context = dict(
            owner_active=owner_active, active=active, handle=handle, rows=a1.shape[0]
        )
        return values, scales, None, ids, weights

    def send_rows(self, a1, topk_weights, topk_ids, num_experts, quant_config):
        """Reference layer: its rows to its experts' ranks now (FP8 or BF16, as the dispatch
        carries them), every rank's cache in the background. The received rows and which are
        active. Over the all-gather transport the rows go once, in the cache's all-gather, joined
        at once, and this rank's experts take theirs from it."""
        block, execution = self.block, self.block.execution
        state = execution.state
        gathered = isinstance(self.buffer, GatherBuffer)
        if isinstance(state, FP8ReferenceState):
            send_ids, send_weights, owner_active = reference_inputs(
                a1,
                topk_ids,
                topk_weights,
                execution.count,
                state.capacity,
                num_experts,
                state.sender,
            )
            fp8_rows, fp8_scales = quantize(state.sender, quant_config)
            state.join()
            if gathered:
                state.gather(execution.epoch, block.index, fp8_rows, fp8_scales, execution.count)
                state.join()
                received, ids, weights, handle = self.buffer.dispatch_gathered(
                    (state.values.flatten(0, 1), state.scales.flatten(0, 1)),
                    send_ids,
                    send_weights,
                    num_experts,
                    state.capacity,
                )
            else:
                received, ids, weights, handle, _ = self.buffer.dispatch(
                    (fp8_rows, fp8_scales),
                    topk_idx=send_ids,
                    topk_weights=send_weights,
                    num_experts=num_experts,
                    num_max_tokens_per_rank=state.capacity,
                    do_expand=False,
                    do_cpu_sync=False,
                    num_sms=self.num_sms,
                    async_with_compute_stream=False,
                )
                state.gather(execution.epoch, block.index, fp8_rows, fp8_scales, execution.count)
            values, scales = received
        else:
            state.join()
            payload, state.sender, send_ids, send_weights, owner_active = reference_rows(
                a1,
                topk_ids.long(),
                topk_weights,
                execution.count,
                state.capacity,
                num_experts,
                self.world,
            )
            state.begin(execution.epoch)
            state.layer.fill_(block.index)
            if gathered:
                state.gather(execution.epoch, block.index, payload, execution.count)
                state.join()
                (values,), ids, weights, handle = self.buffer.dispatch_gathered(
                    (state.received.flatten(0, 1),),
                    send_ids,
                    send_weights,
                    num_experts,
                    state.capacity,
                )
            else:
                received, ids, weights, handle, _ = self.buffer.dispatch(
                    payload,
                    topk_idx=send_ids,
                    topk_weights=send_weights,
                    num_experts=num_experts,
                    num_max_tokens_per_rank=state.capacity,
                    do_expand=False,
                    do_cpu_sync=False,
                    async_with_compute_stream=False,
                )
                state.gather(execution.epoch, block.index, payload, execution.count)
                values = publish_received(state, received, handle)
            scales = None
            owner_active = owner_active[: a1.shape[0]]
        total = handle.psum_num_recv_tokens_per_scaleup_rank[-1]
        active = torch.arange(values.shape[0], device=values.device) < total
        return values, scales, ids, weights, handle, active, owner_active

    def send_code(self, a1, topk_weights, topk_ids, num_experts, expert_map, quant_config):
        """Consuming layer: the code of the residual against this rank's own reference rows. Then
        each received row's cached reference beside its code, once the cache has landed."""
        block, execution = self.block, self.block.execution
        state = execution.state
        fp8 = isinstance(state, FP8ReferenceState)
        weights, owner_active = prepare_routing(
            topk_ids,
            topk_weights,
            expert_map,
            offset=self.rank_expert_offset,
            local=block.local_experts,
        )
        if fp8:
            residual = prepare_sender(
                a1, block.gain, state, execution.epoch, owner_active, reference=block.reference
            )
            payload = quantize(functional.linear(residual, block.encoder), quant_config)
        else:
            residual = local_residual(a1, state.sender[: a1.shape[0]], block.gain, owner_active)
            payload = functional.linear(residual, block.encoder)
        received, ids, weights, handle, _ = self.buffer.dispatch(
            payload,
            topk_idx=topk_ids,
            topk_weights=weights,
            num_experts=num_experts,
            num_max_tokens_per_rank=state.capacity,
            do_expand=False,
            do_cpu_sync=False,
            num_sms=self.num_sms,
            async_with_compute_stream=False,
        )
        state.join()
        metadata = handle.recv_src_metadata
        total = handle.psum_num_recv_tokens_per_scaleup_rank[-1]
        if fp8:
            values, scales, active = assemble_fp8(
                state, execution.epoch, *received, metadata, total, reference=block.reference
            )
        else:
            values, active = assemble(
                state, execution.epoch, received, metadata, total, reference=block.reference
            )
            scales = None
        return values, scales, ids, weights, handle, active, owner_active

    def finalize(
        self,
        output,
        fused_expert_output,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input,
        weight_and_reduce_impl,
    ):
        block, context = self.block, self.context
        self.context = None
        code = fused_expert_output
        # With no rows received vLLM skips the experts and returns an empty tensor of the input's
        # width. The combine still runs, at the code's width.
        if context["active"].numel() == 0 and code.shape == (0, block.input_width):
            code = code.new_empty((0, block.combine_rank))
        code = zero_inactive_(code, context["active"])
        combined, _, _ = self.buffer.combine(
            code, handle=context["handle"], topk_weights=None, async_with_compute_stream=False
        )
        if block.reference is None:
            # the reference layer's dispatch padded its rows to the cache capacity
            combined = combined[: context["rows"]]
        decode_into(output, combined, block.decoder, context["owner_active"])

    def prepare_async(self, *args, **kwargs):
        result = self.prepare(*args, **kwargs)
        return lambda: result

    def finalize_async(self, *args, **kwargs):
        self.finalize(*args, **kwargs)
        return lambda: None


def quantize(rows, quant_config):
    """The sender's FP8 rows: the FP8 experts' own activation quantization (dynamic, block-128),
    else DeepEP's per-token cast (--fp8)."""
    if not quant_config.use_fp8_w8a8:
        return fp8_cast(rows)
    return moe_kernel_quantize_input(
        rows,
        quant_config.a1_scale,
        quant_config.quant_dtype,
        quant_config.per_act_token_quant,
        quant_config.block_shape,
    )


def reference_buffer_hidden(manager, capacity, hidden, reference_hidden, topk):
    """The DeepEP buffer width that covers the reference layers' full-width rows: keep the codes'
    width when its buffer is already as large as the rows need."""
    if isinstance(manager, AgRsAll2AllManager):
        # the all-gather transport (tcp.py) has no buffer to size
        return hidden
    if not manager._gin_checked:
        manager._check_gin_support(manager._device_group)
        manager._gin_checked = True
    kwargs = manager._make_all2all_kwargs(capacity, hidden, topk, False)
    keys = (
        "group",
        "num_max_tokens_per_rank",
        "hidden",
        "num_topk",
        "use_fp8_dispatch",
        "allow_hybrid_mode",
        "allow_multiple_reduction",
    )
    sizing = {key: kwargs[key] for key in keys}
    existing = ElasticBuffer.get_buffer_size_hint(**sizing)
    # multiple reduction only tightens the size bound of the full-width rows
    needed = ElasticBuffer.get_buffer_size_hint(
        **dict(sizing, hidden=reference_hidden, allow_multiple_reduction=True)
    )
    return hidden if existing >= needed else reference_hidden


# --- experts -------------------------------------------------------------------------------------


class PortableTritonExperts(FoldedTritonExperts):
    """vLLM's Triton experts at the folded widths (BF16, or block-128 FP8 taking FP8 dispatch's
    received rows as they are)."""

    fp8_wire = False

    @property
    def expects_unquantized_inputs(self):
        return self.quant_config.use_fp8_w8a8 and not self.fp8_wire


def int4_group32_to_marlin(weight, scale):
    """A group-32 INT4 bank (`[E, N, K/2]` uint8, two offset-8 nibbles per byte, `[E, N, K/32]`
    BF16 scales) in Marlin's layout, as vLLM prepares a compressed-tensors W4A16 MoE checkpoint."""
    experts, out_features, packed_in = weight.shape
    in_features = 2 * packed_in
    # one expert at a time: a whole bank's nibble and int64 word temporaries are several GB
    packed = []
    for expert in range(experts):
        nibbles = torch.stack(
            (weight[expert : expert + 1] & 15, weight[expert : expert + 1] >> 4), -1
        ).reshape(1, out_features, in_features)
        # 8 nibbles per int32 along K, nibble i of word j holding input 8j+i (GPTQ order)
        grouped = (
            nibbles.transpose(1, 2).reshape(1, in_features // 8, 8, out_features).to(torch.int64)
        )
        shifts = (4 * torch.arange(8, device=weight.device, dtype=torch.int64)).view(1, 1, 8, 1)
        words = (grouped << shifts).sum(2)
        words = torch.where(words >= 2**31, words - 2**32, words)  # int32 two's complement
        packed.append(words.to(torch.int32).contiguous()[0])
    empty = torch.empty((experts, 0), dtype=torch.int32, device=weight.device)
    qweight = ops.gptq_marlin_moe_repack(torch.stack(packed), empty, in_features, out_features, 4)
    scales = marlin_moe_permute_scales(
        scale.transpose(1, 2).contiguous(), size_k=in_features, size_n=out_features, group_size=32
    )
    return qweight, scales


class PortableMarlinExperts(MarlinExperts):
    """vLLM's Marlin INT4 experts (the base Kimi-K2.6's kernel) at the folded widths:
    `_fused_marlin_moe` and its reduction, with the down GEMM writing the combine rank's columns."""

    def moe_problem_size(self, a1, w1, w2, topk_ids):
        return w1.shape[0], a1.shape[0], w2.size(1) * 16, w2.size(2) // 2, topk_ids.size(1)

    def apply(
        self,
        output,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        global_num_experts,
        expert_map,
        a1q_scale,
        a2_scale,
        workspace13,
        workspace2,
        expert_tokens_meta,
        apply_router_weight_on_input,
    ):
        empty = torch.empty((w1.size(0), 0), dtype=torch.int32, device=w1.device)
        rows, hidden = hidden_states.shape
        local = w1.size(0)
        top_k = topk_ids.size(1)
        intermediate = w2.size(1) * 16
        out_width = output.shape[1]
        local_rows = (
            math.ceil(rows * local / global_num_experts) if global_num_experts > 0 else rows
        )
        for block_size_m in (8, 16, 32, 48, 64):
            if local_rows * top_k / local / block_size_m < 0.9:
                break
        sorted_ids, expert_ids, post_padded = moe_align_block_size(
            topk_ids, block_size_m, global_num_experts, expert_map, ignore_invalid_experts=True
        )
        workspace = marlin_make_workspace_new(hidden_states.device, 4)
        quant = scalar_types.uint4b8
        gate_up = _resize_cache(workspace2, (rows * top_k, 2 * intermediate))
        gate_up = ops.moe_wna16_marlin_gemm(
            hidden_states, gate_up, w1, None, self.w1_scale, None, None, None, empty, empty,
            workspace, sorted_ids, expert_ids, post_padded, topk_weights,
            moe_block_size=block_size_m, top_k=top_k, mul_topk_weights=False, b_q_type=quant,
            size_m=rows, size_n=2 * intermediate, size_k=hidden, is_k_full=True,
            use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False,
        )  # fmt: skip
        activated = _resize_cache(workspace13, (rows * top_k, intermediate))
        self.activation(
            activation,
            activated,
            gate_up.view(-1, 2 * intermediate),
            topk_ids=topk_ids,
            expert_map=expert_map,
        )
        down = _resize_cache(workspace2, (rows * top_k, out_width))
        down = ops.moe_wna16_marlin_gemm(
            activated, down, w2, None, self.w2_scale, None, None, None, empty, empty, workspace,
            sorted_ids, expert_ids, post_padded, topk_weights, moe_block_size=block_size_m, top_k=1,
            mul_topk_weights=True, b_q_type=quant, size_m=rows * top_k, size_n=out_width,
            size_k=intermediate, is_k_full=True, use_atomic_add=False, use_fp32_reduce=True,
            is_zp_float=False,
        )  # fmt: skip
        ops.moe_sum(down.view(rows, top_k, out_width), output, topk_ids, expert_map)


# --- the folded MoE layer ------------------------------------------------------------------------


class PortableMoE(nn.Module):
    """One folded layer on this rank: vLLM's router (and shared experts), this rank's readers and
    factors, the encoder and gain of a consuming layer, and the owner's decoder. Gemma 4's layer
    computes the router logits and passes them in. With `second_stream` the shared experts run on
    vLLM's auxiliary stream beside the routed experts."""

    def __init__(
        self,
        *,
        layer_index,
        export,
        execution,
        rank,
        world,
        buffer,
        num_sms,
        second_stream,
        vllm_config,
        prefix,
    ):
        super().__init__()
        profile = export.profile
        config = vllm_config.model_config.hf_text_config
        self.index = layer_index
        self.top_k = profile.top_k
        self.num_experts = profile.experts
        self.local_experts = profile.experts // world
        self.expert_offset = rank * self.local_experts
        self.execution = execution
        self.expert_format = export.expert_format
        self.routed_scale = profile.routed_scale
        self.second_stream = second_stream
        gemma = profile.family == "gemma4_text"
        self.activation = MoEActivation.GELU_TANH if gemma else MoEActivation.SILU
        device = execution.state.values.device
        tensors = export.layer(
            layer_index, range(self.expert_offset, self.expert_offset + self.local_experts)
        )
        self.reference = tensors["reference"]
        self.combine_rank = tensors["decoder"].shape[1]
        self.input_width = profile.hidden + (
            0 if self.reference is None else tensors["encoder"].shape[0]
        )

        # vLLM's gate (Gemma 4's is its layer's) and router, and the shared experts of DeepSeek-V3,
        # Kimi-K2.6 and Kanana-2
        with torch.device(device):
            if gemma:
                self.gate = None
                self.per_expert_scale = nn.Parameter(
                    torch.ones(profile.experts), requires_grad=False
                )
            elif profile.family == "qwen3_moe":
                self.gate = ReplicatedLinear(
                    profile.hidden,
                    profile.experts,
                    bias=False,
                    params_dtype=torch.bfloat16,
                    quant_config=None,
                    prefix=f"{prefix}.gate",
                )
            else:
                self.gate = GateLinear(
                    profile.hidden,
                    profile.experts,
                    out_dtype=_get_moe_router_dtype(config),
                    prefix=f"{prefix}.gate",
                )
                self.gate.e_score_correction_bias = (
                    nn.Parameter(torch.empty(profile.experts, dtype=torch.float32))
                    if profile.correction_bias
                    else None
                )
        if gemma:
            # vLLM's Gemma 4 routing: softmax over every expert, top-k, renormalized, times
            # per_expert_scale
            def routing(hidden_states, gating_output, topk, renormalize):
                return gemma4.gemma4_fused_routing_kernel_triton(
                    gating_output, topk, self.per_expert_scale
                )

            self.router = create_fused_moe_router(
                top_k=profile.top_k,
                global_num_experts=profile.experts,
                renormalize=True,
                custom_routing_function=routing,
            )
        elif profile.family == "qwen3_moe":
            self.router = create_fused_moe_router(
                top_k=profile.top_k,
                global_num_experts=profile.experts,
                renormalize=profile.normalized,
            )
        else:
            # vLLM's DeepSeek scales the routed output, not the router weights (a different BF16 rounding)
            self.router = create_fused_moe_router(
                top_k=profile.top_k,
                global_num_experts=profile.experts,
                renormalize=profile.normalized,
                use_grouped_topk=True,
                num_expert_group=profile.groups,
                topk_group=profile.selected_groups,
                scoring_func=profile.scoring_func,
                routed_scaling_factor=1.0,
                e_score_correction_bias=self.gate.e_score_correction_bias,
            )
        self.shared_experts = None
        if profile.shared_experts:
            self.shared_experts = DeepseekV2MLP(
                hidden_size=profile.hidden,
                intermediate_size=profile.ffn * profile.shared_experts,
                hidden_act="silu",
                quant_config=vllm_config.quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        # this rank's experts and the codec, on the device
        for name in (
            "readers",
            "factors",
            "reader_scale",
            "factor_scale",
            "encoder",
            "gain",
            "decoder",
        ):
            value = tensors[name]
            self.register_buffer(
                name, None if value is None else value.to(device=device, copy=True)
            )
        expert_map = torch.full((profile.experts,), -1, device=device, dtype=torch.int32)
        expert_map[self.expert_offset : self.expert_offset + self.local_experts] = torch.arange(
            self.local_experts, device=device, dtype=torch.int32
        )
        self.register_buffer("expert_map", expert_map)

        # vLLM's modular MoE kernel: this layer's transport around the experts in their own format
        adapter = PortablePrepareAndFinalize(
            self, buffer=buffer, rank=rank, world=world, num_sms=num_sms
        )
        self.topk_indices_dtype = adapter.topk_indices_dtype()
        parallel = replace(
            FusedMoEParallelConfig.make_no_parallel(),
            dp_size=world,
            dp_rank=rank,
            ep_size=world,
            ep_rank=rank,
            use_ep=True,
            all2all_backend="deepep_v2",
        )
        moe_config = FusedMoEConfig(
            num_experts=profile.experts,
            experts_per_token=profile.top_k,
            hidden_dim=self.input_width,
            intermediate_size=profile.ffn,
            num_local_experts=self.local_experts,
            num_logical_experts=profile.experts,
            activation=MoEActivation.SILU,
            device=self.readers.device,
            routing_method=self.router.routing_method_type,
            moe_parallel_config=parallel,
            in_dtype=torch.bfloat16,
            max_num_tokens=execution.state.capacity,
        )
        if self.expert_format == "bf16":
            experts = PortableTritonExperts(moe_config, FUSED_MOE_UNQUANTIZED_CONFIG)
        elif self.expert_format == "fp8_block128":
            quant = fp8_w8a8_moe_quant_config(
                self.reader_scale, self.factor_scale, block_shape=[128, 128]
            )
            experts = PortableTritonExperts(moe_config, quant)
            experts.fp8_wire = execution.fp8
        else:
            # the base Kimi-K2.6 runs its INT4 experts on Marlin. Repack once and run them the same way
            self.readers, reader_scale = int4_group32_to_marlin(self.readers, self.reader_scale)
            self.factors, factor_scale = int4_group32_to_marlin(self.factors, self.factor_scale)
            self.reader_scale = self.factor_scale = None
            quant = int4_w4a16_moe_quant_config(reader_scale, factor_scale, block_shape=[0, 32])
            experts = PortableMarlinExperts(moe_config, quant)
        self.kernel = FusedMoEKernel(adapter, experts)

    def forward(self, hidden_states, router_logits=None):
        # rows past the live count are zeroed before the router and their routes masked after it
        rows = hidden_states.shape[0]
        original, active = prepare_router_input(hidden_states.contiguous(), self.execution.count)
        if rows:
            logits = router_logits
            if logits is None:
                logits = self.gate(original)
            if isinstance(logits, tuple):
                logits = logits[0]
            weights, ids = self.router.select_experts(
                original, logits, topk_indices_dtype=self.topk_indices_dtype
            )
        else:
            weights = torch.empty((0, self.top_k), device=original.device, dtype=torch.float32)
            ids = torch.empty(
                (0, self.top_k), device=original.device, dtype=self.topk_indices_dtype
            )
        if (
            ids.dtype in (torch.int32, torch.int64)
            and weights.dtype == torch.float32
            and ids.is_contiguous()
            and weights.is_contiguous()
        ):
            ids, weights = mask_router_output(ids, weights, active)
        else:
            ids = torch.where(active[:, None], ids, -1)
            weights = torch.where(active[:, None], weights, 0)
        shared = done = None
        if self.shared_experts is not None and rows:
            if self.second_stream:
                shared, done = on_side_stream(self.shared_experts, original)
            else:
                shared = self.shared_experts(original)
        routed = self.kernel.apply(
            hidden_states=original,
            w1=self.readers,
            w2=self.factors,
            topk_weights=weights,
            topk_ids=ids,
            activation=self.activation,
            global_num_experts=self.num_experts,
            expert_map=self.expert_map,
            apply_router_weight_on_input=False,
        )
        # vLLM's order: scale the routed output, then add the shared experts' (BF16)
        if self.routed_scale != 1.0:
            routed.mul_(self.routed_scale)
        if shared is None:
            return routed
        if done is not None:
            torch.cuda.current_stream().wait_event(done)
        return zero_inactive_(shared + routed, active)


# --- the models ----------------------------------------------------------------------------------


class PortableMLADecoderLayer(DeepseekV2DecoderLayer):
    """The base DeepSeek-V3 decoder layer with a folded MoE block. The parent's constructor would
    allocate the routed experts the block replaces, so it is not called."""

    def __init__(self, *, vllm_config, prefix, block, moe_layers):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        self.hidden_size = config.hidden_size
        self.layer_idx = int(prefix.rsplit(".", 1)[1])
        self.use_mha = False
        self.use_sequence_parallel_moe = False
        attention = (
            DeepseekV2MLAAttention if vllm_config.model_config.use_mla else DeepseekV2Attention
        )
        self.self_attn = attention(
            vllm_config=vllm_config,
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            max_position_embeddings=config.max_position_embeddings,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.self_attn",
            topk_indices_buffer=None,
            reduce_results=True,
        )
        if self.layer_idx in moe_layers:
            self.mlp = block(layer_index=self.layer_idx, prefix=f"{prefix}.mlp")
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.routed_scaling_factor = config.routed_scaling_factor


class PortableMLAModel(DeepseekV2Model):
    """The base DeepSeek-V3 body with folded MoE blocks. Its constructor's state is set here (no
    torch.compile), its forward is the parent's."""

    def __init__(self, *, vllm_config, prefix, block, moe_layers):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.do_not_compile = True
        self.config = config
        self.device = current_platform.device_type
        self.hidden_size, self.vocab_size = config.hidden_size, config.vocab_size
        self.is_v32, self.use_mha = False, False
        self.is_fused_shared_expert_enabled = False
        self.num_redundant_experts = 0
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: PortableMLADecoderLayer(
                vllm_config=vllm_config, prefix=prefix, block=block, moe_layers=moe_layers
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        self.aux_hidden_state_layers = ()


def gemma4_body(vllm_config, prefix, block):
    """vLLM's Gemma 4 body with folded MoE blocks: its decoder layer builds its Gemma4MoE by the
    module's name, bound to the folded block while the body is built."""

    def folded(config, quant_config=None, prefix=""):
        return block(layer_index=extract_layer_index(prefix), prefix=prefix)

    stock = gemma4.Gemma4MoE
    gemma4.Gemma4MoE = folded
    try:
        return gemma4.Gemma4Model(vllm_config=vllm_config, prefix=prefix)
    finally:
        gemma4.Gemma4MoE = stock


class PortableLanguageModel(nn.Module):
    """The folded student, expert parallel over vLLM's expert-parallel group."""

    family = None
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    }

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        export = PortableExport(Path(vllm_config.model_config.model).parent)
        profile, recipe = export.profile, export.recipe
        self.config = config = vllm_config.model_config.hf_text_config

        # vLLM's DeepEP v2 buffer, wide enough for the codes and the reference layers' rows
        group = get_ep_group()
        manager = group.device_communicator.all2all_manager
        capacity = vllm_config.scheduler_config.max_num_batched_tokens
        hidden = reference_buffer_hidden(
            manager,
            capacity,
            max(recipe.dispatch_rank, recipe.combine_rank),
            profile.hidden,
            profile.top_k,
        )
        buffer = manager.get_handle(
            dict(
                num_max_tokens_per_rank=capacity,
                hidden=hidden,
                num_topk=profile.top_k,
                num_experts=profile.experts,
                use_fp8_dispatch=False,
            )
        )

        # the reference caches, then the layers, each with this rank's experts
        options = vllm_config.load_config.model_loader_extra_config
        self.execution = Execution(
            capacity,
            profile.hidden,
            device=vllm_config.device_config.device,
            group=group.device_group,
            fp8=options["fp8"],
            live_rows=options["live_rows"],
        )
        block = partial(
            PortableMoE,
            export=export,
            execution=self.execution,
            rank=group.rank_in_group,
            world=group.world_size,
            buffer=buffer,
            num_sms=buffer.get_theoretical_num_sms(profile.experts, profile.top_k),
            second_stream=options["second_stream"],
            vllm_config=vllm_config,
        )
        if self.family == "qwen3_moe":
            self.model = Qwen3MoeModel(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "model"),
                decoder_layer_type=partial(FoldedDecoderLayer, block=block),
            )
        elif self.family == "gemma4_text":
            self.model = gemma4_body(vllm_config, maybe_prefix(prefix, "model"), block)
        else:
            self.model = PortableMLAModel(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "model"),
                block=block,
                moe_layers=profile.moe_layers,
            )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
        self.quant_config = vllm_config.quant_config

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        context = get_forward_context()
        rows = (input_ids if inputs_embeds is None else inputs_embeds).shape[0]
        self.execution.select_capacity(int(context.dp_metadata.num_tokens_across_dp_cpu.max()))
        # The live rows come from vLLM's device padding mask. They must form a prefix, since a
        # reference is keyed by its row.
        count = (~context.is_padding).sum(dtype=torch.int64).reshape(1)
        expected = torch.arange(rows, device=context.is_padding.device) >= count[0]
        torch._assert_async(
            (context.is_padding == expected).all(), "Native EP requires prefix-packed live rows"
        )
        with torch.no_grad(), self.execution.forward(count):
            return self.model(input_ids, positions, None, inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states)

    def embed_input_ids(self, input_ids):
        return self.model.embed_input_ids(input_ids)

    def load_weights(self, weights):
        return AutoWeightsLoader(self).load_weights(weights)


class PortableQwen3MoeForCausalLM(PortableLanguageModel):
    family = "qwen3_moe"


class PortableDeepseekV3ForCausalLM(PortableLanguageModel):
    family = "deepseek_v3"


class PortableGemma4ForCausalLM(PortableLanguageModel):
    """As vLLM's Gemma4ForCausalLM: the head tied to the embedding, logits softcapped."""

    family = "gemma4_text"

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size, soft_cap=self.config.final_logit_softcapping
        )

    def load_weights(self, weights):
        """vLLM's Gemma 4 loading (the router's per_expert_scale into the block, the K=V layers' V
        from K). It skips names it does not know, so every parameter must have been loaded."""
        loaded = gemma4.Gemma4ForCausalLM.load_weights(self, weights)
        missing = set(dict(self.named_parameters())) - loaded
        assert not missing, f"not loaded: {sorted(missing)[:8]}"
        return loaded


@MULTIMODAL_REGISTRY.register_processor(
    kimi_k25.KimiK25MultiModalProcessor,
    info=kimi_k25.KimiK25ProcessingInfo,
    dummy_inputs=kimi_k25.KimiK25DummyInputsBuilder,
)
class PortableKimiK25ForConditionalGeneration(kimi_k25.KimiK25ForConditionalGeneration):
    """vLLM's Kimi-K2.6 (its vision tower unchanged) around the folded DeepSeek-V3-style text model."""

    def __init__(self, *, vllm_config, prefix=""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.config = config
        self.use_data_parallel = kimi_k25.is_vit_use_data_parallel(
            config.vision_config.num_attention_heads
        )
        self.hidden_size = config.text_config.hidden_size
        self.device = kimi_k25.current_platform.current_device()
        quant_config = vllm_config.quant_config
        with self._mark_tower_model(vllm_config, "vision_chunk"):
            vision_quant = self._maybe_ignore_quant_config(quant_config)
            self.vision_tower = kimi_k25.MoonViT3dPretrainedModel(
                config.vision_config,
                quant_config=vision_quant,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            self.vision_tower = self.vision_tower.to(
                device=self.device,
                **({} if vision_quant is not None else {"dtype": vllm_config.model_config.dtype}),
            )
            self.mm_projector = kimi_k25.KimiK25MultiModalProjector(
                config=config.vision_config,
                use_data_parallel=self.use_data_parallel,
                quant_config=vision_quant,
                prefix=maybe_prefix(prefix, "mm_projector"),
            ).to(device=self.device, dtype=vllm_config.model_config.dtype)
        self.quant_config = quant_config
        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=[ARCHITECTURES["deepseek_v3"]],
            )
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors
        self.media_placeholder = config.media_placeholder_token_id


class PortableLoader(BaseModelLoader):
    """Load the export's backbone by name. Each folded block read its experts when it was built."""

    def download_model(self, model_config):
        pass

    def load_weights(self, model, model_config):
        model.load_weights(PortableExport(Path(model_config.model).parent).backbone())

    def load_model(self, vllm_config, model_config, prefix=""):
        # built, loaded and converted (FP8 and INT4 backbone layers) on the device
        device = torch.device(self.load_config.device or vllm_config.device_config.device)
        with set_default_torch_dtype(model_config.dtype), device:
            model = initialize_model(
                vllm_config=vllm_config, model_config=model_config, prefix=prefix
            )
            self.load_weights(model, model_config)
            process_weights_after_loading(model, model_config, next(model.parameters()).device)
            model.requires_grad_(False)
        return model.eval()
