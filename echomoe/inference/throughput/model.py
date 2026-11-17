"""The folded student on vLLM 0.29 for throughput runs, data parallel attention and expert parallel
MoE layers over DeepEP v2, and the plugin that registers it.

Each rank builds only its own experts' readers and factors from the export. A reference layer
all-gathers its rows (with their routes) to every rank, which caches them. A consuming layer
dispatches only its code `c_d = E_d (x - a * z)` through DeepEP, and each receiver joins the cached
reference `z` of every row it receives. The experts run through vLLM's grouped Triton GEMMs, the
readers in place of the gate/up weight at width `H + r_d`, the factors in place of the down weight
at width `r_c`. The combine returns each owner its summed code, which it decodes with `D_c`.

Every reference row lives in one cache per capacity bucket (the data-parallel batch rounded up to
a power of two) and carries the epoch of the forward that published it, so a CUDA graph replay
reads only the current forward's rows.
"""

from __future__ import annotations

import os
from dataclasses import replace
from functools import partial

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional
from vllm import ModelRegistry
from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEParallelConfig,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    ExpertTokensMetadata,
    FusedMoEKernel,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_v2 import (
    DeepEPV2PrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.router.router_factory import create_fused_moe_router
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeModel,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, extract_layer_index

from .checkpoint import FoldedExport
from .kernels import (
    assemble,
    assemble_compact,
    compact_routes,
    consumer_send,
    decode_into,
    exchange_reference,
    fp8_cast,
)

ARCHITECTURE = "Qwen3MoeFoldedEPForCausalLM"
LOAD_FORMAT = "echomoe_throughput"
# The measured runtime fused the sender and receiver only at these dispatch widths, hidden size
# and row counts (its unfused path is not ported). The table's eleven configs are all inside.
FUSED_WIDTHS = (256, 512, 768, 1280)
FUSED_HIDDEN = 2048
FUSED_SENDER_ROWS = 4096
FUSED_RECEIVER_ROWS = 131072


def register() -> None:
    """The vLLM plugin (`vllm.general_plugins`): the folded architecture and its loader, and in a
    serving process (serving.py) the throughput rows' patches."""
    register_model_loader(LOAD_FORMAT)(FoldedLoader)
    ModelRegistry.register_model(ARCHITECTURE, f"{__name__}:Qwen3MoeFoldedEPForCausalLM")
    if "ECHOMOE_SERVING" in os.environ:
        from .serving import install

        install()


class ReferenceCache:
    """One capacity bucket's cache of the last reference layer's rows, by (origin rank, row)."""

    def __init__(self, capacity, hidden, *, device, group):
        self.group = group
        self.origins = dist.get_world_size(group)
        self.capacity, self.hidden = capacity, hidden
        self.values = torch.empty(
            (self.origins, capacity, hidden), device=device, dtype=torch.bfloat16
        )
        self.valid = torch.zeros((self.origins, capacity), device=device, dtype=torch.bool)
        # the epoch whose rows these are, and the reference layer that published them
        self.epoch = torch.full((), -1, device=device, dtype=torch.int64)
        self.layer = torch.full((), -1, device=device, dtype=torch.int64)
        # Unused here. Allocated (the receive buffer released at the first exchange) as in the
        # measured runtime, so vLLM's memory profile leaves the KV cache the same size.
        self.row_numbers = torch.arange(capacity, device=device)
        self.received = torch.empty_like(self.values)
        self.counts = torch.empty(self.origins, device=device, dtype=torch.int64)
        self.exchange = None


class Execution:
    """The forward's epoch and live row count, and the reference cache of each capacity bucket."""

    def __init__(self, capacity, hidden, *, device, group):
        self.maximum_capacity, self.hidden = capacity, hidden
        self.device, self.group = device, group
        self.caches = {capacity: ReferenceCache(capacity, hidden, device=device, group=group)}
        self.cache = self.caches[capacity]
        self.epoch = torch.zeros((), device=device, dtype=torch.int64)
        self.count = torch.zeros(1, device=device, dtype=torch.int64)

    def select_capacity(self, rows):
        """Use the bucket of the data-parallel padded batch. Each keeps its tensors for the
        CUDA graphs captured with it."""
        capacity = min(1 << max(rows - 1, 0).bit_length(), self.maximum_capacity)
        if capacity not in self.caches:
            self.caches[capacity] = ReferenceCache(
                capacity, self.hidden, device=self.device, group=self.group
            )
        self.cache = self.caches[capacity]

    def begin(self, count):
        """A new epoch: forget the previous forward's references (replayed with the graph)."""
        self.epoch.add_(1)
        self.cache.valid.zero_()
        self.cache.layer.fill_(-1)
        self.cache.epoch.copy_(self.epoch)
        self.count.copy_(count)


class FoldedPrepareAndFinalize(DeepEPV2PrepareAndFinalize):
    """What one folded layer sends before its experts run and after: at a reference layer the
    all-gather and a reduce-scatter combine, at a consuming layer DeepEP's dispatch of the code
    and its combine. Then the owner's decode. `buffer` is the RailSwitch in front of DeepEP's
    (selection.py), or over TCP the GatherBuffer (tcp.py), which also runs the reference
    collectives. With FP8 activations (`block.fp8`) the dispatch and the all-gather move FP8 rows,
    and the reference layer's combine sums in BF16 within a node and in FP8 between nodes, as
    DeepEP's FP8 combine does."""

    def __init__(self, block, *, buffer, rank, world, top_k):
        super().__init__(
            buffer=buffer,
            num_dispatchers=world,
            dp_size=world,
            rank_expert_offset=block.expert_offset,
            num_experts=block.num_experts,
            num_topk=top_k,
            use_cudagraph=True,
        )
        self.block = block
        self.rank = rank
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
        block, execution = self.block, self.block.execution
        cache = execution.cache
        local = block.local_experts
        if block.reference is None:
            compact = local < self.num_topk
            owner_active, values, ids, weights, active = exchange_reference(
                cache,
                block.index,
                execution.epoch,
                a1,
                topk_ids,
                topk_weights,
                execution.count,
                offset=self.rank_expert_offset,
                local=local,
                compact=compact,
                fp8=block.fp8,
                all_gather=partial(
                    self.buffer.all_gather, group=cache.group, capacity=cache.capacity
                ),
            )
            self.context = dict(
                owner_active=owner_active, active=active, rows=a1.shape[0], cache=cache
            )
            return values, None, None, ids, weights

        # sender: masked routes and the residual against the cached reference, then the code
        assert a1.shape[0] <= FUSED_SENDER_ROWS, a1.shape
        ids, weights, owner_active, residual = consumer_send(
            a1,
            topk_ids,
            topk_weights,
            execution.count,
            block.gain,
            cache,
            execution.epoch,
            origin=self.rank,
            reference=block.reference,
        )
        code = functional.linear(residual, block.encoder)
        received, recv_ids, recv_weights, handle, _ = self.buffer.dispatch(
            fp8_cast(code) if block.fp8 else code,
            topk_idx=ids,
            topk_weights=weights,
            num_experts=num_experts,
            num_max_tokens_per_rank=cache.capacity,
            do_expand=False,
            do_cpu_sync=False,
            num_sms=0,
            async_with_compute_stream=False,
        )
        rows = (received[0] if block.fp8 else received).shape[0]
        assert rows <= FUSED_RECEIVER_ROWS, rows

        # receiver: each row's cached reference beside its code, this rank's routes in slot order
        total = handle.psum_num_recv_tokens_per_scaleup_rank[-1]
        if local < recv_ids.shape[1] or block.fp8:
            values, active, ids, weights = assemble_compact(
                cache,
                execution.epoch,
                received,
                handle.recv_src_metadata,
                total,
                recv_ids,
                recv_weights,
                reference=block.reference,
                offset=self.rank_expert_offset,
                local=local,
            )
        else:
            values, active = assemble(
                cache,
                execution.epoch,
                received,
                handle.recv_src_metadata,
                total,
                reference=block.reference,
            )
            ids, weights = compact_routes(
                recv_ids,
                recv_weights,
                active,
                offset=self.rank_expert_offset,
                local=local,
                experts=num_experts,
            )
        self.context = dict(owner_active=owner_active, active=active, handle=handle)
        # the received count, at which the experts' output sum stops
        meta = ExpertTokensMetadata(None, None, handle.psum_num_recv_tokens_per_scaleup_rank)
        return values, None, meta, ids, weights

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
        if context["active"].numel() == 0 and code.shape == (0, block.readers.shape[-1]):
            code = code.new_empty((0, block.factors.shape[1]))
        # Inactive rows need no clearing. DeepEP's combine reads only received rows, and a
        # reference layer's sum writes zeros for rows with no slot on this rank.
        if block.reference is None:
            cache = context["cache"]
            combined = self.buffer.reduce_scatter(code, cache.group, cache.capacity, block.fp8)
            combined = combined[: context["rows"]]
        else:
            combined, _, _ = self.buffer.combine(
                code, handle=context["handle"], topk_weights=None, async_with_compute_stream=False
            )
        decode_into(output, combined, block.decoder, context["owner_active"])

    def prepare_async(self, *args, **kwargs):
        result = self.prepare(*args, **kwargs)
        return lambda: result

    def finalize_async(self, *args, **kwargs):
        self.finalize(*args, **kwargs)
        return lambda: None


class FoldedTritonExperts(TritonExperts):
    """vLLM's Triton experts at the folded widths: the readers take `H + r_d` columns and the
    factors write `r_c`, so the output workspace is `r_c` wide."""

    def moe_problem_size(self, a1, w1, w2, topk_ids):
        experts, rows, gate_up, _, topk = super().moe_problem_size(a1, w1, w2, topk_ids)
        return experts, rows, gate_up, w2.shape[1], topk


class FoldedMoE(nn.Module):
    """One folded layer on this rank: vLLM's router, this rank's readers and factors, the encoder
    and gain of a consuming layer, and the owner's decoder. `fp8` for FP8 activations."""

    def __init__(
        self,
        *,
        layer_index,
        export,
        execution,
        rank,
        world,
        buffer,
        top_k,
        renormalize,
        fp8,
        prefix,
    ):
        super().__init__()
        folded = export.folded
        self.index = layer_index
        self.fp8 = fp8
        self.top_k = top_k
        self.num_experts = folded["num_experts"]
        self.local_experts = self.num_experts // world
        self.expert_offset = rank * self.local_experts
        self.reference = folded["reference_layer_for"].get(str(layer_index))
        self.execution = execution
        device = execution.cache.values.device
        with torch.device(device):
            self.gate = ReplicatedLinear(
                folded["hidden_size"],
                self.num_experts,
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=None,
                prefix=f"{prefix}.gate",
            )
        self.router = create_fused_moe_router(
            top_k=top_k, global_num_experts=self.num_experts, renormalize=renormalize
        )
        tensors = export.layer(
            layer_index, range(self.expert_offset, self.expert_offset + self.local_experts)
        )
        for name in ("readers", "factors", "encoder", "gain", "decoder"):
            value = tensors[name]
            self.register_buffer(name, None if value is None else value.to(device).clone())
        expert_map = torch.full((self.num_experts,), -1, device=device, dtype=torch.int32)
        expert_map[self.expert_offset : self.expert_offset + self.local_experts] = torch.arange(
            self.local_experts, device=device, dtype=torch.int32
        )
        self.register_buffer("expert_map", expert_map)
        if self.reference is not None:
            assert self.encoder.shape[0] in FUSED_WIDTHS, self.encoder.shape
            assert folded["hidden_size"] == FUSED_HIDDEN, folded["hidden_size"]

        # vLLM's modular MoE kernel: this layer's transport around its Triton experts
        adapter = FoldedPrepareAndFinalize(
            self, buffer=buffer, rank=rank, world=world, top_k=top_k
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
        config = FusedMoEConfig(
            num_experts=self.num_experts,
            experts_per_token=top_k,
            hidden_dim=self.readers.shape[-1],
            intermediate_size=self.factors.shape[-1],
            num_local_experts=self.local_experts,
            num_logical_experts=self.num_experts,
            activation=MoEActivation.SILU,
            device=self.readers.device,
            routing_method=self.router.routing_method_type,
            moe_parallel_config=parallel,
            in_dtype=torch.bfloat16,
            max_num_tokens=execution.cache.capacity,
        )
        experts = FoldedTritonExperts(config, FUSED_MOE_UNQUANTIZED_CONFIG)
        self.kernel = FusedMoEKernel(adapter, experts)

    def forward(self, hidden_states):
        # Routing is row-independent, so rows past the live count route from the unmasked input.
        # The transport masks them.
        if hidden_states.shape[0]:
            logits, _ = self.gate(hidden_states)
            weights, ids = self.router.select_experts(
                hidden_states, logits, topk_indices_dtype=self.topk_indices_dtype
            )
        else:
            weights = torch.empty(
                (0, self.top_k), device=hidden_states.device, dtype=torch.float32
            )
            ids = torch.empty(
                (0, self.top_k), device=hidden_states.device, dtype=self.topk_indices_dtype
            )
        return self.kernel.apply(
            hidden_states=hidden_states,
            w1=self.readers,
            w2=self.factors,
            topk_weights=weights,
            topk_ids=ids,
            activation=MoEActivation.SILU,
            global_num_experts=self.num_experts,
            expert_map=self.expert_map,
            apply_router_weight_on_input=False,
        )


class FoldedDecoderLayer(Qwen3MoeDecoderLayer):
    """The base decoder layer with a folded MoE block. The parent's constructor would allocate
    the routed experts the block replaces, so it is not called."""

    def __init__(self, *, vllm_config, prefix="", block):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_parameters=config.rope_parameters,
            max_position_embeddings=getattr(config, "max_position_embeddings", 8192),
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=vllm_config.cache_config,
            quant_config=None,
            prefix=f"{prefix}.self_attn",
            dual_chunk_attention_config=getattr(config, "dual_chunk_attention_config", None),
        )
        self.mlp = block(layer_index=extract_layer_index(prefix), prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen3MoeFoldedEPForCausalLM(nn.Module):
    """The folded student, expert parallel over vLLM's expert-parallel group."""

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        export = FoldedExport(vllm_config.model_config.model)
        folded = export.folded
        assert config.decoder_sparse_step == 1, "every layer is MoE"
        assert not config.mlp_only_layers, "every layer is MoE"
        assert folded["num_experts"] % vllm_config.parallel_config.data_parallel_size == 0

        # vLLM's DeepEP v2 buffer, sized for the wider of the two codes
        group = get_ep_group()
        capacity = vllm_config.scheduler_config.max_num_batched_tokens
        buffer = group.device_communicator.all2all_manager.get_handle(
            dict(
                num_max_tokens_per_rank=capacity,
                hidden=max(folded["dispatch_rank"], folded["combine_rank"]),
                num_topk=config.num_experts_per_tok,
                num_experts=folded["num_experts"],
                use_fp8_dispatch=False,
            )
        )

        # the reference caches, then the layers, each with this rank's experts
        self.execution = Execution(
            capacity,
            folded["hidden_size"],
            device=vllm_config.device_config.device,
            group=group.device_group,
        )
        block = partial(
            FoldedMoE,
            export=export,
            execution=self.execution,
            rank=group.rank_in_group,
            world=group.world_size,
            buffer=buffer,
            top_k=config.num_experts_per_tok,
            renormalize=config.norm_topk_prob,
            fp8=vllm_config.load_config.model_loader_extra_config["fp8"],
        )
        self.model = Qwen3MoeModel(
            vllm_config=vllm_config,
            prefix="model",
            decoder_layer_type=partial(FoldedDecoderLayer, block=block),
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size, quant_config=None, prefix="lm_head"
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

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
        with torch.no_grad():
            self.execution.begin(count)
            return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states):
        with torch.no_grad():
            return self.logits_processor(self.lm_head, hidden_states)

    def embed_input_ids(self, input_ids):
        with torch.no_grad():
            return self.model.embed_input_ids(input_ids)

    def load_weights(self, weights):
        return AutoWeightsLoader(self).load_weights(weights)


class FoldedLoader(BaseModelLoader):
    """Load the export's backbone by name. Each folded block read its experts when it was built."""

    def download_model(self, model_config):
        pass

    def load_weights(self, model, model_config):
        loaded = model.load_weights(FoldedExport(model_config.model).backbone())
        missing = set(dict(model.named_parameters())) - loaded
        assert not missing, f"the export has no weights for {sorted(missing)}"
