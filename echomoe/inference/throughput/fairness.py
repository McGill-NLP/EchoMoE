"""Patches to stock vLLM 0.29 that both variants run, so the comparison isolates the folded
student: the per-step data-parallel syncs over NCCL instead of gloo, expert work only on the slots
a rank computes (Qwen3-30B-A3B's Triton experts with searched tiles, the larger models'
SiLU-and-mul), and, for the base model, the same route compaction the student's experts see and
(DeepSeek-V3, Kimi-K2.6) the same masking of padding rows' routes. Kanana-2 and Gemma 4 run their
shared experts and dense MLP on a second stream. Each changes only work that no output depends on,
or where it runs, except the tiles, which set the expert GEMMs' block sizes in both variants. A
process installs them once.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path

import torch
from vllm import _custom_ops as ops, forward_context
from vllm.config.parallel import ParallelConfig
from vllm.distributed.device_communicators.all2all import DeepEPV2All2AllManager
from vllm.distributed.parallel_state import get_dp_group
from vllm.model_executor.layers import fused_moe
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts import triton_moe
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExpertsBase
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_v2 import (
    DeepEPV2PrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import GroupedTopKRouter
from vllm.model_executor.models import gemma4
from vllm.utils.torch_utils import aux_stream
from vllm.v1.worker.gpu import dp_utils

from .kernels import (
    align,
    compact_routes,
    compact_routes_tiled,
    fp8_cast,
    live_rows,
    silu_and_mul_live,
    silu_and_mul_sorted,
    sum_live,
)


def install_dp_sync() -> None:
    """Run vLLM's two per-step data-parallel all-reduces on the NCCL group.

    vLLM's V2 model runner all-reduces each step's padding and graph shape (int32 SUM), and the
    offline loop asks whether any rank still has requests (int32 MAX), both on the gloo group.
    Across our nodes gloo costs 20-48 ms a step. Integer sums and maxima are exact, so every rank
    gets the values gloo would have produced.
    """
    original = dp_utils.dist

    class Proxy:
        def __getattr__(self, name):
            return getattr(original, name)

        def all_reduce(self, tensor, group=None, op=None, async_op=False):
            data_parallel = get_dp_group()
            exact = (
                group is data_parallel.cpu_group
                and not async_op
                and tensor.device.type == "cpu"
                and tensor.dtype == torch.int32
                and (op is None or op == original.ReduceOp.SUM)
                and not (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
            )
            if not exact:
                return original.all_reduce(
                    tensor, group=group, async_op=async_op, **({} if op is None else dict(op=op))
                )
            device = tensor.to(data_parallel.device)
            original.all_reduce(device, group=data_parallel.device_group)
            tensor.copy_(device.cpu())
            return None

    original_unfinished = ParallelConfig.__dict__["has_unfinished_dp"]

    def has_unfinished_dp(dp_group, has_unfinished):
        data_parallel = get_dp_group()
        if dp_group is not data_parallel.cpu_group or (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        ):
            return original_unfinished.__func__(dp_group, has_unfinished)
        tensor = torch.tensor([has_unfinished], dtype=torch.int32, device=data_parallel.device)
        original.all_reduce(tensor, op=original.ReduceOp.MAX, group=data_parallel.device_group)
        return bool(tensor.item())

    dp_utils.dist = Proxy()
    ParallelConfig.has_unfinished_dp = staticmethod(has_unfinished_dp)


def install_live_activation() -> None:
    """Apply SiLU-and-mul only to the (token, slot) rows whose expert is on this rank.

    vLLM's Triton experts activate every slot of every received row, though under expert
    parallelism most slots name another rank's experts, whose rows the grouped GEMMs neither write
    nor read. The kernel reproduces vLLM's CUDA kernel's roundings.
    """
    original_apply, original_activation = TritonExperts.apply, TritonExperts.activation

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
        *args,
        **kwargs,
    ):
        self._live_slots = None if expert_map is None else live_rows(topk_ids, expert_map)
        try:
            return original_apply(
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
                *args,
                **kwargs,
            )
        finally:
            self._live_slots = None

    def activation_fn(self, activation, output, input, **kwargs):
        live = getattr(self, "_live_slots", None)
        if (
            live is not None
            and activation == MoEActivation.SILU
            and self.activation_config.clamp_limit is None
            and input.dtype == torch.bfloat16
            and live.shape[0] == input.shape[0]
        ):
            return silu_and_mul_live(output, input, live)
        return original_activation(self, activation, output, input, **kwargs)

    # Marlin (Kimi-K2.6's INT4 experts) passes the routes and the expert map to its activation.
    original_marlin = MarlinExpertsBase.activation

    def marlin_activation(
        self, activation, output, input, *, topk_ids=None, expert_map=None, **kwargs
    ):
        live = None if topk_ids is None or expert_map is None else live_rows(topk_ids, expert_map)
        if (
            live is not None
            and activation == MoEActivation.SILU
            and self.activation_config.clamp_limit is None
            and input.dtype == torch.bfloat16
            and live.shape[0] == input.shape[0]
        ):
            return silu_and_mul_live(output, input, live)
        return original_marlin(
            self, activation, output, input, topk_ids=topk_ids, expert_map=expert_map, **kwargs
        )

    TritonExperts.apply, TritonExperts.activation = apply, activation_fn
    MarlinExpertsBase.activation = marlin_activation


def install_live_experts(tiles) -> None:
    """Run Qwen3-30B-A3B's Triton experts only on the (row, slot) pairs whose expert is on this
    rank, with the tile searched for each expert shape (`tiles`, from tiles.json).

    Most slots of DeepEP's worst-case receive buffer name another rank's expert or no row at all.
    The alignment lists only this rank's slots, SiLU-and-mul reads that list, and the sum stops at
    the received count when the prepare step reports DeepEP's prefix sum. The GEMMs take the tile
    searched at the nearest row count, the first in the file on a tie. An expert shape that was not
    searched keeps vLLM's tile.
    """
    original_prepare = triton_moe._prepare_expert_assignment
    original_apply, original_activation = TritonExperts.apply, TritonExperts.activation
    # this rank's expert count while `apply` runs, and the last alignment
    local = assignment = None

    def prepare(topk_ids, config, num_tokens, top_k_num, global_num_experts, expert_map, **kwargs):
        nonlocal assignment
        if expert_map is not None and local is not None:
            assignment = align(topk_ids, config["BLOCK_SIZE_M"], expert_map, local)
        else:
            kwargs["ignore_invalid_experts"] = True
            assignment = original_prepare(
                topk_ids, config, num_tokens, top_k_num, global_num_experts, expert_map, **kwargs
            )
        return assignment

    def activation_fn(self, activation, output, input, **kwargs):
        if (
            assignment is not None
            and assignment[0] is not None
            and activation == MoEActivation.SILU
            and self.activation_config.clamp_limit is None
            and input.dtype == torch.bfloat16
        ):
            return silu_and_mul_sorted(output, input, assignment[0], assignment[2])
        return original_activation(self, activation, output, input, **kwargs)

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
        nonlocal local
        psum = getattr(expert_tokens_meta, "psum_recv_per_rank", None)
        self._live = (topk_ids, expert_map, None if psum is None else psum[-1:])
        local = w1.shape[0]
        config = None
        table = tiles.get("x".join(map(str, w1.shape)) + "-" + "x".join(map(str, w2.shape)))
        if table is not None:
            rows = hidden_states.shape[0]
            config = dict(table[min(table, key=lambda searched: abs(int(searched) - rows))])
        try:
            with fused_moe.override_config(config) if config else nullcontext():
                return original_apply(
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
                )
        finally:
            self._live = None
            local = None

    def moe_sum(self, input, output):
        ids, expert_map, count = self._live
        if expert_map is None:
            ops.moe_sum(input, output)
        elif count is None:
            ops.moe_sum(input.view(ids.shape[0], ids.shape[1], -1), output, ids, expert_map)
        else:
            sum_live(input.view(ids.shape[0], ids.shape[1], -1), output, ids, expert_map, count)

    triton_moe._prepare_expert_assignment = prepare
    TritonExperts.apply, TritonExperts.moe_sum = apply, moe_sum
    TritonExperts.activation = activation_fn


def install_stock_decode_guard(*, tiled) -> None:
    """In the base model's graphed decode, compact each received row's routes to this rank's
    experts (8 slots to 4 at 32 ranks), as the student's receivers do, with the row-tiled kernel
    if `tiled` (Qwen3-30B-A3B). With as many local experts as slots, point empty slots at a
    foreign expert with weight 0."""
    original = DeepEPV2PrepareAndFinalize._receiver

    def receiver(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if not self.use_cudagraph:
            return result
        rows, scale, metadata, ids, weights = result
        local = self.num_experts // self.num_dispatchers_
        if local < ids.shape[1]:
            count, offset = metadata.psum_recv_per_rank[-1:], self.rank_expert_offset
            if tiled:
                ids, weights = compact_routes_tiled(
                    ids, weights, count, offset=offset, local=local
                )
            else:
                ids, weights = compact_routes(
                    ids,
                    weights,
                    count,
                    offset=offset,
                    local=local,
                    experts=self.num_experts,
                    global_ids=True,
                )
        else:
            # Rank 0 names the first expert of rank 1. Other ranks name expert 0.
            sentinel = local if self.rank_expert_offset == 0 else 0
            valid = ids >= 0
            ids, weights = torch.where(valid, ids, sentinel), torch.where(valid, weights, 0)
        return rows, scale, metadata, ids, weights

    DeepEPV2PrepareAndFinalize._receiver = receiver


def install_fp8_dispatch(cast_back) -> None:
    """Send the base model's dispatch in FP8, cast per token as DeepEP casts, and cast the received
    rows back to BF16 with `cast_back` (kernels.fp8_cast_back, or for a larger model's rows
    portable_kernels.fp8_cast_back_blocks) for its BF16 or INT4 experts. The combine is as DeepEP's
    package does it (the FP8 copy crosses nodes in FP8)."""
    original_dispatch = DeepEPV2PrepareAndFinalize._do_dispatch
    original_receiver = DeepEPV2PrepareAndFinalize._receiver

    def do_dispatch(self, tokens, token_scales, *args, **kwargs):
        if token_scales is None and tokens.dtype == torch.bfloat16:
            tokens, token_scales = fp8_cast(tokens)
        return original_dispatch(self, tokens, token_scales, *args, **kwargs)

    def receiver(
        self,
        event,
        has_scales,
        recv_x,
        recv_topk_idx,
        num_experts,
        recv_expert_num_tokens,
        recv_topk_weights,
        psum_recv_per_rank,
        a1_scale,
        quant_config,
        **kwargs,
    ):
        if (
            isinstance(recv_x, tuple)
            and recv_x[0].dtype == torch.float8_e4m3fn
            and quant_config.quant_dtype is None
        ):
            if event.event is not None:
                event.current_stream_wait()
            recv_x = cast_back(*recv_x, psum_recv_per_rank[-1:])
            has_scales = False
        return original_receiver(
            self,
            event,
            has_scales,
            recv_x,
            recv_topk_idx,
            num_experts,
            recv_expert_num_tokens,
            recv_topk_weights,
            psum_recv_per_rank,
            a1_scale,
            quant_config,
            **kwargs,
        )

    DeepEPV2PrepareAndFinalize._do_dispatch = do_dispatch
    DeepEPV2PrepareAndFinalize._receiver = receiver


def install_bf16_dispatch() -> None:
    """Send the base model's dispatch in BF16 though its experts take FP8 (DeepSeek-V3): they
    quantize the received rows, through the deferred path vLLM keeps for LoRA (without its LoRA
    kernels), and DeepEP's buffers are sized for BF16 rows."""
    TritonExperts.expects_unquantized_inputs = property(lambda self: self.quant_dtype is not None)
    make_kwargs = DeepEPV2All2AllManager._make_all2all_kwargs

    def all2all_kwargs(self, *args, **kwargs):
        return {**make_kwargs(self, *args, **kwargs), "use_fp8_dispatch": False}

    DeepEPV2All2AllManager._make_all2all_kwargs = all2all_kwargs


def install_padding_route_mask() -> None:
    """Give the data-parallel padding rows no routes in vLLM's grouped top-k router (DeepSeek-V3,
    Kimi-K2.6), as vLLM's fused top-k routers and the folded student do: route -1, weight 0."""
    original = GroupedTopKRouter._compute_routing

    def compute_routing(self, hidden_states, router_logits, indices_type, *args, **kwargs):
        weights, ids = original(self, hidden_states, router_logits, indices_type, *args, **kwargs)
        mask = None
        if forward_context.is_forward_context_available():
            mask = getattr(forward_context.get_forward_context(), "is_padding", None)
        if mask is None:
            return weights, ids
        padding = mask[: ids.shape[0]].unsqueeze(1)
        return (
            torch.where(padding, torch.zeros_like(weights), weights),
            torch.where(padding, torch.full_like(ids, -1), ids),
        )

    GroupedTopKRouter._compute_routing = compute_routing


def on_side_stream(fn, *args):
    """fn(*args) on vLLM's auxiliary stream after the work queued so far: its output and the event
    to wait for."""
    stream = aux_stream()
    ready = torch.cuda.Event()
    ready.record()
    with torch.cuda.stream(stream):
        stream.wait_event(ready)
        output = fn(*args)
        done = torch.cuda.Event()
        done.record(stream)
    return output, done


def install_second_stream() -> None:
    """Run the shared experts and Gemma 4's dense MLP on vLLM's auxiliary stream, beside the routed
    experts' dispatch, experts and combine. The base model's shared experts take vLLM's own
    multi-stream path at every batch size: with DeepEP v2 vLLM otherwise overlaps them on one stream.
    The folded student's run on it with `second_stream` (portable.py). The same kernels run in the
    same order, so outputs are unchanged."""
    stock_forward = gemma4.Gemma4DecoderLayer.forward

    def gemma4_layer_forward(
        self, positions, hidden_states, residual, per_layer_input=None, **kwargs
    ):
        """vLLM's Gemma4DecoderLayer.forward with the dense MLP branch on the auxiliary stream
        (MoE layers without per-layer inputs, as Gemma 4 26B-A4B's)."""
        if not self.enable_moe_block or (
            per_layer_input is not None and self.per_layer_input_gate is not None
        ):
            return stock_forward(
                self, positions, hidden_states, residual, per_layer_input, **kwargs
            )
        residual = hidden_states
        hidden_states = self.input_layernorm(residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states, **kwargs)
        hidden_states = self.post_attention_layernorm(hidden_states)
        residual = hidden_states + residual

        def dense(residual_input):
            return self.post_feedforward_layernorm_1(
                self.mlp(self.pre_feedforward_layernorm(residual_input))
            )

        hidden_states_1, done = on_side_stream(dense, residual)
        hidden_states_2 = self.pre_feedforward_layernorm_2(residual)
        hidden_states_2 = self.moe(hidden_states_2, self.router(residual))
        hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)
        torch.cuda.current_stream().wait_event(done)
        hidden_states = self.post_feedforward_layernorm(hidden_states_1 + hidden_states_2)
        hidden_states = hidden_states + residual
        return hidden_states * self.layer_scalar, None

    gemma4.Gemma4DecoderLayer.forward = gemma4_layer_forward
    FusedMoEKernel.can_overlap_shared_experts = property(lambda self: False)
    # read when the engine is built
    os.environ["VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD"] = str(1 << 30)


def install_nic_rate():
    """DeepEP's hybrid mode splits SMs by the NIC rate it reads with `ibstat`, which our
    containers lack. Read the rate of the ports NCCL_IB_HCA allows from sysfs instead."""
    from deep_ep.buffers import elastic

    value = os.environ["NCCL_IB_HCA"]
    assert not value.startswith("^"), f"list the NICs to use in NCCL_IB_HCA, not {value}"
    nics = [part.split(":")[0] for part in value.lstrip("=").split(",")]
    sysfs = Path("/sys/class/infiniband")
    rates = {float((sysfs / nic / "ports/1/rate").read_text().split()[0]) / 8 for nic in nics}
    assert len(rates) == 1, f"the NICs' rates differ: {rates}"
    rate = rates.pop()
    elastic.get_rdma_gbs = lambda *_args, **_kwargs: rate
