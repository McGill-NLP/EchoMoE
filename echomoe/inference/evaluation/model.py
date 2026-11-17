"""The folded student and the full-width baseline as vLLM 0.21 model classes, and the plugin that
registers them. The baseline (`base-blocks`) serves the base model through the same blocks, kernels
and transport at full width, so its scores and bytes compare with the student's like for like.

A folded student's experts read `[z, c_d]` through per-expert folded input projections W_in,i
(`readers`) and write the combine code through folded output projections W_out,i (`factors`, see
`echomoe.codec`). That fits neither vLLM's fused MoE layer nor its weight layout, so each MoE block
is replaced outright. The block still runs vLLM's routing and its grouped expert GEMM, with the
readers standing in for the stacked gate/up weight at a wider `K` and the factors for the down
projection at a narrower `N`. Everything the forward branches on is fixed by the export's folded
config when the block is built, so the forward is static and is captured as a CUDA graph.

The LSH-MoE baseline (`lsh`) serves its export through the full-width block, eagerly and without
expert parallelism, with each rank's batch clustered before the experts and the learned alpha's
offsets added after them (`echomoe.lsh`).

A DeepSeek-V3 student (Kanana-2, Moonlight) is vLLM's own DeepSeek-V3 model with the folded block
in each MoE layer, beside vLLM's gate and shared experts. Its leading dense layers are vLLM's. A
Nemotron-H student is vLLM's own Nemotron-H model, its Mamba and attention layers included, with
the same block in each MoE mixer. A Gemma 4 student is vLLM's own Gemma 4 model with the folded
block in place of its routed experts. The router, the dense MLP and the norms stay vLLM's.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional
from safetensors.torch import load_file
from vllm import ModelRegistry, _custom_ops as ops
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_dp_group, get_ep_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe import FusedMoE, GateLinear
from vllm.model_executor.layers.fused_moe.activation import MoEActivation, apply_moe_activation
from vllm.model_executor.layers.fused_moe.fused_moe import (
    _get_config_dtype_str,
    _prepare_expert_assignment,
    dispatch_fused_moe_kernel,
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe.router.router_factory import create_fused_moe_router
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models import (
    deepseek_v2,
    gemma4 as vllm_gemma4,
    nemotron_h as vllm_nemotron_h,
)
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
)
from vllm.model_executor.models.utils import extract_layer_index, maybe_prefix
from vllm.triton_utils import tl

from echomoe.codec import communication_precision
from echomoe.lsh import (
    ALPHA_FILE,
    LshConfig,
    compensate_pairs,
    node_rows,
    residual_weights,
    rotations,
    served_centroids,
)

from .transport import (
    ExpertParallelTransport,
    expert_parallel_transport,
    fp8_fake_quant,
    local_expert_map,
    local_ids,
    reference_slots,
)

FOLDED_ARCHITECTURE = "Qwen3MoeFoldedForCausalLM"
DEEPSEEK_V3_FOLDED_ARCHITECTURE = "DeepseekV3FoldedForCausalLM"
GEMMA4_FOLDED_ARCHITECTURE = "Gemma4FoldedForCausalLM"
NEMOTRON_H_FOLDED_ARCHITECTURE = "NemotronHFoldedForCausalLM"
FULL_WIDTH_ARCHITECTURE = "Qwen3MoeFullWidthForCausalLM"
# The export's `config.json` keys holding the folded student's config and LSH-MoE's settings, and
# the `--additional-config` key naming the simulated precision
# (`echomoe.codec.COMMUNICATION_PRECISIONS`).
FOLDED_KEY = "folded"
LSH_KEY = "lsh"
PRECISION_KEY = "communication_precision"


def register() -> None:
    """The vLLM plugin (`vllm.general_plugins`), run in every vLLM process."""
    # NCCL registers the buffers a CUDA graph's sends and receives capture (NCCL_GRAPH_REGISTER,
    # on by default). The FP8 combine's exchange captures views of one allocation at a different
    # size in every graph, and with registration on a few percent of tokens arrive wrong under
    # multi-size graphs. NCCL reads the variable when a communicator first captures.
    os.environ["NCCL_GRAPH_REGISTER"] = "0"
    # Named by string, as vLLM recommends, so the classes import where they are built.
    ModelRegistry.register_model(FOLDED_ARCHITECTURE, f"{__name__}:Qwen3MoeFoldedForCausalLM")
    ModelRegistry.register_model(
        DEEPSEEK_V3_FOLDED_ARCHITECTURE, f"{__name__}:DeepseekV3FoldedForCausalLM"
    )
    ModelRegistry.register_model(GEMMA4_FOLDED_ARCHITECTURE, f"{__name__}:Gemma4FoldedForCausalLM")
    ModelRegistry.register_model(
        NEMOTRON_H_FOLDED_ARCHITECTURE, f"{__name__}:NemotronHFoldedForCausalLM"
    )
    ModelRegistry.register_model(
        FULL_WIDTH_ARCHITECTURE, f"{__name__}:Qwen3MoeFullWidthForCausalLM"
    )


@dataclass(frozen=True)
class FoldedConfig:
    """The export's `folded` document. A code width (`dispatch_rank`, `combine_rank`) of None
    switches that side off, `combine_offsets` marks bias or gated bias decoding (`bias`,
    `biasgate`) and `correction_width` is the gate width of `biasgate`."""

    hidden_size: int
    moe_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    num_hidden_layers: int
    dispatch_rank: int | None
    combine_rank: int | None
    combine_offsets: bool
    correction_width: int | None
    correction_epsilon: float
    reference_layers: tuple[int, ...]
    reference_layer_for: dict[int, int]

    @classmethod
    def parse(cls, document: dict[str, Any]) -> FoldedConfig:
        fields = dict(document)
        # Coded references are not part of the method, and every export stores None.
        assert fields.pop("reference_rank") is None, "coded references are not served"
        fields["reference_layers"] = tuple(fields["reference_layers"])
        fields["reference_layer_for"] = {
            int(layer): int(reference)
            for layer, reference in fields["reference_layer_for"].items()
        }
        return cls(**fields)

    def reader_width(self, layer: int) -> int:
        if self.dispatch_rank is None or layer in self.reference_layers:
            return self.hidden_size
        if layer in self.reference_layer_for:
            return self.hidden_size + self.dispatch_rank
        return self.dispatch_rank

    def releases_reference(self, layer: int) -> bool:
        source = self.reference_layer_for[layer]
        return layer == max(
            consumer
            for consumer, reference in self.reference_layer_for.items()
            if reference == source
        )


def served_precision(vllm_config: VllmConfig) -> tuple[bool, bool]:
    name = (vllm_config.additional_config or {}).get(PRECISION_KEY, "bf16")
    precision = communication_precision(name)
    if precision is None:
        return False, False
    return precision.dispatch == "fp8", precision.combine == "fp8"


def fused_expert_codes(
    rows: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    readers: torch.Tensor,
    factors: torch.Tensor,
    *,
    expert_map: torch.Tensor | None,
    pairs: bool = False,
    activation: MoEActivation = MoEActivation.SILU,
) -> torch.Tensor:
    """The summed combine code `c_c = sum_i w_i W_out,i act(W_in,i rows)`, by vLLM's grouped GEMM.
    A gated activation reads the gate half and the up half of the first GEMM's output.
    `fused_experts_impl` is not called only because it sizes its output as its input. With
    `pairs`, `rows` holds one row per (token, slot) pair, token-major, and each pair's expert reads
    its own row."""
    # The kernel scales each expert's output by its routing weight in float32 and rounds once, as
    # the trainer's fp32 router feeds the grouped GEMM.
    topk_weights = topk_weights.float()
    tokens, top_k = topk_ids.shape
    gated_width = readers.shape[1]
    combine_rank = factors.shape[1]
    if tokens == 0:
        return rows.new_zeros((0, combine_rank))
    compute_type = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}[rows.dtype]
    config = try_get_optimal_moe_config(
        readers.size(),
        factors.size(),
        top_k,
        _get_config_dtype_str(
            use_fp8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False, dtype=rows.dtype
        ),
        tokens,
        block_shape=None,
    )
    # The first GEMM's output and the second's share one buffer, as in `fused_experts_impl`.
    shared = torch.empty(
        tokens * top_k * max(gated_width, combine_rank), device=rows.device, dtype=rows.dtype
    )
    gate_up_rows = shared[: tokens * top_k * gated_width].view(tokens, top_k, gated_width)
    code_rows = shared[: tokens * top_k * combine_rank].view(tokens, top_k, combine_rank)
    activated = torch.empty(
        (tokens * top_k, gated_width // 2 if activation.is_gated else gated_width),
        device=rows.device,
        dtype=rows.dtype,
    )
    sorted_token_ids, expert_ids, num_tokens_post_padded = _prepare_expert_assignment(
        topk_ids,
        config,
        tokens,
        top_k,
        readers.shape[0] if expert_map is None else expert_map.shape[0],
        expert_map,
        block_shape=None,
        ignore_invalid_experts=True,
    )
    quantization = {
        "use_fp8_w8a8": False,
        "use_int8_w8a8": False,
        "use_int8_w8a16": False,
        "use_int4_w4a16": False,
        "per_channel_quant": False,
        "block_shape": None,
    }
    dispatch_fused_moe_kernel(
        rows,
        readers,
        gate_up_rows,
        None,
        None,
        None,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        False,
        1 if pairs else top_k,  # the kernel reads row `pair // top_k`
        config,
        compute_type=compute_type,
        **quantization,
    )
    apply_moe_activation(activation, activated, gate_up_rows.view(-1, gated_width))
    if expert_map is not None:
        # An unrouted expert leaves its rows untouched, and this buffer was not zeroed.
        code_rows.zero_()
    dispatch_fused_moe_kernel(
        activated,
        factors,
        code_rows,
        None,
        None,
        None,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        True,
        1,
        config,
        compute_type=compute_type,
        **quantization,
    )
    code = rows.new_empty((tokens, combine_rank))
    ops.moe_sum(code_rows, code)
    return code


def _frozen(*shape: int) -> nn.Parameter:
    return nn.Parameter(torch.empty(shape), requires_grad=False)


def _load_local_experts(offset: int, parameter: torch.Tensor, loaded: torch.Tensor) -> None:
    parameter.data.copy_(loaded[offset : offset + parameter.shape[0]])


class FoldedCorrection(nn.Module):
    """The owner's term `sum_i w_i (1 + alpha_i) o_i` (`echomoe.codec.OwnerCorrection`) in the
    folded layout. Bias decoding (`bias`) holds the offsets alone and adds `sum_i w_i o_i`."""

    def __init__(self, folded: FoldedConfig) -> None:
        super().__init__()
        self.epsilon = folded.correction_epsilon
        experts, hidden = folded.num_experts, folded.hidden_size
        self.offset = _frozen(experts, hidden)
        # The code RMS the trainer measured on its first micro-batch and froze. vLLM's loader
        # diverts any name holding "scale" into kv-scale remapping.
        self.code_rms = nn.Parameter(torch.empty(()), requires_grad=False)
        self.width = folded.correction_width
        if self.width is not None:
            self.input_map = _frozen(self.width, hidden)
            self.code_map = _frozen(self.width, folded.combine_rank)
            self.expert_embedding = _frozen(experts, self.width)
            self.readout = _frozen(experts, self.width)

    def forward(
        self,
        moe_input: torch.Tensor,
        code: torch.Tensor,
        decoded: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.width is None:
            return decoded + weights @ self.offset
        inverse_rms = torch.rsqrt(moe_input.square().mean(-1, keepdim=True) + self.epsilon)
        gate_features = functional.silu(
            functional.linear(moe_input * inverse_rms, self.input_map)
            + functional.linear(code / self.code_rms, self.code_map)
            + weights @ self.expert_embedding
        )
        alpha = gate_features @ self.readout.t()
        return decoded + (weights * (1.0 + alpha)) @ self.offset


class FoldedSparseMoeBlock(nn.Module):
    """A folded layer. A reference layer, or any with the dispatch side off, reads `x_l` through
    the trained gate/up weight in `readers`. A non-reference layer reads `[z, c_d]` and others
    their code."""

    expert_map: torch.Tensor

    def __init__(
        self,
        *,
        folded: FoldedConfig,
        layer_index: int,
        references: dict[int, torch.Tensor],
        renormalize: bool,
        transport: ExpertParallelTransport | None,
        fp8_dispatch: bool,
        fp8_combine: bool,
        quant_config: Any = None,
        prefix: str = "",
        activation: MoEActivation = MoEActivation.SILU,
    ) -> None:
        super().__init__()
        self.activation = activation
        # An FP8 combine is an exchange of partial sums between ranks, and one engine has none.
        assert not (fp8_combine and transport is None), "an FP8 combine needs expert parallelism"
        self.layer_index = layer_index
        self.references = references
        self.transport = transport
        self.fp8_dispatch = fp8_dispatch
        self.hidden_size = folded.hidden_size
        self.num_experts = folded.num_experts
        self.local_experts = folded.num_experts if transport is None else transport.local_experts
        self.dispatch_rank = folded.dispatch_rank
        self.combine_rank = folded.combine_rank
        self.publishes_reference = layer_index in folded.reference_layers
        self.reference_source = folded.reference_layer_for.get(layer_index)
        self.releases_reference = self.reference_source is not None and (
            folded.releases_reference(layer_index)
        )
        self.gate = ReplicatedLinear(
            folded.hidden_size,
            folded.num_experts,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate",
        )
        self.router = create_fused_moe_router(
            top_k=folded.num_experts_per_tok,
            global_num_experts=folded.num_experts,
            renormalize=renormalize,
        )
        width = folded.moe_intermediate_size
        written = folded.hidden_size if folded.combine_rank is None else folded.combine_rank
        # a gated expert reads through [gate ; up], a non-gated one (Nemotron-H) through up alone
        rows = 2 * width if activation.is_gated else width
        self.readers = _frozen(self.local_experts, rows, folded.reader_width(layer_index))
        self.factors = _frozen(self.local_experts, written, width)
        if transport is not None:
            loader = partial(_load_local_experts, transport.expert_offset)
            self.readers.weight_loader = loader
            self.factors.weight_loader = loader
            self.register_buffer(
                "expert_map", local_expert_map(self.local_experts, None), persistent=False
            )
        if folded.combine_rank is not None:
            self.decoder = _frozen(folded.hidden_size, folded.combine_rank)
        if folded.dispatch_rank is not None and not self.publishes_reference:
            self.encoder = _frozen(folded.dispatch_rank, folded.hidden_size)
        if self.reference_source is not None:
            self.reference_gain = _frozen(folded.hidden_size)
        # Direct decoding (`direct`) decompresses alone and holds no correction.
        self.correction = FoldedCorrection(folded) if folded.combine_offsets else None

    def dispatch_rows(self, moe_input: torch.Tensor) -> torch.Tensor:
        # One engine simulates FP8 where the rows would cross between ranks.
        if self.dispatch_rank is None:
            return fp8_fake_quant(moe_input) if self.fp8_dispatch else moe_input
        reference, code = self.dispatch_parts(moe_input)
        if code is None:
            return self.references[self.layer_index]
        if self.fp8_dispatch:
            code = fp8_fake_quant(code)
        return code if reference is None else torch.cat((reference, code), dim=-1)

    def dispatch_parts(
        self, moe_input: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.layer_index == 0:
            self.references.clear()
        if self.publishes_reference:
            # The decoder layer writes the block's output into its input's storage, so a
            # reference read layers later is copied, not aliased. The store keeps the reference
            # as its receivers read it.
            published = moe_input.clone()
            self.references[self.layer_index] = (
                fp8_fake_quant(published) if self.fp8_dispatch else published
            )
            return published, None
        if self.reference_source is None:
            return None, functional.linear(moe_input, self.encoder)
        reference = self.references[self.reference_source]
        code = functional.linear(moe_input - self.reference_gain * reference, self.encoder)
        if self.releases_reference:
            del self.references[self.reference_source]
        return reference, code

    def owner_output(
        self, moe_input: torch.Tensor, code: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        if self.combine_rank is None:
            return code
        decoded = functional.linear(code, self.decoder)
        if self.correction is None:
            return decoded
        return self.correction(moe_input, code, decoded, weights)

    def expert_parallel_output(
        self,
        moe_input: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        transport = self.transport
        transport.begin(self.layer_index, moe_input)
        # a reference layer publishes and dispatches nothing, a non-reference layer dispatches
        # its code, and
        # with the dispatch side off the rows themselves are the payload
        payload = moe_input
        if self.dispatch_rank is not None:
            published, code = self.dispatch_parts(moe_input)
            if code is None:
                transport.publish(self.layer_index, published)
                code = moe_input[:, :0]
            payload = code
        received, ids, received_weights, handle, valid = transport.dispatch(
            payload, topk_ids, topk_weights
        )
        # each receiver reads the published reference of every received row beside its code
        rows = received
        if self.publishes_reference:
            rows = transport.reference_rows(self.layer_index, handle, valid, moe_input.dtype)
        elif self.reference_source is not None:
            reference = transport.reference_rows(
                self.reference_source, handle, valid, moe_input.dtype
            )
            rows = torch.cat((reference, received), dim=-1)
        # the receiver's summed code over its local experts, combined back to the owner
        written = fused_expert_codes(
            rows,
            received_weights,
            local_ids(ids, self.local_experts),
            self.readers,
            self.factors,
            expert_map=self.expert_map,
            activation=self.activation,
        )
        summed = transport.combine(written, handle, valid, topk_ids)
        return self.owner_output(moe_input, summed, weights)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        moe_input = hidden_states.view(-1, self.hidden_size)
        logits, _ = self.gate(moe_input)
        topk_weights, topk_ids = self.router.select_experts(
            hidden_states=moe_input, router_logits=logits
        )
        return self.routed_output(moe_input, topk_weights, topk_ids).view(hidden_states.shape)

    def routed_output(
        self, moe_input: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor
    ) -> torch.Tensor:
        weights = topk_weights.to(moe_input.dtype)
        dense = torch.zeros(
            (moe_input.shape[0], self.num_experts), dtype=moe_input.dtype, device=moe_input.device
        )
        dense.scatter_(1, topk_ids.to(torch.int64), weights)
        if self.transport is not None:
            output = self.expert_parallel_output(moe_input, topk_weights, topk_ids, dense)
        else:
            code = fused_expert_codes(
                self.dispatch_rows(moe_input),
                topk_weights,
                topk_ids.to(torch.int32),
                self.readers,
                self.factors,
                expert_map=None,
                activation=self.activation,
            )
            output = self.owner_output(moe_input, code, dense)
        return output


class SigmoidFoldedMoeBlock(FoldedSparseMoeBlock):
    """A DeepSeek-V3 or Nemotron-H MoE layer: vLLM's gate with its correction bias and vLLM's
    shared experts beside the folded routed experts. The router picks the top-k of sigmoid score
    plus bias and gives the renormalized scores at scale 1. The summed routed output is then
    scaled in place in bf16 and the shared experts' output added, where stock vLLM applies the
    routed scale on CUDA. The trainer's router scales the weights instead, which rounds
    differently in bf16.
    Nemotron-H's gate projects in float32, and its experts are squared-ReLU and not gated."""

    def __init__(self, *, config: Any, **kwargs: Any) -> None:
        nemotron = config.model_type == "nemotron_h"
        super().__init__(
            renormalize=config.norm_topk_prob,
            activation=MoEActivation.RELU2_NO_MUL if nemotron else MoEActivation.SILU,
            **kwargs,
        )
        prefix = kwargs["prefix"]
        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            out_dtype=torch.float32 if nemotron else None,
            force_fp32_compute=nemotron,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32), requires_grad=False
        )
        self.router = create_fused_moe_router(
            top_k=config.num_experts_per_tok,
            global_num_experts=config.n_routed_experts,
            renormalize=config.norm_topk_prob,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            scoring_func="sigmoid" if nemotron else config.scoring_func,
            routed_scaling_factor=1.0,
            e_score_correction_bias=self.gate.e_score_correction_bias,
        )
        self.routed_scaling_factor = config.routed_scaling_factor
        if nemotron:
            self.shared_experts = vllm_nemotron_h.NemotronHMLP(
                config=config,
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_shared_expert_intermediate_size
                * config.n_shared_experts,
                quant_config=kwargs["quant_config"],
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = deepseek_v2.DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
                hidden_act=config.hidden_act,
                quant_config=kwargs["quant_config"],
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        moe_input = hidden_states.view(-1, self.hidden_size)
        logits, _ = self.gate(moe_input)
        topk_weights, topk_ids = self.router.select_experts(
            hidden_states=moe_input, router_logits=logits
        )
        routed = self.routed_output(moe_input, topk_weights, topk_ids)
        routed *= self.routed_scaling_factor
        return (self.shared_experts(moe_input) + routed).view(hidden_states.shape)


class Gemma4FoldedMoeBlock(FoldedSparseMoeBlock):
    """A Gemma 4 layer's routed experts. The decoder layer routes on its residual and passes the
    float32 logits, which the block routes on as vLLM's `Gemma4MoE` does: the top-k of the softmax,
    renormalized and times each expert's `per_expert_scale`. The experts are GELU-tanh gated."""

    def __init__(self, *, config: Any, **kwargs: Any) -> None:
        super().__init__(renormalize=True, activation=MoEActivation.GELU_TANH, **kwargs)
        del self.gate, self.router
        self.top_k = config.top_k_experts
        # vLLM's Gemma 4 loader moves `router.per_expert_scale` here
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts), requires_grad=False)

    def forward(  # type: ignore[override]
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor:
        moe_input = hidden_states.view(-1, self.hidden_size)
        topk_weights, topk_ids = vllm_gemma4.gemma4_fused_routing_kernel_triton(
            router_logits, self.top_k, self.per_expert_scale
        )
        return self.routed_output(moe_input, topk_weights, topk_ids).view(hidden_states.shape)


class FullWidthSparseMoeBlock(nn.Module):
    """The base model's sparse block on the folded block's expert kernel and transport. vLLM's
    fused MoE layer is not built, so its weight rewrite at load and backend choice do not apply.
    With `lsh` every (token, slot) pair's expert reads the centroid of the rank's pairs that share
    its (expert, bucket), and `alpha` (every layer's) scales each pair's offset from it. `counts`
    then holds the tokens, DeepEP rows and clusters of the served forwards by phase."""

    expert_map: torch.Tensor

    def __init__(
        self,
        *,
        config: Any,
        layer_index: int,
        transport: ExpertParallelTransport | None,
        fp8_dispatch: bool,
        fp8_combine: bool,
        lsh: LshConfig | None,
        alpha: torch.Tensor | None,
        quant_config: Any = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        assert not (fp8_combine and transport is None), "an FP8 combine needs expert parallelism"
        assert lsh is None or transport is None, "LSH-MoE clusters what a rank would send"
        self.layer_index = layer_index
        self.hidden_size = config.hidden_size
        self.transport = transport
        self.fp8_dispatch = fp8_dispatch
        self.local_experts = config.num_experts if transport is None else transport.local_experts
        self.num_experts = config.num_experts
        self.expert_offset = 0 if transport is None else transport.expert_offset
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate",
        )
        self.router = create_fused_moe_router(
            top_k=config.num_experts_per_tok,
            global_num_experts=config.num_experts,
            renormalize=config.norm_topk_prob,
        )
        width = config.moe_intermediate_size
        self.readers = _frozen(self.local_experts, 2 * width, config.hidden_size)
        self.factors = _frozen(self.local_experts, config.hidden_size, width)
        self.readers.weight_loader = self._load_expert
        self.factors.weight_loader = self._load_expert
        self.lsh = lsh
        if lsh is not None:
            device = self.readers.device
            projections = rotations(lsh, config.hidden_size).to(device)
            self.register_buffer("rotations", projections, persistent=False)
            self.register_buffer("alpha", alpha[layer_index].to(device), persistent=False)
            counts = torch.zeros(2, 3, dtype=torch.float64, device=device)
            self.register_buffer("counts", counts, persistent=False)
        if transport is not None:
            self.register_buffer(
                "expert_map", local_expert_map(self.local_experts, None), persistent=False
            )

    def _load_expert(
        self,
        parameter: torch.Tensor,
        loaded: torch.Tensor,
        _name: str,
        *,
        shard_id: str,
        expert_id: int,
        return_success: bool,
    ) -> bool:
        local = expert_id - self.expert_offset
        if not 0 <= local < self.local_experts:
            return False
        width = self.factors.shape[-1]
        target = {
            "w1": parameter.data[local, :width],
            "w3": parameter.data[local, width:],
            "w2": parameter.data[local],
        }[shard_id]
        target.copy_(loaded)
        return True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        moe_input = hidden_states.view(-1, self.hidden_size)
        logits, _ = self.gate(moe_input)
        topk_weights, topk_ids = self.router.select_experts(
            hidden_states=moe_input, router_logits=logits
        )
        transport = self.transport
        if self.lsh is not None:
            real, phase = served_tokens(moe_input.shape[0])
            centroids, clusters = served_centroids(
                moe_input, topk_ids, self.rotations, self.lsh.bucket_count, real
            )
            output = fused_expert_codes(
                centroids,
                topk_weights,
                topk_ids.to(torch.int32),
                self.readers,
                self.factors,
                expert_map=None,
                pairs=True,
            )
            weights = residual_weights(topk_weights.float(), self.alpha, topk_ids.long())
            output = compensate_pairs(output, moe_input, centroids, weights)
            if phase is not None:
                routed = torch.zeros(
                    real, self.num_experts, dtype=torch.bool, device=moe_input.device
                )
                routed.scatter_(1, topk_ids[:real].long(), True)
                self.counts[phase, 0] += real
                self.counts[phase, 1] += node_rows(routed)
                self.counts[phase, 2] += clusters
        elif transport is None:
            output = fused_expert_codes(
                fp8_fake_quant(moe_input) if self.fp8_dispatch else moe_input,
                topk_weights,
                topk_ids.to(torch.int32),
                self.readers,
                self.factors,
                expert_map=None,
            )
        else:
            transport.begin(self.layer_index, moe_input)
            received, ids, weights, handle, valid = transport.dispatch(
                moe_input, topk_ids, topk_weights
            )
            written = fused_expert_codes(
                received,
                weights,
                local_ids(ids, self.local_experts),
                self.readers,
                self.factors,
                expert_map=self.expert_map,
            )
            output = transport.combine(written, handle, valid, topk_ids)
        return output.view(hidden_states.shape)


def served_tokens(tokens: int) -> tuple[int, int | None]:
    """The forward's real tokens, before any padding, and its phase: 0 for decodes only, 1 with
    prefill tokens. Outside a served forward (profiling, an idle data-parallel rank) every row is
    real and the phase is None."""
    if not is_forward_context_available():
        return tokens, None
    metadata = get_forward_context().attn_metadata
    if not metadata:
        return tokens, None
    first = next(iter(metadata.values()))
    return first.num_actual_tokens, int(first.num_prefill_tokens > 0)


class Qwen3MoeFoldedDecoderLayer(Qwen3MoeDecoderLayer):
    """The base decoder layer with the folded or full-width block where the sparse one sits."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", block: Any) -> None:
        # The parent's constructor would allocate the 58 GB of fused expert weights these blocks
        # replace, so it is not called.
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
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.self_attn",
            dual_chunk_attention_config=getattr(config, "dual_chunk_attention_config", None),
        )
        self.mlp = block(
            layer_index=extract_layer_index(prefix),
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen3MoeFoldedModel(Qwen3MoeModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_text_config
        folded = FoldedConfig.parse(getattr(config, FOLDED_KEY))
        fp8_dispatch, fp8_combine = served_precision(vllm_config)
        slots = None
        if folded.reference_layers:
            slots = reference_slots(folded.reference_layers, folded.reference_layer_for)
        transport = expert_parallel_transport(
            vllm_config, reference_slots=slots, fp8_dispatch=fp8_dispatch, fp8_combine=fp8_combine
        )
        references: dict[int, torch.Tensor] = {}
        block = partial(
            FoldedSparseMoeBlock,
            folded=folded,
            references=references,
            renormalize=config.norm_topk_prob,
            transport=transport,
            fp8_dispatch=fp8_dispatch,
            fp8_combine=fp8_combine,
        )
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=partial(Qwen3MoeFoldedDecoderLayer, block=block),
        )
        self.num_experts = folded.num_experts

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """The folded export holds no per-expert HuggingFace slots to map."""
        return []


class Qwen3MoeFullWidthModel(Qwen3MoeModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_text_config
        fp8_dispatch, fp8_combine = served_precision(vllm_config)
        transport = expert_parallel_transport(
            vllm_config, reference_slots=None, fp8_dispatch=fp8_dispatch, fp8_combine=fp8_combine
        )
        lsh = alpha = None
        if hasattr(config, LSH_KEY):
            lsh = LshConfig(**getattr(config, LSH_KEY))
            alpha = load_file(Path(vllm_config.model_config.model) / ALPHA_FILE)["alpha"]
        block = partial(
            FullWidthSparseMoeBlock,
            config=config,
            transport=transport,
            fp8_dispatch=fp8_dispatch,
            fp8_combine=fp8_combine,
            lsh=lsh,
            alpha=alpha,
        )
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=partial(Qwen3MoeFoldedDecoderLayer, block=block),
        )
        self.num_experts = config.num_experts

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return [
            (f"mlp.{stacked}", f"mlp.experts.{expert}.{projection}.weight", expert, shard)
            for expert in range(self.num_experts)
            for stacked, projection, shard in (
                ("readers", "gate_proj", "w1"),
                ("readers", "up_proj", "w3"),
                ("factors", "down_proj", "w2"),
            )
        ]


class Qwen3MoeFoldedForCausalLM(Qwen3MoeForCausalLM):
    """The folded student, served through the base model's own kernels and captured at startup."""

    model_class: Any = Qwen3MoeFoldedModel

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # The parent's constructor collects every `FusedMoE` block for expert-parallel load
        # balancing, and a folded model has none.
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        # The readers are indexed by logical expert, and EPLB routes to physical ids.
        assert not vllm_config.parallel_config.enable_eplb, "serve without --enable-eplb"
        assert get_tensor_model_parallel_world_size() == 1, "serve without tensor parallelism"
        assert not config.mlp_only_layers, "the folded student covers every layer"
        assert config.decoder_sparse_step == 1, "the folded student covers every layer"
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.model = self.model_class(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
        experts = self.model.num_experts
        self.expert_weights: list[Any] = []
        self.moe_layers: list[Any] = []
        self.num_moe_layers = 0
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = experts
        self.num_physical_experts = experts
        self.num_local_physical_experts = experts
        self.num_routed_experts = experts
        self.num_redundant_experts = 0

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


class Qwen3MoeFullWidthForCausalLM(Qwen3MoeFoldedForCausalLM):
    """The base export served through full-width blocks on the folded student's transport."""

    model_class = Qwen3MoeFullWidthModel


class DeepseekV3FoldedModel(deepseek_v2.DeepseekV2Model):
    """vLLM's DeepSeek-V3 model with the folded block in each MoE layer. Its decoder layer builds
    the MoE block by the module's name `DeepseekV2MoE`, which the constructor rebinds to the
    folded block for its own length. A block's MoE ordinal is its decoder layer's index less the
    dense layers before it."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_text_config
        folded = FoldedConfig.parse(getattr(config, FOLDED_KEY))
        fp8_dispatch, fp8_combine = served_precision(vllm_config)
        slots = None
        if folded.reference_layers:
            slots = reference_slots(folded.reference_layers, folded.reference_layer_for)
        transport = expert_parallel_transport(
            vllm_config, reference_slots=slots, fp8_dispatch=fp8_dispatch, fp8_combine=fp8_combine
        )
        references: dict[int, torch.Tensor] = {}

        def block(config, parallel_config, quant_config, prefix):
            return SigmoidFoldedMoeBlock(
                config=config,
                folded=folded,
                layer_index=extract_layer_index(prefix) - config.first_k_dense_replace,
                references=references,
                transport=transport,
                fp8_dispatch=fp8_dispatch,
                fp8_combine=fp8_combine,
                quant_config=quant_config,
                prefix=prefix,
            )

        stock = deepseek_v2.DeepseekV2MoE
        deepseek_v2.DeepseekV2MoE = block
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            deepseek_v2.DeepseekV2MoE = stock
        self.num_experts = folded.num_experts


class DeepseekV3FoldedForCausalLM(deepseek_v2.DeepseekV3ForCausalLM):
    """The folded DeepSeek-V3 student, with vLLM's DeepSeek-V3 head and weight loading."""

    model_cls = DeepseekV3FoldedModel

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # The readers are indexed by logical expert, and EPLB routes to physical ids.
        assert not vllm_config.parallel_config.enable_eplb, "serve without --enable-eplb"
        assert get_tensor_model_parallel_world_size() == 1, "serve without tensor parallelism"
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def set_moe_parameters(self) -> None:
        """The parent collects every `DeepseekV2MoE` for expert-parallel load balancing, and a
        folded model has none."""
        experts = self.model.num_experts
        self.expert_weights: list[Any] = []
        self.moe_layers: list[Any] = []
        self.moe_mlp_layers: list[Any] = []
        self.num_moe_layers = 0
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = experts
        self.num_physical_experts = experts
        self.num_local_physical_experts = experts
        self.num_routed_experts = experts
        self.num_redundant_experts = 0


class NemotronHFoldedModel(vllm_nemotron_h.NemotronHModel):
    """vLLM's Nemotron-H model with the folded block in each MoE mixer. Its MoE decoder layer
    builds the mixer by the module's name `NemotronHMoE`, which the constructor rebinds to the
    folded block for its own length. A block's MoE ordinal is its decoder layer's place among the
    pattern's MoE layers (`E`)."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_text_config
        folded = FoldedConfig.parse(getattr(config, FOLDED_KEY))
        fp8_dispatch, fp8_combine = served_precision(vllm_config)
        slots = None
        if folded.reference_layers:
            slots = reference_slots(folded.reference_layers, folded.reference_layer_for)
        transport = expert_parallel_transport(
            vllm_config, reference_slots=slots, fp8_dispatch=fp8_dispatch, fp8_combine=fp8_combine
        )
        references: dict[int, torch.Tensor] = {}
        positions = [
            layer for layer, code in enumerate(config.hybrid_override_pattern) if code == "E"
        ]

        def block(config, quant_config=None, parallel_config=None, prefix=""):
            return SigmoidFoldedMoeBlock(
                config=config,
                folded=folded,
                layer_index=positions.index(extract_layer_index(prefix)),
                references=references,
                transport=transport,
                fp8_dispatch=fp8_dispatch,
                fp8_combine=fp8_combine,
                quant_config=quant_config,
                prefix=prefix,
            )

        stock = vllm_nemotron_h.NemotronHMoE
        vllm_nemotron_h.NemotronHMoE = block
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            vllm_nemotron_h.NemotronHMoE = stock
        self.num_experts = folded.num_experts


class NemotronHFoldedForCausalLM(vllm_nemotron_h.NemotronHForCausalLM):
    """The folded Nemotron-H student, with vLLM's head, Mamba state and weight loading. The
    parent's constructor collects every `NemotronHMoE` mixer for expert-parallel load balancing,
    and a folded model has none, so the pieces are built here."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        # The readers are indexed by logical expert, and EPLB routes to physical ids.
        assert not vllm_config.parallel_config.enable_eplb, "serve without --enable-eplb"
        assert get_tensor_model_parallel_world_size() == 1, "serve without tensor parallelism"
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config
        self.quant_config = vllm_config.quant_config
        self.config = config
        self.model = NemotronHFoldedModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size, prefix=maybe_prefix(prefix, "lm_head")
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
        experts = self.model.num_experts
        self.expert_weights: list[Any] = []
        self.moe_layers: list[Any] = []
        self.num_moe_layers = 0
        self.num_expert_groups = 1
        self.num_shared_experts = config.n_shared_experts
        self.num_logical_experts = experts
        self.num_physical_experts = experts
        self.num_local_physical_experts = experts
        self.num_routed_experts = experts
        self.num_redundant_experts = 0


class Gemma4FoldedModel(vllm_gemma4.Gemma4Model):
    """vLLM's Gemma 4 model with the folded block where each decoder layer builds its routed
    experts by the module's name `Gemma4MoE`, which the constructor rebinds for its own length."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_text_config
        folded = FoldedConfig.parse(getattr(config, FOLDED_KEY))
        fp8_dispatch, fp8_combine = served_precision(vllm_config)
        slots = None
        if folded.reference_layers:
            slots = reference_slots(folded.reference_layers, folded.reference_layer_for)
        transport = expert_parallel_transport(
            vllm_config, reference_slots=slots, fp8_dispatch=fp8_dispatch, fp8_combine=fp8_combine
        )
        references: dict[int, torch.Tensor] = {}

        def block(config, quant_config=None, prefix=""):
            return Gemma4FoldedMoeBlock(
                config=config,
                folded=folded,
                layer_index=extract_layer_index(prefix),
                references=references,
                transport=transport,
                fp8_dispatch=fp8_dispatch,
                fp8_combine=fp8_combine,
                quant_config=quant_config,
                prefix=prefix,
            )

        stock = vllm_gemma4.Gemma4MoE
        vllm_gemma4.Gemma4MoE = block
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            vllm_gemma4.Gemma4MoE = stock
        self.num_experts = folded.num_experts


class Gemma4FoldedForCausalLM(vllm_gemma4.Gemma4ForCausalLM):
    """The folded Gemma 4 student, with vLLM's tied and softcapped head and its weight loading.
    The constructor builds `Gemma4Model` by the module's name, rebound to the folded model."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # The readers are indexed by logical expert, and EPLB routes to physical ids.
        assert not vllm_config.parallel_config.enable_eplb, "serve without --enable-eplb"
        assert get_tensor_model_parallel_world_size() == 1, "serve without tensor parallelism"
        stock = vllm_gemma4.Gemma4Model
        vllm_gemma4.Gemma4Model = Gemma4FoldedModel
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            vllm_gemma4.Gemma4Model = stock
        # the parent counts the experts of its `Gemma4MoE` blocks, and a folded model has none
        experts = self.model.num_experts
        self.num_logical_experts = experts
        self.num_physical_experts = experts
        self.num_local_physical_experts = experts
        self.num_routed_experts = experts


class WorkerExtension:
    """vLLM worker extension (`--worker-extension-cls`): the topology each rank actually serves,
    and LSH-MoE's counts."""

    def served_topology(self) -> list[dict[str, Any]]:
        """Every rank's data- and expert-parallel sizes and expert placement, gathered over the
        expert-parallel group, since vLLM returns only the first engine's result."""
        blocks = (FusedMoE, FoldedSparseMoeBlock, FullWidthSparseMoeBlock)
        placement = {
            (module.local_num_experts, module.global_num_experts)
            if isinstance(module, FusedMoE)
            else (module.local_experts, module.num_experts)
            for module in self.model_runner.model.modules()
            if isinstance(module, blocks)
        }
        ep_group = get_ep_group()
        local = {
            "data_parallel_size": get_dp_group().world_size,
            "expert_parallel_size": ep_group.world_size,
            "experts": sorted(placement),
        }
        reports: list[Any] = [None] * ep_group.world_size
        torch.distributed.all_gather_object(reports, local, group=ep_group.cpu_group)
        return reports

    def lsh_counts(self) -> list[Any]:
        """Every rank's LSH-MoE counts `[layer][phase][tokens, DeepEP rows, clusters]`, gathered
        over the data-parallel group."""
        local = [
            module.counts.tolist()
            for module in self.model_runner.model.modules()
            if isinstance(module, FullWidthSparseMoeBlock)
        ]
        group = get_dp_group()
        reports: list[Any] = [None] * group.world_size
        torch.distributed.all_gather_object(reports, local, group=group.cpu_group)
        return reports
