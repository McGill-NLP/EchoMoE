"""The folded EchoMoE students as transformers models, for
`AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`. Plain PyTorch on one device,
with the math of the vLLM classes in `echomoe.inference.evaluation.model` and no expert
parallelism. Imports only torch and transformers, as remote code must.

Each student is its family's transformers model with a folded block in place of every sparse MoE
block. Every s-th MoE layer is a reference layer. Its experts read the token's MoE input `z = x_l`,
which the non-reference layers after it read too. Such a layer sends the dispatch code
`c_d = C_d (x_l - a * z)` (`encoder`, `reference_gain`), and its experts read `[z, c_d]` through
the folded input projections W_in,i (`readers`). With no reference (plain compression) they read
`C_d x_l`. Each expert writes the combine code through its folded output projection W_out,i
(`factors`). The owner decodes the weighted sum (`decoder`) and adds the bias or gated bias term
(`correction`). A side with no code width reads or writes the full hidden state.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as functional
from transformers import (
    DeepseekV3ForCausalLM,
    Gemma4ForCausalLM,
    NemotronHForCausalLM,
    Qwen3MoeForCausalLM,
)
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MLP
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer, Gemma4TextRouter
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHMLP

from .configuration_echomoe import (
    DeepseekV3FoldedConfig,
    Gemma4FoldedConfig,
    NemotronHFoldedConfig,
    Qwen3MoeFoldedConfig,
)


def at_least_float32(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.promote_types(value.dtype, torch.float32))


def squared_relu(value: torch.Tensor) -> torch.Tensor:
    return functional.relu(value).square()


def gelu_tanh(value: torch.Tensor) -> torch.Tensor:
    return functional.gelu(value, approximate="tanh")


class FoldedCorrection(nn.Module):
    """The owner's term `sum_i w_i (1 + alpha_i) o_i` of gated bias decoding. Bias decoding holds
    the offsets alone and adds `sum_i w_i o_i`. `code_rms` is the combine code's RMS that training
    measured and froze."""

    def __init__(self, folded: dict) -> None:
        super().__init__()
        experts, hidden = folded["num_experts"], folded["hidden_size"]
        self.epsilon = folded["correction_epsilon"]
        self.width = folded["correction_width"]
        self.offset = nn.Parameter(torch.empty(experts, hidden))
        self.code_rms = nn.Parameter(torch.empty(()))
        if self.width is not None:
            self.input_map = nn.Parameter(torch.empty(self.width, hidden))
            self.code_map = nn.Parameter(torch.empty(self.width, folded["combine_rank"]))
            self.expert_embedding = nn.Parameter(torch.empty(experts, self.width))
            self.readout = nn.Parameter(torch.empty(experts, self.width))

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
        features = functional.silu(
            functional.linear(moe_input * inverse_rms, self.input_map)
            + functional.linear(code / self.code_rms, self.code_map)
            + weights @ self.expert_embedding
        )
        alpha = features @ self.readout.t()
        return decoded + (weights * (1.0 + alpha)) @ self.offset


class FoldedMoe(nn.Module):
    """One MoE layer's folded experts and codec. `layer` is the MoE layer's ordinal, and
    `references` holds the reference layers' inputs for the rest of the forward."""

    def __init__(
        self,
        folded: dict,
        layer: int,
        references: dict[int, torch.Tensor],
        activation: Callable[[torch.Tensor], torch.Tensor],
        gated: bool = True,
    ) -> None:
        super().__init__()
        hidden, width = folded["hidden_size"], folded["moe_intermediate_size"]
        self.layer = layer
        self.references = references
        self.activation = activation
        self.gated = gated
        self.dispatch_rank = folded["dispatch_rank"]
        self.combine_rank = folded["combine_rank"]
        self.publishes_reference = layer in folded["reference_layers"]
        self.reference_source = folded["reference_layer_for"].get(str(layer))
        reads = hidden
        if self.dispatch_rank is not None and not self.publishes_reference:
            reads = self.dispatch_rank + (0 if self.reference_source is None else hidden)
            self.encoder = nn.Parameter(torch.empty(self.dispatch_rank, hidden))
        if self.reference_source is not None:
            self.reference_gain = nn.Parameter(torch.empty(hidden))
        writes = hidden if self.combine_rank is None else self.combine_rank
        experts = folded["num_experts"]
        self.readers = nn.Parameter(torch.empty(experts, (2 if gated else 1) * width, reads))
        self.factors = nn.Parameter(torch.empty(experts, writes, width))
        if self.combine_rank is not None:
            self.decoder = nn.Parameter(torch.empty(hidden, self.combine_rank))
        self.correction = FoldedCorrection(folded) if folded["combine_offsets"] else None

    def dispatch_rows(self, moe_input: torch.Tensor) -> torch.Tensor:
        """What each selected expert reads."""
        if self.layer == 0:
            self.references.clear()
        if self.dispatch_rank is None:
            return moe_input
        if self.publishes_reference:
            self.references[self.layer] = moe_input
            return moe_input
        if self.reference_source is None:
            return functional.linear(moe_input, self.encoder)
        # a device map can place the reference layer on another device
        reference = self.references[self.reference_source].to(moe_input.device)
        code = functional.linear(moe_input - self.reference_gain * reference, self.encoder)
        return torch.cat((reference, code), dim=-1)

    def routed_output(
        self, moe_input: torch.Tensor, weights: torch.Tensor, experts: torch.Tensor
    ) -> torch.Tensor:
        """The owner's output for tokens `moe_input` routed to `experts` with `weights`."""
        rows = self.dispatch_rows(moe_input)
        code = rows.new_zeros(rows.shape[0], self.factors.shape[1])
        for expert in experts.unique().tolist():
            token, slot = torch.where(experts == expert)
            projected = functional.linear(rows[token], self.readers[expert])
            if self.gated:
                gate, up = projected.chunk(2, dim=-1)
                hidden = self.activation(gate) * up
            else:
                hidden = self.activation(projected)
            written = at_least_float32(functional.linear(hidden, self.factors[expert]))
            # each pair is weighted in float32 and rounded once, as vLLM's grouped GEMM does
            weighted = written * weights[token, slot, None].to(written.dtype)
            code.index_add_(0, token, weighted.to(code.dtype))
        if self.combine_rank is None:
            return code
        decoded = functional.linear(code, self.decoder)
        if self.correction is None:
            return decoded
        dense = moe_input.new_zeros(moe_input.shape[0], self.factors.shape[0])
        dense.scatter_(1, experts, weights.to(moe_input.dtype))
        return self.correction(moe_input, code, decoded, dense)


class Qwen3MoeFoldedBlock(FoldedMoe):
    """A Qwen3-MoE layer: the top-k of the router's softmax, renormalized, and SiLU-gated
    experts."""

    def __init__(
        self, config: Qwen3MoeFoldedConfig, layer: int, references: dict[int, torch.Tensor]
    ) -> None:
        super().__init__(config.folded, layer, references, functional.silu)
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        moe_input = hidden_states.reshape(-1, hidden_states.shape[-1])
        probabilities = torch.softmax(at_least_float32(self.gate(moe_input)), dim=-1)
        weights, experts = torch.topk(probabilities, self.top_k, dim=-1)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return self.routed_output(moe_input, weights, experts).view(hidden_states.shape)


class Qwen3MoeFoldedForCausalLM(Qwen3MoeForCausalLM):
    """A folded Qwen3-MoE student: transformers' Qwen3-MoE with the folded block in every
    layer."""

    config_class = Qwen3MoeFoldedConfig

    def __init__(self, config: Qwen3MoeFoldedConfig) -> None:
        super().__init__(config)
        references: dict[int, torch.Tensor] = {}
        for index, layer in enumerate(self.model.layers):
            layer.mlp = Qwen3MoeFoldedBlock(config, index, references)
        self.post_init()


class SigmoidRouter(nn.Module):
    """A DeepSeek-V3 or Nemotron-H gate: its weight, and the bias its expert choice adds to the
    scores. `float32_logits` projects in float32 (Nemotron-H), else in the model's dtype."""

    def __init__(
        self, config: DeepseekV3FoldedConfig | NemotronHFoldedConfig, float32_logits: bool
    ) -> None:
        super().__init__()
        self.float32_logits = float32_logits
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        self.register_buffer("e_score_correction_bias", torch.empty(config.n_routed_experts))

    def forward(self, moe_input: torch.Tensor) -> torch.Tensor:
        if self.float32_logits:
            return functional.linear(at_least_float32(moe_input), at_least_float32(self.weight))
        return functional.linear(moe_input, self.weight)


class SigmoidFoldedBlock(FoldedMoe):
    """A DeepSeek-V3 or Nemotron-H layer. The top-k of the sigmoid scores plus the bias, among the
    best groups, weighted by their scores and renormalized. The routed output is then scaled in
    the model's dtype and the shared experts' output added, as vLLM serves it."""

    def __init__(
        self,
        config: DeepseekV3FoldedConfig | NemotronHFoldedConfig,
        layer: int,
        references: dict[int, torch.Tensor],
        *,
        activation: Callable[[torch.Tensor], torch.Tensor],
        gated: bool,
        float32_logits: bool,
        shared_experts: nn.Module,
    ) -> None:
        super().__init__(config.folded, layer, references, activation, gated)
        self.gate = SigmoidRouter(config, float32_logits)
        self.shared_experts = shared_experts
        self.top_k = config.num_experts_per_tok
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        moe_input = hidden_states.reshape(-1, hidden_states.shape[-1])
        scores = at_least_float32(self.gate(moe_input)).sigmoid()
        groups = (scores + self.gate.e_score_correction_bias).view(len(scores), self.n_group, -1)
        best = groups.topk(2, dim=-1)[0].sum(-1).topk(self.topk_group, dim=-1, sorted=False)[1]
        chosen = torch.zeros_like(groups[..., 0]).scatter_(1, best, 1).bool()
        choice = groups.masked_fill(~chosen[..., None], float("-inf")).view(len(scores), -1)
        experts = choice.topk(self.top_k, dim=-1, sorted=False)[1]
        weights = scores.gather(1, experts)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        routed = self.routed_output(moe_input, weights, experts) * self.routed_scaling_factor
        return (self.shared_experts(moe_input) + routed).view(hidden_states.shape)


class DeepseekV3FoldedForCausalLM(DeepseekV3ForCausalLM):
    """A folded DeepSeek-V3 student (Kanana-2, Moonlight): transformers' DeepSeek-V3 with the
    folded block in each MoE layer, after the leading dense layers."""

    config_class = DeepseekV3FoldedConfig

    def __init__(self, config: DeepseekV3FoldedConfig) -> None:
        super().__init__(config)
        references: dict[int, torch.Tensor] = {}
        dense = config.first_k_dense_replace
        shared = config.moe_intermediate_size * config.n_shared_experts
        for index, layer in enumerate(self.model.layers[dense:], start=dense):
            layer.mlp = SigmoidFoldedBlock(
                config,
                index - dense,
                references,
                activation=functional.silu,
                gated=True,
                float32_logits=False,
                shared_experts=DeepseekV3MLP(config, intermediate_size=shared),
            )
        self.post_init()


class NemotronHFoldedForCausalLM(NemotronHForCausalLM):
    """A folded Nemotron-H student: transformers' Nemotron-H, its Mamba and attention layers
    included, with the folded block as each MoE layer's mixer. Its experts are squared-ReLU and
    not gated."""

    config_class = NemotronHFoldedConfig

    def __init__(self, config: NemotronHFoldedConfig) -> None:
        super().__init__(config)
        references: dict[int, torch.Tensor] = {}
        blocks = [block for block in self.model.layers if block.block_type == "moe"]
        for ordinal, block in enumerate(blocks):
            block.mixer = SigmoidFoldedBlock(
                config,
                ordinal,
                references,
                activation=squared_relu,
                gated=False,
                float32_logits=True,
                shared_experts=NemotronHMLP(
                    config, intermediate_size=config.moe_shared_expert_intermediate_size
                ),
            )
        self.post_init()


class Gemma4FoldedRouter(Gemma4TextRouter):
    """Gemma 4's router as vLLM serves it: float32 logits of the normalized and scaled residual,
    the top-k of their softmax, renormalized and times each expert's `per_expert_scale`."""

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scaled = self.norm(hidden_states)
        scaled = scaled * torch.tensor(self.scalar_root_size, dtype=scaled.dtype)
        scaled = scaled * self.scale.to(scaled.dtype)
        logits = functional.linear(at_least_float32(scaled), at_least_float32(self.proj.weight))
        probabilities = torch.softmax(logits, dim=-1)
        weights, experts = torch.topk(probabilities, self.config.top_k_experts, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.per_expert_scale[experts].to(weights.dtype)
        return probabilities, weights, experts


class Gemma4FoldedExperts(FoldedMoe):
    """Gemma 4's routed experts, folded and GELU-tanh gated, called as transformers calls its
    experts."""

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        return self.routed_output(hidden_states, top_k_weights, top_k_index)


class Gemma4FoldedDecoderLayer(Gemma4TextDecoderLayer):
    """Gemma 4's decoder layer with the folded experts under `moe`, the export's name, where
    transformers' forward calls `experts`."""

    @property
    def experts(self) -> Gemma4FoldedExperts:  # type: ignore[override]
        return self.moe

    def __init__(
        self, config: Gemma4FoldedConfig, layer: int, references: dict[int, torch.Tensor]
    ) -> None:
        super().__init__(config, layer)
        del self._modules["experts"]
        self.router = Gemma4FoldedRouter(config)
        self.moe = Gemma4FoldedExperts(config.folded, layer, references, gelu_tanh)


class Gemma4FoldedForCausalLM(Gemma4ForCausalLM):
    """A folded Gemma 4 student: transformers' Gemma 4 text model with the folded experts beside
    each layer's dense MLP."""

    config_class = Gemma4FoldedConfig

    def __init__(self, config: Gemma4FoldedConfig) -> None:
        super().__init__(config)
        references: dict[int, torch.Tensor] = {}
        self.model.layers = nn.ModuleList(
            Gemma4FoldedDecoderLayer(config, index, references)
            for index in range(config.num_hidden_layers)
        )
        self.post_init()
