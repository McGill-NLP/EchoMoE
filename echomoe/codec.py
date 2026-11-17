"""The EchoMoE codec: what each MoE layer sends between expert-parallel ranks, and how the
compression is folded into the experts.

Dispatch (owner to each selected expert). Every s-th layer is a reference layer. It publishes the
token's full-width MoE input `z = x_l` once to every rank and sends no dispatch payload. The
non-reference layers after it send only what the reference fails to predict,

    c_d = C_d (x_l - a * z)        (dispatch against the reference, `residual`)

and the receiver reads it back as `x_hat = D_d c_d + G z`. Plain compression (`plain`) sends
C_d x_l and has no reference. `a` and the reconstruction gain G are per-dimension gains that the
fit initializes equal. G is diagonal here and stored as the vector `b`, G = diag(b).

Combine (each expert to the owner). Each expert sends `C_c y_i`. The owner decompresses the
weighted sum once, `y_hat = D_c sum_i w_i C_c y_i`, and adds an owner-side term that sends nothing.
Direct decoding (`direct`) adds nothing, bias decoding (`bias`) adds `sum_i w_i o_i` and gated bias
decoding (`biasgate`) scales each `o_i` by `1 + alpha_i`.

Folding. For serving, the dispatch decompressor and the combine compressor are multiplied into the
expert matrices next to them. A non-reference layer's experts read `[z, c_d]` through the folded
input projection `W_in,i = W_i [G, D_d]` (`readers`), with W_i the stacked gate and up projections,
and write codes through the folded output projection `W_out,i = C_c W_down,i` (`factors`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional

# Every communicated width is a multiple of 128.
RANK_GRANULARITY = 128
DISPATCH_VARIANTS = ("plain", "residual")
COMBINE_OWNERS = ("direct", "bias", "biasgate")
SCALE_FORMATS = ("float32", "ue8m0")

DISPATCH_COMPRESSOR_MODULE_NAME = "dispatch_compressor"
COMBINE_COMPRESSOR_MODULE_NAME = "combine_compressor"
DISPATCH_REFERENCE_MODULE_NAME = "dispatch_reference"
COMBINE_CORRECTION_MODULE_NAME = "combine_correction"
# The attachment names a side and the model names an attribute.
MODULE_ATTRIBUTE_NAMES = {
    "dispatch": DISPATCH_COMPRESSOR_MODULE_NAME,
    "combine": COMBINE_COMPRESSOR_MODULE_NAME,
    "dispatch_reference": DISPATCH_REFERENCE_MODULE_NAME,
    "combine_correction": COMBINE_CORRECTION_MODULE_NAME,
}


# --- Modules ----------------------------------------------------------------------------------


class Compressor(nn.Module):
    """An explicit code of width `rank`: compressor C (`encoder`), decompressor D (`decoder`)."""

    def __init__(self, hidden_size: int, rank: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(hidden_size, rank, bias=False)
        self.decoder = nn.Linear(rank, hidden_size, bias=False)

    @property
    def rank(self) -> int:
        return self.encoder.out_features

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        return functional.linear(value, self.encoder.weight)

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        return functional.linear(code, self.decoder.weight)


class ReferenceGain(nn.Module):
    """The sender's gain `a` (`encoder_gains`) and the receiver's `b` (`decoder_gains`), the
    diagonal of the reconstruction gain G. Their length-one leading axis is kept so stored
    artifacts and checkpoints load unchanged."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder_gains = nn.Parameter(torch.ones(1, hidden_size))
        self.decoder_gains = nn.Parameter(torch.ones(1, hidden_size))

    def encoder_gain(self, staleness: torch.Tensor) -> torch.Tensor:
        return self.encoder_gains[0].expand(*staleness.shape, self.hidden_size)

    def decoder_gain(self, staleness: torch.Tensor) -> torch.Tensor:
        return self.decoder_gains[0].expand(*staleness.shape, self.hidden_size)


class OwnerCorrection(nn.Module):
    """The owner's term `sum_i w_i (1 + alpha_i) o_i`, added to the decompressed combine code.
    `width=None` is bias decoding (`bias`, alpha = 0). A width makes it gated bias decoding
    (`biasgate`), with its readout at zero."""

    code_scale: torch.Tensor
    # Training sets the data-parallel group, so every rank measures the same code scale.
    reduce_group: Any = None

    def __init__(
        self, hidden_size: int, rank: int, *, num_experts: int, width: int | None
    ) -> None:
        super().__init__()
        self.width = width
        self.offset = nn.Parameter(torch.zeros(num_experts, hidden_size))
        # Measured on the first micro-batch and frozen, so the gate reads a unit-scale code. Zero
        # means not yet measured.
        self.register_buffer("code_scale", torch.zeros(()))
        self._unmeasured = True
        self.input_map = self.code_map = self.expert_embedding = self.readout = None
        if width is not None:
            self.input_map = nn.Parameter(torch.randn(width, hidden_size) * hidden_size**-0.5)
            self.code_map = nn.Parameter(torch.randn(width, rank) * rank**-0.5)
            self.expert_embedding = nn.Parameter(torch.randn(num_experts, width) * 0.1)
            self.readout = nn.Parameter(torch.zeros(num_experts, width))

    def forward(
        self,
        moe_input: torch.Tensor,
        code: torch.Tensor,
        decoded: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        gains = weights.to(dtype=self.offset.dtype)
        if self.width is not None:
            if self._unmeasured:
                self._measure_code_scale(code)
            inverse_rms = torch.rsqrt(moe_input.square().mean(-1, keepdim=True) + 1.0e-6)
            gate_features = functional.silu(
                functional.linear(moe_input * inverse_rms, self.input_map)
                + functional.linear(code / self.code_scale, self.code_map)
                + gains @ self.expert_embedding
            )
            gains = gains * (1.0 + gate_features @ self.readout.t())
        return decoded + gains @ self.offset

    def _measure_code_scale(self, code: torch.Tensor) -> None:
        self._unmeasured = False
        if float(self.code_scale) > 0.0:  # restored by a resume or installed from the fit
            return
        values = code.detach().to(torch.float32)
        totals = torch.stack(
            (
                values.square().sum(),
                torch.tensor(float(values.numel()), dtype=torch.float32, device=values.device),
            )
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            assert self.reduce_group is not None, "set reduce_group to the data-parallel group"
            torch.distributed.all_reduce(totals, group=self.reduce_group)
        scale = float((totals[0] / totals[1]).sqrt())
        assert 0.0 < scale < math.inf, f"the combine code RMS is {scale}"
        with torch.no_grad():
            self.code_scale.fill_(scale)


class FoldedDispatchCompressor(nn.Module):
    def __init__(self, hidden_size: int, rank: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(hidden_size, rank, bias=False)

    @property
    def rank(self) -> int:
        return self.encoder.out_features

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        return functional.linear(value, self.encoder.weight)


class FoldedCombineCompressor(nn.Module):
    def __init__(self, hidden_size: int, rank: int) -> None:
        super().__init__()
        self.decoder = nn.Linear(rank, hidden_size, bias=False)

    @property
    def rank(self) -> int:
        return self.decoder.in_features

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        return functional.linear(code, self.decoder.weight)


class FoldedReferenceGain(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder_gains = nn.Parameter(torch.ones(1, hidden_size))

    def encoder_gain(self, staleness: torch.Tensor) -> torch.Tensor:
        return self.encoder_gains[0].expand(*staleness.shape, self.hidden_size)


# --- Configuration ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceConfig:
    """The reference layers, and the lead before a non-reference layer may read one."""

    lead: int
    references: tuple[int, ...]

    def reference_layer_for(self, layer: int) -> int | None:
        """The newest reference the lead has had time to deliver, if any."""
        delivered = [reference for reference in self.references if reference <= layer - self.lead]
        return max(delivered) if delivered else None

    def staleness(self, layer: int) -> int | None:
        reference = self.reference_layer_for(layer)
        return None if reference is None else layer - reference


@dataclass(frozen=True)
class PrecisionConfig:
    """Simulated `bf16` or `fp8` precision of each side, with the FP8 block and scale format. The
    dispatch side covers the published reference."""

    dispatch: str
    combine: str
    block: int
    scale_format: str


@dataclass(frozen=True)
class CompressorConfig:
    """One compressor configuration. A code width (`dispatch_rank`, `combine_rank`) of None
    leaves that side uncompressed."""

    dispatch_rank: int | None
    combine_rank: int | None
    dispatch_variant: str | None
    reference: ReferenceConfig | None
    route_mass_atol: float
    combine_owner: str | None = None  # None exactly when combine is off
    combine_correction_width: int | None = None  # gate width of gated bias decoding (`biasgate`)
    precision: PrecisionConfig | None = None  # None is bf16 on every side

    def dispatch_role(self, layer: int) -> str:
        if self.dispatch_rank is None:
            return "off"
        if self.reference is None:
            return "plain"
        if layer in self.reference.references:
            return "reference"
        return "plain" if self.reference.reference_layer_for(layer) is None else "consumer"

    def build_modules(
        self, *, layer: int, hidden_size: int, num_experts: int, folded: bool = False
    ) -> tuple[nn.Module | None, nn.Module | None, nn.Module | None, OwnerCorrection | None]:
        """(dispatch, combine, reference gain, owner correction) for one layer. A folded student
        keeps only what the fold leaves outside the experts."""
        role = self.dispatch_role(layer)
        dispatch = combine = gain = correction = None
        if role in ("plain", "consumer"):
            dispatch = (FoldedDispatchCompressor if folded else Compressor)(
                hidden_size, self.dispatch_rank
            )
        if self.combine_rank is not None:
            combine = (FoldedCombineCompressor if folded else Compressor)(
                hidden_size, self.combine_rank
            )
        if role == "consumer":
            gain = (FoldedReferenceGain if folded else ReferenceGain)(hidden_size)
        if self.combine_owner in ("bias", "biasgate"):
            correction = OwnerCorrection(
                hidden_size,
                self.combine_rank,
                num_experts=num_experts,
                width=self.combine_correction_width,
            )
        return dispatch, combine, gain, correction

    def validate_model(
        self, *, hidden_size: int, num_layers: int, payload_dtype: torch.dtype
    ) -> None:
        if payload_dtype != torch.bfloat16:
            raise ValueError("DeepEP compressor communication requires BF16 payloads")
        for side, rank in (("dispatch", self.dispatch_rank), ("combine", self.combine_rank)):
            width = hidden_size if rank is None else rank
            if width > hidden_size:
                raise ValueError("compressor rank cannot exceed the model hidden size")
            if width % RANK_GRANULARITY:
                raise ValueError(f"{side} width {width} must be a multiple of {RANK_GRANULARITY}")
        if self.reference is not None:
            if self.reference.references[-1] >= num_layers:
                raise ValueError(f"reference layers outside a {num_layers}-layer model")
            consumed = {
                self.reference.reference_layer_for(layer)
                for layer in range(num_layers)
                if self.dispatch_role(layer) == "consumer"
            }
            unread = [layer for layer in self.reference.references if layer not in consumed]
            if unread:
                raise ValueError(f"reference layers without a consumer: {unread}")

    def validate_recompute_chunk(self, *, num_layers: int, chunk_layers: int) -> None:
        """Refuse full-recompute chunks that split a reference from any layer that reads it. The
        chunks hold `chunk_layers` MoE layers each, the last one fewer when they do not divide."""
        for layer in range(num_layers):
            if self.dispatch_role(layer) == "consumer":
                reference = self.reference.reference_layer_for(layer)
                if reference // chunk_layers != layer // chunk_layers:
                    raise ValueError(
                        f"recompute chunks of {chunk_layers} layers split reference {reference} "
                        f"from its consumer {layer}"
                    )


# --- Fold -------------------------------------------------------------------------------------
#
# The fold rewrites a fitted or trained compressor into the student's expert matrices in float64.
# Both rewrites are linear identities, so the gate compares the folded products against the ones
# they replace on random rows. It does not test bf16 rounding. The start validation's KL measures
# that cost end to end.

# The largest relative Frobenius distance a folded product may sit from the unfolded one.
FOLD_LIMIT = 1.0e-6
GATE_ROWS = 64


@dataclass(frozen=True)
class ExpertWeights:
    """One layer's local expert matrices, read before the fold replaces the modules."""

    gate_up: tuple[torch.Tensor, ...]
    down: tuple[torch.Tensor, ...]


def fold_reader(
    gate_up: torch.Tensor,
    *,
    role: str,
    decoder: torch.Tensor,
    reconstruction_gain: torch.Tensor | None,
) -> torch.Tensor:
    """The folded input projection `W_in,i = W_i [G, D_d]`, or `W_i D_d` for a plain layer."""
    if role == "plain":
        return gate_up @ decoder
    return torch.cat((gate_up * reconstruction_gain, gate_up @ decoder), dim=1)


def fold_factor(down: torch.Tensor, encoder: torch.Tensor) -> torch.Tensor:
    """The folded output projection `W_out,i = C_c W_down,i`: the expert writes the combine code
    directly."""
    return encoder @ down


def fold_student(
    *,
    attachment: Any,
    weights: Mapping[int, ExpertWeights],
    tensors: Mapping[str, torch.Tensor],
    seed: int,
) -> list[dict[str, Any]]:
    """Fold this rank's layers into their `readers` and `factors`, and return each layer's gate."""
    modules = attachment.modules()
    assert sorted(modules) == sorted(weights), (sorted(modules), sorted(weights))
    return [
        _fold_layer(
            layer,
            role=attachment.adapters[layer].dispatch_role,
            layer_modules=modules[layer],
            readers=attachment.folded_readers.get(layer),
            factors=attachment.folded_experts.get(layer),
            weights=weights[layer],
            tensors=tensors,
            seed=seed,
        )
        for layer in sorted(modules)
    ]


def _fold_layer(
    layer: int,
    *,
    role: str,
    layer_modules: Mapping[str, Any],
    readers: Any,
    factors: Any,
    weights: ExpertWeights,
    tensors: Mapping[str, torch.Tensor],
    seed: int,
) -> dict[str, Any]:
    device = weights.down[0].device

    def artifact(side: str, name: str) -> torch.Tensor:
        return tensors[f"{layer}/{side}/{name}"].to(device=device, dtype=torch.float64)

    # One host generator per layer, so every rank draws the same rows. The combine gate reads it
    # first and the dispatch gate second.
    generator = torch.Generator()
    generator.manual_seed((seed * 1_000_003 + layer) * 2 + 1)

    def random_rows(width: int) -> torch.Tensor:
        return torch.randn(GATE_ROWS, width, generator=generator, dtype=torch.float64).to(device)

    # combine: each expert's down weight absorbs C_c. The owner keeps D_c and its term as stored
    combine_error = 0.0
    if "combine" in layer_modules:
        encoder = artifact("combine", "encoder.weight")
        decoder = artifact("combine", "decoder.weight")
        _install(layer_modules["combine"].decoder.weight, decoder)
        offset = None
        if "combine_correction" in layer_modules:
            correction = layer_modules["combine_correction"]
            correction.load_state_dict(
                {name: artifact("combine_correction", name) for name in correction.state_dict()}
            )
            offset = artifact("combine_correction", "offset")
        activations = random_rows(weights.down[0].shape[1])
        expert_weights = random_rows(len(weights.down))
        hidden_size, code_width = encoder.shape[1], encoder.shape[0]
        folded_code = torch.zeros(GATE_ROWS, code_width, dtype=torch.float64, device=device)
        unfolded_output = torch.zeros(GATE_ROWS, hidden_size, dtype=torch.float64, device=device)
        offset_sum = torch.zeros(GATE_ROWS, hidden_size, dtype=torch.float64, device=device)
        first_expert = int(factors.local_expert_offset)
        for expert, stored in enumerate(weights.down):
            down = stored.to(torch.float64)
            factor = fold_factor(down, encoder)
            _install(factors.factors[expert], factor)
            weight = expert_weights[:, expert : expert + 1]
            folded_code += weight * (activations @ factor.t())
            unfolded_output += weight * (activations @ down.t())
            if offset is not None:
                offset_sum += weight * offset[first_expert + expert]
        combine_error = _relative(
            folded_code @ decoder.t() + offset_sum,
            (unfolded_output @ encoder.t()) @ decoder.t() + offset_sum,
        )

    # dispatch: an off layer's experts read `x_l` and a reference layer's read `z = x_l`, so
    # neither folds. Otherwise the sender keeps C_d and `a`, and each W_in,i absorbs D_d and G
    if role in ("off", "reference"):
        assert not any(key.startswith(f"{layer}/dispatch") for key in tensors), layer
        dispatch_error = None if role == "reference" else 0.0
    else:
        dispatch_encoder = artifact("dispatch", "encoder.weight")
        _install(layer_modules["dispatch"].encoder.weight, dispatch_encoder)
        decoder = artifact("dispatch", "decoder.weight")
        reconstruction_gain = None
        if role == "consumer":
            reference_rows = random_rows(weights.down[0].shape[0])
            reconstruction_gain = artifact("dispatch_reference", "decoder_gains")[0]
            sender_gain = artifact("dispatch_reference", "encoder_gains")
            _install(layer_modules["dispatch_reference"].encoder_gains, sender_gain)
        code_rows = random_rows(decoder.shape[1])
        if role == "plain":
            sent, decoded = code_rows, code_rows @ decoder.t()
        else:
            sent = torch.cat((reference_rows, code_rows), dim=1)
            decoded = reconstruction_gain * reference_rows + code_rows @ decoder.t()
        errors = []
        for expert, stored in enumerate(weights.gate_up):
            gate_up = stored.to(torch.float64)
            reader = fold_reader(
                gate_up, role=role, decoder=decoder, reconstruction_gain=reconstruction_gain
            )
            _install(readers.readers[expert], reader)
            errors.append(_relative(sent @ reader.t(), decoded @ gate_up.t()))
        dispatch_error = max(errors)
    return {
        "layer": layer,
        "reference": role == "reference",
        "dispatch_error": dispatch_error,
        "combine_error": combine_error,
    }


def _install(parameter: torch.Tensor, value: torch.Tensor) -> None:
    assert parameter.shape == value.shape, (tuple(parameter.shape), tuple(value.shape))
    with torch.no_grad():
        parameter.copy_(value)


def _relative(folded: torch.Tensor, exact: torch.Tensor) -> float:
    return float(torch.linalg.norm(folded - exact) / torch.linalg.norm(exact))


def fold_document(
    folds: Sequence[Mapping[str, Any]], *, seed: int, group: Any = None
) -> dict[str, Any]:
    """Merge every rank's folds into one record: each layer's worst error over the ranks."""
    rank_folds = [list(folds)]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank_folds = [None] * torch.distributed.get_world_size(group)
        torch.distributed.all_gather_object(rank_folds, list(folds), group=group)
    layers = []
    for layer_folds in zip(*rank_folds, strict=True):
        dispatch_errors = [
            entry["dispatch_error"] for entry in layer_folds if entry["dispatch_error"] is not None
        ]
        layers.append(
            {
                "layer": layer_folds[0]["layer"],
                "reference": layer_folds[0]["reference"],
                "dispatch_error": max(dispatch_errors) if dispatch_errors else None,
                "combine_error": max(entry["combine_error"] for entry in layer_folds),
            }
        )
    return {
        "limit": FOLD_LIMIT,
        "gate_rows": GATE_ROWS,
        "seed": seed,
        "ranks": len(rank_folds),
        "layers": layers,
    }


def require_exact_fold(document: Mapping[str, Any]) -> None:
    assert document["layers"], "no layer was folded"
    for layer in document["layers"]:
        for side in ("dispatch_error", "combine_error"):
            if layer[side] is not None and layer[side] > FOLD_LIMIT:
                raise ValueError(
                    f"layer {layer['layer']} folds its {side.removesuffix('_error')} side to a "
                    f"relative error {layer[side]}, above {FOLD_LIMIT}"
                )


# --- Stored compressor documents --------------------------------------------------------------


def compressor_config_document(config: CompressorConfig) -> dict[str, Any]:
    document: dict[str, Any] = {
        "sides": {
            "dispatch": config.dispatch_rank is not None,
            "combine": config.combine_rank is not None,
        },
        "skipped_layers": [],
        "validation": {"route_mass_atol": config.route_mass_atol},
    }
    if config.dispatch_rank is not None:
        document["dispatch"] = {
            "rank": config.dispatch_rank,
            "variant": config.dispatch_variant,
            "bypass_layers": [],
        }
        if config.reference is not None:
            document["dispatch"]["reference"] = {
                "schedule": "fixed",
                "lead": config.reference.lead,
                "references": list(config.reference.references),
            }
    if config.combine_rank is not None:
        document["combine"] = {
            "rank": config.combine_rank,
            "variant": "plain",
            "owner": config.combine_owner,
        }
        if config.combine_correction_width is not None:
            document["combine"]["correction"] = {"width": config.combine_correction_width}
    if config.precision is not None:
        document["precision"] = vars(config.precision).copy()
    return document


def parse_compressor_config(value: Any) -> CompressorConfig:
    """Parse a compressor mapping as written in recipes, fit artifacts and exports."""
    if not isinstance(value, Mapping):
        raise ValueError("compressor configuration must be a mapping")
    sides = _fields(value.get("sides"), {"dispatch", "combine"}, "compressor.sides")
    fields = {"sides", "skipped_layers", "validation"} | {side for side in sides if sides[side]}
    # A precision section alone is the uncompressed model with simulated FP8 sides.
    fields |= {"precision"} & set(value)
    if not (sides["dispatch"] or sides["combine"] or "precision" in value):
        raise ValueError("at least one of compressor.sides.dispatch or .combine must be true")
    section = _fields(value, fields, "compressor")
    _empty(section["skipped_layers"], "compressor.skipped_layers")

    # dispatch: plain compression (`plain`), or dispatch against the reference (`residual`) on a
    # fixed list of reference layers
    dispatch_rank = variant = reference = None
    if sides["dispatch"]:
        name = "compressor.dispatch"
        variant = _choice(section["dispatch"].get("variant"), DISPATCH_VARIANTS, f"{name}.variant")
        extra = {"reference"} if variant == "residual" else set()
        dispatch = _fields(section["dispatch"], {"rank", "variant", "bypass_layers"} | extra, name)
        _empty(dispatch["bypass_layers"], f"{name}.bypass_layers")
        dispatch_rank = _positive_int(dispatch["rank"], f"{name}.rank")
        if variant == "residual":
            schedule = _fields(
                dispatch["reference"], {"schedule", "lead", "references"}, f"{name}.reference"
            )
            _choice(schedule["schedule"], ("fixed",), f"{name}.reference.schedule")
            references = tuple(schedule["references"])
            if not references or list(references) != sorted(set(references)) or references[0] < 0:
                raise ValueError(
                    f"{name}.reference.references must be sorted, unique and non-empty"
                )
            reference = ReferenceConfig(
                _positive_int(schedule["lead"], f"{name}.reference.lead"), references
            )

    # combine: plain, with the owner's term
    combine_rank = owner = correction_width = None
    if sides["combine"]:
        name = "compressor.combine"
        owner = _choice(section["combine"].get("owner"), COMBINE_OWNERS, f"{name}.owner")
        extra = {"correction"} if owner == "biasgate" else set()
        combine = _fields(section["combine"], {"rank", "variant", "owner"} | extra, name)
        _choice(combine["variant"], ("plain",), f"{name}.variant")
        combine_rank = _positive_int(combine["rank"], f"{name}.rank")
        if owner == "biasgate":
            correction = _fields(combine["correction"], {"width"}, f"{name}.correction")
            correction_width = _positive_int(correction["width"], f"{name}.correction.width")

    name = "compressor.validation"
    atol = _fields(section["validation"], {"route_mass_atol"}, name)["route_mass_atol"]
    if isinstance(atol, bool) or not isinstance(atol, (int, float)) or not 0 <= atol < math.inf:
        raise ValueError(f"{name}.route_mass_atol must be finite and non-negative")
    precision = parse_precision_config(section["precision"]) if "precision" in section else None
    return CompressorConfig(
        dispatch_rank,
        combine_rank,
        variant,
        reference,
        float(atol),
        owner,
        correction_width,
        precision,
    )


def parse_precision_config(value: Any) -> PrecisionConfig:
    name = "compressor.precision"
    section = _fields(value, {"dispatch", "combine", "block", "scale_format"}, name)
    dispatch = _choice(section["dispatch"], ("bf16", "fp8"), f"{name}.dispatch")
    combine = _choice(section["combine"], ("bf16", "fp8"), f"{name}.combine")
    if "fp8" not in (dispatch, combine):
        raise ValueError(f"{name} names no fp8 side. Omit it for bf16 on every side")
    return PrecisionConfig(
        dispatch,
        combine,
        _positive_int(section["block"], f"{name}.block"),
        _choice(section["scale_format"], SCALE_FORMATS, f"{name}.scale_format"),
    )


# The named communication precisions, as (dispatch, combine).
COMMUNICATION_PRECISIONS: dict[str, tuple[str, str] | None] = {
    "bf16": None,
    "fp8-dispatch": ("fp8", "bf16"),
    "fp8-combine": ("bf16", "fp8"),
    "fp8-all": ("fp8", "fp8"),
}


def communication_precision(name: str) -> PrecisionConfig | None:
    if name not in COMMUNICATION_PRECISIONS:
        raise ValueError(
            f"unknown communication precision {name!r}. One of {sorted(COMMUNICATION_PRECISIONS)}"
        )
    if COMMUNICATION_PRECISIONS[name] is None:
        return None
    dispatch, combine = COMMUNICATION_PRECISIONS[name]
    # DeepEP's per-token block-128 cast with float32 scales, the only one the transport sends.
    return PrecisionConfig(dispatch, combine, 128, "float32")


def _fields(value: Any, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        got = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError(f"{name} must have exactly the fields {sorted(fields)}, got {got}")
    return value


def _choice(value: Any, allowed: tuple[str, ...], name: str) -> str:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {list(allowed)}, got {value!r}")
    return value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _empty(value: Any, name: str) -> None:
    # Kept in every stored document, but per-layer skips and bypasses are not part of the method.
    if value != []:
        raise ValueError(f"{name} must be an empty list")
