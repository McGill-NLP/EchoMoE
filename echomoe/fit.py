"""The closed-form initialization: FP64 moments of the uncompressed model's MoE activations, then
a PCA fit of each compressor config, written as one `<config>.safetensors` keyed
`{layer}/{side}/{name}`, with the compressor document and method as JSON metadata. Layers are
numbered by MoE ordinal (`moe_layout`).

    python -m echomoe.fit moments <capture root> outputs/qwen3-30b-a3b/fit
    python -m echomoe.fit fits outputs/qwen3-30b-a3b/fit \
        --moments outputs/qwen3-30b-a3b/fit/moments.safetensors \
        --export outputs/qwen3-30b-a3b/teacher/hf \
        dres512-s4-c1280-direct dfull-c1024-biasgate-q128 ...
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from echomoe.codec import (
    CompressorConfig,
    ReferenceConfig,
    compressor_config_document,
    parse_compressor_config,
)
from echomoe.runs import atomic_write_json, write_meta

# FP64 addition is not associative, so the chunking and the part order are part of the moments.
CHUNK_ROWS = 2048
MAX_STALENESS = 16  # cross moments x_l x_{l-s}^T for s = 1..16
RIDGE = 1.0e-6
READ_METRIC_EPS = 1.0e-6
CORRECTION_SEED = 42
# The selection frequencies `[layers, experts]` a dispatch fit's read metric was built from.
READ_METRIC_RHO_NAME = "read_metric/rho"


def moe_layout(config: Mapping[str, Any]) -> tuple[list[int], int]:
    """An HF config's MoE decoder layers, in MoE order, and its routed experts. DeepSeek-V3's first
    `first_k_dense_replace` layers are dense. Nemotron-H's pattern places MoE layers (`E`) among
    Mamba (`M`) and attention (`*`) layers."""
    decoder_layers = range(int(config["num_hidden_layers"]))
    if config["model_type"] == "deepseek_v3":
        dense_layers = int(config["first_k_dense_replace"])
        return list(decoder_layers[dense_layers:]), int(config["n_routed_experts"])
    if config["model_type"] == "nemotron_h":
        pattern = config["hybrid_override_pattern"]
        return [layer for layer in decoder_layers if pattern[layer] == "E"], int(
            config["n_routed_experts"]
        )
    if config["model_type"] == "gemma4_text":
        return list(decoder_layers), int(config["num_experts"])
    return list(decoder_layers), int(config["num_local_experts"])


def accumulate_moments(capture: Path, *, device: str = "cpu") -> dict[str, torch.Tensor]:
    first_meta = json.loads((capture / "part-00000/meta.json").read_text())
    layer_count = len(first_meta["layers"])
    hidden_size, expert_count = first_meta["hidden_size"], first_meta["num_experts"]
    # a sigmoid router's rule (DeepSeek-V3), which a softmax router's capture does not record
    router = first_meta.get("router")
    if router is not None:
        expert_biases = torch.tensor(router["expert_biases"], dtype=torch.float64, device=device)
    # Gemma 4's router scales each renormalized weight by its expert's learned scale
    expert_scales = first_meta.get("expert_scales")
    if expert_scales is not None:
        expert_scales = torch.tensor(expert_scales, dtype=torch.float64, device=device)
    layer_pairs = [
        (layer, staleness)
        for layer in range(layer_count)
        for staleness in range(1, min(MAX_STALENESS, layer) + 1)
    ]
    fp64 = {"dtype": torch.float64, "device": device}
    x_moment = torch.zeros((layer_count, hidden_size, hidden_size), **fp64)
    y_moment = torch.zeros_like(x_moment)
    cross_moment = torch.zeros((layer_count, MAX_STALENESS, hidden_size, hidden_size), **fp64)
    # E[w w^T] and E[w y^T] of the routing weights w, stored as `ww` and `wy`
    weight_moment = torch.zeros((layer_count, expert_count, expert_count), **fp64)
    weight_output_moment = torch.zeros((layer_count, expert_count, hidden_size), **fp64)
    selected = torch.zeros((layer_count, expert_count), dtype=torch.int64, device=device)
    row_count = 0
    for moe_input, moe_output, router_logits in _chunks(capture):
        # the activation moments
        moe_input = moe_input.to(device=device, dtype=torch.float64)
        moe_output = moe_output.to(device=device, dtype=torch.float64)
        for layer in range(layer_count):
            x_moment[layer].addmm_(moe_input[layer].T, moe_input[layer])
            y_moment[layer].addmm_(moe_output[layer].T, moe_output[layer])
        for layer, staleness in layer_pairs:
            cross_moment[layer, staleness - 1].addmm_(
                moe_input[layer].T, moe_input[layer - staleness]
            )
        # the routing moments, on the model's dense weights: softmax, top-k, renormalized. A
        # sigmoid router picks the top-k of score plus bias and scales the renormalized scores
        for layer in range(layer_count):
            scores = router_logits[layer].to(device).double()
            if router is None:
                probabilities = torch.softmax(scores, dim=-1)
                top_weights, top_experts = probabilities.topk(first_meta["topk"], dim=-1)
            else:
                probabilities = torch.sigmoid(scores)
                biased = probabilities + expert_biases[layer]
                top_experts = biased.topk(first_meta["topk"], dim=-1).indices
                top_weights = probabilities.gather(-1, top_experts)
            if first_meta["renormalize"]:
                top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
            if router is not None:
                top_weights = top_weights * router["routed_scaling_factor"]
            if expert_scales is not None:
                top_weights = top_weights * expert_scales[layer][top_experts]
            weights = torch.zeros_like(probabilities).scatter_(-1, top_experts, top_weights)
            weight_output_moment[layer].addmm_(weights.T, moe_output[layer])
            weight_moment[layer].addmm_(weights.T, weights)
            selected[layer] += (weights > 0).sum(0)
        row_count += moe_input.shape[1]

    # the least-squares channel gain of x_l on x_{l-s}, per staleness s
    input_diagonal = x_moment.diagonal(dim1=-2, dim2=-1)
    gain = torch.zeros((layer_count, MAX_STALENESS, hidden_size), **fp64)
    for layer, staleness in layer_pairs:
        cross_diagonal = cross_moment[layer, staleness - 1].diagonal()
        gain[layer, staleness - 1] = cross_diagonal / input_diagonal[layer - staleness]
    # the mean table Ybar = E[w w^T]^-1 E[w y^T], solved in FP64 and stored in FP32. The combine
    # offsets are computed from the stored table
    identity = torch.eye(expert_count, **fp64)
    mean_output = torch.stack(
        [
            torch.linalg.solve(
                weight_moment[layer] / row_count + RIDGE * identity,
                weight_output_moment[layer] / row_count,
            )
            for layer in range(layer_count)
        ]
    ).to(torch.float32)
    moments = {
        "x_moment": x_moment,
        "y_moment": y_moment,
        "cross_moment": cross_moment,
        "gain": gain,
        "ww": weight_moment,
        "wy": weight_output_moment,
        "selected": selected,
        "count": torch.full((layer_count,), row_count, dtype=torch.int64),
        "rho": (selected.to(torch.float64) / row_count).to(torch.float32),
        "mean_output": mean_output,
    }
    return {name: value.cpu() for name, value in moments.items()}


def layer_fit(
    moment: torch.Tensor, read_metric: torch.Tensor | None, ranks: Sequence[int]
) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor, float]], float]:
    """PCA of a moment in a read metric `R`, if given: `{rank: (C, D, retention)}` and the energy.
    Every code width is a prefix of one basis `Q`, with `C = Q^T R^{1/2}` and `D = R^{-1/2} Q`."""
    moment = moment.to(torch.float64)
    metric_root = inverse_root = None
    if read_metric is not None:
        metric = read_metric.to(torch.float64)
        eigenvalues, eigenvectors = torch.linalg.eigh((metric + metric.T) * 0.5)
        scale = (eigenvalues.clamp_min(0.0) + READ_METRIC_EPS).sqrt()
        metric_root = (eigenvectors * scale) @ eigenvectors.T
        inverse_root = (eigenvectors / scale) @ eigenvectors.T
        moment = metric_root @ moment @ metric_root
    # eigh needs exact symmetry, which the accumulation only holds to rounding. Its signs are
    # arbitrary, so each eigenvector is flipped to make its largest entry positive
    basis = torch.linalg.eigh((moment + moment.T) * 0.5)[1].flip(1)
    pivots = basis.abs().argmax(dim=0)
    signs = basis[pivots, torch.arange(basis.shape[1], device=basis.device)].sign()
    signs[signs == 0] = 1.0
    basis = (basis * signs).contiguous()
    fits = {}
    for rank in ranks:
        columns = basis[:, :rank]
        if metric_root is None:
            encoder, decoder = columns.T.contiguous(), columns.contiguous()
        else:
            encoder = (columns.T @ metric_root).contiguous()
            decoder = (inverse_root @ columns).contiguous()
        # the share of the moment's trace this code width keeps
        retention = float((moment @ columns).mul(columns).sum() / torch.trace(moment))
        fits[rank] = (encoder, decoder, retention)
    return fits, float(moment.diagonal().sum())


def expert_read_metric(export: Path, layer: int, rho: torch.Tensor, device: str) -> torch.Tensor:
    """`R = sum_i rho_i (W_gate_i^T W_gate_i + W_up_i^T W_up_i)` over the experts of decoder layer
    `layer`: how strongly the experts that fire read each input direction. Non-gated experts
    (Nemotron-H's squared ReLU) read through `W_up_i` alone."""
    weight_map = json.loads((export / "model.safetensors.index.json").read_text())["weight_map"]
    stacked = f"model.layers.{layer}.experts.gate_up_proj"
    if stacked in weight_map:
        # Gemma 4 stacks a layer's experts as one [experts, 2 * width, hidden] tensor, gate rows
        # first
        with safe_open(export / weight_map[stacked], framework="pt") as handle:
            readers = handle.get_tensor(stacked).to(device).chunk(2, dim=1)
    else:
        prefix = f"model.layers.{layer}.mlp.experts."
        projections = ("gate_proj", "up_proj")
        if f"backbone.layers.{layer}.mixer.experts.0.up_proj.weight" in weight_map:
            prefix, projections = f"backbone.layers.{layer}.mixer.experts.", ("up_proj",)
        names = [
            f"{prefix}{expert}.{projection}.weight"
            for projection in projections
            for expert in range(len(rho))
        ]
        tensors = {}
        for shard in sorted({weight_map[name] for name in names}):
            with safe_open(export / shard, framework="pt") as handle:
                tensors.update({n: handle.get_tensor(n) for n in names if weight_map[n] == shard})
        readers = [
            torch.stack([tensors[f"{prefix}{i}.{name}.weight"] for i in range(len(rho))]).to(
                device
            )
            for name in projections
        ]
    metric = torch.zeros((readers[0].shape[-1],) * 2, dtype=torch.float64, device=device)
    for expert, frequency in enumerate(rho.tolist()):
        for projection in (reader[expert] for reader in readers):
            projection = projection.to(torch.float64)
            metric.addmm_(projection.T, projection, alpha=frequency)
    return (metric + metric.T) * 0.5


def fit(
    moments: Path, export: Path, config_ids: Sequence[str], *, device: str = "cpu"
) -> dict[str, tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, list[float | None]]]]:
    """Fit every config in one pass over the layers: `{config_id: (tensors, header, retention)}`.
    `moments` is the file the `moments` command writes."""
    with safe_open(moments, framework="pt") as handle:
        # the large moments are read a layer at a time
        def read(name: str, index: tuple[slice, ...]) -> torch.Tensor:
            return handle.get_slice(name)[index].to(device=device, dtype=torch.float64)

        row_count = float(handle.get_tensor("count")[0])
        rho = handle.get_tensor("rho")
        mean_output = handle.get_tensor("mean_output")
        num_layers, num_experts = rho.shape
        decoder_layers = moe_layout(json.loads((export / "config.json").read_text()))[0]

        # every code width of a side is a prefix of one basis: collect the widths each basis serves
        configs = {name: parse_config_id(name, num_layers) for name in config_ids}
        combine_ranks = sorted({c.combine_rank for c in configs.values() if c.combine_rank})
        dispatch_ranks: dict[ReferenceConfig | None, set[int]] = {}  # None: plain dispatch
        for config in configs.values():
            if config.dispatch_rank:
                dispatch_ranks.setdefault(config.reference, set()).add(config.dispatch_rank)
        if any(reference is not None for reference in dispatch_ranks):
            gains = handle.get_tensor("gain")
        tensors: dict[str, dict[str, torch.Tensor]] = {name: {} for name in configs}
        retention = {
            name: {
                side: []
                for side, rank in (("dispatch", c.dispatch_rank), ("combine", c.combine_rank))
                if rank
            }
            for name, c in configs.items()
        }

        for layer in range(num_layers):
            # combine: PCA of the raw merged-output moment E[y y^T]
            if combine_ranks:
                output_moment = read("y_moment", (slice(layer, layer + 1),))[0]
                combine_fits, output_energy = layer_fit(
                    output_moment / row_count, None, combine_ranks
                )

            # dispatch: PCA in the read metric, of E[x_l x_l^T] (plain) or of the difference's
            # moment E[(x_l - a x_{l-s})(x_l - a x_{l-s})^T] (dispatch against the reference). A
            # reference layer sends no code
            dispatch_fits = {}
            if dispatch_ranks:
                metric = expert_read_metric(export, decoder_layers[layer], rho[layer], device)
                input_moment = read("x_moment", (slice(layer, layer + 1),))[0] / row_count
            for reference, ranks in dispatch_ranks.items():
                if reference is None:
                    dispatch_fits[None] = layer_fit(input_moment, metric, sorted(ranks))[0], None
                    continue
                if layer in reference.references:
                    continue
                staleness = reference.staleness(layer)
                gain = gains[layer, staleness - 1].to(device=device, dtype=torch.float64)
                source = slice(layer - staleness, layer - staleness + 1)
                reference_moment = read("x_moment", (source,))[0] / row_count
                cross_moment = read(
                    "cross_moment", (slice(layer, layer + 1), slice(staleness - 1, staleness))
                )[0, 0]
                scaled = gain[:, None] * (cross_moment / row_count).T
                residual_moment = (
                    input_moment
                    - scaled
                    - scaled.T
                    + gain[:, None] * reference_moment * gain[None, :]
                )
                dispatch_fits[reference] = (
                    layer_fit(residual_moment, metric, sorted(ranks))[0],
                    gain,
                )

            # each config takes its code widths' prefixes, in FP32
            for name, config in configs.items():
                out, scores = tensors[name], retention[name]
                if config.dispatch_rank and config.reference not in dispatch_fits:
                    scores["dispatch"].append(None)  # a reference layer
                elif config.dispatch_rank:
                    fits, gain = dispatch_fits[config.reference]
                    encoder, decoder, kept = fits[config.dispatch_rank]
                    out[f"{layer}/dispatch/encoder.weight"] = encoder.float().cpu()
                    out[f"{layer}/dispatch/decoder.weight"] = decoder.float().cpu()
                    if gain is not None:  # a = b, the channel gains at this layer's staleness
                        gain_row = gain.float().cpu()[None]
                        out[f"{layer}/dispatch_reference/encoder_gains"] = gain_row
                        out[f"{layer}/dispatch_reference/decoder_gains"] = gain_row.clone()
                    scores["dispatch"].append(kept)
                rank, correction_width = config.combine_rank, config.combine_correction_width
                if not rank:
                    continue
                encoder, decoder, kept = combine_fits[rank]
                out[f"{layer}/combine/encoder.weight"] = encoder.float().cpu()
                out[f"{layer}/combine/decoder.weight"] = decoder.float().cpu()
                scores["combine"].append(kept)
                if config.combine_owner == "direct":
                    continue
                # the owner's offsets o_i = (I - D_c C_c) ybar_i
                prefix = f"{layer}/combine_correction/"
                ybar = mean_output[layer].to(dtype=torch.float64, device=encoder.device)
                offset = ybar - (ybar @ encoder.to(torch.float64).T) @ decoder.to(torch.float64).T
                out[f"{prefix}offset"] = offset.cpu().float()
                # the code scale of gated bias decoding (`biasgate`), the RMS entry of C_c y in
                # closed form (bias decoding publishes the unmeasured zero), and its gate: seeded
                # maps over a zero readout, drawn from slot 0 of the per-layer streams whose slot 1
                # the fold's gate rows use (codec)
                code_scale = math.sqrt(kept * output_energy / rank) if correction_width else 0.0
                out[f"{prefix}code_scale"] = torch.tensor(code_scale, dtype=torch.float32)
                if correction_width:
                    seed = (CORRECTION_SEED * 1_000_003 + layer) * 2
                    generator = torch.Generator().manual_seed(seed)
                    hidden_size = mean_output.shape[-1]
                    for key, shape, std in (
                        ("input_map", (correction_width, hidden_size), hidden_size**-0.5),
                        ("code_map", (correction_width, rank), rank**-0.5),
                        ("expert_embedding", (num_experts, correction_width), 0.1),
                    ):
                        draw = torch.randn(shape, generator=generator, dtype=torch.float64)
                        out[prefix + key] = (draw * std).float()
                    out[f"{prefix}readout"] = torch.zeros(num_experts, correction_width)

    method = {"moments": str(moments), "export": str(export)}
    fitted = {}
    for name, config in configs.items():
        if config.dispatch_rank:
            tensors[name][READ_METRIC_RHO_NAME] = rho
        header = {"compressor": compressor_config_document(config), "method": method}
        fitted[name] = tensors[name], header, retention[name]
    return fitted


_CONFIG_ID = re.compile(
    r"(?:dres(?P<rank>\d+)-s(?P<stride>\d+)|dplain(?P<plain>\d+)|dfull)"
    r"-(?:c(?P<combine>\d+)-(?P<owner>direct|bias|biasgate)(?:-q(?P<correction>\d+))?|cfull)"
)


def parse_config_id(config_id: str, num_layers: int) -> CompressorConfig:
    """The compressor a config id names, such as `dres512-s4-c1280-direct`, `dplain768-cfull` or
    `dfull-c1024-biasgate-q128`. With dispatch against the reference, references sit every `s`
    layers from 0."""
    match = _CONFIG_ID.fullmatch(config_id)
    if (
        match is None
        or config_id == "dfull-cfull"
        or bool(match["correction"]) != (match["owner"] == "biasgate")
    ):
        raise ValueError(f"{config_id!r} is not a config id such as dres512-s4-c1280-direct")
    dispatch_rank, stride, combine_rank, correction_width = (
        int(value) if value else None
        for value in (
            match["rank"] or match["plain"],
            match["stride"],
            match["combine"],
            match["correction"],
        )
    )
    variant = None if dispatch_rank is None else "plain" if stride is None else "residual"
    reference = None if stride is None else ReferenceConfig(1, tuple(range(0, num_layers, stride)))
    return CompressorConfig(
        dispatch_rank=dispatch_rank,
        combine_rank=combine_rank,
        dispatch_variant=variant,
        reference=reference,
        route_mass_atol=1.0e-6,
        combine_owner=match["owner"],
        combine_correction_width=correction_width,
    )


def _chunks(capture: Path) -> Iterator[tuple[torch.Tensor, ...]]:
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, part in enumerate(sorted(capture.glob("part-*"))):
            meta = json.loads((part / "meta.json").read_text())
            assert meta["metadata"]["part_index"] == index, f"capture part {index} is missing"
            file_names = {key.split("/")[2] for key in meta["files"] if key.startswith("fit/")}
            for name in sorted(file_names):
                with ExitStack() as stack:
                    handles = [
                        stack.enter_context(
                            safe_open(part / f"fit/layer-{layer:04d}/{name}", "pt")
                        )
                        for layer in meta["layers"]
                    ]
                    total_rows = handles[0].get_slice("moe_input").get_shape()[0]
                    for start in range(0, total_rows, CHUNK_ROWS):
                        chunk = slice(start, min(start + CHUNK_ROWS, total_rows))
                        pending = [
                            [
                                pool.submit(handle.get_slice(key).__getitem__, chunk)
                                for handle in handles
                            ]
                            for key in ("moe_input", "moe_output", "router_logits")
                        ]
                        yield tuple(
                            torch.stack([read.result() for read in reads]) for reads in pending
                        )


def save_fit(path: Path, tensors: Mapping[str, torch.Tensor], header: Mapping[str, Any]) -> None:
    save_file(dict(tensors), path, {key: json.dumps(value) for key, value in header.items()})


def load_fit(
    path: str | Path, compressor: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """One fit's header and tensors, refusing a fit of another compressor."""
    with safe_open(path, framework="pt") as handle:
        header = {key: json.loads(value) for key, value in handle.metadata().items()}
    expected = compressor_config_document(parse_compressor_config(compressor))
    if header["compressor"] != expected:
        raise ValueError(f"{path} is a fit of compressor {header['compressor']}, not {expected}")
    return header, load_file(path)


def install_fit(
    tensors: Mapping[str, torch.Tensor], modules: Mapping[int, Mapping[str, torch.nn.Module]]
) -> None:
    """Copy a fit into `modules[layer][side]`, which must own exactly the fit's tensors."""
    unread = set(tensors) - {READ_METRIC_RHO_NAME}
    for layer, layer_modules in sorted(modules.items()):
        for side, module in sorted(layer_modules.items()):
            state = {}
            for name, current in module.state_dict().items():
                key = f"{layer}/{side}/{name}"
                state[name] = tensors[key].to(device=current.device, dtype=current.dtype)
                unread.discard(key)
            module.load_state_dict(state, strict=True)
    assert not unread, f"no module owns {sorted(unread)[0]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    moments = commands.add_parser("moments", help="accumulate the moments over a capture")
    moments.add_argument("capture", type=Path)
    moments.add_argument("output", type=Path)
    fits = commands.add_parser("fits", help="fit config ids from the moments")
    fits.add_argument("output", type=Path)
    fits.add_argument("configs", nargs="+")
    fits.add_argument("--moments", type=Path, required=True)
    fits.add_argument("--export", type=Path, required=True, help="the teacher's HF export")
    for command in (moments, fits):
        command.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.command == "moments":
        moments = accumulate_moments(args.capture, device=args.device)
        save_file(moments, args.output / "moments.safetensors")
        write_meta(args.output, inputs={"capture": args.capture}, rows=int(moments["count"][0]))
        return

    # each fit, and its retention per layer with the mean over the layers that send a code
    for name, (tensors, header, retention) in fit(
        args.moments, args.export, args.configs, device=args.device
    ).items():
        save_fit(args.output / f"{name}.safetensors", tensors, header)
        scores = {}
        for side, values in retention.items():
            kept = [value for value in values if value is not None]
            scores[side] = {"layers": values, "mean": sum(kept) / len(kept)}
        atomic_write_json(args.output / f"{name}.json", scores)


if __name__ == "__main__":
    main()
