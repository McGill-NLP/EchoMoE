"""Turn one training checkpoint into the servable folded model `runs/<run>/export/step-N/`, or the
teacher's SFT checkpoint into the HuggingFace directory `teacher/hf/` every later stage reads.

    torchrun --standalone --nproc_per_node=8 -m echomoe.inference.evaluation.export \
        --checkpoint outputs/qwen3-30b-a3b/runs/<run>/checkpoints/step-N
    torchrun --standalone --nproc_per_node=8 -m echomoe.inference.evaluation.export \
        --checkpoint outputs/qwen3-30b-a3b/teacher/checkpoints/step-0002230 \
        --output outputs/qwen3-30b-a3b/teacher/hf

A student takes two steps. In step 1 every rank loads the student as the run trained it and the
Megatron Bridge writes the backbone. In step 2 rank 0 writes the export that
`Qwen3MoeFoldedForCausalLM` (`DeepseekV3FoldedForCausalLM`, `Gemma4FoldedForCausalLM`) loads, and
folds a dev mode (backbone frozen) student layer by layer in float64. The codec's MoE layer k sits
in the decoder layer `echomoe.fit.moe_layout` places it in. The export also carries the
transformers remote code of `echomoe.hf`, which `auto_map` in `config.json` names. An
LSH-MoE run is exported as the teacher is, with its LSH settings under `lsh` in `config.json` and
its learned alpha in `lsh_alpha.safetensors`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import yaml
from megatron.bridge import AutoBridge
from megatron.core import parallel_state
from safetensors import safe_open
from safetensors.torch import save_file

from echomoe.codec import (
    MODULE_ATTRIBUTE_NAMES,
    CompressorConfig,
    ExpertWeights,
    compressor_config_document,
    fold_document,
    fold_student,
    parse_compressor_config,
    require_exact_fold,
)
from echomoe.fit import moe_layout
from echomoe.hf import add_remote_code
from echomoe.lsh import ALPHA_FILE
from echomoe.runs import staged_directory, write_meta
from echomoe.training.checkpoint import load_model, step_meta
from echomoe.training.model import (
    COMPENSATION_MODULE_NAME,
    ExpertBank,
    add_codec_modules,
    add_compensation_scales,
    moe_layers,
)

# The vLLM class that serves the export by model type (model.py, which runs in the other
# environment).
FOLDED_ARCHITECTURES = {
    "qwen3_moe": "Qwen3MoeFoldedForCausalLM",
    "deepseek_v3": "DeepseekV3FoldedForCausalLM",
    "gemma4_text": "Gemma4FoldedForCausalLM",
    "nemotron_h": "NemotronHFoldedForCausalLM",
}
# Each codec tensor under the parameter name the served block registers. vLLM's loader diverts any
# name holding "scale" into kv-scale remapping, so the code RMS travels as `code_rms`.
SERVED_NAMES = {
    "combine/decoder.weight": "decoder",
    "combine_correction/code_map": "correction.code_map",
    "combine_correction/code_scale": "correction.code_rms",
    "combine_correction/expert_embedding": "correction.expert_embedding",
    "combine_correction/input_map": "correction.input_map",
    "combine_correction/offset": "correction.offset",
    "combine_correction/readout": "correction.readout",
    "dispatch/encoder.weight": "encoder",
    "dispatch_reference/encoder_gains": "reference_gain",
}
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
)
# OwnerCorrection's normalization epsilon, which the served block is told rather than loads.
CORRECTION_EPSILON = 1.0e-6
FOLD_SEED = 0
SHARD_BYTES = 4 * 1024**3


# --- 1. The Megatron model --------------------------------------------------------------------


def bridge_model(
    base: Path, prepare: Callable[[list[Any]], list[Any]] | None = None
) -> tuple[AutoBridge, list[Any]]:
    bridge = AutoBridge.from_hf_pretrained(str(base))
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.context_parallel_size = 1
    # Expert parallel over every launched rank, as the checkpoints shard it. WORLD_SIZE comes from
    # the launcher, since torch.distributed is initialized only below.
    provider.expert_model_parallel_size = int(os.environ.get("WORLD_SIZE", "1"))
    provider.gradient_accumulation_fusion = False
    provider.moe_router_dtype = "fp32"
    provider.finalize()
    if not parallel_state.is_initialized():
        provider.initialize_model_parallel(seed=0)
    if prepare is not None:
        provider.register_pre_wrap_hook(prepare)
    return bridge, provider.provide_distributed_model(wrap_with_ddp=False, bf16=True)


def require_base_names(hf: Path, base: Path) -> None:
    """Assert the bridge wrote every `base` weight. It skips with a warning any it cannot map."""
    exported, expected = (
        set(json.loads((path / "model.safetensors.index.json").read_text())["weight_map"])
        for path in (hf, base)
    )
    assert exported == expected, f"the HF export differs from {base} in {exported ^ expected}"


def restore_layer_pattern(hf: Path, base: Path) -> None:
    """Write back a Nemotron-H config's layer pattern string and layer count, which transformers
    5.8.1 saves only as `layers_block_type` and vLLM's own config reads only as a string."""
    config = json.loads((hf / "config.json").read_text())
    if config["model_type"] != "nemotron_h":
        return
    pattern = json.loads((base / "config.json").read_text())
    config |= {key: pattern[key] for key in ("hybrid_override_pattern", "num_hidden_layers")}
    (hf / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def export_teacher(
    checkpoint: Path, base: Path, parts: Path, output: Path, lsh: dict | None = None
) -> None:
    # an LSH-MoE student's alpha modules as training adds them, so the checkpoint loads them
    bridge, model = bridge_model(base, None if lsh is None else add_compensation_scales)
    load_model(model, checkpoint, prefix="module.")
    if lsh is not None:
        alpha = take_alpha(model)
    rank = torch.distributed.get_rank()
    if rank == 0:
        shutil.rmtree(parts, ignore_errors=True)
    torch.distributed.barrier()
    parts.mkdir(parents=True, exist_ok=True)
    bridge.save_hf_pretrained(model, str(parts), distributed_save=True)
    if rank == 0:
        require_base_names(parts, base)
        restore_layer_pattern(parts, base)
        # the base model's tokenizer files, in place of the bridge's, which move the chat template
        # into a file of its own
        for name in TOKENIZER_FILES:
            if (base / name).is_file():
                shutil.copy2(base / name, parts / name)
            else:
                (parts / name).unlink(missing_ok=True)
        if lsh is not None:
            config = json.loads((parts / "config.json").read_text()) | {"lsh": lsh}
            (parts / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
            save_file({"alpha": alpha}, parts / ALPHA_FILE)
        write_meta(parts, inputs={"checkpoint": checkpoint, "model": base})
        parts.rename(output)


def take_alpha(model: list[Any]) -> torch.Tensor:
    """Every layer's alpha, `[layers, experts]` in float32, from each rank's local experts. The
    modules are then removed, as the bridge has no HF name for them and refuses an unmapped
    parameter."""
    layers = moe_layers(model)
    first = getattr(layers[0][1], COMPENSATION_MODULE_NAME)
    alpha = torch.zeros(
        len(layers), first.num_global_experts, dtype=torch.float32, device=first.raw.device
    )
    for layer, moe in layers:
        scale = getattr(moe, COMPENSATION_MODULE_NAME)
        start = scale.local_expert_offset
        alpha[layer, start : start + scale.raw.shape[0]] = scale.alpha.detach()
        delattr(moe, COMPENSATION_MODULE_NAME)
    torch.distributed.all_reduce(alpha)  # the export is expert parallel over every rank
    return alpha.cpu()


def export_student(checkpoint: Path, parts: Path) -> tuple[CompressorConfig, Path, bool]:
    """Step 1: the backbone as `parts/hf`, the codec tensors as `parts/codec.safetensors` and a
    folded (full-model adaptation) student's banks as `parts/bank-<first expert>.safetensors`."""
    # read the run's config: the teacher it trained from, its compressor, and whether it folded
    run = yaml.safe_load((checkpoint.parent.parent / "config.yaml").read_text())
    compressor = parse_compressor_config(run["compressor"])
    folded = run["models"]["student"]["folded"]
    teacher = Path(run["models"]["teacher"])
    base = teacher / "hf"

    # build the student expert parallel over every launched rank, as the checkpoint sharded it
    codec_modules, dense = {}, {}

    def prepare(chunks: list[Any]) -> list[Any]:
        # the codec modules as training adds them, before the bf16 wrap so it casts them with the
        # rest. The dense expert modules a folded student's banks replace are kept for the bridge.
        for layer, moe in moe_layers(chunks):
            dense[layer] = (moe.experts.linear_fc1, moe.experts.linear_fc2)
            codec_modules[layer] = add_codec_modules(moe, layer, compressor, folded=folded)
        return chunks

    bridge, model = bridge_model(base, prepare)

    # load the checkpoint, and a dev-mode run's frozen backbone from the teacher
    codec_only = step_meta(checkpoint)["codec_only"]
    if codec_only:
        load_model(model, teacher / "megatron-ep8", backbone_only=True, prefix="module.")
    load_model(model, checkpoint, codec_only=codec_only, prefix="module.")

    # take a folded student's banks out of the expert slots, which get their dense modules back,
    # zeroed, so the bridge maps every base name, and then the codec modules, which it cannot map
    banks, first = {}, None
    with torch.no_grad():
        for layer, moe in moe_layers(model):
            for slot, module in zip(("linear_fc1", "linear_fc2"), dense[layer], strict=True):
                bank = getattr(moe.experts, slot)
                if isinstance(bank, ExpertBank):
                    weight = bank.get_parameter(bank.name)
                    banks[f"{layer}/{bank.name}"] = weight.detach().cpu().contiguous()
                    first = bank.local_expert_offset
                    setattr(moe.experts, slot, module.to(dtype=weight.dtype))
                    for parameter in module.parameters():
                        parameter.zero_()
            for name in MODULE_ATTRIBUTE_NAMES.values():
                if hasattr(moe, name):
                    delattr(moe, name)

    # write the backbone, the codec tensors and this rank's banks, over any failed attempt's
    rank = torch.distributed.get_rank()
    if rank == 0:
        shutil.rmtree(parts, ignore_errors=True)
    torch.distributed.barrier()
    (parts / "hf").mkdir(parents=True, exist_ok=True)
    bridge.save_hf_pretrained(model, str(parts / "hf"), distributed_save=True)
    if rank == 0:
        restore_layer_pattern(parts / "hf", base)
        tensors = {
            f"{layer}/{side}/{name}": value.detach().cpu().contiguous()
            for layer, layer_modules in sorted(codec_modules.items())
            for side, module in sorted(layer_modules.items())
            for name, value in module.state_dict().items()
        }
        save_file(tensors, parts / "codec.safetensors")
    if folded:
        save_file(banks, parts / f"bank-{first:05d}.safetensors")
    torch.distributed.barrier()
    return compressor, base, folded


# --- 2. The servable export -------------------------------------------------------------------


class _Shards:
    """One shard mapped at a time, as a 57 GB backbone mapped whole exceeds a 32 GiB data-segment
    limit. Passes walk the shards in order. `get` copies since the next shard drops the mapping."""

    def __init__(self, directory: Path) -> None:
        weight_map = json.loads((directory / "model.safetensors.index.json").read_text())
        self.index = {name: directory / shard for name, shard in weight_map["weight_map"].items()}
        self._path: Path | None = None
        self._handle: Any = None

    def get(self, name: str) -> torch.Tensor:
        if self.index[name] != self._path:
            if self._handle is not None:
                self._handle.__exit__(None, None, None)
            self._path = self.index[name]
            self._handle = safe_open(self._path, framework="pt")
            self._handle.__enter__()
        return self._handle.get_tensor(name).clone()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.__exit__(None, None, None)
            self._handle = None


class _ShardWriter:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.weight_map: dict[str, str] = {}
        self.total_bytes = 0
        self._pending: dict[str, torch.Tensor] = {}
        self._pending_bytes = 0
        self._shards: list[str] = []

    def add(self, name: str, tensor: torch.Tensor) -> None:
        assert name not in self.weight_map | self._pending, name
        value = tensor.contiguous()
        self._pending[name] = value
        self._pending_bytes += value.numel() * value.element_size()
        if self._pending_bytes >= SHARD_BYTES:
            self._flush()

    def finish(self) -> None:
        self._flush()
        width = max(5, len(str(len(self._shards))))
        names = {
            staged: f"model-{index + 1:0{width}d}-of-{len(self._shards):0{width}d}.safetensors"
            for index, staged in enumerate(self._shards)
        }
        for staged, published in names.items():
            (self.directory / staged).rename(self.directory / published)
        weight_map = {name: names[shard] for name, shard in self.weight_map.items()}
        index = {
            "metadata": {"total_size": self.total_bytes},
            "weight_map": dict(sorted(weight_map.items())),
        }
        (self.directory / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n"
        )

    def _flush(self) -> None:
        if not self._pending:
            return
        shard = f"staged-{len(self._shards):05d}.safetensors"
        save_file(self._pending, self.directory / shard)
        for name in self._pending:
            self.weight_map[name] = shard
        self._shards.append(shard)
        self.total_bytes += self._pending_bytes
        self._pending = {}
        self._pending_bytes = 0


def folded_config(compressor: CompressorConfig, config: Mapping[str, Any]) -> dict[str, Any]:
    """The config the folded model class reads from `config.json["folded"]`."""
    decoder_layers, experts = moe_layout(config)
    roles = {layer: compressor.dispatch_role(layer) for layer in range(len(decoder_layers))}
    top_k = "top_k_experts" if config["model_type"] == "gemma4_text" else "num_experts_per_tok"
    return {
        "hidden_size": int(config["hidden_size"]),
        "moe_intermediate_size": int(config["moe_intermediate_size"]),
        "num_experts": experts,
        "num_experts_per_tok": int(config[top_k]),
        "num_hidden_layers": len(decoder_layers),
        "dispatch_rank": compressor.dispatch_rank,
        # Coded references are not part of the method and are never set.
        "reference_rank": None,
        "combine_rank": compressor.combine_rank,
        "combine_offsets": compressor.combine_owner in ("bias", "biasgate"),
        "correction_width": compressor.combine_correction_width,
        "correction_epsilon": CORRECTION_EPSILON,
        "reference_layers": [layer for layer, role in roles.items() if role == "reference"],
        "reference_layer_for": {
            str(layer): compressor.reference.reference_layer_for(layer)
            for layer, role in roles.items()
            if role == "consumer"
        },
    }


def expert_layout(config: Mapping[str, Any], decoder_layer: int) -> tuple[str, str, bool]:
    """Where a decoder layer's routed experts are named, where the served block's tensors go, and
    whether the experts are gated. Gemma 4's experts sit beside its dense `mlp` and vLLM serves
    them as `moe`. Nemotron-H's sit in its `mixer`, with no gate (squared ReLU)."""
    if config["model_type"] == "gemma4_text":
        layer = f"model.layers.{decoder_layer}."
        return layer + "experts.", layer + "moe.", True
    if config["model_type"] == "nemotron_h":
        layer = f"backbone.layers.{decoder_layer}.mixer."
        return layer + "experts.", layer, False
    layer = f"model.layers.{decoder_layer}.mlp."
    return layer + "experts.", layer, True


def fold_layer(
    compressor: CompressorConfig,
    layer: int,
    backbone: _Shards,
    tensors: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None, torch.Tensor | None, dict[str, Any]]:
    """Fold MoE layer `layer` of an unfolded compressor into its experts. The reconstruction gain
    G (`b`) goes into the folded input projections W_in,i (`readers`), the sender gain `a` stays
    and the owner term keeps its measured code scale."""
    decoder_layers, experts = moe_layout(config)
    width = int(config["moe_intermediate_size"])
    hidden, dtype = int(config["hidden_size"]), getattr(torch, config["dtype"])
    role = compressor.dispatch_role(layer)
    prefix, _served, gated = expert_layout(config, decoder_layers[layer])

    def read(expert: int, projection: str) -> torch.Tensor:
        return backbone.get(f"{prefix}{expert}.{projection}_proj.weight").to(device)

    # A layer whose experts read `x_l` itself folds nothing into its gate/up weight. Gemma 4
    # stacks a layer's experts, gate rows first.
    if config["model_type"] == "gemma4_text":
        weights = ExpertWeights(
            gate_up=()
            if role in ("off", "reference")
            else backbone.get(prefix + "gate_up_proj").to(device).unbind(0),
            down=backbone.get(prefix + "down_proj").to(device).unbind(0),
        )
    else:
        projections = ("gate", "up") if gated else ("up",)
        weights = ExpertWeights(
            gate_up=()
            if role in ("off", "reference")
            else tuple(
                torch.cat([read(expert, projection) for projection in projections])
                for expert in range(experts)
            ),
            down=tuple(read(expert, "down") for expert in range(experts)),
        )
    built = compressor.build_modules(
        layer=layer, hidden_size=hidden, num_experts=experts, folded=True
    )
    layer_modules = {
        side: module.to(dtype)
        for side, module in zip(MODULE_ATTRIBUTE_NAMES, built)
        if module is not None
    }
    readers = factors = None
    if role in ("plain", "consumer"):
        reads = (hidden if role == "consumer" else 0) + compressor.dispatch_rank
        readers = torch.zeros(experts, (2 if gated else 1) * width, reads, dtype=dtype)
    if compressor.combine_rank is not None:
        factors = torch.zeros(experts, compressor.combine_rank, width, dtype=dtype)
    attachment = SimpleNamespace(
        modules=lambda: {layer: layer_modules},
        adapters={layer: SimpleNamespace(dispatch_role=role)},
        folded_readers={} if readers is None else {layer: SimpleNamespace(readers=readers)},
        folded_experts={}
        if factors is None
        else {layer: SimpleNamespace(factors=factors, local_expert_offset=0)},
    )
    (fold,) = fold_student(
        attachment=attachment, weights={layer: weights}, tensors=tensors, seed=FOLD_SEED
    )
    # The served gate divides the code by the scale, and zero is the unmeasured sentinel.
    owner = layer_modules.get("combine_correction")
    assert owner is None or owner.width is None or float(owner.code_scale) > 0.0, layer
    folded = {
        f"{layer}/{side}/{name}": value.detach().cpu().contiguous()
        for side, module in sorted(layer_modules.items())
        for name, value in module.state_dict().items()
    }
    return folded, readers, factors, fold


def _stacked_bank(banks: list[Path], layer: int, name: str) -> torch.Tensor:
    parts = []
    for path in banks:
        with safe_open(path, framework="pt") as handle:
            parts.append(handle.get_tensor(f"{layer}/{name}").clone())
    return torch.cat(parts)


def write_export(
    *,
    source: Path,
    compressor: CompressorConfig,
    tensors: Mapping[str, torch.Tensor],
    banks: list[Path] | None,
    base: Path,
    destination: Path,
    inputs: Mapping[str, Any],
) -> None:
    """Step 2: write the export from the backbone directory `source`, the codec `tensors` and a
    folded student's `banks` (sorted by first expert). `banks=None` folds the student here."""
    config = json.loads((source / "config.json").read_text())
    decoder_layers, experts = moe_layout(config)
    hidden, width = int(config["hidden_size"]), int(config["moe_intermediate_size"])
    expert_prefixes = tuple(expert_layout(config, layer)[0] for layer in decoder_layers)
    dtype = getattr(torch, config["dtype"])
    backbone = _Shards(source)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = []
    with staged_directory(destination) as staging:
        writer = _ShardWriter(staging)
        # the backbone without its expert slots
        for _shard, name in sorted((shard, name) for name, shard in backbone.index.items()):
            if not name.startswith(expert_prefixes):
                writer.add(name, backbone.get(name))

        # each MoE layer's readers, factors (W_out,i) and codec tensors, in its decoder layer
        for layer, decoder_layer in enumerate(decoder_layers):
            prefix, served, gated = expert_layout(config, decoder_layer)
            role = compressor.dispatch_role(layer)
            if banks is None:
                layer_tensors, readers, factors, fold = fold_layer(
                    compressor, layer, backbone, tensors, config, device
                )
                folds.append(fold)
            else:
                layer_tensors = {
                    name: tensor
                    for name, tensor in tensors.items()
                    if name.startswith(f"{layer}/")
                }
                readers = factors = None
                if role in ("plain", "consumer"):
                    readers = _stacked_bank(banks, layer, "readers")
                if compressor.combine_rank is not None:
                    factors = _stacked_bank(banks, layer, "factors")
            # A side that folded nothing keeps the trained matrix the backbone carries: the
            # gate/up weight of a layer whose experts read `x_l`, the down weight with no combine.
            if config["model_type"] == "gemma4_text":
                readers = backbone.get(prefix + "gate_up_proj") if readers is None else readers
                factors = backbone.get(prefix + "down_proj") if factors is None else factors
            if readers is None:
                halves = ("gate", "up") if gated else ("up",)
                readers = torch.empty(experts, len(halves) * width, hidden, dtype=dtype)
                for index, half in enumerate(halves):
                    for expert in range(experts):
                        name = f"{prefix}{expert}.{half}_proj.weight"
                        readers[expert, index * width : (index + 1) * width] = backbone.get(name)
            if factors is None:
                factors = torch.empty(experts, hidden, width, dtype=dtype)
                for expert in range(experts):
                    factors[expert] = backbone.get(f"{prefix}{expert}.down_proj.weight")
            writer.add(served + "readers", readers)
            writer.add(served + "factors", factors)
            placed = [key for key in sorted(SERVED_NAMES) if f"{layer}/{key}" in layer_tensors]
            assert len(placed) == len(layer_tensors), f"layer {layer} has tensors with no place"
            for key in placed:
                value = layer_tensors[f"{layer}/{key}"]
                if key == "dispatch_reference/encoder_gains":
                    # Stored with a leading axis of one and served as a vector.
                    assert tuple(value.shape) == (1, hidden), value.shape
                    value = value[0]
                writer.add(served + SERVED_NAMES[key], value)
        writer.finish()
        # an open mapping would keep the shard, and so `parts/`, on network storage after deletion
        backbone.close()

        # the fold's gate record, config.json, the tokenizer and provenance
        record = fold_document(folds, seed=FOLD_SEED) if folds else None
        if record is not None:
            require_exact_fold(record)
        config["architectures"] = [FOLDED_ARCHITECTURES[config["model_type"]]]
        config["folded"] = folded_config(compressor, config)
        config = add_remote_code(staging, config)
        (staging / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        for name in TOKENIZER_FILES:
            if (base / name).is_file():
                shutil.copy2(base / name, staging / name)
        write_meta(
            staging, inputs=inputs, compressor=compressor_config_document(compressor), fold=record
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True, help="<run>/checkpoints/step-N")
    parser.add_argument("--output", type=Path, help="default <run>/export/step-N")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output or checkpoint.parent.parent / "export" / checkpoint.name
    assert not output.exists(), f"{output} already exists"
    parts = output.parent / f".{output.name}.parts"

    # An SFT run (the teacher) names its base model at the top level of its config.
    run = yaml.safe_load((checkpoint.parent.parent / "config.yaml").read_text())
    if "model" in run:
        export_teacher(checkpoint, Path(run["model"]), parts, output)
        torch.distributed.destroy_process_group()
        return
    if "lsh" in run["compressor"]:
        base = Path(run["models"]["teacher"]) / "hf"
        export_teacher(checkpoint, base, parts, output, lsh=run["compressor"]["lsh"])
        torch.distributed.destroy_process_group()
        return
    compressor, base, folded = export_student(checkpoint, parts)
    rank = torch.distributed.get_rank()
    # Step 2 runs on rank 0 alone, outside the process group: a fold here gathers its record over
    # the group when there is one, and the other ranks are done.
    torch.distributed.destroy_process_group()
    if rank != 0:
        return
    hf = parts / "hf"
    require_base_names(hf, base)
    with safe_open(parts / "codec.safetensors", framework="pt") as handle:
        tensors = {key: handle.get_tensor(key).clone() for key in handle.keys()}  # noqa: SIM118
    write_export(
        source=hf,
        compressor=compressor,
        tensors=tensors,
        banks=sorted(parts.glob("bank-*.safetensors")) if folded else None,
        base=base,
        destination=output,
        inputs={"checkpoint": checkpoint, "student": base},
    )
    shutil.rmtree(parts)


if __name__ == "__main__":
    main()
