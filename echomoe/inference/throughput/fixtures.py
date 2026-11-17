"""Random-weight checkpoints of a larger MoE model for the throughput runs (Tables 37 and 38): the
base model and its folded student, full size and depth. They measure execution cost only. Every
weight is seeded noise with fan-in scaling, so they have no language quality or trained routing.

    . scripts/env.sh vllm029
    python -m echomoe.inference.throughput.fixtures \
        --model-config configs/inference/throughput/qwen3-235b-a22b.yaml \
        --official <the official release's Hugging Face snapshot> --output outputs/qwen3-235b-a22b/fixtures

`base/` has exactly the official release's tensor names, dtypes and shapes, read from its
safetensors headers (no official weight is read), and its config and tokenizer files, so stock
vLLM serves it. `folded/` is the folded student (`checkpoint.PortableExport`): the same backbone,
read back from `base/`, and random readers, factors and codec, its experts stored in the base
model's format. Each tensor draws from its own generator seeded by its name, so the bytes do not
depend on the file layout or on the number of workers.
"""

import argparse
import hashlib
import json
import math
import multiprocessing
import re
import shutil
import struct
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from pathlib import Path

import torch
import yaml
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
from safetensors import safe_open
from safetensors.torch import save_file

from echomoe.runs import write_meta

from .checkpoint import codec_recipe, model_profile, reference_for

SEED = 20260922
# bytes per element of each safetensors dtype
ELEMENT_BYTES = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "I32": 4}
NORM_GAIN = (
    r"(^|\.)(input_layernorm|post_attention_layernorm|norm|q_norm|k_norm|kv_a_layernorm"
    r"|q_a_layernorm|norm0|norm1|final_layernorm|pre_norm)\.weight$"
)
# config, tokenizer and remote-code files copied from the official release
METADATA_SUFFIXES = (".json", ".py", ".jinja", ".model", ".txt")


def tensor_seed(name: str) -> int:
    # A tensor's random stream: its name hashed with the base seed (the measured fixtures' rule).
    return int(hashlib.sha256(f"{SEED}:{name}".encode()).hexdigest()[:15], 16)


def layer_of(name):
    match = re.match(r"(?:language_model\.)?model\.layers\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def official_tensors(official: Path) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Each official tensor's dtype and shape from the shard headers, without the optional layers
    past the decoder's depth (DeepSeek-V3's multi-token prediction layer)."""
    index = json.loads((official / "model.safetensors.index.json").read_text())["weight_map"]
    tensors = {}
    for shard in sorted(set(index.values())):
        with open(official / shard, "rb") as stream:
            (length,) = struct.unpack("<Q", stream.read(8))
            header = json.loads(stream.read(length))
        header.pop("__metadata__", None)
        tensors.update(
            {name: (entry["dtype"], tuple(entry["shape"])) for name, entry in header.items()}
        )
    config = json.loads((official / "config.json").read_text())
    depth = config.get("text_config", config)["num_hidden_layers"]
    return {
        name: spec
        for name, spec in tensors.items()
        if layer_of(name) is None or layer_of(name) < depth
    }


def normal_rows(name, shape, std, dtype):
    """Seeded N(0, std^2), generated a row chunk at a time (FP32 staging up to 256 MiB)."""
    generator = torch.Generator().manual_seed(tensor_seed(name))
    out = torch.empty(shape, dtype=dtype)
    rows = max(1, (1 << 26) // shape[1])
    for start in range(0, shape[0], rows):
        stop = min(shape[0], start + rows)
        out[start:stop] = (torch.randn((stop - start, shape[1]), generator=generator) * std).to(
            dtype
        )
    return out


def bf16_tensor(name, shape):
    """Norm gains 1, biases N(0, 0.02^2), embeddings N(0, 1), every other matrix N(0, 1/fan-in)
    (a convolution's fan-in is C*kh*kw, position embeddings N(0, 0.02^2))."""
    if len(shape) == 1:
        if re.search(NORM_GAIN, name):
            return torch.ones(shape, dtype=torch.bfloat16)
        assert name.endswith(".bias"), name
        std = 0.02
    elif name.endswith("patch_embed.proj.weight") and len(shape) == 4:
        std = 1.0 / math.sqrt(math.prod(shape[1:]))
    elif name.endswith("pos_emb.weight") and len(shape) == 3:
        std = 0.02
    else:
        std = 1.0 if name.endswith("embed_tokens.weight") else 1.0 / math.sqrt(shape[1])
    if len(shape) != 2:
        generator = torch.Generator().manual_seed(tensor_seed(name))
        return (torch.randn(shape, generator=generator) * std).to(torch.bfloat16)
    return normal_rows(name, shape, std, torch.bfloat16)


def parameter_base(name):
    """A quantized weight's name for its scale and shape tensors too."""
    for suffix in ("_scale_inv", "_packed", "_scale", "_shape"):
        if name.endswith(".weight" + suffix):
            return name[: -len(suffix)]
    return name


def make_parameter(base, members):
    """Every tensor of one parameter: a BF16 tensor, DeepSeek-V3's FP32 router bias, an FP8 weight
    with its 128x128 block scales (`weight_scale_inv`, w ~ q * s), or a compressed-tensors INT4
    weight (`weight_packed`, group-32 BF16 `weight_scale`, `weight_shape`). A quantized weight is
    its N(0, 1/fan-in) FP32 master quantized."""
    if set(members) == {base}:
        dtype, shape = members[base]
        if dtype == "BF16":
            return {base: bf16_tensor(base, shape)}
        assert dtype == "F32", base
        generator = torch.Generator().manual_seed(tensor_seed(base))
        return {base: torch.randn(shape, generator=generator) * 0.02}
    if base + "_scale_inv" in members:
        shape = members[base][1]
        master = normal_rows(base, shape, 1.0 / math.sqrt(shape[1]), torch.float32)
        out_features, in_features = master.shape
        out_blocks, in_blocks = -(-out_features // 128), -(-in_features // 128)
        padded = torch.zeros(out_blocks * 128, in_blocks * 128, dtype=torch.float32)
        padded[:out_features, :in_features] = master
        blocks = padded.view(out_blocks, 128, in_blocks, 128)
        scale = blocks.abs().amax((1, 3)).clamp_min(1e-12) / 448.0
        quantized = (blocks / scale[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        return {
            base: quantized.view(out_blocks * 128, in_blocks * 128)[
                :out_features, :in_features
            ].contiguous(),
            base + "_scale_inv": scale.contiguous(),
        }
    out_features, in_features = (
        members[base + "_packed"][1][0],
        members[base + "_packed"][1][1] * 8,
    )
    master = normal_rows(
        base, (out_features, in_features), 1.0 / math.sqrt(in_features), torch.float32
    )
    groups = master.view(out_features, in_features // 32, 32)
    scale = (groups.abs().amax(-1).clamp_min(1e-8) / 7).to(torch.bfloat16)
    quantized = (
        (groups / scale.float()[..., None])
        .round()
        .clamp(-8, 7)
        .to(torch.int8)
        .view(out_features, in_features)
    )
    return {
        base + "_packed": pack_to_int32(quantized, 4),
        base + "_scale": scale.contiguous(),
        base + "_shape": torch.tensor([out_features, in_features], dtype=torch.int32),
    }


def base_shards(tensors, expert_shard_bytes):
    """The base model's files: the backbone in ~4 GiB shards, then the routed experts layer by layer
    (in shards of `expert_shard_bytes` if not None). A weight never leaves its scales."""
    groups = {}
    for name in tensors:
        groups.setdefault(parameter_base(name), []).append(name)

    def size(base):
        return sum(math.prod(tensors[n][1]) * ELEMENT_BYTES[tensors[n][0]] for n in groups[base])

    def order(name):
        return (layer_of(name) if layer_of(name) is not None else -1, name)

    def chunk(bases, limit):
        result, current, total = [], [], 0
        for base in bases:
            if current and limit is not None and total + size(base) > limit:
                result.append(current)
                current, total = [], 0
            current += sorted(groups[base])
            total += size(base)
        return [*result, current] if current else result

    shards = chunk(sorted((b for b in groups if ".mlp.experts." not in b), key=order), 4 * 1024**3)
    by_layer = {}
    for base in (b for b in groups if ".mlp.experts." in b):
        by_layer.setdefault(layer_of(base), []).append(base)
    for layer in sorted(by_layer):
        shards += chunk(sorted(by_layer[layer], key=order), expert_shard_bytes)
    return {
        f"model-{index + 1:05d}-of-{len(shards):05d}.safetensors": names
        for index, names in enumerate(shards)
    }


def write_base_shard(task):
    """One file of the base model: its tensors (name -> dtype and shape), parameter by parameter."""
    path, specs = task
    torch.set_num_threads(1)
    values = {}
    for base in dict.fromkeys(parameter_base(name) for name in specs):
        values.update(
            make_parameter(base, {n: specs[n] for n in specs if parameter_base(n) == base})
        )
    save_file({name: values[name] for name in specs}, path, metadata={"format": "pt"})
    return path


def write_base(official: Path, out: Path, expert_format: str, workers: int) -> None:
    tensors = official_tensors(official)
    # the measured fixtures kept a BF16 release's experts one file per layer
    shards = base_shards(tensors, None if expert_format == "bf16" else 4 * 1024**3)
    out.mkdir(parents=True)
    tasks = [
        (str(out / file), {name: tensors[name] for name in names})
        for file, names in shards.items()
    ]
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for path in pool.map(write_base_shard, tasks):
            print(path, flush=True)
    for path in sorted(official.iterdir()):
        if (
            path.is_file()
            and path.suffix in METADATA_SUFFIXES
            and path.name != "model.safetensors.index.json"
            and path.stat().st_size < 64 * 1024**2
        ):
            shutil.copyfile(path, out / path.name)
    total = sum(math.prod(shape) * ELEMENT_BYTES[dtype] for dtype, shape in tensors.values())
    weight_map = {name: file for file, names in shards.items() for name in names}
    index = dict(metadata=dict(total_size=total), weight_map=weight_map)
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")


def encode_weight(weight, expert_format):
    """A BF16 `[experts, outputs, inputs]` bank in the experts' storage format, and its scales:
    block-128 FP8 (FP32 scale per 128x128 block) or group-32 INT4 (two offset-8 nibbles per byte,
    a BF16 scale per 32 inputs)."""
    if expert_format == "bf16":
        return weight.contiguous().clone(), None
    experts, outputs, inputs = weight.shape
    if expert_format == "fp8_block128":
        blocks = (
            weight.float()
            .reshape(experts, outputs // 128, 128, inputs // 128, 128)
            .permute(0, 1, 3, 2, 4)
        )
        scale = blocks.abs().amax((-1, -2)).clamp_min(1e-8) / 448.0
        quant = (blocks / scale[..., None, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        return quant.permute(0, 1, 3, 2, 4).reshape(
            experts, outputs, inputs
        ).contiguous(), scale.contiguous()
    groups = weight.float().reshape(experts, outputs, inputs // 32, 32)
    scale = (groups.abs().amax(-1).clamp_min(1e-8) / 7).bfloat16()
    quant = (groups / scale.float()[..., None]).round().clamp(-8, 7).to(torch.int16) + 8
    quant = quant.reshape(experts, outputs, inputs).to(torch.uint8)
    return (quant[..., 0::2] | (quant[..., 1::2] << 4)).contiguous(), scale.contiguous()


def stored_bytes(expert_format, outputs, inputs):
    """One expert matrix's bytes in its storage format, scales included."""
    if expert_format == "bf16":
        return outputs * inputs * 2
    if expert_format == "fp8_block128":
        return outputs * inputs + (outputs // 128) * (inputs // 128) * 4
    return outputs * inputs // 2 + outputs * (inputs // 32) * 2


def write_folded(base: Path, out: Path, expert_format: str, prefetch: int) -> None:
    config = json.loads((base / "config.json").read_text())
    profile = model_profile(config)
    recipe = codec_recipe(profile)
    hidden, ffn, dispatch_rank, combine_rank = (
        profile.hidden,
        profile.ffn,
        recipe.dispatch_rank,
        recipe.combine_rank,
    )
    # a file holds one EP32 rank's consuming-layer experts (and backbone shards the same size)
    shard_bytes = max(1, profile.experts // 32) * (
        stored_bytes(expert_format, 2 * ffn, hidden + dispatch_rank)
        + stored_bytes(expert_format, combine_rank, ffn)
    )
    out.mkdir(parents=True)
    (out / "hf").mkdir()
    (out / "banks").mkdir()

    def write_shards(groups, name):
        """Save groups of tensors in files of up to `shard_bytes` (a group is never split), the
        i-th named `name(i)`. Return each tensor's file."""
        files, shard, size, number = {}, {}, 0, 0
        for group in groups:
            count = sum(value.numel() * value.element_size() for value in group.values())
            if shard and size + count > shard_bytes:
                save_file(shard, str(out / name(number)))
                shard, size, number = {}, 0, number + 1
            shard.update({key: value.contiguous() for key, value in group.items()})
            size += count
            files.update(dict.fromkeys(group, name(number)))
        save_file(shard, str(out / name(number)))
        return files

    # the backbone, read back from the base model
    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]

    def backbone():
        for key in (tensor_name for tensor_name in index if ".mlp.experts." not in tensor_name):
            with safe_open(str(base / index[key]), framework="pt", device="cpu") as reader:
                yield {key: reader.get_tensor(key).clone()}

    files = write_shards(backbone(), lambda i: f"hf/backbone-{i:05d}.safetensors")
    weight_map = {key: file.removeprefix("hf/") for key, file in files.items()}
    (out / "hf/config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "hf/model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}, indent=2) + "\n"
    )
    for path in sorted(base.iterdir()):
        if path.suffix != ".safetensors" and path.name not in (
            "config.json",
            "model.safetensors.index.json",
        ):
            shutil.copyfile(path, out / "hf" / path.name)

    # random readers, factors and codec: readers N(0, 1/input width), factors N(0, 1/F),
    # decoder N(0, 1/r_c), encoder N(0, 1/H), gain N(0.9, 0.08^2) clamped to [0.5, 1.3]
    # (the trained Qwen3-30B-A3B gains: mean 0.90, std 0.08)
    def normal(shape, std, kind, layer, expert=None):
        generator = torch.Generator().manual_seed(tensor_seed(f"folded:{kind}:{layer}:{expert}"))
        return (torch.randn(shape, generator=generator) * std).to(torch.bfloat16)

    def experts(layer, pool):
        """Each expert's readers and factors in the storage format, drawn `2 * prefetch` ahead."""
        width = hidden + (0 if reference_for(recipe, layer) is None else dispatch_rank)

        def draw(expert):
            readers = normal((2 * ffn, width), 1 / math.sqrt(width), "readers", layer, expert)
            factors = normal((combine_rank, ffn), 1 / math.sqrt(ffn), "factors", layer, expert)
            return readers, factors

        pending = {}
        for expert in range(profile.experts):
            for ahead in range(expert, min(expert + 2 * prefetch + 1, profile.experts)):
                if ahead not in pending:
                    pending[ahead] = pool.submit(draw, ahead)
            group = {}
            for kind, scale_kind, value in zip(
                ("readers", "factors"),
                ("reader_scale", "factor_scale"),
                pending.pop(expert).result(),
            ):
                quant, scale = encode_weight(value[None], expert_format)
                group[f"{layer}/{kind}/{expert}"] = quant[0]
                if scale is not None:
                    group[f"{layer}/{scale_kind}/{expert}"] = scale[0]
            yield group

    tensors = {}
    with ThreadPoolExecutor(prefetch) as pool:
        for layer in profile.moe_layers:
            codec = {
                f"{layer}/decoder": normal(
                    (hidden, combine_rank), 1 / math.sqrt(combine_rank), "decoder", layer
                )
            }
            if reference_for(recipe, layer) is not None:
                codec[f"{layer}/encoder"] = normal(
                    (dispatch_rank, hidden), 1 / math.sqrt(hidden), "encoder", layer
                )
                generator = torch.Generator().manual_seed(tensor_seed(f"folded:gain:{layer}:None"))
                gain = (0.9 + 0.08 * torch.randn((1, hidden), generator=generator)).clamp(0.5, 1.3)
                codec[f"{layer}/gain"] = gain.to(torch.bfloat16)
            save_file(codec, str(out / f"banks/codec-{layer:03d}.safetensors"))
            tensors.update(dict.fromkeys(codec, f"banks/codec-{layer:03d}.safetensors"))
            name = partial("banks/layer-{:03d}-{:04d}.safetensors".format, layer)
            tensors.update(write_shards(experts(layer, pool), name))
            print(f"layer {layer}", flush=True)
    manifest = dict(
        expert_format=expert_format,
        tensors={key: dict(file=file, key=key) for key, file in tensors.items()},
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True, help="the official HF snapshot")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6, help="processes writing base shards")
    parser.add_argument("--prefetch", type=int, default=8, help="threads drawing folded experts")
    args = parser.parse_args()
    expert_format = yaml.safe_load(args.model_config.read_text())["expert_format"]
    write_base(args.official, args.output / "base", expert_format, args.workers)
    write_folded(args.output / "base", args.output / "folded", expert_format, args.prefetch)
    write_meta(
        args.output,
        inputs={"official": args.official, "model_config": args.model_config},
        seed=SEED,
    )


if __name__ == "__main__":
    main()
