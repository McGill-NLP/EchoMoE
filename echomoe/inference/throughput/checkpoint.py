"""Read a folded student's export for the throughput runtime: its folded config, one layer's
tensors for this rank's experts, and the backbone vLLM loads by name. `FoldedExport` reads a
trained student (`runs/<run>/export/step-N`, written by `echomoe.inference.evaluation.export`),
`PortableExport` a larger model's student: random weights written by `fixtures.py`, or the trained
Kanana-2 and Gemma 4 students in the same layout."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from torch.nn import functional

# The per-layer folded tensors (`model.layers.L.mlp.<name>`). Every other tensor is backbone.
FOLDED_NAMES = ("readers", "factors", "encoder", "decoder", "reference_gain")


class FoldedExport:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.folded = json.loads((self.path / "config.json").read_text())["folded"]
        index = json.loads((self.path / "model.safetensors.index.json").read_text())
        self.index = index["weight_map"]
        # The runtime serves a direct owner with both sides compressed, and caches one reference
        # at a time, so a consuming layer must read the nearest earlier reference layer.
        folded = self.folded
        assert not folded["combine_offsets"], "direct owners only"
        assert folded["correction_width"] is None, "direct owners only"
        assert folded["dispatch_rank"] is not None, "both sides compressed"
        assert folded["combine_rank"] is not None, "both sides compressed"
        references = folded["reference_layers"]
        assert references[0] == 0, "every layer before the first reference would read no cache"
        for layer, source in folded["reference_layer_for"].items():
            nearest = max(earlier for earlier in references if earlier < int(layer))
            assert source == nearest, (layer, source)

    def read(self, name: str, experts: range | None = None) -> torch.Tensor:
        with safe_open(self.path / self.index[name], framework="pt", device="cpu") as handle:
            if experts is None:
                tensor = handle.get_tensor(name)
            else:
                tensor = handle.get_slice(name)[experts.start : experts.stop]
            # a copy, so the shard's mapping closes: every shard of an export mapped at once would
            # pass a 32 GiB data-segment limit
            return tensor.clone()

    def layer(self, index: int, experts: range) -> dict[str, torch.Tensor | None]:
        """Layer `index`'s readers and factors for `experts`, its decoder and, at a consuming
        layer, its encoder and reference gain (as `[1, H]`)."""
        folded = self.folded
        hidden, width = folded["hidden_size"], folded["moe_intermediate_size"]
        dispatch, combine = folded["dispatch_rank"], folded["combine_rank"]
        consuming = str(index) in folded["reference_layer_for"]
        name = f"model.layers.{index}.mlp."
        tensors = {
            "readers": self.read(name + "readers", experts),
            "factors": self.read(name + "factors", experts),
            "encoder": self.read(name + "encoder") if consuming else None,
            "gain": self.read(name + "reference_gain")[None] if consuming else None,
            "decoder": self.read(name + "decoder"),
        }
        shapes = {
            "readers": (len(experts), 2 * width, hidden + (dispatch if consuming else 0)),
            "factors": (len(experts), combine, width),
            "encoder": (dispatch, hidden),
            "gain": (1, hidden),
            "decoder": (hidden, combine),
        }
        for key, value in tensors.items():
            assert value is None or (value.shape, value.dtype) == (shapes[key], torch.bfloat16), (
                f"layer {index} {key}: {value.shape} {value.dtype}"
            )
        return tensors

    def backbone(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Every tensor but the folded ones, shard by shard."""
        for shard in sorted(set(self.index.values())):
            with safe_open(self.path / shard, framework="pt", device="cpu") as handle:
                for name in handle.keys():  # noqa: SIM118
                    if name.rsplit(".", 1)[1] not in FOLDED_NAMES:
                        yield name, handle.get_tensor(name)


def model_profile(config: dict) -> SimpleNamespace:
    """The routed experts of a Qwen3-MoE, Gemma 4 or DeepSeek-V3-style config (Kimi-K2.6 wraps
    one)."""
    text = config.get("text_config", config)
    layers = text["num_hidden_layers"]
    top_k = text.get("num_experts_per_tok")
    if text["model_type"] == "gemma4_text":
        # a MoE block beside the dense MLP of every layer, with GELU-tanh experts
        assert text["enable_moe_block"]
        assert text["hidden_activation"] == "gelu_pytorch_tanh"
        experts, top_k = text["num_experts"], text["top_k_experts"]
        moe_layers = list(range(layers))
        shared, scoring, groups, selected, scale, bias = 0, "softmax", 1, 1, 1.0, False
    elif text["model_type"] == "qwen3_moe":
        dense, step = text.get("mlp_only_layers", []), text.get("decoder_sparse_step", 1)
        experts = text["num_experts"]
        moe_layers = [
            layer for layer in range(layers) if layer not in dense and (layer + 1) % step == 0
        ]
        shared, scoring, groups, selected, scale, bias = 0, "softmax", 1, 1, 1.0, False
    else:
        first, step = text.get("first_k_dense_replace", 0), text.get("moe_layer_freq", 1)
        experts = text["n_routed_experts"]
        moe_layers = [layer for layer in range(first, layers) if layer % step == 0]
        shared = text.get("n_shared_experts") or 0
        scoring = text.get("scoring_func", "softmax")
        groups, selected = text.get("n_group", 1), text.get("topk_group", 1)
        scale = text.get("routed_scaling_factor", 1.0)
        bias = text.get("topk_method") == "noaux_tc"
    return SimpleNamespace(
        family=config["model_type"],
        hidden=text["hidden_size"],
        ffn=text["moe_intermediate_size"],
        experts=experts,
        top_k=top_k,
        layers=layers,
        moe_layers=moe_layers,
        shared_experts=shared,
        scoring_func=scoring,
        normalized=text.get("norm_topk_prob", True),
        groups=groups,
        selected_groups=selected,
        routed_scale=float(scale),
        correction_bias=bias,
    )


def codec_recipe(profile: SimpleNamespace) -> SimpleNamespace:
    """The larger models' compression: dispatch rank H/4 and combine rank H/2 (rounded up to
    multiples of 128) and a reference every 4 MoE layers."""
    hidden = profile.hidden
    return SimpleNamespace(
        dispatch_rank=((hidden + 511) // 512) * 128,
        combine_rank=((hidden + 255) // 256) * 128,
        stride=4,
        references=profile.moe_layers[::4],
    )


def reference_for(recipe: SimpleNamespace, layer: int) -> int | None:
    """The reference layer a MoE layer reads (None at a reference layer)."""
    if layer in recipe.references:
        return None
    return max(earlier for earlier in recipe.references if earlier < layer)


class PortableExport:
    """A larger model's folded student: `manifest.json` names each folded tensor's file and key
    (`<layer>/<readers|factors|reader_scale|factor_scale>/<expert>`, `<layer>/<encoder|gain|decoder>`)
    and, for a trained student, its recipe. `hf/` holds the config and the backbone.

    DeepEP's combine takes rows of a multiple of 256 values (hybrid_combine.cuh), so the combine
    codes travel zero-padded to one (Gemma 4: 1408 to 1536): the factors gain zero output rows and
    the decoder zero input columns, which leaves the output unchanged."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        manifest = json.loads((self.path / "manifest.json").read_text())
        self.expert_format = manifest["expert_format"]
        self.tensors = manifest["tensors"]
        self.config = json.loads((self.path / "hf/config.json").read_text())
        self.profile = model_profile(self.config)
        self.recipe = codec_recipe(self.profile)
        if "recipe" in manifest:
            trained = manifest["recipe"]
            recipe = self.recipe
            assert (trained["dispatch_rank"], trained["combine_rank"], trained["references"]) == (
                recipe.dispatch_rank,
                recipe.combine_rank,
                recipe.references,
            ), trained
        self.recipe.combine_rank = -(-self.recipe.combine_rank // 256) * 256
        index = json.loads((self.path / "hf/model.safetensors.index.json").read_text())
        self.index = index["weight_map"]

    def read(self, name: str) -> torch.Tensor:
        entry = self.tensors[name]
        with safe_open(self.path / entry["file"], framework="pt", device="cpu") as handle:
            return handle.get_tensor(entry["key"]).clone()

    def layer(self, index: int, experts: range) -> dict[str, torch.Tensor | int | None]:
        """Layer `index`'s readers and factors for `experts` (with their scales unless BF16), its
        decoder and, at a consuming layer, its encoder, gain (`[1, H]`) and reference layer."""
        reference = reference_for(self.recipe, index)
        quantized = self.expert_format != "bf16"
        factors = torch.stack([self.read(f"{index}/factors/{expert}") for expert in experts])
        decoder = self.read(f"{index}/decoder")
        padding = self.recipe.combine_rank - decoder.shape[1]
        if padding:
            factors = functional.pad(factors, (0, 0, 0, padding))
            decoder = functional.pad(decoder, (0, padding))
        return {
            "readers": torch.stack([self.read(f"{index}/readers/{expert}") for expert in experts]),
            "factors": factors,
            "reader_scale": torch.stack(
                [self.read(f"{index}/reader_scale/{expert}") for expert in experts]
            )
            if quantized
            else None,
            "factor_scale": torch.stack(
                [self.read(f"{index}/factor_scale/{expert}") for expert in experts]
            )
            if quantized
            else None,
            "encoder": None if reference is None else self.read(f"{index}/encoder"),
            "gain": None if reference is None else self.read(f"{index}/gain"),
            "decoder": decoder,
            "reference": reference,
        }

    def backbone(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Every backbone tensor, in the index's order."""
        for name, shard in self.index.items():
            with safe_open(self.path / "hf" / shard, framework="pt", device="cpu") as handle:
                yield name, handle.get_tensor(name).clone()
