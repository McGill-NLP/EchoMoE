"""The model a teacher is fine-tuned from, as `configs/models/<model>.yaml` pins it: the base
weights with the instruct model's chat template and tokens.

    python -m echomoe.training.base_model configs/models/qwen3-30b-a3b.yaml \
        outputs/qwen3-30b-a3b/base

Qwen3-30B-A3B grafts the Instruct template, end-of-turn token and padding token onto the Base
tokenizer. Kanana-2 takes the instruct tokenizer and template, the template also writing bos
before the first turn. Moonlight's tiktoken tokenizer is converted to `tokenizer.json`, so no
remote code runs, and takes the Instruct template. Gemma 4 keeps the text weights of its
multimodal checkpoint as Gemma4ForCausalLM, the base tokenizer and the instruct template. Nemotron
3.5 Lightning keeps its base weights without the multi-token-prediction block and takes the
post-trained template with thinking off by default.
"""

import argparse
import json
import os
import shutil
import struct
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

from echomoe.runs import staged_directory, write_meta

# the files each model takes from its base and instruct snapshots, beside the weight shards
FILES = {
    "qwen3-30b-a3b": (
        (
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
        ),
        ("tokenizer_config.json",),
    ),
    "kanana-2-30b-a3b": (
        ("config.json", "model.safetensors.index.json"),
        ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"),
    ),
    "moonlight-16b-a3b": (
        ("config.json", "model.safetensors.index.json", "tiktoken.model", "tokenizer_config.json"),
        ("tokenizer_config.json",),
    ),
    "gemma4-26b-a4b": (
        ("config.json", "model.safetensors.index.json", "tokenizer.json"),
        ("tokenizer_config.json", "chat_template.jinja"),
    ),
    "nemotron-3.5-lightning-30b-a3b": (
        ("config.json", "model.safetensors.index.json"),
        ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"),
    ),
}
KANANA_TEMPLATE_BOS = "{{- '<|begin_of_text|>' }}"
# The split pattern of Moonlight's remote code (`tokenization_moonshot.py`), which reserves 256
# special ids plus two after the BPE ranks.
MOONLIGHT_PATTERN = "|".join(
    [
        r"""[\p{Han}]+""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*"""
        r"""[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+"""
        r"""[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)
MOONLIGHT_SPECIAL_TOKENS = 256 + 2
# Megatron's import maps each weight shard whole on every rank, which a node refuses on 8 ranks for
# Gemma 4's 49.9 GB shard. Copied weights are regrouped into shards of at most this size.
SHARD_BYTES = 4_000_000_000
GEMMA_TEXT_PREFIX = "model.language_model."
# With thinking off, the instruct template's generation prompt opens and closes an empty thought
# channel that the same conversation rendered for training does not have. Without it a prompt is a
# prefix of its training rendering.
GEMMA_EMPTY_THOUGHT = """        {%- if not enable_thinking -%}
            {{- '<|channel>thought\\n<channel|>' -}}
        {%- endif -%}
"""
NEMOTRON_THINKING_DEFAULT = (
    "{%- set enable_thinking = enable_thinking if enable_thinking is defined else True %}"
)
NEMOTRON_THINKING_OFF = NEMOTRON_THINKING_DEFAULT.replace("else True", "else False")
NEMOTRON_LAYER_CODES = {"mamba": "M", "moe": "E", "attention": "*", "mlp": "-"}


def _write_json(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def moonlight_tokenizer(tiktoken_model: Path, added_tokens_decoder: dict):
    """Moonlight's byte-level BPE, split pattern and special tokens as a `tokenizers.Tokenizer`.
    Special ids without a name are `<|reserved_token_{id}|>`, as the remote code names them."""
    from tiktoken.load import load_tiktoken_bpe
    from tokenizers import AddedToken
    from transformers.convert_slow_tokenizer import TikTokenConverter

    base = len(load_tiktoken_bpe(str(tiktoken_model)))
    names = {int(index): entry["content"] for index, entry in added_tokens_decoder.items()}
    tokenizer = TikTokenConverter(
        vocab_file=str(tiktoken_model), pattern=MOONLIGHT_PATTERN
    ).converted()
    tokenizer.add_special_tokens(
        [
            AddedToken(
                names.get(index, f"<|reserved_token_{index}|>"), special=True, normalized=False
            )
            for index in range(base, base + MOONLIGHT_SPECIAL_TOKENS)
        ]
    )
    return tokenizer


def copy_weights(base: Path, shards: list[str], staging: Path, rename) -> None:
    """The shards and index of the tensors `rename` gives a name (None drops one), copied byte for
    byte in their stored order into shards of at most SHARD_BYTES."""
    groups, filled, metadata = [[]], 0, None
    for shard in shards:
        with (base / shard).open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(length))
        if metadata is None:
            metadata = header.get("__metadata__")
        stored = sorted(
            (entry["data_offsets"], name)
            for name, entry in header.items()
            if name != "__metadata__" and rename(name) is not None
        )
        for (begin, end), name in stored:
            if groups[-1] and filled + end - begin > SHARD_BYTES:
                groups.append([])
                filled = 0
            groups[-1].append((base / shard, 8 + length + begin, end - begin, name, header[name]))
            filled += end - begin
    weight_map = {}
    for number, group in enumerate(groups, start=1):
        shard = f"model-{number:05d}-of-{len(groups):05d}.safetensors"
        layout, offset = {} if metadata is None else {"__metadata__": metadata}, 0
        for _path, _start, size, name, entry in group:
            name = rename(name)
            layout[name] = {
                "dtype": entry["dtype"],
                "shape": entry["shape"],
                "data_offsets": [offset, offset + size],
            }
            weight_map[name] = shard
            offset += size
        encoded = json.dumps(layout, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        with (staging / shard).open("wb") as out:
            out.write(struct.pack("<Q", len(encoded)) + encoded)
            for path, start, size, _name, _entry in group:
                with path.open("rb") as source:
                    source.seek(start)
                    while size:
                        block = source.read(min(size, 1 << 26))
                        out.write(block)
                        size -= len(block)
    total_size = sum((staging / shard).stat().st_size for shard in set(weight_map.values()))
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    _write_json(staging / "model.safetensors.index.json", index)


def build(model: Path, destination: Path) -> None:
    spec, name = yaml.safe_load(model.read_text()), model.stem
    base_files, instruct_files = FILES[name]
    tokens = spec["tokens"]
    with staged_directory(destination) as staging:
        # download the base snapshot and the instruct files at their pinned revisions
        base, instruct = spec["base"], spec["instruct"]
        index = Path(
            snapshot_download(
                base["repo"],
                revision=base["revision"],
                allow_patterns=["model.safetensors.index.json"],
            )
        )
        index = json.loads((index / "model.safetensors.index.json").read_text())
        shards = sorted(set(index["weight_map"].values()))
        base = Path(
            snapshot_download(
                base["repo"], revision=base["revision"], allow_patterns=[*base_files, *shards]
            )
        )
        instruct = Path(
            snapshot_download(
                instruct["repo"], revision=instruct["revision"], allow_patterns=instruct_files
            )
        )

        # hard-link the weight shards where the filesystem allows. Gemma 4 keeps its text weights
        # and Nemotron drops its multi-token-prediction block, which no stage trains.
        if name == "gemma4-26b-a4b":
            copy_weights(
                base,
                shards,
                staging,
                lambda tensor: (
                    "model." + tensor.removeprefix(GEMMA_TEXT_PREFIX)
                    if tensor.startswith(GEMMA_TEXT_PREFIX)
                    else None
                ),
            )
        elif name == "nemotron-3.5-lightning-30b-a3b":
            copy_weights(
                base, shards, staging, lambda tensor: None if tensor.startswith("mtp.") else tensor
            )
        else:
            shutil.copy2(base / "model.safetensors.index.json", staging)
            for shard in shards:
                try:
                    os.link((base / shard).resolve(), staging / shard)
                except OSError:
                    shutil.copy2(base / shard, staging / shard)

        # the config, tokenizer and template of each model
        config = json.loads((base / "config.json").read_text())
        if name == "qwen3-30b-a3b":
            for file in ("tokenizer.json", "vocab.json", "merges.txt"):
                shutil.copy2(base / file, staging / file)
            config["eos_token_id"] = tokens["eos"]
            tokenizer = json.loads((base / "tokenizer_config.json").read_text())
            graft = json.loads((instruct / "tokenizer_config.json").read_text())
            tokenizer |= {key: graft[key] for key in ("chat_template", "eos_token", "pad_token")}
            _write_json(staging / "tokenizer_config.json", tokenizer | {"model_max_length": 32768})
        elif name == "kanana-2-30b-a3b":
            config["eos_token_id"] = [tokens["eos"], tokens["eod"]]
            for file in ("tokenizer.json", "tokenizer_config.json"):
                shutil.copy2(instruct / file, staging / file)
            template = (instruct / "chat_template.jinja").read_text()
            (staging / "chat_template.jinja").write_text(KANANA_TEMPLATE_BOS + template)
        elif name == "gemma4-26b-a4b":
            config = config["text_config"] | {
                "architectures": ["Gemma4ForCausalLM"],
                "eos_token_id": [tokens["eod"], tokens["eos"]],
                "transformers_version": config["transformers_version"],
            }
            shutil.copy2(base / "tokenizer.json", staging / "tokenizer.json")
            shutil.copy2(instruct / "tokenizer_config.json", staging / "tokenizer_config.json")
            template = (instruct / "chat_template.jinja").read_text()
            assert template.count(GEMMA_EMPTY_THOUGHT) == 1, "the instruct template changed"
            (staging / "chat_template.jinja").write_text(template.replace(GEMMA_EMPTY_THOUGHT, ""))
        elif name == "nemotron-3.5-lightning-30b-a3b":
            # vLLM's own Nemotron-H config reads the layer pattern only as a string
            config["num_nextn_predict_layers"] = 0
            config["hybrid_override_pattern"] = "".join(
                NEMOTRON_LAYER_CODES[layer] for layer in config["layers_block_type"]
            )
            config["eos_token_id"] = [tokens["eod"], tokens["eos"]]
            for file in ("tokenizer.json", "tokenizer_config.json"):
                shutil.copy2(instruct / file, staging / file)
            # The template writes every assistant turn after `<think></think>`, which its
            # generation prompt opens only with thinking off.
            template = (instruct / "chat_template.jinja").read_text()
            assert template.count(NEMOTRON_THINKING_DEFAULT) == 1, "the template changed"
            (staging / "chat_template.jinja").write_text(
                template.replace(NEMOTRON_THINKING_DEFAULT, NEMOTRON_THINKING_OFF)
            )
        else:
            # Moonlight. Without its remote code, transformers' and vLLM's own DeepSeek-V3
            # classes load it
            del config["auto_map"]
            config["eos_token_id"] = [tokens["eos"], tokens["eod"]]
            tokenizer = json.loads((base / "tokenizer_config.json").read_text())
            moonlight_tokenizer(base / "tiktoken.model", tokenizer["added_tokens_decoder"]).save(
                str(staging / "tokenizer.json")
            )
            for key in ("auto_map", "chat_template"):
                del tokenizer[key]
            tokenizer["tokenizer_class"] = "PreTrainedTokenizerFast"
            _write_json(staging / "tokenizer_config.json", tokenizer)
            template = json.loads((instruct / "tokenizer_config.json").read_text())
            (staging / "chat_template.jinja").write_text(template["chat_template"])
        _write_json(staging / "config.json", config)
        generation = {
            "bos_token_id": tokens["bos"],
            "do_sample": False,
            # Gemma 4's and Nemotron's instruct configs list the document end first
            "eos_token_id": (
                [tokens["eod"], tokens["eos"]]
                if name in ("gemma4-26b-a4b", "nemotron-3.5-lightning-30b-a3b")
                else [tokens["eos"], tokens["eod"]]
            ),
            "pad_token_id": tokens["pad"],
            "max_new_tokens": 2048,
        }
        _write_json(staging / "generation_config.json", generation)
        write_meta(
            staging,
            inputs={
                "model": str(model),
                "base": f"{spec['base']['repo']}@{spec['base']['revision']}",
                "instruct": f"{spec['instruct']['repo']}@{spec['instruct']['revision']}",
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path, help="configs/models/<model>.yaml")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    build(args.model, args.destination)
