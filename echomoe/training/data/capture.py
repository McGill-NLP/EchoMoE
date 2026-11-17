"""Activation capture for the offline fit: the seeded token selection and the on-disk format of the
MoE inputs, outputs and router logits recorded for it (`teacher.py` runs the vLLM pass).

A part `part-NNNNN/` holds safetensors files `<split>/layer-LLLL/part-FFFFFF`, sorted by
coordinate, with int64 `dataset_rows`, `token_indices`, bf16 `moe_input`, `moe_output` and
`router_logits` in the dtype the model routes on (bf16 for Qwen3 and DeepSeek-V3, float32 for
Gemma 4). Layers are MoE ordinals. Its `meta.json` holds the `files` list, `metadata`, `layers`,
`hidden_size`, `num_experts`, `topk` and `renormalize`, a sigmoid router's `router` (its
`routed_scaling_factor` and every layer's `expert_biases`) and Gemma 4's every layer's
`expert_scales`.
"""

import hashlib
import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import save_file

from echomoe.runs import atomic_write_json, publish_staged_directory

SPLITS = ("fit", "validation")


@dataclass(frozen=True, order=True)
class ActivationCoordinate:
    split: str
    dataset_row: int
    token_index: int


def _select_split(layout, *, split, rows, seed, max_tokens_per_document):
    # A segment of n tokens is an n-1 token prompt, so its last position has no hidden state. The
    # Dolci train layout holds 1.16 billion such positions, so spans are kept and sampling indexes
    # into them.
    spans = defaultdict(list)
    for dataset_row, start, length in zip(
        layout.dataset_rows.tolist(),
        layout.token_starts.tolist(),
        layout.raw_token_lengths.tolist(),
    ):
        spans[dataset_row].append(range(start, start + length - 1))
    counts = {row: sum(len(span) for span in items) for row, items in spans.items()}
    # The "codec" namespace predates the rename. Changing it changes which tokens are drawn.
    key = f"codec-pca-selection:{max_tokens_per_document}:{seed}:{split}"
    generator = random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))
    document_order = sorted(spans)
    generator.shuffle(document_order)
    selected = []
    remaining = rows
    for dataset_row in document_order:
        take = min(max_tokens_per_document, counts[dataset_row], remaining)
        for index in generator.sample(range(counts[dataset_row]), take):
            for span in spans[dataset_row]:
                if index < len(span):
                    break
                index -= len(span)
            selected.append(ActivationCoordinate(split, dataset_row, span[index]))
        remaining -= take
        if not remaining:
            break
    if remaining:
        raise ValueError(f"{split} needs {rows} rows but only {rows - remaining} are selectable")
    return selected


def select_coordinates(
    fit_layout, validation_layout, *, fit_rows, validation_rows, seed, max_tokens_per_document
):
    common = {"seed": seed, "max_tokens_per_document": max_tokens_per_document}
    fit = _select_split(fit_layout, split="fit", rows=fit_rows, **common)
    validation = _select_split(
        validation_layout, split="validation", rows=validation_rows, **common
    )
    return (*fit, *validation)


def partition_coordinates(coordinates, *, index, count, seed):
    """Part `index` of `count`: documents dealt round-robin in a seeded order, per split."""
    selected = []
    for split in SPLITS:
        by_document = defaultdict(list)
        for coordinate in coordinates:
            if coordinate.split == split:
                by_document[coordinate.dataset_row].append(coordinate)
        # The hashed key fixes which worker owns which document, so its exact byte form is part
        # of the artifact: changing it repartitions old captures.
        deal_order = sorted(
            by_document,
            key=lambda row: hashlib.sha256(
                json.dumps([seed, split, row], separators=(",", ":")).encode()
            ).digest(),
        )
        for ordinal, dataset_row in enumerate(deal_order):
            if ordinal % count == index:
                selected.extend(sorted(by_document[dataset_row]))
    return tuple(selected)


@dataclass(frozen=True)
class CaptureMetadata:
    part_index: int
    part_count: int


@dataclass(frozen=True)
class ActivationCapture:
    path: Path
    meta: dict
    metadata: CaptureMetadata
    layers: tuple
    hidden_size: int
    num_experts: int
    topk: int
    renormalize: bool


def load_activation_capture(path):
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    part = meta["metadata"]
    return ActivationCapture(
        path,
        meta,
        CaptureMetadata(part["part_index"], part["part_count"]),
        tuple(meta["layers"]),
        *(meta[key] for key in ("hidden_size", "num_experts", "topk", "renormalize")),
    )


class ActivationCaptureWriter:
    """Buffers rows per (layer, split) into files of `rows_per_file` rows. `finish` renames the
    staged part into place, so an interrupted part restarts."""

    def __init__(self, path, *, expected_coordinates, rows_per_file, record):
        self.path = Path(path)
        self.staging = self.path.with_name(f".{self.path.name}.incomplete")
        shutil.rmtree(self.staging, ignore_errors=True)
        self.staging.mkdir(parents=True)
        self.rows_per_file = rows_per_file
        self.record = dict(record)
        self.expected = {
            split: {item for item in expected_coordinates if item.split == split}
            for split in SPLITS
        }
        self.files = []
        self.written = defaultdict(int)
        self.seen = defaultdict(set)
        self.buffer = {}

    def add(self, layer, coordinates, moe_input, moe_output, router_logits):
        assert moe_input.dtype == moe_output.dtype == torch.bfloat16
        assert router_logits.dtype in (torch.bfloat16, torch.float32)
        for split in SPLITS:
            key = (layer, split)
            rows = [index for index, item in enumerate(coordinates) if item.split == split]
            if not rows:
                continue
            batch = frozenset(coordinates[row] for row in rows)
            assert len(batch) == len(rows), "repeated coordinate"
            assert not batch & self.seen[key], "repeated coordinate"
            assert batch <= self.expected[split], "coordinate was never selected"
            self.seen[key] |= batch
            index = torch.tensor(rows, dtype=torch.int64, device=moe_input.device)
            fresh = [
                value.index_select(0, index).detach().to("cpu")
                for value in (moe_input, moe_output, router_logits)
            ]
            held = self.buffer.get(key)
            coordinates_held = tuple(coordinates[row] for row in rows)
            if held is not None:
                coordinates_held = held[0] + coordinates_held
                fresh = [torch.cat((old, new)) for old, new in zip(held[1:], fresh)]
            self.buffer[key] = (coordinates_held, *fresh)
            while key in self.buffer and len(self.buffer[key][0]) >= self.rows_per_file:
                self._write_file(key)

    def _write_file(self, key):
        coordinates, moe_input, moe_output, logits = self.buffer[key]
        rows = min(len(coordinates), self.rows_per_file)
        order = sorted(range(rows), key=lambda position: coordinates[position])
        ordered = [coordinates[position] for position in order]
        index = torch.tensor(order, dtype=torch.int64)
        layer, split = key
        relative = f"{split}/layer-{layer:04d}/part-{self.written[key]:06d}.safetensors"
        self.written[key] += 1
        (self.staging / relative).parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "dataset_rows": torch.tensor(
                    [coordinate.dataset_row for coordinate in ordered], dtype=torch.int64
                ),
                "token_indices": torch.tensor(
                    [coordinate.token_index for coordinate in ordered], dtype=torch.int64
                ),
                "moe_input": moe_input.index_select(0, index).contiguous(),
                "moe_output": moe_output.index_select(0, index).contiguous(),
                "router_logits": logits.index_select(0, index).contiguous(),
            },
            self.staging / relative,
        )
        self.files.append(relative)
        if rows == len(coordinates):
            del self.buffer[key]
        else:
            self.buffer[key] = (
                coordinates[rows:],
                moe_input[rows:],
                moe_output[rows:],
                logits[rows:],
            )

    def finish(self):
        while self.buffer:
            self._write_file(next(iter(self.buffer)))
        for layer in self.record["layers"]:
            for split in SPLITS:
                if self.seen[layer, split] != self.expected[split]:
                    raise ValueError(f"layer {layer} is missing {split} rows")
        atomic_write_json(self.staging / "meta.json", self.record | {"files": sorted(self.files)})
        publish_staged_directory(self.staging, self.path)
