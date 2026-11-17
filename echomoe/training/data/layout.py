"""Token banks and their packed 8192-token layouts.

A token bank holds Arrow shards in `data/` that `datasets.load_from_disk` opens, an `index.npz`
and a `meta.json` with its `kind`. A layout packs bank rows into packs of `SEQUENCE_LENGTH` slots,
and its `meta.json` records its kind and partition. A segment of n tokens trains as n-1 shifted
positions padded to a multiple of `ALIGNMENT`, and its flat segment index keys both target caches.
A pack is padded with the model's padding id (`pad_token_id`).
"""

import heapq
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from datasets import load_from_disk

SEQUENCE_LENGTH = 8192
ALIGNMENT = 8
ARRAYS = ("pack_offsets", "dataset_rows", "token_starts", "raw_token_lengths")


@dataclass(frozen=True)
class PackedLayout:
    path: Path
    meta: dict
    pack_offsets: np.ndarray
    dataset_rows: np.ndarray
    token_starts: np.ndarray
    raw_token_lengths: np.ndarray

    def __len__(self):
        return len(self.pack_offsets) - 1


@dataclass(frozen=True)
class TokenizedDataset:
    path: Path
    meta: dict
    dataset: object
    row_ids: np.ndarray
    raw_token_lengths: np.ndarray
    shifted_response_counts: np.ndarray | None

    def __len__(self):
        return len(self.row_ids)

    def __getitem__(self, dataset_row):
        row = self.dataset[dataset_row]
        result = {"input_ids": list(row["input_ids"])}
        if self.shifted_response_counts is not None:
            result["response_mask"] = [bool(value) for value in row["response_mask"]]
        return result


def load_tokenized_dataset(path):
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    index = np.load(path / "index.npz")
    shifted = index["shifted_response_counts"] if meta["kind"] == "sft" else None
    dataset = load_from_disk(str(path / "data"), keep_in_memory=False)
    return TokenizedDataset(
        path, meta, dataset, index["row_ids"], index["raw_token_lengths"], shifted
    )


def pad_token_id(model):
    """The padding id of a model directory, which its generation config holds. The base model
    writes it from `configs/models/<model>.yaml` and every export copies it."""
    return json.loads((Path(model) / "generation_config.json").read_text())["pad_token_id"]


def load_layout(path):
    path = Path(path)
    arrays = [np.load(path / f"{name}.npy", mmap_mode="r") for name in ARRAYS]
    return PackedLayout(path, json.loads((path / "meta.json").read_text()), *arrays)


class _Fenwick:
    def __init__(self, size):
        self.tree = [0] * (size + 1)

    def add(self, index, delta):
        index += 1
        while index < len(self.tree):
            self.tree[index] += delta
            index += index & -index

    def prefix(self, index):
        total, index = 0, index + 1
        while index:
            total += self.tree[index]
            index -= index & -index
        return total

    def first_with_prefix(self, target):
        index, bit = 0, 1 << ((len(self.tree) - 1).bit_length() - 1)
        while bit:
            candidate = index + bit
            if candidate < len(self.tree) and self.tree[candidate] < target:
                index = candidate
                target -= self.tree[candidate]
            bit >>= 1
        return index


def best_fit_decreasing(segments):
    units = SEQUENCE_LENGTH // ALIGNMENT
    buckets = [[] for _ in range(units + 1)]
    for segment in segments:
        buckets[(segment[2] - 1 + ALIGNMENT - 1) // ALIGNMENT].append(segment)
    remaining_buckets = [[] for _ in range(units + 1)]
    population = _Fenwick(units + 1)
    packs = []
    # largest first, each into the fullest pack it fits (ties to the lowest pack id), a size class
    # in (dataset_row, token_start) order
    for size in range(units, 0, -1):
        for segment in sorted(buckets[size]):
            before = population.prefix(size - 1)
            if before == population.prefix(units):
                pack_id, remaining = len(packs), units
                packs.append([])
            else:
                remaining = population.first_with_prefix(before + 1)
                pack_id = heapq.heappop(remaining_buckets[remaining])
                population.add(remaining, -1)
            packs[pack_id].append(segment)
            heapq.heappush(remaining_buckets[remaining - size], pack_id)
            population.add(remaining - size, 1)
    return packs


def build_layout(raw_token_lengths, dataset_rows, *, kind):
    """The layout arrays of bank rows `dataset_rows`. Dolmino rows longer than a pack become
    overlap-one spans, so every target is scored once. Dolci rows that do not fit are dropped."""
    # cut rows into (dataset_row, token_start, raw_length) segments
    segments = []
    for dataset_row, raw_length in zip(dataset_rows.tolist(), raw_token_lengths.tolist()):
        if kind == "pretrain":
            for token_start in range(0, raw_length - 1, SEQUENCE_LENGTH):
                segments.append(
                    (dataset_row, token_start, min(SEQUENCE_LENGTH + 1, raw_length - token_start))
                )
        elif 2 <= raw_length <= SEQUENCE_LENGTH + 1:
            segments.append((dataset_row, 0, raw_length))
    # pack them, then flatten the packs into the four layout arrays
    packs = best_fit_decreasing(segments)
    flat = [segment for pack in packs for segment in pack]
    columns = np.asarray(flat, dtype=np.int64).reshape(-1, 3)
    return {
        "pack_offsets": np.cumsum([0, *map(len, packs)], dtype=np.int64),
        "dataset_rows": columns[:, 0].copy(),
        "token_starts": columns[:, 1].astype(np.int32),
        "raw_token_lengths": columns[:, 2].astype(np.int32),
    }


def write_layout(path, arrays, *, kind, partition):
    path.mkdir(parents=True)
    for name, array in arrays.items():
        np.save(path / f"{name}.npy", array)
    meta = {
        "kind": kind,
        "partition": partition,
        "sequence_length": SEQUENCE_LENGTH,
        "alignment": ALIGNMENT,
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
