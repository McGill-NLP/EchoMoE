"""Teacher target caches: the top-1024 next-token targets distillation reads.

Rows are keyed by layout segment. Segment s contributes `raw_token_lengths[s] - 1` rows in layout
order, exactly the positions its packed batch's prediction mask selects. A part `part-NNNNN/` owns
the segments of the packs whose index is its index modulo `part_count`, in shards of whole
segments. A shard holds int32 `segment_ids` plus int32 `ids` and f32 `logprobs` `[rows, top_m]`
and f32 `tail`. The part's `meta.json` is rewritten after each shard, so a rerun computes only
what is missing. `teacher.py` runs the vLLM pass that fills them.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from echomoe.runs import atomic_write_json

from .batches import derive_pack_order, planned_segments, source_queue


def top_m_targets(logits, top_m):
    """The top-M ids and log-probabilities of teacher logits at temperature one, and the log-mass
    outside them."""
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    values, indices = torch.topk(logprobs, top_m, dim=-1)
    # The tail is a log-sum-exp rather than log1p(-top mass), which keeps full precision where
    # the top-M mass rounds to one. It is -inf when the top M cover the vocabulary, and a consumer
    # must apply 0 log 0 = 0 there.
    tail = torch.logsumexp(logprobs.scatter_(1, indices, float("-inf")), dim=-1)
    return {
        "ids": indices.to(torch.int32),
        "logprobs": values.to(torch.float32),
        "tail": tail.to(torch.float32),
    }


def covered_segments(part):
    part = Path(part)
    if not (part / "meta.json").is_file():
        return set()
    covered = set()
    for name in json.loads((part / "meta.json").read_text())["files"]:
        with safe_open(part / name, framework="pt") as handle:
            covered.update(handle.get_tensor("segment_ids").tolist())
    return covered


class SegmentWriter:
    """Holds rows per segment until all its positions are in, then writes whole-segment shards."""

    def __init__(self, path, *, segments, segment_rows, rows_per_file, record):
        self.path = Path(path)
        self.segment_rows = dict(zip(map(int, segments), map(int, segment_rows)))
        self.rows_per_file = rows_per_file
        self.record = dict(record)
        self.path.mkdir(parents=True, exist_ok=True)
        # adopt the shards meta.json lists and delete whatever else a preempted pass left
        meta = self.path / "meta.json"
        self.files = list(json.loads(meta.read_text())["files"]) if meta.is_file() else []
        for item in self.path.iterdir():
            if item.name not in {"meta.json", *self.files}:
                item.unlink()
        self.done = covered_segments(self.path)
        # Re-running one layout under a different part_count would move segments between parts.
        if not self.done <= set(self.segment_rows):
            raise ValueError(f"{self.path} holds segments outside this part's plan")
        self.partial, self.buffer, self.buffered_rows = {}, [], 0

    def add(self, segment, start, tensors):
        assert segment not in self.done, f"segment {segment} is already published"
        # A preempted request is recomputed from its first position, so a restart replaces
        # whatever that segment already buffered.
        held = [] if start == 0 else self.partial.get(segment, [])
        assert sum(len(item["logprobs"]) for item in held) == start, "non-contiguous span"
        held.append({name: value.detach().to("cpu") for name, value in tensors.items()})
        filled = start + len(tensors["logprobs"])
        assert filled <= self.segment_rows[segment], f"segment {segment} got too many positions"
        if filled < self.segment_rows[segment]:
            self.partial[segment] = held
            return
        self.partial.pop(segment, None)
        self.done.add(segment)
        self.buffer.append(
            (segment, {name: torch.cat([part[name] for part in held]) for name in held[0]})
        )
        self.buffered_rows += filled
        while self.buffered_rows >= self.rows_per_file:
            self._write_shard()

    def _write_shard(self):
        # the lowest buffered segments that fit one shard, never splitting a segment
        ordered = sorted(self.buffer, key=lambda item: item[0])
        taken = rows = 0
        for _segment, tensors in ordered:
            if taken and rows + len(tensors["logprobs"]) > self.rows_per_file:
                break
            taken += 1
            rows += len(tensors["logprobs"])
        held = ordered[:taken]
        name = f"shard-{len(self.files):06d}.safetensors"
        shard = {"segment_ids": torch.tensor([segment for segment, _ in held], dtype=torch.int32)}
        shard |= {key: torch.cat([tensors[key] for _, tensors in held]) for key in held[0][1]}
        save_file(shard, self.path / f"{name}.tmp")
        (self.path / f"{name}.tmp").replace(self.path / name)
        self.files.append(name)
        self.buffer = ordered[taken:]
        self.buffered_rows -= rows
        atomic_write_json(self.path / "meta.json", self.record | {"files": self.files})

    def finish(self):
        while self.buffer:
            self._write_shard()
        missing = set(self.segment_rows) - self.done
        if missing:
            raise ValueError(f"{self.path} is missing {len(missing)} planned segments")
        atomic_write_json(self.path / "meta.json", self.record | {"files": self.files})


def plan_part(config, part_index, layout):
    """This part's segments of the run's budgeted pack stream. Parts own whole packs."""
    offsets = np.asarray(layout.pack_offsets, dtype=np.int64)
    orders = derive_pack_order(config["order_seed"], config["order_epochs"], len(layout))
    queue = source_queue(layout, orders, token_budget=config["token_budget"])
    planned = planned_segments(layout, queue)
    pack_of_segment = np.searchsorted(offsets, planned, side="right") - 1
    return planned[pack_of_segment % config["part_count"] == part_index]


@dataclass(frozen=True)
class TargetCache:
    shards: tuple
    shard_of_segment: np.ndarray
    first_row_of_segment: np.ndarray
    rows_of_segment: np.ndarray
    row_shapes: dict
    dtypes: dict

    def read(self, segments):
        """The cached rows of these segments, concatenated in the order given."""
        wanted = np.asarray(segments, dtype=np.int64)
        uncovered = wanted[self.shard_of_segment[wanted] < 0]
        if len(uncovered):
            raise ValueError(f"the target cache does not cover layout segment {uncovered[0]}")
        rows = self.rows_of_segment[wanted]
        starts = np.concatenate(([0], np.cumsum(rows)))
        output = {
            name: torch.empty((int(starts[-1]), *shape), dtype=self.dtypes[name])
            for name, shape in self.row_shapes.items()
        }
        # read each shard once, copying every requested segment's rows into place
        spans = {}
        for position, segment in enumerate(wanted.tolist()):
            spans.setdefault(int(self.shard_of_segment[segment]), []).append(
                (
                    int(starts[position]),
                    int(self.first_row_of_segment[segment]),
                    int(rows[position]),
                )
            )
        for shard, group in sorted(spans.items()):
            with safe_open(self.shards[shard], framework="pt") as handle:
                for name, values in output.items():
                    stored = handle.get_slice(name)
                    for destination, source, count in group:
                        values[destination : destination + count] = stored[source : source + count]
        return output


def load_target_cache(directory, layout):
    """Index the shards of a cache's parts against the layout they were built from."""
    lengths = np.asarray(layout.raw_token_lengths, dtype=np.int64) - 1
    shard_of_segment = np.full(len(lengths), -1, dtype=np.int64)
    first_row_of_segment = np.zeros(len(lengths), dtype=np.int64)
    shards, row_shapes, dtypes = [], {}, {}
    for part in sorted(Path(directory).glob("part-?????")):
        for name in json.loads((part / "meta.json").read_text())["files"]:
            with safe_open(part / name, framework="pt") as handle:
                segments = handle.get_tensor("segment_ids").numpy().astype(np.int64)
                rows = handle.get_slice("logprobs").get_shape()[0]
                for key in sorted(set(handle.keys()) - {"segment_ids"} - set(row_shapes)):
                    first = handle.get_slice(key)[0:1]
                    row_shapes[key], dtypes[key] = tuple(first.shape[1:]), first.dtype
            if (shard_of_segment[segments] >= 0).any():
                raise ValueError(f"target cache parts hold the same segment twice: {part / name}")
            offsets = np.concatenate(([0], np.cumsum(lengths[segments])))
            if int(offsets[-1]) != rows:
                raise ValueError(f"{part / name} rows differ from this layout's segments")
            shard_of_segment[segments] = len(shards)
            first_row_of_segment[segments] = offsets[:-1]
            shards.append(part / name)
    return TargetCache(
        tuple(shards), shard_of_segment, first_row_of_segment, lengths, row_shapes, dtypes
    )
