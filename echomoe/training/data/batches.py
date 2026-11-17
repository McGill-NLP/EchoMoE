"""Seeded pack order, the annealed two-source mixture plan, and THD batch assembly.

A compressor run mixes a token-budgeted prefix of the Dolmino stream with the Dolci stream into
optimizer steps of `packs_per_update` packs, both sources in random slots of every step. The Dolci
share stays at `f0`, then rises linearly to its end value, with `f0` solved so the Dolci queue is
consumed exactly.
"""

import math

import numpy as np
import torch

MINIMUM_START_FRACTION = 0.05
SEQUENCE_KEYS = (
    "tokens",
    "labels",
    "lm_mask",
    "prediction_mask",
    "padding_mask",
    "position_ids",
    "source_index",
    "target_ids",
    "target_logprobs",
    "target_tail_logprob",
)


def derive_pack_order(seed, epochs, pack_count):
    """One seeded permutation of the layout's packs per epoch."""
    return tuple(
        np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, epoch])))
        .permutation(pack_count)
        .astype(np.int64, copy=False)
        for epoch in range(epochs)
    )


def sft_pack_id(
    epoch_orders,
    *,
    microbatch_index,
    pack_index,
    data_parallel_rank,
    data_parallel_size,
    micro_batch_size,
):
    """The SFT pack at stream slot `(microbatch * mbs + pack) * dp + rank`, None past the end."""
    stream_index = (microbatch_index * micro_batch_size + pack_index) * data_parallel_size
    stream_index += data_parallel_rank
    if stream_index >= sum(len(order) for order in epoch_orders):
        return None
    epoch, offset = divmod(stream_index, len(epoch_orders[0]))
    return int(epoch_orders[epoch][offset])


def sft_total_microbatches(
    epoch_orders, *, data_parallel_size, gradient_accumulation_steps, micro_batch_size
):
    """Microbatches per rank that cover every pack, padded to whole updates."""
    packs_per_update = data_parallel_size * gradient_accumulation_steps * micro_batch_size
    updates = math.ceil(sum(len(order) for order in epoch_orders) / packs_per_update)
    return updates * gradient_accumulation_steps


def planned_segments(layout, pack_ids):
    """The sorted unique layout segments of these packs."""
    offsets = np.asarray(layout.pack_offsets, dtype=np.int64)
    taken = np.zeros(len(offsets) - 1, dtype=bool)
    taken[pack_ids] = True
    return np.flatnonzero(np.repeat(taken, np.diff(offsets)))


def source_queue(layout, epoch_orders, *, token_budget):
    """The pack ids one source contributes to a run, in order: the concatenated epochs up to and
    including the pack that reaches `token_budget` scored tokens, or all of them."""
    offsets = np.asarray(layout.pack_offsets, dtype=np.int64)
    scored_by_pack = np.add.reduceat(
        np.asarray(layout.raw_token_lengths, dtype=np.int64) - 1, offsets[:-1]
    )
    stream = np.concatenate(epoch_orders)
    consumed = np.cumsum(scored_by_pack[stream])
    if token_budget is None:
        packs = len(stream)
    else:
        if token_budget > int(consumed[-1]):
            raise ValueError(f"token budget {token_budget} exceeds the layout's {consumed[-1]}")
        packs = int(np.searchsorted(consumed, token_budget, side="left")) + 1
    return np.asarray(stream[:packs], dtype=np.int64)


def anneal_curve(first_packs, second_packs, *, packs_per_update, anneal_start, end_dolci_fraction):
    if not (0.0 < anneal_start < 1.0 and 0.0 < end_dolci_fraction <= 1.0):
        raise ValueError("anneal_start must lie in (0, 1) and end_dolci_fraction in (0, 1]")
    width = packs_per_update
    real_packs = first_packs + second_packs
    steps = -(-real_packs // width)
    capacity = np.full(steps, width, dtype=np.int64)
    capacity[-1] = real_packs - width * (steps - 1)
    ramp_start = math.ceil(anneal_start * steps)
    if ramp_start >= steps:
        raise ValueError(f"anneal_start {anneal_start} leaves no step for the ramp")
    ramp = np.zeros(steps, dtype=np.float64)
    ramp[ramp_start:] = np.arange(1, steps - ramp_start + 1) / (steps - ramp_start)
    f0 = float(
        (second_packs - end_dolci_fraction * (capacity * ramp).sum())
        / (capacity * (1.0 - ramp)).sum()
    )
    if not MINIMUM_START_FRACTION <= f0 <= end_dolci_fraction:
        raise ValueError(f"the queue sizes imply a constant share {f0:.4f} outside the anneal")
    return steps, capacity, f0, f0 * (1.0 - ramp) + end_dolci_fraction * ramp


def build_annealed_schedule(
    first, second, *, packs_per_update, mix_seed, anneal_start, end_dolci_fraction
):
    """The queue (0 `first`, 1 `second`) and pack id of every slot of every optimizer step, as two
    `[steps, packs_per_update]` arrays with fillers -1."""
    width = packs_per_update
    steps, capacity, _f0, fraction = anneal_curve(
        len(first),
        len(second),
        packs_per_update=width,
        anneal_start=anneal_start,
        end_dolci_fraction=end_dolci_fraction,
    )
    # round the second queue's per-step share by largest remainder, ties to the later step
    targets = fraction * capacity
    second_counts = np.floor(targets).astype(np.int64)
    short = len(second) - int(second_counts.sum())
    order = np.lexsort((-np.arange(len(targets)), -(targets - second_counts)))
    second_counts[order[:short]] += 1
    if (capacity - second_counts < 0).any():
        raise ValueError("the last step cannot hold its share of the second source")

    slot_source = np.full((steps, width), -1, dtype=np.int8)
    slot_pack = np.full((steps, width), -1, dtype=np.int64)
    cursors = [0, 0]
    for step in range(steps):
        labels = np.full(width, -1, dtype=np.int8)
        labels[: second_counts[step]] = 1
        labels[second_counts[step] : capacity[step]] = 0
        generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence([mix_seed, step])))
        slot_source[step] = labels[generator.permutation(width)]
        for index, queue in enumerate((first, second)):
            slots = np.flatnonzero(slot_source[step] == index)
            slot_pack[step, slots] = queue[cursors[index] : cursors[index] + len(slots)]
            cursors[index] += len(slots)
    return slot_source, slot_pack


def assemble_packed_batch(bank, layout, *, pack_id, pad_id, sequence_length, alignment):
    """One layout pack as a THD row, each segment shifted on its own and padded to `alignment`.
    A `pack_id` of None is a filler pack that scores nothing."""
    columns = {name: [] for name in SEQUENCE_KEYS[:6]}

    def extend(*values):
        for column, value in zip(columns.values(), values, strict=True):
            column.extend(value)

    cu_unpadded, cu_padded, padded_lengths = [0], [0], []
    start = stop = 0
    if pack_id is not None:
        start, stop = int(layout.pack_offsets[pack_id]), int(layout.pack_offsets[pack_id + 1])
    for segment in range(start, stop):
        token_start = int(layout.token_starts[segment])
        raw_length = int(layout.raw_token_lengths[segment])
        row = bank[int(layout.dataset_rows[segment])]
        ids = row["input_ids"][token_start : token_start + raw_length]
        mask = row["response_mask"] if "response_mask" in row else [False] * len(row["input_ids"])
        mask = mask[token_start : token_start + raw_length]
        shifted = raw_length - 1
        aligned = (shifted + alignment - 1) // alignment * alignment
        padding = aligned - shifted
        extend(
            ids[:-1],
            ids[1:],
            map(bool, mask[1:]),
            [True] * shifted,
            [False] * shifted,
            range(shifted),
        )
        extend(
            [pad_id] * padding,
            [0] * padding,
            [False] * padding,
            [False] * padding,
            [True] * padding,
            [0] * padding,
        )
        cu_unpadded.append(cu_unpadded[-1] + shifted)
        cu_padded.append(cu_padded[-1] + aligned)
        padded_lengths.append(aligned)
    trailing = sequence_length - len(columns["tokens"])
    assert trailing >= 0, "packed layout exceeds the sequence length"
    if trailing:
        extend(*([value] * trailing for value in (pad_id, 0, False, False, True, 0)))
        cu_unpadded.append(cu_unpadded[-1])
        cu_padded.append(sequence_length)
        padded_lengths.append(trailing)
    dtypes = (torch.int64, torch.int64, torch.bool, torch.bool, torch.bool, torch.int64)
    batch = {
        name: torch.tensor([values], dtype=dtype)
        for (name, values), dtype in zip(columns.items(), dtypes, strict=True)
    }
    return batch | {
        "cu_seqlens_unpadded": torch.tensor(cu_unpadded, dtype=torch.int32),
        "cu_seqlens_padded": torch.tensor(cu_padded, dtype=torch.int32),
        "max_seqlen": torch.tensor([max(padded_lengths)], dtype=torch.int32),
        "pack_id": torch.tensor(-1 if pack_id is None else pack_id, dtype=torch.int64),
        "segment_count": torch.tensor(stop - start, dtype=torch.int32),
    }


def concatenate_packed_batches(batches):
    """Join packs into one THD row with shifted cu_seqlens and a vector of pack ids."""
    length = int(batches[0]["tokens"].shape[1])
    unpadded = [batches[0]["cu_seqlens_unpadded"]]
    padded = [batches[0]["cu_seqlens_padded"]]
    for index, batch in enumerate(batches[1:], start=1):
        unpadded.append(batch["cu_seqlens_unpadded"][1:] + unpadded[-1][-1])
        padded.append(batch["cu_seqlens_padded"][1:] + index * length)
    joined = {
        name: torch.cat([batch[name] for batch in batches], dim=1)
        for name in SEQUENCE_KEYS
        if name in batches[0]
    }
    return joined | {
        "cu_seqlens_unpadded": torch.cat(unpadded),
        "cu_seqlens_padded": torch.cat(padded),
        "max_seqlen": torch.stack([batch["max_seqlen"] for batch in batches]).max(dim=0).values,
        "pack_id": torch.stack([batch["pack_id"].reshape(()) for batch in batches]),
        "segment_count": torch.stack([batch["segment_count"] for batch in batches]).sum(),
    }


def packed_batch_to_device(batch, device):
    """Move the model tensors. `pack_id` and `segment_count` stay on the host for accounting."""
    host = {"pack_id", "segment_count"}
    return {
        name: value.to(device) if torch.is_tensor(value) and name not in host else value
        for name, value in batch.items()
    }
