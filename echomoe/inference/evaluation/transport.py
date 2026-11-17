"""What the served folded student and its full-width baseline send between expert-parallel ranks.

vLLM runs attention data-parallel and hands each MoE layer one data-parallel rank's rows. With
tensor parallelism 1 the expert-parallel group is that group, and a rank owns a contiguous block of
experts. The owner routes its rows and sends what the experts read through DeepEP's elastic buffer:
nothing at a reference layer, the code at a non-reference layer, the activation where the dispatch
side is off. A reference layer publishes its rows along rails and then inside nodes, and a receiver
finds each row's reference from the source rank and row the dispatch handle records. Each receiver
sums its experts' weighted outputs per row and the combine sums those rows back into the owner's.

An eager forward waits on the host for each receive count. A graphed forward (FULL_DECODE_ONLY's
decode batches) cannot read anything back, so every rank holds the same padded batch and receives
a fixed `batch * ranks` rows, with the rows past the device-side count masked. Simulated FP8 is
DeepEP's per-token cast, one float8_e4m3fn byte per element and a float32 scale per 128. It covers
the dispatched rows, the published reference and each expert rank's per-token partial sums, which
the owners exchange directly and add as they read.
"""

from __future__ import annotations

import socket
from collections.abc import Sequence
from typing import Any

import torch
import triton
import triton.language as language
from deep_ep import ElasticBuffer
from torch.nn import functional
from vllm.config import CompilationMode, CUDAGraphMode
from vllm.distributed import get_ep_group
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.forward_context import get_forward_context

FP8_MAX = 448.0
AMAX_FLOOR = 1e-4
BLOCK = 128
# DeepEP's queue-pair budget assumes a bare ConnectX-7 and asks for more than the cluster's SR-IOV
# functions allocate. 65 is the largest count that allocates there and the fastest measured.
ALLOCATED_QPS = 65
# Streaming multiprocessors per dispatch and combine kernel (measured fastest at EP32).
NUM_SMS = 24
# The elastic combine kernel is compiled per width and takes multiples of 256 elements.
PAYLOAD_GRANULARITY = 256


# --- FP8 casts --------------------------------------------------------------------------------


def fp8_cast(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows `[n, width]` as float8_e4m3fn with a float32 scale per block of 128, `[n, blocks]`."""
    if values.device.type == "cuda" and values.shape[0] > 0:
        return _fp8_cast_fused(values)
    return _fp8_cast_rows(values)


def fp8_cast_back(cast: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if cast.device.type == "cuda" and cast.shape[0] > 0:
        return _fp8_cast_back_fused(cast, scale, dtype)
    return _fp8_cast_back_rows(cast, scale).to(dtype)


def fp8_fake_quant(values: torch.Tensor) -> torch.Tensor:
    """`values` cast to FP8 and back over its last dimension."""
    rows = values.reshape(-1, values.shape[-1])
    payload, scale = fp8_cast(rows)
    return fp8_cast_back(payload, scale, values.dtype).reshape(values.shape)


def _fp8_cast_rows(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # DeepEP 2.1.0's per_token_cast_to_fp8, used for empty batches and as the kernel's reference.
    rows, width = values.shape[0], values.shape[1]
    count = -(-width // BLOCK)
    blocks = functional.pad(values, (0, -width % BLOCK)).view(rows, count, BLOCK)
    amax = blocks.abs().float().amax(dim=2, keepdim=True).clamp(AMAX_FLOOR)
    cast = (blocks * (FP8_MAX / amax)).to(torch.float8_e4m3fn)
    scale = amax / FP8_MAX
    return cast.view(rows, count * BLOCK)[:, :width].contiguous(), scale.view(rows, count)


def _fp8_cast_back_rows(cast: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    rows, width = cast.shape
    count = -(-width // BLOCK)
    blocks = functional.pad(cast.to(torch.float32), (0, -width % BLOCK)).view(rows, count, BLOCK)
    return (blocks * scale.view(rows, count, 1)).view(rows, count * BLOCK)[:, :width]


@triton.jit
def _cast_kernel(
    x_ptr, out_ptr, scale_ptr, width, count, block: language.constexpr, floor, maximum
):
    row = language.program_id(0)
    index = language.program_id(1)
    offsets = index * block + language.arange(0, block)
    inside = offsets < width
    values = language.load(x_ptr + row * width + offsets, mask=inside, other=0.0)
    values = values.to(language.float32)
    amax = language.maximum(language.max(language.abs(values)), floor)
    # torch divides a python float by a tensor as a multiply by its correctly rounded reciprocal.
    # Triton's fast float32 divide lands an ulp away and would move a payload byte on a tie.
    reciprocal = language.inline_asm_elementwise(
        "rcp.rn.f32 $0, $1;", "=r,r", [amax], dtype=language.float32, is_pure=True, pack=1
    )
    language.store(
        out_ptr + row * width + offsets,
        (values * (reciprocal * maximum)).to(language.float8e4nv),
        mask=inside,
    )
    language.store(scale_ptr + row * count + index, amax / maximum)


@triton.jit
def _cast_back_kernel(cast_ptr, scale_ptr, out_ptr, width, count, block: language.constexpr):
    row = language.program_id(0)
    index = language.program_id(1)
    offsets = index * block + language.arange(0, block)
    inside = offsets < width
    values = language.load(cast_ptr + row * width + offsets, mask=inside, other=0.0)
    scale = language.load(scale_ptr + row * count + index)
    restored = values.to(language.float32) * scale
    language.store(
        out_ptr + row * width + offsets, restored.to(out_ptr.dtype.element_ty), mask=inside
    )


def _fp8_cast_fused(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, width = values.shape
    count = -(-width // BLOCK)
    payload = torch.empty((rows, width), dtype=torch.float8_e4m3fn, device=values.device)
    scale = torch.empty((rows, count), dtype=torch.float32, device=values.device)
    _cast_kernel[(rows, count)](
        values.contiguous(),
        payload,
        scale,
        width,
        count,
        block=BLOCK,
        floor=AMAX_FLOOR,
        maximum=FP8_MAX,
        # The reference multiplies by a float32 factor, and a contraction would not match it.
        enable_fp_fusion=False,
    )
    return payload, scale


def _fp8_cast_back_fused(
    cast_rows: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    rows, width = cast_rows.shape
    count = -(-width // BLOCK)
    out = torch.empty((rows, width), dtype=dtype, device=cast_rows.device)
    _cast_back_kernel[(rows, count)](
        cast_rows.contiguous(),
        scale.contiguous(),
        out,
        width,
        count,
        block=BLOCK,
        enable_fp_fusion=False,
    )
    return out


@triton.jit
def _scatter_add_kernel(
    payload_ptr,
    scales_ptr,
    rows_ptr,
    out_ptr,
    width: language.constexpr,
    blocks: language.constexpr,
    topk: language.constexpr,
    block: language.constexpr,
    size: language.constexpr,
):
    token = language.program_id(0)
    offsets = language.program_id(1) * size + language.arange(0, size)
    inside = offsets < width
    total = language.zeros((size,), dtype=language.float32)
    for slot in range(topk):
        row = language.load(rows_ptr + token * topk + slot)
        taken = row >= 0
        source = language.where(taken, row, 0)
        values = language.load(
            payload_ptr + source * width + offsets, mask=inside & taken, other=0.0
        ).to(language.float32)
        scale = language.load(
            scales_ptr + source * blocks + offsets // block, mask=inside & taken, other=0.0
        )
        total += values * scale
    language.store(out_ptr + token * width + offsets, total, mask=inside)


def fp8_scatter_add(
    payload: torch.Tensor, scales: torch.Tensor, rows: torch.Tensor, width: int, dtype: torch.dtype
) -> torch.Tensor:
    """Each token's FP8 partial sums, dequantized as read and added in float32 in `k` order.
    `rows[t, k]` is the row carrying the k-th partial sum of token t, or -1."""
    tokens, topk = rows.shape
    out = torch.empty((tokens, width), dtype=dtype, device=payload.device)
    size = 256
    _scatter_add_kernel[(tokens, -(-width // size))](
        payload,
        scales,
        rows,
        out,
        width=width,
        blocks=scales.shape[1],
        topk=topk,
        block=BLOCK,
        size=size,
        # Fusing the multiply and add would not match the float32 reference.
        enable_fp_fusion=False,
    )
    return out


# --- Layout -----------------------------------------------------------------------------------


def payload_width(width: int) -> int:
    return width + (-width % PAYLOAD_GRANULARITY)


def padded(values: torch.Tensor) -> torch.Tensor:
    padding = payload_width(values.shape[-1]) - values.shape[-1]
    return values if padding == 0 else functional.pad(values, (0, padding))


def received_sources(handle: Any, received_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    # The elastic kernel writes `source_rank * num_max_tokens_per_rank + row`.
    encoded = handle.recv_src_metadata[:received_rows, 0].to(torch.int64)
    source = torch.div(encoded, handle.num_max_tokens_per_rank, rounding_mode="floor")
    return source, encoded % handle.num_max_tokens_per_rank


def group_hosts(group: Any) -> list[str]:
    hosts: list[Any] = [None] * torch.distributed.get_world_size(group)
    torch.distributed.all_gather_object(hosts, socket.gethostname(), group=group)
    return hosts


def node_layout(
    hosts: Sequence[str], global_ranks: Sequence[int]
) -> tuple[list[int], list[list[int]], list[list[int]]]:
    """The rail then node all-gather over `hosts`: each rank's block of the gathered rows, the
    rails (ranks with one local index) and the nodes, by global rank as new_group orders them."""
    size = len(global_ranks)
    members: dict[str, list[int]] = {}
    for rank, host in enumerate(hosts):
        members.setdefault(host, []).append(rank)
    counts = {len(ranks) for ranks in members.values()}
    assert len(counts) == 1, "every host must hold the same number of expert-parallel ranks"
    nodes = [sorted(ranks, key=global_ranks.__getitem__) for ranks in members.values()]
    per_node = counts.pop()
    local = {rank: position for ranks in nodes for position, rank in enumerate(ranks)}
    rails = [
        sorted(
            (rank for rank in range(size) if local[rank] == index),
            key=global_ranks.__getitem__,
        )
        for index in range(per_node)
    ]
    rail_position = {rank: position for rail in rails for position, rank in enumerate(rail)}
    blocks = [local[rank] * len(nodes) + rail_position[rank] for rank in range(size)]
    return blocks, rails, nodes


def reference_slots(
    reference_layers: Sequence[int], reference_layer_for: dict[int, int]
) -> dict[int, int]:
    """A store slot for each reference layer, held from the layer to the last layer reading it."""
    last = {
        reference: max(
            (layer for layer, source in reference_layer_for.items() if source == reference),
            default=reference,
        )
        for reference in reference_layers
    }
    slots: dict[int, int] = {}
    held: dict[int, int] = {}
    for reference in sorted(reference_layers):
        free = sorted(slot for slot, until in held.items() if until < reference)
        slot = free[0] if free else len(held)
        slots[reference] = slot
        held[slot] = last[reference]
    return slots


def local_ids(ids: torch.Tensor, experts: int) -> torch.Tensor:
    """Received top-k ids, with another rank's slots (-1) moved to the spare expert that
    `local_expert_map` drops."""
    return ids.masked_fill(ids < 0, experts).to(torch.int32)


def local_expert_map(experts: int, device: Any) -> torch.Tensor:
    """Map this rank's expert ids to themselves and the spare id to -1."""
    mapping = torch.arange(experts + 1, dtype=torch.int32, device=device)
    mapping[experts] = -1
    return mapping


def forward_state() -> tuple[torch.Tensor, bool]:
    context = get_forward_context()
    graphed = context.cudagraph_runtime_mode != CUDAGraphMode.NONE
    return context.dp_metadata.num_tokens_across_dp_cpu, graphed


# --- Communicators ----------------------------------------------------------------------------


def _communicator(group: Any, ranks: list[int]) -> Any:
    if len(ranks) == 1:
        return None
    global_ranks = torch.distributed.get_process_group_ranks(group)
    gloo = torch.distributed.new_group(
        [global_ranks[rank] for rank in ranks], backend="gloo", use_local_synchronization=True
    )
    communicator = PyNcclCommunicator(gloo, device=torch.cuda.current_device())
    assert not communicator.disabled, "vLLM's NCCL wrapper did not initialize"
    return communicator


def _gather(communicator: Any, output: torch.Tensor, rows: torch.Tensor, sizes: list[int]) -> None:
    if communicator is None:
        output.copy_(rows)
    elif len(set(sizes)) == 1:
        communicator.all_gather(output, rows)
    else:
        communicator.all_gatherv(output, rows, sizes)


def _exchange(
    communicator: Any, rank: int, blocks: tuple[tuple[torch.Tensor, torch.Tensor], ...]
) -> None:
    size = blocks[0][0].shape[0]
    for sent, back in blocks:
        back[rank].copy_(sent[rank])
    if communicator is None:
        return
    communicator.group_start()
    for peer in range(size):
        if peer == rank:
            continue
        for sent, back in blocks:
            communicator.send(sent[peer], peer)
            communicator.recv(back[peer], peer)
    communicator.group_end()


def _exchange_exact(
    communicator: Any,
    rank: int,
    sends: Sequence[int],
    backs: Sequence[int],
    pairs: tuple[tuple[torch.Tensor, torch.Tensor], ...],
) -> None:
    send_offsets = [sum(sends[:peer]) for peer in range(len(sends))]
    back_offsets = [sum(backs[:peer]) for peer in range(len(backs))]
    for sent, back in pairs:
        back[back_offsets[rank] : back_offsets[rank] + backs[rank]].copy_(
            sent[send_offsets[rank] : send_offsets[rank] + sends[rank]]
        )
    if communicator is None:
        return
    communicator.group_start()
    for peer in range(len(sends)):
        if peer == rank:
            continue
        for sent, back in pairs:
            if sends[peer]:
                communicator.send(
                    sent[send_offsets[peer] : send_offsets[peer] + sends[peer]], peer
                )
            if backs[peer]:
                communicator.recv(
                    back[back_offsets[peer] : back_offsets[peer] + backs[peer]], peer
                )
    communicator.group_end()


# --- Transport --------------------------------------------------------------------------------


class ExpertParallelTransport:
    """One worker's elastic buffer, reference store and expert placement."""

    def __init__(
        self,
        *,
        group: Any,
        rank: int,
        size: int,
        num_experts: int,
        router_topk: int,
        hidden_size: int,
        tokens_per_rank: int,
        reference_slots: dict[int, int] | None,
        fp8_dispatch: bool,
        fp8_combine: bool,
    ) -> None:
        self.group = group
        self.rank = rank
        self.size = size
        self.num_experts = num_experts
        self.local_experts = num_experts // size
        self.expert_offset = rank * self.local_experts
        self.router_topk = router_topk
        # The widest payload a layer sends is the full-width row.
        self.buffer_hidden = payload_width(hidden_size)
        self.tokens_per_rank = tokens_per_rank
        self.reference_slots = reference_slots
        self.fp8_dispatch = fp8_dispatch
        self.fp8_combine = fp8_combine
        self.buffer: Any = None
        self.num_qps = 0
        self.rail: Any = None
        self.node: Any = None
        self.everyone: Any = None
        self.exchange: tuple[torch.Tensor, ...] = ()
        self.rail_size = 1
        self.rails: list[list[int]] = []
        self.rail_members: list[int] = []
        self.node_members: list[int] = []
        self.block_ranks: list[int] = []
        self.rail_sizes: list[int] = []
        self.node_sizes: list[int] = []
        self.gathered_rows = 0
        self.block_offsets: torch.Tensor | None = None
        self.blocks: torch.Tensor | None = None
        # The gathered rows per slot (bf16, or FP8 bytes and float32 scales), and along one rail.
        self.stores: dict[int, tuple[torch.Tensor, ...]] = {}
        self.rail_rows: tuple[torch.Tensor, ...] = ()
        self.rows_per_rank = 0
        self.graphed = False

    def _connect(self, device: Any) -> None:
        # The first NCCL communicator created after the elastic buffer took its queue pairs failed
        # at EP=16, so the rail and node communicators come first.
        hosts = group_hosts(self.group)
        if self.reference_slots is not None:
            global_ranks = torch.distributed.get_process_group_ranks(self.group)
            self.set_layout(*node_layout(hosts, global_ranks), device=device)
            self.rail = _communicator(self.group, self.rail_members)
            self.node = _communicator(self.group, self.node_members)
        if self.fp8_combine and self.size > 1:
            self.everyone = _communicator(self.group, list(range(self.size)))
        # torch creates a subgroup's NCCL communicator at its first collective, and DeepEP takes
        # the communicator from the group.
        torch.distributed.all_reduce(torch.zeros(1, device="cuda"), group=self.group)
        self.buffer = ElasticBuffer(
            self.group,
            num_max_tokens_per_rank=self.tokens_per_rank,
            hidden=self.buffer_hidden,
            num_topk=self.router_topk,
            num_allocated_qps=ALLOCATED_QPS,
        )
        self.num_qps = min(self.buffer.get_theoretical_num_qps(NUM_SMS), ALLOCATED_QPS)
        nodes = len(set(hosts))
        domain = tuple(self.buffer.get_logical_domain_size())
        assert domain == (nodes, self.size // nodes), f"elastic logical domains {domain}"

    def set_layout(
        self, blocks: list[int], rails: list[list[int]], nodes: list[list[int]], *, device: Any
    ) -> None:
        self.rails = rails
        self.rail_members = next(ranks for ranks in rails if self.rank in ranks)
        self.node_members = next(ranks for ranks in nodes if self.rank in ranks)
        self.rail_size = len(self.rail_members)
        self.block_ranks = sorted(range(len(blocks)), key=blocks.__getitem__)
        self.blocks = torch.tensor(blocks, dtype=torch.int64, device=device)

    def _reference_layout(self, tokens: list[int], device: Any) -> None:
        # References travel unpadded, so a source rank's rows start at its block's offset.
        self.rail_sizes = [tokens[rank] for rank in self.rail_members]
        # A node member at local position p gathered rail p.
        self.node_sizes = [sum(tokens[rank] for rank in rail) for rail in self.rails]
        if self.graphed:
            # One padded batch on every rank: the offsets need nothing from the host at replay.
            self.block_offsets = None
            return
        starts = [0]
        for rank in self.block_ranks[:-1]:
            starts.append(starts[-1] + tokens[rank])
        self.block_offsets = torch.tensor(starts, dtype=torch.int64, device=device)

    def begin(self, layer: int, rows: torch.Tensor) -> None:
        """Open a layer. Layer 0 reads the batch layout and the path, which every rank agrees on
        because vLLM takes the minimum cudagraph mode over the data-parallel ranks every step."""
        if self.buffer is None:
            self._connect(rows.device)
        if layer != 0:
            return
        tokens, self.graphed = forward_state()
        assert int(tokens[self.rank]) == rows.shape[0], "forward context and batch disagree"
        self.rows_per_rank = int(tokens.max())
        assert not self.graphed or int(tokens.min()) == self.rows_per_rank, tokens.tolist()
        batches = [int(value) for value in tokens.tolist()]
        # One rank receives at most every other rank's rows.
        self.gathered_rows = sum(batches)
        if self.reference_slots is not None:
            self._reference_layout(batches, rows.device)

    def publish(self, layer: int, rows: torch.Tensor) -> None:
        """Gather this layer's reference from every rank, each rank's rows as they are."""
        published = rows.contiguous()
        # FP8 bytes travel as uint8, which NCCL reduces nothing over and copies exactly.
        parts = (
            tuple(
                part.view(torch.uint8) if part.dtype == torch.float8_e4m3fn else part
                for part in fp8_cast(published)
            )
            if self.fp8_dispatch
            else (published,)
        )
        slot = self.reference_slots[layer]
        if slot not in self.stores:
            self.stores[slot] = tuple(
                part.new_empty((self.tokens_per_rank * self.size, part.shape[1])) for part in parts
            )
        if not self.rail_rows:
            self.rail_rows = tuple(
                part.new_empty((self.tokens_per_rank * self.rail_size, part.shape[1]))
                for part in parts
            )
        stores = tuple(store[: self.gathered_rows] for store in self.stores[slot])
        across = tuple(rail[: sum(self.rail_sizes)] for rail in self.rail_rows)
        for part, along, store in zip(parts, across, stores):
            _gather(self.rail, along, part, self.rail_sizes)
            _gather(self.node, store, along, self.node_sizes)

    def dispatch(
        self, payload: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any, torch.Tensor]:
        """Send each row to its experts' ranks. Returns the received rows, their local expert ids
        (-1 for another rank's slot or a row past the receive count), weights, handle and mask."""
        # Every rank sends one width, so a zero-width reference-layer dispatch stays bf16.
        fp8 = self.fp8_dispatch and payload.shape[1] > 0
        tokens_per_rank = self.rows_per_rank if self.graphed else None
        received, ids, weights, handle, _event = self.buffer.dispatch(
            fp8_cast(payload) if fp8 else payload.contiguous(),
            topk_idx=topk_ids.to(torch.int64),
            topk_weights=topk_weights.float(),
            num_experts=self.num_experts,
            num_max_tokens_per_rank=tokens_per_rank,
            num_sms=NUM_SMS,
            num_qps=self.num_qps,
            do_cpu_sync=tokens_per_rank is None,
        )
        if fp8:
            received = fp8_cast_back(*received, dtype=payload.dtype)
        count = handle.psum_num_recv_tokens_per_scaleup_rank[-1]
        valid = torch.arange(received.shape[0], device=received.device) < count
        ids = ids.masked_fill(~valid[:, None], -1)
        weights = weights.masked_fill(~valid[:, None], 0.0)
        return received, ids, weights, handle, valid

    def reference_rows(
        self, layer: int, handle: Any, valid: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        """The published reference of each received row. Masked rows read row 0."""
        source, row = received_sources(handle, valid.shape[0])
        source = source.masked_fill(~valid, 0)
        row = row.masked_fill(~valid, 0)
        blocks = self.blocks.index_select(0, source)
        if self.block_offsets is None:
            positions = blocks * self.rows_per_rank + row
        else:
            positions = self.block_offsets.index_select(0, blocks) + row
        stores = self.stores[self.reference_slots[layer]]
        if not self.fp8_dispatch:
            return stores[0].index_select(0, positions)
        payload, scales = (store.index_select(0, positions) for store in stores)
        return fp8_cast_back(payload.view(torch.float8_e4m3fn), scales, dtype=dtype)

    def combine(
        self, rows: torch.Tensor, handle: Any, valid: torch.Tensor, topk_ids: torch.Tensor
    ) -> torch.Tensor:
        """Sum every receiver's rows back into their owners' rows. `valid` marks the receiver's
        real rows and `topk_ids`, the owner's routing, says where its rows come back from."""
        width = rows.shape[-1]
        if not self.fp8_combine:
            combined, _weights, _event = self.buffer.combine(
                padded(rows).contiguous(), handle, num_qps=self.num_qps
            )
            return combined[:, :width]
        payload, scales = fp8_cast(rows)
        owner, row = received_sources(handle, valid.shape[0])
        blocks = scales.shape[1]
        if self.graphed:
            return self._square_combine(
                payload, scales, owner, row, valid, topk_ids, width, blocks, rows.dtype
            )
        return self._exact_combine(
            payload, scales, owner, row, valid, topk_ids, width, blocks, rows.dtype
        )

    def _square_combine(
        self,
        payload: torch.Tensor,
        scales: torch.Tensor,
        owner: torch.Tensor,
        row: torch.Tensor,
        valid: torch.Tensor,
        topk_ids: torch.Tensor,
        width: int,
        blocks: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # A graph exchanges a fixed square of `rows_per_rank` slots per pair of ranks, and slot
        # `(r, t)` of what comes back is rank r's partial sum of this rank's token t.
        slots = self.rows_per_rank
        # One spare row past the square takes the received rows past the receive count. A slot of
        # the square that no row lands in is one this rank was not routed, so nobody reads it.
        square = slots * self.size
        place = torch.where(valid, owner * slots + row, square)
        sent_payload, sent_scales, _, back_payload, back_scales, _ = self._exchange_rows(
            square, width, blocks, payload.device
        )
        sent_payload.index_copy_(0, place, payload.view(torch.uint8))
        sent_scales.index_copy_(0, place, scales)
        _exchange(
            self.everyone,
            self.rank,
            (
                (
                    sent_payload[:square].view(self.size, slots, width),
                    back_payload.view(self.size, slots, width),
                ),
                (
                    sent_scales[:square].view(self.size, slots, blocks),
                    back_scales.view(self.size, slots, blocks),
                ),
            ),
        )
        return fp8_scatter_add(
            back_payload.view(torch.float8_e4m3fn),
            back_scales,
            self._returned_rows(topk_ids, slots),
            width,
            dtype,
        )

    def _exact_combine(
        self,
        payload: torch.Tensor,
        scales: torch.Tensor,
        owner: torch.Tensor,
        row: torch.Tensor,
        valid: torch.Tensor,
        topk_ids: torch.Tensor,
        width: int,
        blocks: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # An eager forward packs the rows by owner and exchanges them at exact per-pair counts.
        size = self.size
        # Rows sorted by owner, then by the owner's token, with rows past the receive count last.
        # Each owner then gets its rows back from every rank in ascending token order.
        key = torch.where(valid, owner * self.tokens_per_rank + row, size * self.tokens_per_rank)
        order = torch.argsort(key)
        sends_here = torch.bincount(owner.masked_fill(~valid, size), minlength=size + 1)[:size]
        # What comes back from rank r is one row per token this rank routed to r.
        ranks = torch.div(topk_ids.to(torch.int64), self.local_experts, rounding_mode="floor")
        ordered, _ = torch.sort(ranks, dim=1)
        once = torch.ones_like(ordered, dtype=torch.bool)
        once[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
        backs_here = torch.bincount(ordered.masked_fill(~once, size).flatten(), minlength=size + 1)
        # The one host read of the layer. The eager dispatch already waits on its receive count.
        sends, backs = torch.stack([sends_here, backs_here[:size]]).tolist()
        total_sent, total_back = sum(sends), sum(backs)

        sent_payload, sent_scales, sent_rows, back_payload, back_scales, back_rows = (
            self._exchange_rows(max(total_sent, total_back), width, blocks, payload.device)
        )
        order = order[:total_sent]
        torch.index_select(payload.view(torch.uint8), 0, order, out=sent_payload[:total_sent])
        torch.index_select(scales, 0, order, out=sent_scales[:total_sent])
        torch.index_select(row.to(torch.int32), 0, order, out=sent_rows[:total_sent])
        _exchange_exact(
            self.everyone,
            self.rank,
            sends,
            backs,
            ((sent_payload, back_payload), (sent_scales, back_scales), (sent_rows, back_rows)),
        )
        # Returned rows come in rank order and ascend by token within a rank, so a stable sort by
        # token lists each token's partial sums in ascending rank order. The square does the same,
        # so both paths sum identically.
        token = back_rows[:total_back].to(torch.int64)
        permutation = torch.argsort(token, stable=True)
        by_token = token.index_select(0, permutation)
        first = torch.searchsorted(by_token, by_token)
        slot = torch.arange(total_back, device=token.device) - first
        table = torch.full(tuple(topk_ids.shape), -1, dtype=torch.int64, device=token.device)
        table[by_token, slot] = permutation
        return fp8_scatter_add(
            back_payload[:total_back].view(torch.float8_e4m3fn),
            back_scales[:total_back],
            table,
            width,
            dtype,
        )

    def _exchange_rows(
        self, square: int, width: int, blocks: int, device: Any
    ) -> tuple[torch.Tensor, ...]:
        # Send then receive buffers (payload, scales, each row's token), allocated once because a
        # captured graph records the addresses its sends and receives name.
        if not self.exchange:
            most = self.tokens_per_rank * self.size
            self.exchange = (
                torch.empty((most + 1, width), dtype=torch.uint8, device=device),
                torch.empty((most + 1, blocks), dtype=torch.float32, device=device),
                torch.empty((most + 1,), dtype=torch.int32, device=device),
                torch.empty((most, width), dtype=torch.uint8, device=device),
                torch.empty((most, blocks), dtype=torch.float32, device=device),
                torch.empty((most,), dtype=torch.int32, device=device),
            )
        sent_payload, sent_scales, sent_rows, back_payload, back_scales, back_rows = self.exchange
        assert (sent_payload.shape[1], sent_scales.shape[1]) == (width, blocks), "one width"
        return (
            sent_payload[: square + 1],
            sent_scales[: square + 1],
            sent_rows[: square + 1],
            back_payload[:square],
            back_scales[:square],
            back_rows[:square],
        )

    def _returned_rows(self, topk_ids: torch.Tensor, slots: int) -> torch.Tensor:
        ranks = torch.div(topk_ids.to(torch.int64), self.local_experts, rounding_mode="floor")
        # Ascending rank, so an owner adds its partial sums in one order however it routed, and a
        # rank it routed to twice contributes once.
        ordered, _ = torch.sort(ranks, dim=1)
        once = torch.ones_like(ordered, dtype=torch.bool)
        once[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
        token = torch.arange(topk_ids.shape[0], device=topk_ids.device).unsqueeze(1)
        return torch.where(once, ordered * slots + token, torch.full_like(ordered, -1))


def expert_parallel_transport(
    vllm_config: Any,
    *,
    reference_slots: dict[int, int] | None,
    fp8_dispatch: bool,
    fp8_combine: bool,
) -> ExpertParallelTransport | None:
    """The worker's transport when the engine runs expert parallelism over more than one rank.
    `reference_slots` is None for a model that publishes nothing."""
    parallel = vllm_config.parallel_config
    if not parallel.enable_expert_parallel:
        return None
    group = get_ep_group()
    if group.world_size == 1:
        return None
    # The forward context indexes rows by data-parallel rank.
    assert parallel.tensor_parallel_size == 1, "the transport serves tensor parallelism 1"
    assert group.world_size == parallel.data_parallel_size, "expert parallel = data parallel"
    assert group.rank_in_group == parallel.data_parallel_rank, "expert-parallel rank = DP rank"
    compilation = vllm_config.compilation_config
    if compilation.mode != CompilationMode.NONE or compilation.cudagraph_mode not in (
        CUDAGraphMode.NONE,
        CUDAGraphMode.FULL,
        CUDAGraphMode.FULL_DECODE_ONLY,
    ):
        raise ValueError(
            "the expert-parallel transport runs uncompiled, eager or as whole-model graphs: serve "
            'with --compilation-config \'{"mode": "NONE", "cudagraph_mode": "FULL_DECODE_ONLY"}\''
        )
    config = vllm_config.model_config.hf_text_config
    return ExpertParallelTransport(
        group=group.device_group,
        rank=group.rank_in_group,
        size=group.world_size,
        num_experts=int(
            config.n_routed_experts
            if config.model_type in ("deepseek_v3", "nemotron_h")
            else config.num_experts
        ),
        router_topk=int(
            config.top_k_experts
            if config.model_type == "gemma4_text"
            else config.num_experts_per_tok
        ),
        hidden_size=int(config.hidden_size),
        tokens_per_rank=int(vllm_config.scheduler_config.max_num_batched_tokens),
        reference_slots=reference_slots,
        fp8_dispatch=fp8_dispatch,
        fp8_combine=fp8_combine,
    )
