"""The Triton kernels the larger models' folded layers launch (`portable.py`) besides those of
`kernels.py`. The kernel bodies are the measured runtime's, unchanged. Each wrapper allocates the
outputs and launches its kernel with the arguments the measured runtime used.

Every layer first zeroes the rows past the live count (`prepare_router_input`) and masks their
routes (`mask_router_output`). A reference layer sends its rows to its experts' ranks through
DeepEP (`reference_rows`, `publish_received`) and all-gathers them in the background into the cache
(`publish_reference`). A consuming layer encodes its residual against its own reference rows
(`local_residual`). With FP8 dispatch (DeepSeek-V3) a reference layer pads its rows into the sender
store (`reference_inputs`), a consuming layer reads its residual from it (`prepare_sender`), the
background all-gather moves FP8 rows with their scales (`pack4`, `unpack4`, `publish_valid`) and a
receiver joins the cached FP8 reference to each code (`assemble_fp8`). With FP8 dispatch of BF16 or
INT4 experts (the CLI's --fp8), the receiver casts the rows back (`fp8_cast_back_blocks`).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .kernels import FP8_BLOCK, PROGRAMS

# --- every layer: router input and output ------------------------------------------------------


@triton.jit
def _router_input(
    X, COUNT, Y, ACTIVE, GUARD, N, H: tl.constexpr, B: tl.constexpr, CHECK: tl.constexpr
):
    row = tl.program_id(0)
    h = tl.arange(0, B)
    count = tl.load(COUNT)
    live = (row < N) & (row < count)
    if CHECK and row == 0:
        tl.store(GUARD, (count >= 0) & (count <= N))
    # Column tiles: one iteration (the original single block) when H <= 4096.
    for c in tl.static_range(0, H, B):
        x = tl.load(X + row * H + c + h, mask=live & (c + h < H), other=0)
        tl.store(Y + row * H + c + h, x, mask=(row < N) & (c + h < H))
    tl.store(ACTIVE + row, live, mask=row < N)


def prepare_router_input(hidden_states, count):
    """`hidden_states` with the rows past the live `count` zeroed, and which rows are live."""
    rows, hidden = hidden_states.shape
    out = torch.empty_like(hidden_states)
    active = torch.empty(rows, device=hidden_states.device, dtype=torch.bool)
    guard = torch.empty((0,), device=hidden_states.device, dtype=torch.bool)
    _router_input[(max(rows, 1),)](
        hidden_states,
        count,
        out,
        active,
        guard,
        rows,
        hidden,
        min(triton.next_power_of_2(hidden), 4096),
        False,
    )
    return out, active


@triton.jit
def _router_output(IDS, W, ACTIVE, OUT_IDS, OUT_W, N, K: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    valid = i < N * K
    live = tl.load(ACTIVE + i // K, mask=valid, other=False)
    ids = tl.load(IDS + i, mask=valid, other=-1)
    w = tl.load(W + i, mask=valid, other=0)
    tl.store(OUT_IDS + i, tl.where(live, ids, -1), mask=valid)
    tl.store(OUT_W + i, tl.where(live, w, 0), mask=valid)


def mask_router_output(ids, weights, active):
    """The router's top-k with inactive rows' routes set to -1 and weight 0."""
    out_ids = torch.empty_like(ids)
    out_weights = torch.empty_like(weights)
    rows, top_k = ids.shape
    if rows:
        _router_output[(triton.cdiv(rows * top_k, 256),)](
            ids, weights, active, out_ids, out_weights, rows, top_k, 256
        )
    return out_ids, out_weights


# --- consuming layers: routes ------------------------------------------------------------------


@triton.jit
def _routing(
    IDS,
    WEIGHTS,
    MAP,
    OUT,
    ACTIVE,
    DENSE,
    GUARD,
    N,
    K: tl.constexpr,
    E: tl.constexpr,
    OFFSET: tl.constexpr,
    LOCAL: tl.constexpr,
    BK: tl.constexpr,
    BE: tl.constexpr,
    WRITE_DENSE: tl.constexpr,
    CHECK: tl.constexpr,
):
    row = tl.program_id(0)
    k = tl.arange(0, BK)
    e = tl.arange(0, BE)
    if CHECK and row == 0:
        actual = tl.load(MAP + e, mask=e < E, other=-1)
        expected = tl.where((e >= OFFSET) & (e < OFFSET + LOCAL), e - OFFSET, -1)
        map_ok = tl.sum(((e >= E) | (actual == expected)).to(tl.int32), 0) == BE
        tl.store(GUARD + N, map_ok)
    if row < N:
        ids = tl.load(IDS + row * K + k, mask=k < K, other=-1).to(tl.int64)
        raw = tl.load(WEIGHTS + row * K + k, mask=k < K, other=0)
        w = tl.where(ids >= 0, raw, 0)
        if CHECK:
            ids_ok = tl.sum(((ids >= -1) & (ids < E)).to(tl.int32), 0) == BK
            weights_ok = (
                tl.sum(((w == w) & (tl.abs(w) != float("inf")) & (w >= 0)).to(tl.int32), 0) == BK
            )
            duplicate = (
                (k[:, None] < k[None, :]) & (ids[:, None] >= 0) & (ids[:, None] == ids[None, :])
            )
            unique = tl.sum(tl.sum(duplicate.to(tl.int32), 1), 0) == 0
        live = tl.sum((ids >= 0).to(tl.int32), 0) > 0
        if WRITE_DENSE:
            dense = tl.sum(tl.where(e[:, None] == ids[None, :], w[None, :], 0), 1)
        tl.store(OUT + row * K + k, w, mask=k < K)
        tl.store(ACTIVE + row, live)
        if WRITE_DENSE:
            tl.store(DENSE + row * E + e, dense, mask=e < E)
        if CHECK:
            tl.store(GUARD + row, ids_ok & weights_ok & unique)


def prepare_routing(ids, weights, expert_map, *, offset, local):
    """The sender's route weights (0 where the route is empty) and which rows route anywhere."""
    rows, top_k = ids.shape
    experts = expert_map.numel()
    out = torch.empty_like(weights)
    active = torch.empty(rows, device=ids.device, dtype=torch.bool)
    dense = torch.empty((0,), device=ids.device, dtype=torch.float32)
    guard = torch.empty(0, device=ids.device, dtype=torch.bool)
    _routing[(max(rows, 1),)](
        ids,
        weights,
        expert_map,
        out,
        active,
        dense,
        guard,
        rows,
        top_k,
        experts,
        offset,
        local,
        triton.next_power_of_2(top_k),
        triton.next_power_of_2(experts),
        False,
        False,
    )
    return out, active


@triton.jit
def _receive_routes(
    IDS,
    W,
    ACTIVE,
    MAP,
    OUT_IDS,
    OUT_W,
    GUARD,
    N,
    K: tl.constexpr,
    E: tl.constexpr,
    OFFSET: tl.constexpr,
    LOCAL: tl.constexpr,
    BK: tl.constexpr,
    BE: tl.constexpr,
    ERROR,
    AGGREGATE: tl.constexpr,
    INLINE_CHECK: tl.constexpr,
):
    row = tl.program_id(0)
    k = tl.arange(0, BK)
    if not (AGGREGATE or INLINE_CHECK) and row == 0:
        e = tl.arange(0, BE)
        actual = tl.load(MAP + e, mask=e < E, other=-1)
        expected = tl.where((e >= OFFSET) & (e < OFFSET + LOCAL), e - OFFSET, -1)
        tl.store(GUARD + N, tl.sum(((e >= E) | (actual == expected)).to(tl.int32), 0) == BE)
    if row < N:
        active = tl.load(ACTIVE + row)
        ids = tl.load(IDS + row * K + k, mask=k < K, other=-1).to(tl.int64)
        weights = tl.load(W + row * K + k, mask=k < K, other=0)
        live = active & (ids >= 0) & (k < K)
        bad = tl.sum((live & (ids >= LOCAL)).to(tl.int32), 0) > 0
        if INLINE_CHECK:
            tl.device_assert(~bad, "Received expert outside local bank")
        if AGGREGATE:  # noqa: SIM102 (the kernel as measured)
            if bad:
                tl.atomic_or(ERROR, 4, sem="relaxed")
            # Keep invalid transport metadata away from expert-map indexing,
            # even though the forward will fail at its final guard.
        if AGGREGATE or INLINE_CHECK:
            live = live & (ids < LOCAL)
        # Contiguous local bank: rank0's first foreign ID is LOCAL. Every
        # other bank's first foreign ID is zero. No per-call argmax needed.
        sentinel = LOCAL if OFFSET == 0 else 0
        tl.store(OUT_IDS + row * K + k, tl.where(live, ids + OFFSET, sentinel), mask=k < K)
        tl.store(OUT_W + row * K + k, tl.where(live, weights, 0), mask=k < K)
        if not (AGGREGATE or INLINE_CHECK):
            tl.store(GUARD + row, ~bad)


def receive_routes(ids, weights, active, expert_map, *, offset, local):
    """DeepEP's received (local) routes as global expert ids, empty slots naming a foreign expert
    with weight 0 (as many local experts as top-k slots or more)."""
    rows, top_k = ids.shape
    experts = expert_map.numel()
    out_ids = torch.empty_like(ids, dtype=torch.int64)
    out_weights = torch.empty_like(weights)
    guard = torch.empty(0, device=ids.device, dtype=torch.bool)
    _receive_routes[(max(rows, 1),)](
        ids,
        weights,
        active,
        expert_map,
        out_ids,
        out_weights,
        guard,
        rows,
        top_k,
        experts,
        offset,
        local,
        triton.next_power_of_2(top_k),
        triton.next_power_of_2(experts),
        guard,
        False,
        True,
        debug=True,
    )
    return out_ids, out_weights


# --- reference layers, BF16 dispatch: rows to the experts, the cache in the background ----------


@triton.jit
def _reference(
    X,
    IDS,
    W,
    COUNT,
    LOCAL,
    OUT,
    SCALE,
    OID,
    OW,
    ACTIVE,
    COMPLEMENT,
    N,
    CAP: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    WORLD: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
    BW: tl.constexpr,
    FP8: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.arange(0, BH)
    k = tl.arange(0, BK)
    r = tl.arange(0, BW)
    count = tl.load(COUNT)
    live = (row < N) & (row < count)
    tl.device_assert((count >= 0) & (count <= N), "Reference count exceeds input")
    ids = tl.load(IDS + row * K + k, mask=live & (k < K), other=-1).to(tl.int64)
    weights = tl.load(W + row * K + k, mask=live & (k < K), other=0)
    valid = (ids >= 0) & (k < K)
    tl.device_assert(tl.sum((valid & (ids >= E)).to(tl.int32), 0) == 0, "Invalid reference route")
    tl.device_assert(
        tl.sum(
            (
                valid & ((weights != weights) | (weights < 0) | (tl.abs(weights) == float("inf")))
            ).to(tl.int32),
            0,
        )
        == 0,
        "Invalid reference weights",
    )
    x = tl.load(X + row * H + h, mask=live & (h < H), other=0).to(tl.float32)
    tl.device_assert(
        tl.sum(((h < H) & ((x != x) | (tl.abs(x) == float("inf")))).to(tl.int32), 0) == 0,
        "Nonfinite reference input",
    )
    # Store original sender state before FP8 rounding in the SAME launch.
    tl.store(LOCAL + row * H + h, x.to(tl.bfloat16), mask=h < H)
    if FP8:
        grouped = tl.reshape(x, (BH // 128, 128))
        scale = tl.maximum(tl.max(tl.abs(grouped), 1), 1.0e-8) / 448.0
        q = tl.minimum(tl.maximum(grouped / scale[:, None], -448.0), 448.0).to(tl.float8e4nv)
        tl.store(OUT + row * H + h, tl.reshape(q, (BH,)), mask=h < H)
        g = tl.arange(0, BH // 128)
        tl.store(SCALE + row * (H // 128) + g, scale, mask=g < H // 128)
    else:
        tl.store(OUT + row * H + h, x.to(tl.bfloat16), mask=h < H)
    tl.store(OID + row * K + k, ids, mask=k < K)
    tl.store(OW + row * K + k, tl.where(valid, weights, 0), mask=k < K)
    tl.store(ACTIVE + row, live)
    destinations = ids // (E // WORLD)
    selected = tl.sum(((destinations[:, None] == r[None, :]) & valid[:, None]).to(tl.int32), 0) > 0
    tl.store(COMPLEMENT + row * WORLD + r, tl.where(live & ~selected, r, -1), mask=r < WORLD)


def reference_rows(hidden_states, ids, weights, count, capacity, experts, world):
    """The reference layer's rows padded to `capacity` (twice: the payload and this rank's own
    copy), their routes and weights (none past the live `count`), and which rows are live."""
    rows, hidden = hidden_states.shape
    top_k = ids.shape[1]
    local = torch.empty((capacity, hidden), device=hidden_states.device, dtype=torch.bfloat16)
    payload = torch.empty_like(local)
    scale = torch.empty(
        (capacity, hidden // 128), device=hidden_states.device, dtype=torch.float32
    )
    out_ids = torch.empty((capacity, top_k), device=hidden_states.device, dtype=torch.int64)
    out_weights = torch.empty_like(out_ids, dtype=torch.float32)
    active = torch.empty(capacity, device=hidden_states.device, dtype=torch.bool)
    complement = torch.empty((capacity, world), device=hidden_states.device, dtype=torch.int64)
    _reference[(capacity,)](
        hidden_states,
        ids,
        weights,
        count,
        local,
        payload,
        scale,
        out_ids,
        out_weights,
        active,
        complement,
        rows,
        capacity,
        hidden,
        top_k,
        experts,
        world,
        triton.next_power_of_2(hidden),
        triton.next_power_of_2(top_k),
        triton.next_power_of_2(world),
        False,
        debug=True,
    )
    return payload, local, out_ids, out_weights, active


@triton.jit
def _scatter(
    WIRE,
    SCALE,
    META,
    TOTAL,
    CACHE,
    VALID,
    ROWS,
    H: tl.constexpr,
    MAX_SOURCE: tl.constexpr,
    MS: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    B: tl.constexpr,
    FP8: tl.constexpr,
    OUT,
    MATERIALIZE: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.arange(0, B)
    total = tl.load(TOTAL)
    tl.device_assert((total >= 0) & (total <= ROWS), "Invalid sparse reference receive count")
    live = row < total
    source = tl.load(META + row * MS, mask=live, other=-1).to(tl.int64)
    valid = (source >= 0) & (source < MAX_SOURCE)
    tl.device_assert(~live | valid, "Invalid sparse reference source")
    x = tl.load(WIRE + row * H + h, mask=live & valid & (h < H), other=0.0).to(tl.float32)
    if FP8:
        scale = tl.load(SCALE + row * S0 + (h // 128) * S1, mask=live & valid & (h < H), other=0)
        x = x * scale
    tl.store(CACHE + source * H + h, x.to(tl.bfloat16), mask=live & valid & (h < H))
    tl.store(VALID + source, True, mask=live & valid)
    if MATERIALIZE:
        tl.store(OUT + row * H + h, x.to(tl.bfloat16), mask=h < H)


def publish_received(state, rows, handle):
    """Write the rows DeepEP delivered into the cache by source, mark them valid, and return them
    as the reference experts' input (inactive receive slots zero)."""
    received, hidden = rows.shape
    metadata = handle.recv_src_metadata
    total = handle.psum_num_recv_tokens_per_scaleup_rank[-1]
    output = torch.empty((received, hidden), device=rows.device, dtype=torch.bfloat16)
    if received:
        _scatter[(received,)](
            rows,
            rows,
            metadata,
            total,
            state.values,
            state.valid,
            received,
            hidden,
            state.origins * state.capacity,
            metadata.stride(0),
            0,
            0,
            triton.next_power_of_2(hidden),
            False,
            output,
            True,
            debug=True,
        )
    return output


@triton.jit
def _residual_prepare(X, STALE, GAIN, ACTIVE, OUT, N, H: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    h = tl.arange(0, B)
    active = tl.load(ACTIVE + row)
    x = tl.load(X + row * H + h, mask=h < H, other=0).to(tl.float32)
    stale = tl.load(STALE + row * H + h, mask=h < H, other=0).to(tl.float32)
    gain = tl.load(GAIN + h, mask=h < H, other=0).to(tl.float32)
    residual = (x - (stale * gain).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    tl.store(OUT + row * H + h, tl.where(active, residual, 0.0), mask=h < H)


def local_residual(hidden_states, reference, gain, active):
    """The encoder input `x - gain * z` against this rank's own reference rows `z`, inactive rows 0."""
    result = torch.empty_like(hidden_states)
    if hidden_states.shape[0]:
        _residual_prepare[(hidden_states.shape[0],)](
            hidden_states,
            reference,
            gain,
            active,
            result,
            hidden_states.shape[0],
            hidden_states.shape[1],
            triton.next_power_of_2(hidden_states.shape[1]),
            enable_fp_fusion=False,
        )
    return result


@triton.jit
def _publish(
    SOURCE,
    COUNTS,
    EXECUTION,
    VALUES,
    VALID,
    EPOCH,
    REFERENCE,
    LAYER,
    CAPACITY: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.arange(0, BLOCK)
    count = tl.load(COUNTS + row // CAPACITY)
    valid = (count >= 0) & (count <= CAPACITY) & (row % CAPACITY < count)
    # Column tiles: one iteration (the original single block) when HIDDEN <= 4096.
    for c in tl.static_range(0, HIDDEN, BLOCK):
        value = tl.load(SOURCE + row * HIDDEN + c + h, mask=c + h < HIDDEN, other=0)
        tl.store(VALUES + row * HIDDEN + c + h, value, mask=c + h < HIDDEN)
    tl.store(VALID + row, valid)
    if row == 0:
        tl.store(EPOCH, tl.load(EXECUTION))
        tl.store(REFERENCE, LAYER)


def publish_reference(state, layer, epoch, values, counts):
    """Replace the cache with every origin's gathered rows, valid below its count, for `epoch`."""
    _publish[(state.origins * state.capacity,)](
        values,
        counts,
        epoch,
        state.values,
        state.valid,
        state.epoch,
        state.layer,
        layer,
        state.capacity,
        state.hidden,
        min(triton.next_power_of_2(state.hidden), 4096),
    )


@triton.jit
def _publish_rows(
    SOURCE,
    COUNTS,
    EXECUTION,
    VALUES,
    VALID,
    EPOCH,
    REFERENCE,
    LAYER,
    ROWS,
    CAPACITY: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0)
    origin, row = index // ROWS, index % ROWS
    h = tl.arange(0, BLOCK)
    count = tl.load(COUNTS + origin)
    valid = (count >= 0) & (count <= ROWS) & (row < count)
    for c in tl.static_range(0, HIDDEN, BLOCK):
        value = tl.load(SOURCE + index * HIDDEN + c + h, mask=c + h < HIDDEN, other=0)
        tl.store(VALUES + (origin * CAPACITY + row) * HIDDEN + c + h, value, mask=c + h < HIDDEN)
    tl.store(VALID + origin * CAPACITY + row, valid)
    if index == 0:
        tl.store(EPOCH, tl.load(EXECUTION))
        tl.store(REFERENCE, LAYER)


def publish_rows(state, layer, epoch, values, counts, rows):
    """publish_reference for `rows` gathered rows of every origin, laid out contiguously. The cache
    rows from `rows` up stay invalid."""
    _publish_rows[(state.origins * rows,)](
        values,
        counts,
        epoch,
        state.values,
        state.valid,
        state.epoch,
        state.layer,
        layer,
        rows,
        state.capacity,
        state.hidden,
        min(triton.next_power_of_2(state.hidden), 4096),
    )


# --- FP8 dispatch (DeepSeek-V3)----------------------------------------------------------------


@triton.jit
def _prepare_sender(
    X,
    GAIN,
    CACHE,
    VALID,
    EPOCH,
    EXPECTED,
    LAYER,
    ACTIVE,
    ORIGINAL,
    RESIDUAL,
    GUARD,
    H: tl.constexpr,
    CAP: tl.constexpr,
    ORIGIN: tl.constexpr,
    REFERENCE: tl.constexpr,
    B: tl.constexpr,
    KEEP_ORIGINAL: tl.constexpr,
    ERROR,
    AGGREGATE: tl.constexpr,
    INLINE_CHECK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    live = tl.load(ACTIVE + row)
    present = tl.load(VALID + ORIGIN * CAP + row)
    epoch = tl.load(EPOCH)
    present = present & (epoch >= 0) & (epoch == tl.load(EXPECTED)) & (tl.load(LAYER) == REFERENCE)
    x = tl.load(X + row * H + col, mask=col < H, other=0).to(tl.float32)
    x = tl.where(live, x, 0)
    # Even inactive but present cache rows retain their faithful residual.
    # Transport routing, not a changed codec equation, excludes these rows.
    stale = tl.load(CACHE + (ORIGIN * CAP + row) * H + col, mask=present & (col < H), other=0).to(
        tl.float32
    )
    gain = tl.load(GAIN + col, mask=col < H, other=0).to(tl.float32)
    prediction = (gain * stale).to(tl.bfloat16).to(tl.float32)
    residual = (x - prediction).to(tl.bfloat16)
    if KEEP_ORIGINAL:
        tl.store(ORIGINAL + row * H + col, x.to(tl.bfloat16), mask=col < H)
    tl.store(RESIDUAL + row * H + col, residual, mask=col < H)
    if INLINE_CHECK:
        tl.device_assert((~live) | present, "Missing active sender reference")
    elif AGGREGATE:
        if live & ~present:
            tl.atomic_or(ERROR, 2, sem="relaxed")
    else:
        tl.store(GUARD + row, (~live) | present)


def prepare_sender(hidden_states, gain, state, epoch, active, *, reference):
    """The encoder input `x - gain * z` against this rank's BF16 reference rows `z` in the sender
    store of `state` (which must hold the current epoch's `reference`), inactive rows 0."""
    rows, hidden = hidden_states.shape
    residual = torch.empty_like(hidden_states)
    guard = torch.empty(0, device=hidden_states.device, dtype=torch.bool)
    if rows:
        # this rank's rows as a one-origin cache: the sender store and its own validity row
        origin = state.origin
        _prepare_sender[(rows,)](
            hidden_states,
            gain,
            state.sender.view(1, state.capacity, hidden),
            state.valid[origin : origin + 1],
            state.epoch,
            epoch,
            state.layer,
            active,
            hidden_states,
            residual,
            guard,
            hidden,
            state.capacity,
            0,
            reference,
            triton.next_power_of_2(hidden),
            False,
            guard,
            False,
            True,
            enable_fp_fusion=False,
            debug=True,
        )
    return residual


@triton.jit
def _reference_send_prepare(
    X,
    IDS,
    W,
    COUNT,
    PAD,
    OUT_IDS,
    OUT_W,
    ACTIVE,
    GUARD,
    N,
    CAP: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
    CHECK: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.arange(0, BH)
    k = tl.arange(0, BK)
    count = tl.load(COUNT)
    live = (row < N) & (row < count)
    count_ok = (count >= 0) & (count <= N)
    ids = tl.load(IDS + row * K + k, mask=live & (k < K), other=-1).to(tl.int64)
    raw = tl.load(W + row * K + k, mask=live & (k < K), other=0)
    weights = tl.where(ids >= 0, raw, 0)
    if CHECK:
        ids_ok = tl.sum(((ids >= -1) & (ids < E)).to(tl.int32), 0) == BK
        weights_ok = (
            tl.sum(
                ((weights == weights) & (tl.abs(weights) != float("inf")) & (weights >= 0)).to(
                    tl.int32
                ),
                0,
            )
            == BK
        )
    # Column tiles: one iteration (the original single block) when H <= 4096.
    for c in tl.static_range(0, H, BH):
        x = tl.load(X + row * H + c + h, mask=live & (c + h < H), other=0)
        tl.store(PAD + row * H + c + h, x, mask=c + h < H)
    tl.store(OUT_IDS + row * K + k, ids, mask=k < K)
    tl.store(OUT_W + row * K + k, weights, mask=k < K)
    tl.store(ACTIVE + row, live & (tl.sum((ids >= 0).to(tl.int32), 0) > 0), mask=row < N)
    if CHECK:
        tl.store(GUARD + row, count_ok & ids_ok & weights_ok)


def reference_inputs(hidden_states, ids, weights, count, capacity, experts, out):
    """The reference layer's rows written into `out` padded to `capacity` (zero past the live
    `count`), their routes and weights, and which rows route anywhere."""
    rows, hidden = hidden_states.shape
    top_k = ids.shape[1]
    out_ids = torch.empty(capacity, top_k, device=hidden_states.device, dtype=torch.int64)
    out_weights = torch.empty(capacity, top_k, device=hidden_states.device, dtype=weights.dtype)
    active = torch.empty(rows, device=hidden_states.device, dtype=torch.bool)
    guard = torch.empty(0, device=hidden_states.device, dtype=torch.bool)
    _reference_send_prepare[(capacity,)](
        hidden_states,
        ids,
        weights,
        count,
        out,
        out_ids,
        out_weights,
        active,
        guard,
        rows,
        capacity,
        hidden,
        top_k,
        experts,
        min(triton.next_power_of_2(hidden), 4096),
        triton.next_power_of_2(top_k),
        False,
    )
    return out_ids, out_weights, active


@triton.jit
def _pack4(
    A_IN,
    B_IN,
    C_IN,
    D_IN,
    OUT,
    A: tl.constexpr,
    B: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A_IN + i, mask=i < A, other=0)
    b = tl.load(B_IN + i - A, mask=(i >= A) & (i < A + B), other=0)
    c = tl.load(C_IN + i - A - B, mask=(i >= A + B) & (i < A + B + C), other=0)
    d = tl.load(D_IN + i - A - B - C, mask=(i >= A + B + C) & (i < A + B + C + D), other=0)
    tl.store(
        OUT + i,
        tl.where(i < A, a, tl.where(i < A + B, b, tl.where(i < A + B + C, c, d))),
        mask=i < A + B + C + D,
    )


@triton.jit
def _unpack4(
    IN,
    A_OUT,
    B_OUT,
    C_OUT,
    D_OUT,
    A: tl.constexpr,
    B: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rank = tl.program_id(1)
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(IN + rank * (A + B + C + D) + i, mask=i < A + B + C + D, other=0)
    tl.store(A_OUT + rank * A + i, x, mask=i < A)
    tl.store(B_OUT + rank * B + i - A, x, mask=(i >= A) & (i < A + B))
    tl.store(C_OUT + rank * C + i - A - B, x, mask=(i >= A + B) & (i < A + B + C))
    tl.store(D_OUT + rank * D + i - A - B - C, x, mask=(i >= A + B + C) & (i < A + B + C + D))


def words(tensor):
    """A contiguous tensor's bytes as int32 words."""
    return tensor.reshape(-1).view(torch.int32)


def pack4(fields, out):
    """`[f0; f1; ...]` (two to four fields) in one word buffer, bits unchanged."""
    views = [words(field) for field in fields]
    sizes = [view.numel() for view in views]
    # unused fields reuse a real view with size 0
    views, sizes = views + [views[0]] * (4 - len(views)), sizes + [0] * (4 - len(sizes))
    _pack4[(triton.cdiv(out.numel(), 1024),)](*views, out, *sizes, 1024)


def unpack4(packed, fields, world):
    """Split `world` gathered `pack4` messages back into `fields` (each `world` times as long)."""
    views = [words(field) for field in fields]
    sizes = [view.numel() // world for view in views]
    views, sizes = views + [views[0]] * (4 - len(views)), sizes + [0] * (4 - len(sizes))
    _unpack4[(triton.cdiv(sum(sizes), 1024), world)](packed, *views, *sizes, 1024)


@triton.jit
def _publish_valid(
    COUNTS,
    EXECUTION,
    VALID,
    EPOCH,
    REFERENCE,
    LAYER,
    ORIGIN,
    CAP: tl.constexpr,
    LOCAL: tl.constexpr,
):
    # reference_publish._publish's validity rule. LOCAL marks only this origin's rows from its own count.
    row = tl.program_id(0)
    origin = row // CAP
    if LOCAL:
        count = tl.load(COUNTS)
        mine = origin == ORIGIN
    else:
        count = tl.load(COUNTS + origin)
        mine = origin >= 0
    tl.store(VALID + row, mine & (count >= 0) & (count <= CAP) & (row % CAP < count))
    if row == 0:
        tl.store(EPOCH, tl.load(EXECUTION))
        tl.store(REFERENCE, LAYER)


def publish_valid(state, layer, epoch, counts, *, local=False):
    """Mark the cache rows below each origin's count valid for `epoch` (only this rank's own with
    `local`, from its one count)."""
    _publish_valid[(state.origins * state.capacity,)](
        counts,
        epoch,
        state.valid,
        state.epoch,
        state.layer,
        layer,
        state.origin,
        state.capacity,
        local,
    )


@triton.jit
def _assemble_fp8(
    WIRE,
    WSCALE,
    META,
    TOTAL,
    CACHE,
    CSCALE,
    VALID,
    EPOCH,
    EXPECTED,
    LAYER,
    OUT,
    OSCALE,
    ACTIVE,
    ORIGINS,
    ROWS,
    GUARD,
    N,
    H4: tl.constexpr,
    D4: tl.constexpr,
    G: tl.constexpr,
    GD: tl.constexpr,
    CAP: tl.constexpr,
    WORLD: tl.constexpr,
    MS: tl.constexpr,
    WS0: tl.constexpr,
    WS1: tl.constexpr,
    REFERENCE: tl.constexpr,
    BW: tl.constexpr,
    BG: tl.constexpr,
    INLINE_CHECK: tl.constexpr,
):
    # receive_assembly._assemble's source decoding and guards. Values move as 4-byte words.
    i = tl.program_id(0)
    j = tl.arange(0, BW)
    g = tl.arange(0, BG)
    total = tl.load(TOTAL).to(tl.int64)
    active = i < total
    source = tl.load(META + i * MS, mask=active, other=-1).to(tl.int64)
    rem = source % CAP
    origin = source // CAP - (rem < 0).to(tl.int64)
    row = rem + tl.where(rem < 0, CAP, 0)
    bounds = (source >= 0) & (origin < WORLD) & (row < CAP)
    index = origin * CAP + row
    valid = tl.load(VALID + index, mask=bounds, other=0)
    epoch = tl.load(EPOCH)
    present = (
        bounds
        & valid
        & (epoch >= 0)
        & (epoch == tl.load(EXPECTED))
        & (tl.load(LAYER) == REFERENCE)
    )
    required = (total >= 0) & (total <= N) & ((~active) | present)
    take = active & present
    # One pass over the output row, as the BF16 kernel does: [cache words | wire words].
    for c in tl.static_range(0, H4 + D4, BW):
        k = c + j
        ref = tl.load(CACHE + index * H4 + k, mask=take & (k < H4), other=0)
        code = tl.load(WIRE + i * D4 + k - H4, mask=active & (k >= H4) & (k < H4 + D4), other=0)
        tl.store(OUT + i * (H4 + D4) + k, tl.where(k < H4, ref, code), mask=k < H4 + D4)
    for c in tl.static_range(0, G + GD, BG):
        k = c + g
        ref_scale = tl.load(CSCALE + index * G + k, mask=take & (k < G), other=0.0)
        code_scale = tl.load(
            WSCALE + i * WS0 + (k - G) * WS1, mask=active & (k >= G) & (k < G + GD), other=0.0
        )
        tl.store(
            OSCALE + i * (G + GD) + k, tl.where(k < G, ref_scale, code_scale), mask=k < G + GD
        )
    tl.store(ACTIVE + i, active)
    tl.store(ORIGINS + i, origin)
    tl.store(ROWS + i, row)
    if INLINE_CHECK:
        tl.device_assert(required, "Invalid receive count or missing active reference")
    else:
        tl.store(GUARD + i, required)


def assemble_fp8(state, epoch, codes, scales, metadata, total, *, reference):
    """Each received row's `[z ; c_d]` in FP8 and `[z scales ; c_d scales]`, the cached reference
    `z` joined by source, and which receive slots are active."""
    received, code_width = codes.shape
    hidden, groups = state.hidden, state.groups
    values = torch.empty(
        (received, hidden + code_width), device=codes.device, dtype=torch.float8_e4m3fn
    )
    out_scales = torch.empty(
        (received, groups + code_width // 128), device=codes.device, dtype=torch.float32
    )
    active = torch.empty(received, device=codes.device, dtype=torch.bool)
    origins = torch.empty(received, device=codes.device, dtype=torch.int64)
    source_rows = torch.empty_like(origins)
    guard = torch.empty(0, device=codes.device, dtype=torch.bool)
    if received:
        _assemble_fp8[(received,)](
            words(codes),
            scales,
            metadata,
            total,
            words(state.values),
            state.scales,
            state.valid,
            state.epoch,
            epoch,
            state.layer,
            words(values),
            out_scales,
            active,
            origins,
            source_rows,
            guard,
            received,
            hidden // 4,
            code_width // 4,
            groups,
            code_width // 128,
            state.capacity,
            state.origins,
            metadata.stride(0),
            scales.stride(0),
            scales.stride(1),
            reference,
            min(triton.next_power_of_2((hidden + code_width) // 4), 4096),
            triton.next_power_of_2(groups + code_width // 128),
            True,
            debug=True,
        )
    else:
        torch._assert_async((total == 0).all(), "Invalid empty receive count")
    return values, out_scales, active


# --- FP8 dispatch of BF16 or INT4 experts ----------------------------------------------------------


@triton.jit
def _fp8_cast_back_blocks(
    CAST, SCALE, OUT, COUNT, H: tl.constexpr, BLOCKS: tl.constexpr, BLOCK: tl.constexpr
):
    cols = tl.arange(0, BLOCK)
    for row in range(tl.program_id(0), tl.load(COUNT), tl.num_programs(0)):
        row = tl.cast(row, tl.int64)
        for block in range(BLOCKS):
            offsets = block * BLOCK + cols
            values = tl.load(CAST + row * H + offsets).to(tl.float32)
            scale = tl.load(SCALE + row * BLOCKS + block)
            tl.store(OUT + row * H + offsets, (values * scale).to(tl.bfloat16))


def fp8_cast_back_blocks(payload, scale, count):
    """kernels.fp8_cast_back one block of 128 columns at a time: the folded layers' received rows
    are up to 8960 wide, and one load per row would need a vector of the next power of two."""
    rows, width = payload.shape
    out = torch.empty((rows, width), dtype=torch.bfloat16, device=payload.device)
    if rows:
        _fp8_cast_back_blocks[(PROGRAMS,)](
            payload, scale, out, count, width, scale.shape[1], FP8_BLOCK, enable_fp_fusion=False
        )
    return out
