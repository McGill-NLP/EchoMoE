"""The Triton kernels of the folded student's MoE layers on vLLM 0.29, and of the patches both
variants share. The kernel bodies are the measured runtimes', without the options the measured
runs left off. Each wrapper allocates the outputs and launches its kernel with the arguments the
measured runtime used.

A consuming layer runs `consumer_send` (route masking, the residual against the cached reference),
the encoder GEMM, DeepEP's dispatch and `assemble_compact` (each received row's reference beside
its code, and the routes of this rank's experts in slot order). A reference layer sends its rows
and routes in one all-gather (`exchange_reference`) and publishes them as the cache later layers
read. The owner decodes the combined code into vLLM's output (`decode_into`). The larger models'
layers also clear inactive rows of their expert output before the combine (`zero_inactive_`).
With FP8 activations, rows cross the network as DeepEP casts them (`fp8_cast`), and a reference
layer's combine sums in BF16 within a node and in FP8 between nodes (`reduce_scatter_fp8_nodes`).

Qwen3-30B-A3B's experts, in both variants, compute only the (row, slot) pairs whose expert is on
this rank: `align` lists them, `silu_and_mul_sorted` activates them and `sum_live` sums them up to
the received count. The larger models' experts activate only those pairs (`silu_and_mul_live`).
"""

from __future__ import annotations

import functools
import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from triton.language.extra import libdevice

# --- consuming layers: sender ------------------------------------------------------------------


@triton.jit
def _consumer_send(
    X,
    IDS,
    W,
    COUNT,
    GAIN,
    CACHE,
    VALID,
    EPOCH,
    EXPECTED,
    LAYER,
    OUT_IDS,
    OUT_W,
    ACTIVE,
    RESIDUAL,
    N,
    K: tl.constexpr,
    H: tl.constexpr,
    CAP: tl.constexpr,
    ORIGIN: tl.constexpr,
    REFERENCE: tl.constexpr,
    BK: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    k = tl.arange(0, BK)
    col = tl.arange(0, B)
    count = tl.load(COUNT)
    live0 = (row < N) & (row < count)
    raw_ids = tl.load(IDS + row * K + k, mask=k < K, other=-1)
    raw_w = tl.load(W + row * K + k, mask=k < K, other=0)
    ids = tl.where(live0, raw_ids, -1)
    w = tl.where(live0, raw_w, 0)
    w = tl.where(ids >= 0, w, 0)
    live = tl.sum((ids >= 0).to(tl.int32), 0) > 0
    tl.store(OUT_IDS + row * K + k, ids, mask=k < K)
    tl.store(OUT_W + row * K + k, w, mask=k < K)
    tl.store(ACTIVE + row, live)
    present = tl.load(VALID + ORIGIN * CAP + row)
    epoch = tl.load(EPOCH)
    present = present & (epoch >= 0) & (epoch == tl.load(EXPECTED)) & (tl.load(LAYER) == REFERENCE)
    x = tl.load(X + row * H + col, mask=live0 & (col < H), other=0).to(tl.float32)
    x = tl.where(live, x, 0)
    stale = tl.load(CACHE + (ORIGIN * CAP + row) * H + col, mask=present & (col < H), other=0).to(
        tl.float32
    )
    gain = tl.load(GAIN + col, mask=col < H, other=0).to(tl.float32)
    prediction = (gain * stale).to(tl.bfloat16).to(tl.float32)
    residual = (x - prediction).to(tl.bfloat16)
    tl.store(RESIDUAL + row * H + col, residual, mask=col < H)
    tl.device_assert((~live) | present, "Missing active sender reference")


def consumer_send(hidden_states, ids, weights, count, gain, cache, epoch, *, origin, reference):
    """From the router's raw top-k: the routes and weights of rows below the live `count`, which
    rows route anywhere, and the encoder input `x - gain * z` against the cached reference `z`."""
    rows, hidden = hidden_states.shape
    top_k = ids.shape[1]
    out_ids = torch.empty_like(ids)
    out_weights = torch.empty_like(weights)
    active = torch.empty(rows, device=hidden_states.device, dtype=torch.bool)
    residual = torch.empty_like(hidden_states)
    if rows:
        _consumer_send[(rows,)](
            hidden_states,
            ids,
            weights,
            count,
            gain,
            cache.values,
            cache.valid,
            cache.epoch,
            epoch,
            cache.layer,
            out_ids,
            out_weights,
            active,
            residual,
            rows,
            top_k,
            hidden,
            cache.capacity,
            origin,
            reference,
            triton.next_power_of_2(top_k),
            triton.next_power_of_2(hidden),
            enable_fp_fusion=False,
            debug=True,
        )
    return out_ids, out_weights, active, residual


# --- consuming layers: receiver ----------------------------------------------------------------


@triton.jit
def _assemble_compact(
    CODES,
    META,
    TOTAL,
    CACHE,
    VALID,
    EPOCH,
    EXPECTED,
    LAYER,
    OUT,
    ACTIVE,
    RIDS,
    RW,
    OUT_IDS,
    OUT_W,
    N,
    H: tl.constexpr,
    D: tl.constexpr,
    CAP: tl.constexpr,
    WORLD: tl.constexpr,
    MS: tl.constexpr,
    REFERENCE: tl.constexpr,
    B: tl.constexpr,
    K: tl.constexpr,
    OUT_K: tl.constexpr,
    OFFSET: tl.constexpr,
    LOCAL: tl.constexpr,
    BK: tl.constexpr,
    BO: tl.constexpr,
    R: tl.constexpr,
    FP8: tl.constexpr,
    CODE_SCALES,
    DB: tl.constexpr,
):
    first = tl.program_id(0).to(tl.int64) * R
    rows = first + tl.arange(0, R)
    inside = rows < N
    total = tl.load(TOTAL).to(tl.int64)
    active = rows < total
    tl.store(ACTIVE + rows, active, mask=inside)
    j = tl.arange(0, B)
    epoch = tl.load(EPOCH)
    expected = tl.load(EXPECTED)
    layer = tl.load(LAYER)
    for i in range(R):
        row = first + i
        if (row < N) & (row < total):
            source = tl.load(META + row * MS).to(tl.int64)
            rem = source % CAP
            origin = source // CAP - (rem < 0).to(tl.int64)
            slot_row = rem + tl.where(rem < 0, CAP, 0)
            bounds = (source >= 0) & (origin < WORLD) & (slot_row < CAP)
            index = origin * CAP + slot_row
            valid = tl.load(VALID + index, mask=bounds, other=0)
            present = bounds & valid & (epoch >= 0) & (epoch == expected) & (layer == REFERENCE)
            ref = tl.load(CACHE + index * H + j, mask=present & (j < H), other=0)
            in_code = (j >= H) & (j < H + D)
            if FP8:
                raw = tl.load(CODES + row * D + j - H, mask=in_code, other=0.0).to(tl.float32)
                scale = tl.load(CODE_SCALES + row * DB + (j - H) // 128, mask=in_code, other=0.0)
                code = (raw * scale).to(tl.bfloat16)
            else:
                code = tl.load(CODES + row * D + j - H, mask=in_code, other=0.0)
            tl.store(OUT + row * (H + D) + j, tl.where(j < H, ref, code), mask=j < H + D)
    slot = tl.arange(0, BK)
    destination = tl.arange(0, BO)
    load = (active & inside)[:, None] & (slot < K)[None, :]
    ids = tl.load(RIDS + rows[:, None] * K + slot[None, :], mask=load, other=-1)
    weights = tl.load(RW + rows[:, None] * K + slot[None, :], mask=load, other=0.0)
    live = load & (ids >= 0)
    position = tl.cumsum(live.to(tl.int32), 1) - 1
    match = live[:, :, None] & (position[:, :, None] == destination[None, None, :])
    source_slot = tl.min(tl.where(match, slot[None, :, None], BK), 1)
    selected = source_slot < BK
    safe = tl.minimum(source_slot, BK - 1)
    selected_ids = tl.gather(ids, safe, 1)
    selected_weights = tl.gather(weights, safe, 1)
    sentinel = LOCAL if OFFSET == 0 else 0
    store = inside[:, None] & (destination < OUT_K)[None, :]
    tl.store(
        OUT_IDS + rows[:, None] * OUT_K + destination[None, :],
        tl.where(selected, selected_ids + OFFSET, sentinel),
        mask=store,
    )
    tl.store(
        OUT_W + rows[:, None] * OUT_K + destination[None, :],
        tl.where(selected, selected_weights, 0.0),
        mask=store,
    )


def assemble_compact(
    cache, epoch, codes, metadata, total, ids, weights, *, reference, offset, local
):
    """Each received row's `[z ; c_d]` and active flag, and its live routes to this rank's `local`
    experts packed in slot order (fewer local experts than top-k slots), 8 rows per program. Rows
    past the received count get no values: the experts read only rows with a live route. `codes`
    may be DeepEP's FP8 pair (values, float32 scale per 128 columns), cast back as it is read."""
    fp8 = isinstance(codes, tuple)
    codes, code_scales = codes if fp8 else (codes, None)
    received, code_width = codes.shape
    top_k = ids.shape[1]
    width = min(top_k, local)
    values = torch.empty(
        (received, cache.hidden + code_width), device=codes.device, dtype=torch.bfloat16
    )
    active = torch.empty(received, device=codes.device, dtype=torch.bool)
    out_ids = torch.empty((received, width), device=codes.device, dtype=torch.int64)
    out_weights = torch.empty((received, width), device=codes.device, dtype=torch.float32)
    if received:
        _assemble_compact[(triton.cdiv(received, 8),)](
            codes,
            metadata,
            total,
            cache.values,
            cache.valid,
            cache.epoch,
            epoch,
            cache.layer,
            values,
            active,
            ids,
            weights,
            out_ids,
            out_weights,
            received,
            cache.hidden,
            code_width,
            cache.capacity,
            cache.origins,
            metadata.stride(0),
            reference,
            triton.next_power_of_2(cache.hidden + code_width),
            top_k,
            width,
            offset,
            local,
            triton.next_power_of_2(top_k),
            triton.next_power_of_2(width),
            8,
            fp8,
            code_scales,
            0 if code_scales is None else code_scales.shape[1],
            enable_fp_fusion=False,
        )
    return values, active, out_ids, out_weights


@triton.jit
def _assemble(
    CODES,
    META,
    TOTAL,
    CACHE,
    VALID,
    EPOCH,
    EXPECTED,
    LAYER,
    OUT,
    ACTIVE,
    ORIGINS,
    ROWS,
    GUARD,
    N,
    H: tl.constexpr,
    D: tl.constexpr,
    CAP: tl.constexpr,
    WORLD: tl.constexpr,
    MS: tl.constexpr,
    REFERENCE: tl.constexpr,
    B: tl.constexpr,
    ERROR,
    AGGREGATE: tl.constexpr,
    INLINE_CHECK: tl.constexpr,
    SCALE,
    S0: tl.constexpr,
    S1: tl.constexpr,
    FP8: tl.constexpr,
):
    i = tl.program_id(0)
    j = tl.arange(0, B)
    total = tl.load(TOTAL).to(tl.int64)
    active = i < total
    source = tl.load(META + i * MS, mask=active, other=-1).to(tl.int64)
    # Triton integer division truncates toward zero. Match Torch floor division
    # and positive remainder even for negative padding/invalid source IDs.
    rem = source % CAP
    origin = source // CAP - (rem < 0).to(tl.int64)
    row = rem + tl.where(rem < 0, CAP, 0)
    bounds = (source >= 0) & (origin < WORLD) & (row < CAP)
    index = origin * CAP + row
    valid = tl.load(VALID + index, mask=bounds, other=0)
    epoch = tl.load(EPOCH)
    expected = tl.load(EXPECTED)
    layer = tl.load(LAYER)
    present = bounds & valid & (epoch >= 0) & (epoch == expected) & (layer == REFERENCE)
    required = (total >= 0) & (total <= N) & ((~active) | present)
    ref = tl.load(CACHE + index * H + j, mask=active & present & (j < H), other=0)
    code = tl.load(CODES + i * D + j - H, mask=active & (j >= H) & (j < H + D), other=0.0)
    if FP8:
        scale = tl.load(
            SCALE + i * S0 + ((j - H) // 128) * S1, mask=active & (j >= H) & (j < H + D), other=0.0
        )
        code = (code.to(tl.float32) * scale).to(tl.bfloat16)
    result = tl.where(j < H, ref, code)
    tl.store(OUT + i * (H + D) + j, result, mask=j < H + D)
    tl.store(ACTIVE + i, active)
    tl.store(ORIGINS + i, origin)
    tl.store(ROWS + i, row)
    if INLINE_CHECK:
        tl.device_assert(required, "Invalid receive count or missing active reference")
    elif AGGREGATE:
        if not required:
            tl.atomic_or(ERROR, 1, sem="relaxed")
    else:
        tl.store(GUARD + i, required)


def assemble(cache, epoch, codes, metadata, total, *, reference):
    """Each received row's `[z ; c_d]` and active flag (as many local experts as top-k slots)."""
    received, code_width = codes.shape
    values = torch.empty(
        (received, cache.hidden + code_width), device=codes.device, dtype=torch.bfloat16
    )
    active = torch.empty(received, device=codes.device, dtype=torch.bool)
    origins = torch.empty(received, device=codes.device, dtype=torch.int64)
    source_rows = torch.empty_like(origins)
    guard = torch.empty(0, device=codes.device, dtype=torch.bool)
    if received:
        _assemble[(received,)](
            codes,
            metadata,
            total,
            cache.values,
            cache.valid,
            cache.epoch,
            epoch,
            cache.layer,
            values,
            active,
            origins,
            source_rows,
            guard,
            received,
            cache.hidden,
            code_width,
            cache.capacity,
            cache.origins,
            metadata.stride(0),
            reference,
            triton.next_power_of_2(cache.hidden + code_width),
            guard,
            False,
            True,
            codes,
            0,
            0,
            False,
            debug=True,
        )
    else:
        torch._assert_async((total == 0).all(), "Invalid empty receive count")
    return values, active


@triton.jit
def _compact_routes(
    IDS,
    W,
    ACTIVE,
    OUT_IDS,
    OUT_W,
    N,
    K: tl.constexpr,
    OUT_K: tl.constexpr,
    OFFSET: tl.constexpr,
    LOCAL: tl.constexpr,
    BK: tl.constexpr,
    BO: tl.constexpr,
    E: tl.constexpr,
    GLOBAL_IDS: tl.constexpr,
    ACTIVE_COUNT: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.arange(0, BK)
    destination = tl.arange(0, BO)
    active = row < tl.load(ACTIVE) if ACTIVE_COUNT else tl.load(ACTIVE + row)
    ids = tl.load(IDS + row * K + slot, mask=slot < K, other=-1)
    weights = tl.load(W + row * K + slot, mask=slot < K, other=0)
    if GLOBAL_IDS:
        tl.device_assert(
            tl.sum((active & (slot < K) & ((ids < -1) | (ids >= E))).to(tl.int32), 0) == 0,
            "Invalid global compact expert",
        )
        local_ids = ids - OFFSET
        live = active & (slot < K) & (local_ids >= 0) & (local_ids < LOCAL)
        ids = local_ids
    else:
        live = active & (slot < K) & (ids >= 0)
    if ACTIVE_COUNT:
        tl.device_assert(
            (tl.load(ACTIVE) >= 0) & (tl.load(ACTIVE) <= N), "Invalid compact receive count"
        )
    count = tl.sum(live.to(tl.int32), 0)
    # Never truncate an unexpected route: stock top-k has distinct experts,
    # but the diagnostic candidate still fails before its outputs are consumed.
    tl.device_assert(
        tl.sum((live & (ids >= LOCAL)).to(tl.int32), 0) == 0, "Invalid compact local expert"
    )
    tl.device_assert(count <= OUT_K, "Too many routes for compact local bank")
    position = tl.cumsum(live.to(tl.int32), 0) - 1
    source = tl.min(
        tl.where(live[None, :] & (position[None, :] == destination[:, None]), slot[None, :], BK), 1
    )
    selected = source < BK
    safe = tl.minimum(source, BK - 1)
    selected_ids = tl.gather(ids, safe, 0)
    selected_weights = tl.gather(weights, safe, 0)
    sentinel = LOCAL if OFFSET == 0 else 0
    tl.store(
        OUT_IDS + row * OUT_K + destination,
        tl.where(selected, selected_ids + OFFSET, sentinel),
        mask=destination < OUT_K,
    )
    tl.store(
        OUT_W + row * OUT_K + destination,
        tl.where(selected, selected_weights, 0),
        mask=destination < OUT_K,
    )


def compact_routes(ids, weights, active, *, offset, local, experts, global_ids=False):
    """Each row's live routes to the `local` experts from `offset` on, in slot order, in
    `min(top-k, local)` slots. A free slot names a foreign expert with weight 0. `active` is a
    row mask, or a one-element count of leading active rows."""
    rows, top_k = ids.shape
    width = min(top_k, local)
    out_ids = torch.empty((rows, width), device=ids.device, dtype=torch.int64)
    out_weights = torch.empty((rows, width), device=ids.device, dtype=torch.float32)
    if rows:
        _compact_routes[(rows,)](
            ids,
            weights,
            active,
            out_ids,
            out_weights,
            rows,
            top_k,
            width,
            offset,
            local,
            triton.next_power_of_2(top_k),
            triton.next_power_of_2(width),
            experts,
            global_ids,
            active.dtype != torch.bool,
            debug=True,
        )
    return out_ids, out_weights


@triton.jit
def _compact_routes_tiled(
    IDS,
    W,
    ACTIVE,
    OUT_IDS,
    OUT_W,
    N,
    K: tl.constexpr,
    OUT_K: tl.constexpr,
    OFFSET: tl.constexpr,
    LOCAL: tl.constexpr,
    BK: tl.constexpr,
    BO: tl.constexpr,
    R: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    inside = rows < N
    active = rows < tl.load(ACTIVE)
    slot = tl.arange(0, BK)
    destination = tl.arange(0, BO)
    load = (active & inside)[:, None] & (slot < K)[None, :]
    ids = tl.load(IDS + rows[:, None] * K + slot[None, :], mask=load, other=-1)
    weights = tl.load(W + rows[:, None] * K + slot[None, :], mask=load, other=0.0)
    local_ids = ids - OFFSET
    live = load & (local_ids >= 0) & (local_ids < LOCAL)
    position = tl.cumsum(live.to(tl.int32), 1) - 1
    match = live[:, :, None] & (position[:, :, None] == destination[None, None, :])
    source = tl.min(tl.where(match, slot[None, :, None], BK), 1)
    selected = source < BK
    safe = tl.minimum(source, BK - 1)
    selected_ids = tl.gather(local_ids, safe, 1)
    selected_weights = tl.gather(weights, safe, 1)
    sentinel = LOCAL if OFFSET == 0 else 0
    store = inside[:, None] & (destination < OUT_K)[None, :]
    tl.store(
        OUT_IDS + rows[:, None] * OUT_K + destination[None, :],
        tl.where(selected, selected_ids + OFFSET, sentinel),
        mask=store,
    )
    tl.store(
        OUT_W + rows[:, None] * OUT_K + destination[None, :],
        tl.where(selected, selected_weights, 0.0),
        mask=store,
    )


def compact_routes_tiled(ids, weights, count, *, offset, local):
    """`compact_routes` for global ids and a one-element count of leading active rows, 64 rows
    per program."""
    rows, top_k = ids.shape
    width = min(top_k, local)
    out_ids = torch.empty((rows, width), device=ids.device, dtype=torch.int64)
    out_weights = torch.empty((rows, width), device=ids.device, dtype=torch.float32)
    if rows:
        _compact_routes_tiled[(triton.cdiv(rows, 64),)](
            ids,
            weights,
            count,
            out_ids,
            out_weights,
            rows,
            top_k,
            width,
            offset,
            local,
            triton.next_power_of_2(top_k),
            triton.next_power_of_2(width),
            64,
        )
    return out_ids, out_weights


# --- FP8 activations: DeepEP's per-token cast ---------------------------------------------------

FP8_MAX, AMAX_FLOOR, FP8_BLOCK = 448.0, 1e-4, 128


@triton.jit
def _fp8_cast(
    X, OUT, SCALE, H: tl.constexpr, BLOCKS: tl.constexpr, BLOCK: tl.constexpr, floor, maximum
):
    row = tl.program_id(0).to(tl.int64)
    index = tl.program_id(1)
    offsets = index * BLOCK + tl.arange(0, BLOCK)
    inside = offsets < H
    values = tl.load(X + row * H + offsets, mask=inside, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(values)), floor)
    reciprocal = tl.inline_asm_elementwise(
        "rcp.rn.f32 $0, $1;", "=r,r", [amax], dtype=tl.float32, is_pure=True, pack=1
    )
    tl.store(
        OUT + row * H + offsets, (values * (reciprocal * maximum)).to(tl.float8e4nv), mask=inside
    )
    tl.store(SCALE + row * BLOCKS + index, amax / maximum)


def fp8_cast(values):
    """`values` as float8_e4m3fn and one float32 scale per 128 columns, as DeepEP casts them."""
    rows, width = values.shape
    blocks = triton.cdiv(width, FP8_BLOCK)
    payload = torch.empty((rows, width), dtype=torch.float8_e4m3fn, device=values.device)
    scale = torch.empty((rows, blocks), dtype=torch.float32, device=values.device)
    if rows:
        _fp8_cast[(rows, blocks)](
            values.contiguous(),
            payload,
            scale,
            width,
            blocks,
            FP8_BLOCK,
            AMAX_FLOOR,
            FP8_MAX,
            enable_fp_fusion=False,
        )
    return payload, scale


@triton.jit
def _fp8_cast_back(
    CAST,
    SCALE,
    OUT,
    COUNT,
    H: tl.constexpr,
    BLOCKS: tl.constexpr,
    BH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    cols = tl.arange(0, BH)
    mask = cols < H
    for row in range(tl.program_id(0), tl.load(COUNT), tl.num_programs(0)):
        row = row.to(tl.int64)
        values = tl.load(CAST + row * H + cols, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(SCALE + row * BLOCKS + cols // BLOCK, mask=mask, other=0.0)
        tl.store(OUT + row * H + cols, (values * scale).to(tl.bfloat16), mask=mask)


def fp8_cast_back(payload, scale, count):
    """BF16 rows from `fp8_cast`'s output for the first `count` rows (a one-element tensor). The
    rest are left unwritten."""
    rows, width = payload.shape
    out = torch.empty((rows, width), dtype=torch.bfloat16, device=payload.device)
    if rows:
        _fp8_cast_back[(PROGRAMS,)](
            payload,
            scale,
            out,
            count,
            width,
            scale.shape[1],
            triton.next_power_of_2(width),
            FP8_BLOCK,
            num_warps=8 if width >= 2048 else 4,
            enable_fp_fusion=False,
        )
    return out


# --- reference layers ---------------------------------------------------------------------------


def row_layout(hidden, topk):
    """Byte layout of one all-gathered row `[x | ids | weights | count | pad]`, 16-byte aligned.
    Offsets in units of the element type that reads them."""
    assert hidden % 4 == 0, hidden
    assert topk % 2 == 0, topk
    assert 0 < topk <= 16, topk
    x_bytes, ids_bytes, w_bytes = 2 * hidden, 8 * topk, 4 * topk
    used = x_bytes + ids_bytes + w_bytes + 8
    row = -(-used // 16) * 16
    return dict(
        row_bytes=row,
        ids=(x_bytes) // 8,
        weights=(x_bytes + ids_bytes) // 4,
        count=(x_bytes + ids_bytes + w_bytes) // 8,
    )


def fp8_row_layout(hidden, topk):
    """`row_layout` with the row as `[x in FP8 | float32 scale per 128 | ids | weights | count |
    pad]`."""
    blocks = triton.cdiv(hidden, FP8_BLOCK)
    x_bytes, s_bytes, ids_bytes, w_bytes = hidden, 4 * blocks, 8 * topk, 4 * topk
    assert x_bytes % 8 == 0
    assert (x_bytes + s_bytes) % 8 == 0
    used = x_bytes + s_bytes + ids_bytes + w_bytes + 8
    return dict(
        row_bytes=-(-used // 16) * 16,
        scales=x_bytes // 4,
        ids=(x_bytes + s_bytes) // 8,
        weights=(x_bytes + s_bytes + ids_bytes) // 4,
        count=(x_bytes + s_bytes + ids_bytes + w_bytes) // 8,
    )


@triton.jit
def _send_reference(
    X,
    IDS,
    W,
    COUNT,
    WB,
    WI,
    WF,
    ACTIVE,
    N,
    H: tl.constexpr,
    K: tl.constexpr,
    RB2: tl.constexpr,
    RB4: tl.constexpr,
    RB8: tl.constexpr,
    IDO: tl.constexpr,
    WO: tl.constexpr,
    CO: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, BH)
    k = tl.arange(0, BK)
    count = tl.load(COUNT)
    live = (row < N) & (row < count)
    ids = tl.load(IDS + row * K + k, mask=live & (k < K), other=-1).to(tl.int64)
    raw = tl.load(W + row * K + k, mask=live & (k < K), other=0)
    weights = tl.where(ids >= 0, raw, 0)
    for c in tl.static_range(0, H, BH):
        x = tl.load(X + row * H + c + h, mask=live & (c + h < H), other=0)
        tl.store(WB + row * RB2 + c + h, x, mask=c + h < H)
    tl.store(WI + row * RB8 + IDO + k, ids, mask=k < K)
    tl.store(WF + row * RB4 + WO + k, weights, mask=k < K)
    tl.store(WI + row * RB8 + CO, count)
    tl.store(ACTIVE + row, live & (tl.sum((ids >= 0).to(tl.int32), 0) > 0), mask=row < N)


@triton.jit
def _send_reference_fp8(
    X,
    IDS,
    W,
    COUNT,
    W1,
    WI,
    WF,
    ACTIVE,
    N,
    H: tl.constexpr,
    K: tl.constexpr,
    RB1: tl.constexpr,
    RB4: tl.constexpr,
    RB8: tl.constexpr,
    SO: tl.constexpr,
    IDO: tl.constexpr,
    WO: tl.constexpr,
    CO: tl.constexpr,
    BK: tl.constexpr,
    floor,
    maximum,
):
    """_send_reference with the row cast as fp8_cast casts it (zeros past the count)."""
    row = tl.program_id(0).to(tl.int64)
    k = tl.arange(0, BK)
    count = tl.load(COUNT)
    live = (row < N) & (row < count)
    ids = tl.load(IDS + row * K + k, mask=live & (k < K), other=-1).to(tl.int64)
    raw = tl.load(W + row * K + k, mask=live & (k < K), other=0)
    weights = tl.where(ids >= 0, raw, 0)
    h = tl.arange(0, 128)
    for c in tl.static_range(0, H, 128):
        x = tl.load(X + row * H + c + h, mask=live & (c + h < H), other=0).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(x)), floor)
        reciprocal = tl.inline_asm_elementwise(
            "rcp.rn.f32 $0, $1;", "=r,r", [amax], dtype=tl.float32, is_pure=True, pack=1
        )
        cast = (x * (reciprocal * maximum)).to(tl.float8e4nv)
        tl.store(W1 + row * RB1 + c + h, cast.to(tl.uint8, bitcast=True), mask=c + h < H)
        tl.store(WF + row * RB4 + SO + c // 128, amax / maximum)
    tl.store(WI + row * RB8 + IDO + k, ids, mask=k < K)
    tl.store(WF + row * RB4 + WO + k, weights, mask=k < K)
    tl.store(WI + row * RB8 + CO, count)
    tl.store(ACTIVE + row, live & (tl.sum((ids >= 0).to(tl.int32), 0) > 0), mask=row < N)


@triton.jit
def _receive_reference(
    WB,
    WI,
    WF,
    EXECUTION,
    CACHE,
    VALID,
    EPOCH,
    REFERENCE,
    LAYER,
    OUT_IDS,
    OUT_W,
    ACTIVE,
    CAP: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    OUT_K: tl.constexpr,
    RB2: tl.constexpr,
    RB4: tl.constexpr,
    RB8: tl.constexpr,
    IDO: tl.constexpr,
    WO: tl.constexpr,
    CO: tl.constexpr,
    OFFSET: tl.constexpr,
    LOCAL: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
    BO: tl.constexpr,
    COMPACT: tl.constexpr,
    FP8: tl.constexpr,
    W1,
    SO: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, BH)
    slot = tl.arange(0, BK)
    count = tl.load(WI + row * RB8 + CO)
    valid = (count >= 0) & (count <= CAP) & (row % CAP < count)
    ids = tl.load(WI + row * RB8 + IDO + slot, mask=slot < K, other=-1)
    raw = tl.load(WF + row * RB4 + WO + slot, mask=slot < K, other=0)
    active = valid & (tl.sum((ids >= 0).to(tl.int32), 0) > 0)
    for c in tl.static_range(0, H, BH):
        if FP8:
            # W1: the row's bytes. The scales start at float32 offset SO.
            payload = tl.load(W1 + row * (RB4 * 4) + c + h, mask=c + h < H, other=0)
            scale = tl.load(WF + row * RB4 + SO + (c + h) // 128, mask=c + h < H, other=0.0)
            value = (payload.to(tl.float8e4nv, bitcast=True).to(tl.float32) * scale).to(
                tl.bfloat16
            )
        else:
            value = tl.load(WB + row * RB2 + c + h, mask=c + h < H, other=0)
        tl.store(CACHE + row * H + c + h, value, mask=c + h < H)
    tl.store(VALID + row, valid)
    if row == 0:
        tl.store(EPOCH, tl.load(EXECUTION))
        tl.store(REFERENCE, LAYER)
    sentinel = LOCAL if OFFSET == 0 else 0
    routed = active & (ids >= 0)
    weight = tl.where(routed, raw, 0)
    global_ids = tl.where(routed, ids, sentinel)
    tl.store(ACTIVE + row, active)
    if COMPACT:
        destination = tl.arange(0, BO)
        local_ids = global_ids - OFFSET
        live = active & (slot < K) & (local_ids >= 0) & (local_ids < LOCAL)
        position = tl.cumsum(live.to(tl.int32), 0) - 1
        source = tl.min(
            tl.where(
                live[None, :] & (position[None, :] == destination[:, None]), slot[None, :], BK
            ),
            1,
        )
        selected = source < BK
        safe = tl.minimum(source, BK - 1)
        tl.store(
            OUT_IDS + row * OUT_K + destination,
            tl.where(selected, tl.gather(local_ids, safe, 0) + OFFSET, sentinel),
            mask=destination < OUT_K,
        )
        tl.store(
            OUT_W + row * OUT_K + destination,
            tl.where(selected, tl.gather(weight, safe, 0), 0),
            mask=destination < OUT_K,
        )
    else:
        tl.store(OUT_IDS + row * K + slot, global_ids, mask=slot < K)
        tl.store(OUT_W + row * K + slot, weight, mask=slot < K)


def exchange_reference(
    cache,
    layer,
    epoch,
    hidden_states,
    ids,
    weights,
    count,
    *,
    offset,
    local,
    compact,
    fp8,
    all_gather,
):
    """Send one reference layer's rows and routes to every rank in one all-gather, run as
    `all_gather(out, inp)` (the rows in FP8 if `fp8`), and publish the gathered rows as the cache.
    Returns which own rows route anywhere, the expert input, the kernel routes and weights, and
    which gathered rows are active. The expert input is the cache itself: the experts read only rows
    with a live route, and those rows are cached as sent."""
    capacity, hidden = cache.capacity, cache.hidden
    own_rows, top_k = ids.shape
    if cache.exchange is None:
        # The send block and gathered rows live as long as the cache, for graph replay. The
        # cache's receive buffer is released once they exist.
        layout = fp8_row_layout(hidden, top_k) if fp8 else row_layout(hidden, top_k)
        cache.exchange = (
            layout,
            torch.empty(
                (capacity, layout["row_bytes"]), device=hidden_states.device, dtype=torch.uint8
            ),
            torch.empty(
                (cache.origins * capacity, layout["row_bytes"]),
                device=hidden_states.device,
                dtype=torch.uint8,
            ),
        )
        cache.received = None
    layout, send, gathered = cache.exchange
    row_bytes = layout["row_bytes"]
    send_b, send_i, send_f = (
        send.view(dtype) for dtype in (torch.bfloat16, torch.int64, torch.float32)
    )
    active = torch.empty(own_rows, device=hidden_states.device, dtype=torch.bool)
    hidden_block = min(triton.next_power_of_2(hidden), 4096)
    top_k_block = triton.next_power_of_2(top_k)
    if fp8:
        _send_reference_fp8[(capacity,)](
            hidden_states,
            ids,
            weights,
            count,
            send,
            send_i,
            send_f,
            active,
            own_rows,
            hidden,
            top_k,
            row_bytes,
            row_bytes // 4,
            row_bytes // 8,
            layout["scales"],
            layout["ids"],
            layout["weights"],
            layout["count"],
            top_k_block,
            AMAX_FLOOR,
            FP8_MAX,
            enable_fp_fusion=False,
        )
    else:
        _send_reference[(capacity,)](
            hidden_states,
            ids,
            weights,
            count,
            send_b,
            send_i,
            send_f,
            active,
            own_rows,
            hidden,
            top_k,
            row_bytes // 2,
            row_bytes // 4,
            row_bytes // 8,
            layout["ids"],
            layout["weights"],
            layout["count"],
            hidden_block,
            top_k_block,
        )
    all_gather(gathered, send)
    rows = cache.origins * capacity
    width = min(top_k, local) if compact else top_k
    out_ids = torch.empty(rows, width, device=hidden_states.device, dtype=torch.int64)
    out_weights = torch.empty(rows, width, device=hidden_states.device, dtype=torch.float32)
    received_active = torch.empty(rows, device=hidden_states.device, dtype=torch.bool)
    got_b, got_i, got_f = (
        gathered.view(dtype) for dtype in (torch.bfloat16, torch.int64, torch.float32)
    )
    _receive_reference[(rows,)](
        got_b,
        got_i,
        got_f,
        epoch,
        cache.values,
        cache.valid,
        cache.epoch,
        cache.layer,
        layer,
        out_ids,
        out_weights,
        received_active,
        capacity,
        hidden,
        top_k,
        width,
        row_bytes // 2,
        row_bytes // 4,
        row_bytes // 8,
        layout["ids"],
        layout["weights"],
        layout["count"],
        offset,
        local,
        hidden_block,
        top_k_block,
        triton.next_power_of_2(width),
        compact,
        fp8,
        gathered,
        layout.get("scales", 0),
        enable_fp_fusion=False,
    )
    return active, cache.values.view(rows, hidden), out_ids, out_weights, received_active


# --- every layer: combine and owner output -------------------------------------------------------


@triton.jit
def _mask_rows(Y, ACTIVE, N, H: tl.constexpr, B: tl.constexpr):
    # One CTA per row: active rows do not touch the matrix at all. A flat
    # elementwise mask needlessly reloads the same flag for every column tile.
    row = tl.program_id(0)
    if row < N:  # noqa: SIM102 (the kernel as measured)
        if not tl.load(ACTIVE + row):
            col = tl.arange(0, B)
            tl.store(Y + row * H + col, 0, mask=col < H)


def zero_inactive_(values, active):
    """Zero the rows of `values` whose `active` flag is false, in place."""
    rows, hidden = values.shape
    if rows:
        _mask_rows[(rows,)](values, active, rows, hidden, triton.next_power_of_2(hidden))
    return values


def decode_into(output, code, decoder, active):
    """The owner's output `D_c c_c` written into vLLM's output, inactive rows zeroed."""
    torch.mm(code, decoder.t(), out=output)
    return zero_inactive_(output, active)


@triton.jit
def _fp8_cast_packed(X, OUT, H: tl.constexpr, STRIDE: tl.constexpr, floor, maximum):
    """fp8_cast into one byte row per input row: `[FP8 payload (H) | float32 scale per 128]`."""
    row = tl.program_id(0).to(tl.int64)
    index = tl.program_id(1)
    offsets = index * 128 + tl.arange(0, 128)
    inside = offsets < H
    values = tl.load(X + row * H + offsets, mask=inside, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(values)), floor)
    reciprocal = tl.inline_asm_elementwise(
        "rcp.rn.f32 $0, $1;", "=r,r", [amax], dtype=tl.float32, is_pure=True, pack=1
    )
    cast = (values * (reciprocal * maximum)).to(tl.float8e4nv)
    tl.store(OUT + row * STRIDE + offsets, cast.to(tl.uint8, bitcast=True), mask=inside)
    scales = (OUT + row * STRIDE + H).to(tl.pointer_type(tl.float32))
    tl.store(scales + index, amax / maximum)


@triton.jit
def _sum_nodes_fp8(
    PACKED,
    OWN,
    OUT,
    ROWS,
    SELF,
    N: tl.constexpr,
    K: tl.constexpr,
    STRIDE: tl.constexpr,
    BK: tl.constexpr,
):
    """out[r] = sum over nodes m, in node order and float32, of node m's partial for row r: this
    node's in BF16 (`OWN`), every other node's dequantized from its FP8 row."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BK)
    mask = cols < K
    total = tl.zeros((BK,), dtype=tl.float32)
    for m in range(N):
        base = PACKED + (m * ROWS + row) * STRIDE
        values = (
            tl.load(base + cols, mask=mask, other=0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        )
        scale = tl.load(
            (base + K).to(tl.pointer_type(tl.float32)) + cols // 128, mask=mask, other=0.0
        )
        own = tl.load(OWN + (m * ROWS + row) * K + cols, mask=mask, other=0.0).to(tl.float32)
        total = total + tl.where(m == SELF, own, values * scale)
    tl.store(OUT + row * K + cols, total.to(tl.bfloat16), mask=mask)


@functools.cache
def node_groups(group):
    """This rank's node group and rail group (the same local rank on every node) of `group`, made
    once on every rank in the same order, as new_group requires."""
    ranks = dist.get_process_group_ranks(group)
    local = int(os.environ["LOCAL_WORLD_SIZE"])
    mine = dist.get_rank()
    node = rail = None
    for start in range(0, len(ranks), local):
        members = ranks[start : start + local]
        created = dist.new_group(members)
        node = created if mine in members else node
    for offset in range(local):
        members = ranks[offset::local]
        created = dist.new_group(members)
        rail = created if mine in members else rail
    return node, rail


def reduce_scatter_fp8_nodes(code, group, capacity):
    """A reference layer's combine in FP8 with the in-node sum first, as DeepEP's FP8 combine sums.
    A BF16 reduce-scatter in each node hands every rank its node's partials of the rows its rail
    peers own (the same local rank on every node). They go to those peers as FP8 in one all-to-all
    over the rail and are summed there, the owner node's partial kept in BF16. `code` is
    `[ranks * capacity, r_c]` in rank order."""
    node, rail = node_groups(group)
    local, nodes = dist.get_world_size(node), dist.get_world_size(rail)
    rows, width = code.shape
    # rank-major (node, local) to (local, node): each node rank's chunk is its rail's rows
    by_local = code.view(nodes, local, capacity, width).transpose(0, 1).contiguous()
    partial = torch.empty((nodes * capacity, width), dtype=code.dtype, device=code.device)
    dist.reduce_scatter_tensor(partial, by_local.view(rows, width), group=node)
    blocks = triton.cdiv(width, FP8_BLOCK)
    stride = width + 4 * blocks
    packed = torch.empty((nodes * capacity, stride), dtype=torch.uint8, device=code.device)
    _fp8_cast_packed[(nodes * capacity, blocks)](
        partial, packed, width, stride, AMAX_FLOOR, FP8_MAX, enable_fp_fusion=False
    )
    back = torch.empty_like(packed)
    dist.all_to_all_single(back, packed, group=rail)
    # every node's partial of this rank's rows in node order. This node's own is `partial`'s chunk
    # at this rank's place in the rail (its node).
    out = torch.empty((capacity, width), dtype=torch.bfloat16, device=code.device)
    _sum_nodes_fp8[(capacity,)](
        back,
        partial,
        out,
        capacity,
        dist.get_rank(rail),
        nodes,
        width,
        stride,
        triton.next_power_of_2(width),
        enable_fp_fusion=False,
    )
    return out


# --- both variants: the expert slots this rank computes -----------------------------------------


@triton.jit
def _silu_mul_live(IN, OUT, LIVE, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    live = tl.load(LIVE + row)
    if live != 0:
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(IN + row * (2 * N) + cols, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(IN + row * (2 * N) + N + cols, mask=mask, other=0.0).to(tl.float32)
        silu = tl.div_rn(x, 1.0 + libdevice.exp(-x)).to(tl.bfloat16).to(tl.float32)
        tl.store(OUT + row * N + cols, (silu * y).to(tl.bfloat16), mask=mask)


def silu_and_mul_live(output, input, live):
    """`output[r] = silu(input[r, :N]) * input[r, N:]` for rows with `live[r] != 0`, with vLLM's
    CUDA kernel's roundings (fp32 SiLU with IEEE division, bf16 SiLU, bf16 product). Other rows
    are left as they are: no later kernel reads them."""
    rows, gate_up = input.shape
    width = gate_up // 2
    block = 1024 if width >= 1024 else triton.next_power_of_2(width)
    if rows:
        _silu_mul_live[(rows, triton.cdiv(width, block))](
            input, output, live, N=width, BLOCK=block
        )


def live_rows(topk_ids, expert_map):
    """[M * topk] uint8: 1 where a slot's expert is on this rank, 0 for foreign or empty slots."""
    ids = topk_ids.reshape(-1).long()
    valid = ids >= 0
    local = expert_map[ids.clamp_min(0)] >= 0
    return (valid & local).to(torch.uint8)


# Programs of the kernels that loop over rows: 4 per SM of an H100 (132 SMs).
PROGRAMS = 132 * 4


@triton.jit
def _silu_mul_sorted(IN, OUT, SORTED, POST, NUMEL, N: tl.constexpr, BLOCK: tl.constexpr):
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    for i in range(tl.program_id(0), tl.load(POST), tl.num_programs(0)):
        row = tl.load(SORTED + i).to(tl.int64)
        if row < NUMEL:
            x = tl.load(IN + row * (2 * N) + cols, mask=mask, other=0.0).to(tl.float32)
            y = tl.load(IN + row * (2 * N) + N + cols, mask=mask, other=0.0).to(tl.float32)
            silu = tl.div_rn(x, 1.0 + libdevice.exp(-x)).to(tl.bfloat16).to(tl.float32)
            tl.store(OUT + row * N + cols, (silu * y).to(tl.bfloat16), mask=mask)


def silu_and_mul_sorted(output, input, sorted_ids, post):
    """`silu_and_mul_live` of the slots `align` listed: the first `post` entries of `sorted_ids`,
    whose padding entries equal the slot count."""
    rows, gate_up = input.shape
    width = gate_up // 2
    _silu_mul_sorted[(PROGRAMS,)](
        input,
        output,
        sorted_ids,
        post,
        rows,
        N=width,
        BLOCK=triton.next_power_of_2(width),
        num_warps=4,
    )


@triton.jit
def _sum_live(IN, IDS, MAP, OUT, COUNT, K: tl.constexpr, TOPK: tl.constexpr, BLOCK: tl.constexpr):
    cols = tl.arange(0, BLOCK)
    mask = cols < K
    end = tl.load(COUNT)
    for row in range(tl.program_id(0), end, tl.num_programs(0)):
        row = row.to(tl.int64)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for slot in tl.static_range(TOPK):
            expert = tl.load(IDS + row * TOPK + slot)
            local = tl.load(MAP + tl.maximum(expert, 0))
            if (expert >= 0) & (local >= 0):
                value = tl.load(IN + (row * TOPK + slot) * K + cols, mask=mask, other=0.0).to(
                    tl.float32
                )
                acc = acc + value
        tl.store(OUT + row * K + cols, acc.to(tl.bfloat16), mask=mask)


def sum_live(input, output, ids, expert_map, count):
    """`output[r]`, the sum of row r's slots whose expert is on this rank, for the rows below the
    one-element `count`. Later rows are left as they are: the combine reads only received rows."""
    rows, top_k = ids.shape
    width = output.shape[1]
    if rows:
        _sum_live[(PROGRAMS,)](
            input,
            ids,
            expert_map,
            output,
            count,
            K=width,
            TOPK=top_k,
            BLOCK=triton.next_power_of_2(width),
            num_warps=8 if width >= 2048 else 4,
        )


@triton.jit
def _align_count(
    IDS, MAP, COUNTS, NUMEL, E: tl.constexpr, BINS: tl.constexpr, CHUNK: tl.constexpr
):
    program = tl.program_id(0)
    index = program * CHUNK + tl.arange(0, CHUNK)
    expert = tl.load(IDS + index, mask=index < NUMEL, other=-1)
    local = tl.load(MAP + tl.maximum(expert, 0), mask=expert >= 0, other=-1)
    histogram = tl.histogram(tl.where(local >= 0, local, E), BINS)
    tl.store(COUNTS + program * E + tl.arange(0, BINS), histogram, mask=tl.arange(0, BINS) < E)


@triton.jit
def _align_scan(
    COUNTS,
    OFFSETS,
    SORTED,
    EXPERT_IDS,
    POST,
    PROGRAMS,
    NUMEL,
    E: tl.constexpr,
    BINS: tl.constexpr,
    P: tl.constexpr,
    BM: tl.constexpr,
):
    programs = tl.arange(0, P)
    experts = tl.arange(0, BINS)
    mask = (programs < PROGRAMS)[:, None] & (experts < E)[None, :]
    counts = tl.load(COUNTS + programs[:, None] * E + experts[None, :], mask=mask, other=0)
    within = tl.cumsum(counts, 0) - counts
    totals = tl.sum(counts, 0)
    padded = (totals + BM - 1) // BM * BM
    starts = tl.cumsum(padded, 0) - padded
    tl.store(
        OFFSETS + programs[:, None] * E + experts[None, :], within + starts[None, :], mask=mask
    )
    tl.store(POST, tl.sum(padded, 0))
    for e in tl.static_range(E):
        start = tl.sum(tl.where(experts == e, starts, 0), 0)
        total = tl.sum(tl.where(experts == e, totals, 0), 0)
        size = tl.sum(tl.where(experts == e, padded, 0), 0)
        pad = tl.arange(0, BM)
        tl.store(SORTED + start + total + pad, NUMEL, mask=total + pad < size)
        for block in range(start // BM, (start + size) // BM):
            tl.store(EXPERT_IDS + block, e)


@triton.jit
def _align_place(IDS, MAP, OFFSETS, SORTED, NUMEL, E: tl.constexpr, CHUNK: tl.constexpr):
    program = tl.program_id(0)
    index = program * CHUNK + tl.arange(0, CHUNK)
    expert = tl.load(IDS + index, mask=index < NUMEL, other=-1)
    local = tl.load(MAP + tl.maximum(expert, 0), mask=expert >= 0, other=-1)
    for e in tl.static_range(E):
        match = local == e
        rank = tl.cumsum(match.to(tl.int32), 0) - 1
        base = tl.load(OFFSETS + program * E + e)
        tl.store(SORTED + base + rank, index.to(tl.int32), mask=match)


def align(topk_ids, block_m, expert_map, local_experts):
    """vLLM's `moe_align_block_size` with `ignore_invalid_experts`, for a few local experts: the
    slots whose expert is on this rank grouped by local expert in slot order, each group padded to
    `block_m` with the slot count. Returns the sorted slots, each block's local expert and the
    padded length."""
    numel = topk_ids.numel()
    device = topk_ids.device
    chunk = min(4096, max(256, triton.next_power_of_2(triton.cdiv(numel, 128))))
    programs = triton.cdiv(numel, chunk)
    bins = triton.next_power_of_2(local_experts + 1)
    capacity = numel + local_experts * (block_m - 1)
    counts = torch.empty((programs, local_experts), dtype=torch.int32, device=device)
    offsets = torch.empty_like(counts)
    sorted_ids = torch.empty(capacity, dtype=torch.int32, device=device)
    expert_ids = torch.full(
        (triton.cdiv(capacity, block_m),), -1, dtype=torch.int32, device=device
    )
    post = torch.empty(1, dtype=torch.int32, device=device)
    ids = topk_ids.reshape(-1)
    _align_count[(programs,)](ids, expert_map, counts, numel, local_experts, bins, chunk)
    _align_scan[(1,)](
        counts,
        offsets,
        sorted_ids,
        expert_ids,
        post,
        programs,
        numel,
        local_experts,
        bins,
        triton.next_power_of_2(programs),
        block_m,
    )
    _align_place[(programs,)](ids, expert_map, offsets, sorted_ids, numel, local_experts, chunk)
    return sorted_ids, expert_ids, post
