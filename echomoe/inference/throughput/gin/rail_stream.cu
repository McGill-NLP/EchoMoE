// Streaming MoE dispatch and combine at EP over GIN rails (NCCL device API), for prefill sizes: rail_moe.cu's two
// hops (InfiniBand to the same local rank on each other node, then NVLink), with each token sent once per node and
// every remote transfer split into chunks that move on as they land. See rails.py for the outputs.
//
// Dispatch, at source s = (node, local rank l):
//   The worker CTAs route s's tokens together (each a share of them, into State; then each copies all of it in).
//   They then write one bundle per node (this one's too) in s's window: a head (each of the node's GPUs' record
//   count, where s's combine rows from that node go, the chunking, and per chunk each GPU's records before it and
//   where its list starts), then the chunks, in slices spread over the CTAs: the records (the row, then the token
//   index, top-k ids, weights and the token's first row in s's combine block for the node) of the tokens bound for
//   the node, once each, in token order, then each GPU's list of the records it takes. A remote bundle goes to s's
//   rail peer there, the head and then each chunk (by the CTA finishing its last slice) in its own put and signal,
//   over a GIN context picked by destination and chunk; the relay CTA there flags each one in its node-mates'
//   windows as it lands. The own node's pieces are flagged by their CTAs.
//   Every rank waits for all heads, packs the row counts into psum and pulls its rows over NVLink from wherever
//   each source's bundle is (the source, on this node; else the source's rail peer here), each chunk once flagged,
//   straight into the outputs.
// Combine sums each token's rows in fp32 in destination order, as rail_moe.cu's: each received row goes to its
//   source over NVLink if on-node, else into the source's block at the source's rail peer here (the aggregator).
//   A source's block for a node holds its tokens' rows there by token, then destination; the aggregator's sender
//   CTA sends it in order over one GIN context, in pieces of at least `chunk` bytes as they fill, straight into the
//   source's receive area, where the rows landed so far are a prefix: the source sums each token once its rows are
//   in. Rows are not added on the way, so the bits are rail_moe.cu's. With `reduce`, the aggregator's CTAs add
//   each token's rows there (in fp32, in destination order, rounded to bf16) as the block fills (some warps from
//   the start, the rest once they have written), and the sender sends those partials instead; the source adds the
//   nodes' partials (its own node's from its rows the same way) in fp32 in node order.
// NVLink arrivals are published with a system fence and relaxed adds or stores. Grids are `ctas` cooperative CTAs:
// ctas - 1 workers, then the relay (dispatch) or sender (combine). Signals and counters only grow: a kernel waits
// for its call's arrivals above the values in State; flags hold the dispatch call number.
// The trace stores (Trace events, globaltimer) and the non-reduce branches stay compiled in so that the kernels are
// the measured ones, and the host never enables them: every combine reduces node by node.
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

#define NCCL_CHECK(cmd)                                                                      \
    do {                                                                                     \
        ncclResult_t r_ = (cmd);                                                             \
        if (r_ != ncclSuccess) throw std::runtime_error(std::string(#cmd ": ") + ncclGetErrorString(r_)); \
    } while (0)
#define CUDA_CHECK(cmd)                                                                      \
    do {                                                                                     \
        cudaError_t e_ = (cmd);                                                              \
        if (e_ != cudaSuccess) throw std::runtime_error(std::string(#cmd ": ") + cudaGetErrorString(e_)); \
    } while (0)

namespace {

constexpr int kThreads = 512, kWarps = kThreads / 32, kMaxT = 4096, kMaxRanks = 32, kMaxNodes = 8, kMaxTopk = 8;
constexpr int kMaxLsa = 8, kMaxContexts = 8, kUnroll = 8, kMaxChunks = 64;
constexpr int kRows = 32;  // rows per warp work item, at most
// bytes per dispatch pull item, combine write item and dispatch pack slice, about
constexpr size_t kItemBytes = 16 << 10, kWriteBytes = 8 << 10, kSliceBytes = 256 << 10;
constexpr int kSliceRecords = 1024;  // at most
constexpr int kReduceWarps = 4;     // per combine CTA, with reduce
constexpr int kBatch = 8;           // records per claimed batch of partials, with reduce
// GIN signals, per context.
enum Signal {
    kHeadFrom,                                         // + node: the head from that node's rail peer landed
    kChunkFrom = kHeadFrom + kMaxNodes,                // + node * kMaxChunks + c: chunk c of its bundle landed
    kCombFrom = kChunkFrom + kMaxNodes * kMaxChunks,   // + node: combine rows from that node's aggregator (adds the
                                                       //   rows, and 1 with the last put)
    kNumSignals = kCombFrom + kMaxNodes,
};
// Window counters; k counts remote nodes from this one's next.
enum Counter {
    kCombArrive,                                // a node-mate CTA wrote its combine rows
    kGatherDone,                                // + k: a node-mate CTA wrote all its rows for remote node k's sources
    kPiece = kGatherDone + kMaxNodes,           // + k * kMaxChunks + p: rows written into piece p of node k's block
    kNumCounters = kPiece + kMaxNodes * kMaxChunks,
};
// Trace slots, per kernel.
enum Trace {
    kDStart, kDRouted, kDHeadsSent, kDPacked, kDHeadsRelayed, kDSeen, kDLastRelay, kDEnd, kTraceSlots = 32,
    kCStart = 0, kCWritten, kCFirstPut, kCPut, kCArrived = kCPut + kMaxNodes - 1, kCEnd,
};

// Device memory beside the window: the signal and counter values of the calls so far, and per-call scratch.
struct State {
    uint64_t signal[kMaxContexts][kNumSignals];
    uint64_t counter[kNumCounters];
    uint64_t calls;                          // dispatch calls; flags hold calls + 1 during a call
    unsigned packed[kMaxNodes][kMaxChunks];  // slices of each chunk written (reset by the last)
    unsigned exits, routed, routed_base;     // routed: worker CTAs past their share of the routing
    unsigned reduced[kMaxNodes][kMaxChunks];  // with reduce: partials written into each piece (reset by the sender)
    unsigned claimed;                          // with reduce: batches of partials claimed this call
    uint32_t mask[kMaxT], ballot[kMaxT / 32 * (kMaxRanks + 2 * kMaxNodes)];  // the routing, as Route's
};

// A bundle's head. Chunk c follows at c * per_chunk * (record bytes + 2 * lsa) past it: its records, then each
// GPU's list (uint16 record indices) of the records it takes there, GPU j's from list[j][c] on.
struct Head {
    int count[kMaxLsa];                  // records for each GPU of the node
    int total, nbase, chunks, per_chunk;  // records; where the node's combine rows go in the source's comb area
    int row_bytes;                        // a record's bytes before its metadata
    int first[kMaxLsa][kMaxChunks + 1];   // each GPU's records in earlier chunks
    int list[kMaxLsa][kMaxChunks + 1];
};

// Window byte offsets. flags: [local rank][node][chunk, then the head] of the call number; heads: 16 bytes per
// source (row count, combine block base, combine block rows); roffs: each received row's row in its source's
// combine block for this node; bundle and fwd: one area per node (bundles written here, and landed here from that
// node's rail peer); gather: one block per remote node k of up to min(lsa, k) * t rows; comb: k * t rows (and
// nodes * t more with reduce); partial (with reduce): one block of t rows per remote node k.
struct Layout {
    size_t flags, heads, roffs, bundle, fwd, area, gather, gather_area, comb, partial, counters, bytes;
};

struct Args {
    ncclDevComm comm;
    ncclWindow_t win;
    Layout at;
    int t, n, k, width, experts_per_rank, ctas, contexts, reduce;
    int vbytes, sbytes;         // dispatch: each row's value bytes (2 * width in bf16) and scale bytes (FP8: a
                                //   float32 per 128 values; 0 in bf16)
    int fp8;                    // combine (with reduce): the partials between nodes in FP8
    size_t chunk;               // bytes per put, at least
    const void* x;              // dispatch: [n, vbytes]
    const float* xs;            // [n, sbytes / 4] or null
    const int64_t* topk_idx;    // [n, k]
    const float* topk_weights;  // [n, k]
    void* recv_x;               // [ranks * t, vbytes]
    float* recv_xs;             // [ranks * t, sbytes / 4] or null
    int64_t* recv_idx;          // [ranks * t, k]
    float* recv_weights;        // [ranks * t, k]
    int* recv_src;              // [ranks * t]
    int* psum;                  // [ranks]
    int* comb_rows;             // [t, k]: each token's rows in comb, in destination order, then -1
    int* node_rows;             // [kMaxNodes]: rows each node returns in combine
    const __nv_bfloat16* y;     // combine: [ranks * t, width]
    __nv_bfloat16* out;         // [n, width]
    State* state;
    uint64_t* trace;            // [kTraceSlots] or null
};

__host__ __device__ size_t align16(size_t v) { return (v + 15) / 16 * 16; }
// token index, top-k ids, top-k weights, first combine row
__host__ __device__ int meta_bytes(int k) { return (4 * (2 + 2 * k) + 15) / 16 * 16; }
__host__ __device__ size_t area_bytes(int t, int width, int k, int lsa) {
    return align16(sizeof(Head)) + size_t(t) * (size_t(width) * 2 + meta_bytes(k) + 2 * lsa);
}

// The pull tables of a dispatch CTA, by source rank.
struct Pull {
    const char* src[kMaxRanks];  // the source's bundle for this node, where this rank reads it
    int prev[kMaxRanks], count[kMaxRanks], chunks[kMaxRanks], total[kMaxRanks], per_chunk[kMaxRanks];
    int first[kMaxRanks][kMaxChunks + 1], list[kMaxRanks][kMaxChunks + 1];
    int items[kMaxRanks * kMaxChunks + 1];  // the first work item of each (source, chunk) group, in pull order
    uint16_t recs[kWarps][kRows];
};
// Dynamic shared memory of a dispatch CTA: Route's mask, ballots and bases; the pull tables; each warp's staged
// record scales (sb bytes, 16-byte aligned) and metadata.
__host__ __device__ size_t scale_bytes(int width) { return align16(4 * size_t((width + 127) / 128)); }
__host__ __device__ size_t route_bytes(int t, int ranks, int nodes) {
    const size_t groups = (t + 31) / 32, cols = ranks + 2 * nodes;
    return align16(4 * size_t(t)) + 2 * align16(4 * groups * cols);
}
__host__ __device__ size_t smem_bytes(int t, int ranks, int nodes, size_t sb) {
    return route_bytes(t, ranks, nodes) + align16(sizeof(Pull)) + size_t(kWarps) * kRows * (sb + meta_bytes(kMaxTopk));
}

__device__ uint64_t globaltimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ uint64_t ld_acquire(const uint64_t* p) {
    uint64_t v;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
__device__ unsigned ld_acquire32(const unsigned* p) {
    unsigned v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ void wait_ge(const uint64_t* p, uint64_t v) {
    while (ld_acquire(p) < v) {}
}
// After fence_sys, relaxed adds and stores publish everything this thread wrote or saw before (with the CTA's or
// warp's writes once synchronized with them).
__device__ void fence_sys() { asm volatile("fence.acq_rel.sys;" ::: "memory"); }
__device__ void add_relaxed(uint64_t* p, uint64_t v) {
    asm volatile("red.relaxed.sys.global.add.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ void st_relaxed32(unsigned* p, unsigned v) {
    asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ int inclusive_scan(int v, int lane) {
    for (int o = 1; o < 32; o <<= 1) {
        const int u = __shfl_up_sync(~0u, v, o);
        if (lane >= o) v += u;
    }
    return v;
}
// Lane d gets bit d of every lane's x, as bit (lane) of the result (a 32x32 bit transpose).
__device__ uint32_t transpose(uint32_t x, int lane) {
    uint32_t keep = 0x0000FFFFu;
    for (int j = 16; j != 0; j >>= 1, keep ^= keep << j) {
        const uint32_t y = __shfl_xor_sync(~0u, x, j);
        x = lane & j ? (x & ~keep) | (y >> j & keep) : (x & keep) | (y & keep) << j;
    }
    return x;
}

// 8 bf16 (one int4) added into acc in fp32
__device__ void accumulate(float* acc, const int4& v) {
    const auto* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const float2 f = __bfloat1622float2(h[e]);
        acc[2 * e] += f.x, acc[2 * e + 1] += f.y;
    }
}
// DeepEP's combine_reduce for at most two slots: a native bf16 add, without fp32
__device__ int4 add_bf16(const int4& a, const int4& b) {
    int4 o;
    auto* h = reinterpret_cast<__nv_bfloat162*>(&o);
    const auto* x = reinterpret_cast<const __nv_bfloat162*>(&a);
    const auto* y = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
    for (int e = 0; e < 4; ++e) h[e] = __hadd2(x[e], y[e]);
    return o;
}
__device__ int4 to_bf16(const float* acc) {
    int4 o;
    auto* h = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
    for (int e = 0; e < 4; ++e) h[e] = __floats2bfloat162_rn(acc[2 * e], acc[2 * e + 1]);
    return o;
}
// FP8 as deepep_fp8c's combine casts its in-node sums: per 128 values amax = max(max |v|, 1e-4), v * (rcp(amax) *
// 448) to e4m3 (round to nearest, saturating; kernels.fp8_cast's values), scale amax / 448 rounded (kernels.fp8_cast's
// Triton cast has amax * (1 / 448) instead, a unit in the last place apart in about half the blocks). The
// whole warp calls it, each lane with 8 bf16 values (segment g of a row of `width` values; the 16 lanes of a
// 128-value block together): values at row + 8 g, scales (float32 per block) at row + width.
__device__ void quantize(char* row, int g, int width, const int4& values, bool live) {
    const auto* h = reinterpret_cast<const __nv_bfloat162*>(&values);
    float f[8], amax = 0;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const float2 p = __bfloat1622float2(h[e]);
        f[2 * e] = p.x, f[2 * e + 1] = p.y;
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) amax = fmaxf(amax, fabsf(f[e]));
    for (int m = 8; m >= 1; m >>= 1) amax = fmaxf(amax, __shfl_xor_sync(~0u, amax, m));
    amax = fmaxf(amax, 1e-4f);
    const float factor = __fmul_rn(__frcp_rn(amax), 448.0f);
    if (!live) return;
    __nv_fp8x2_storage_t q[4];
#pragma unroll
    for (int e = 0; e < 4; ++e)
        q[e] = __nv_cvt_float2_to_fp8x2(make_float2(__fmul_rn(f[2 * e], factor), __fmul_rn(f[2 * e + 1], factor)),
                                        __NV_SATFINITE, __NV_E4M3);
    *reinterpret_cast<int2*>(row + 8 * size_t(g)) = *reinterpret_cast<const int2*>(q);
    if (g % 16 == 0) reinterpret_cast<float*>(row + width)[g / 16] = __fdiv_rn(amax, 448.0f);
}
// acc += 8 FP8 values (q.x, q.y) times their scale, each product rounded, then added (no fused multiply-add)
__device__ void dequantize_add(float* acc, const int4& q, float scale) {
    const auto* pairs = reinterpret_cast<const __nv_fp8x2_storage_t*>(&q);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const float2 f = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(pairs[e], __NV_E4M3)));
        acc[2 * e] = __fadd_rn(acc[2 * e], __fmul_rn(f.x, scale));
        acc[2 * e + 1] = __fadd_rn(acc[2 * e + 1], __fmul_rn(f.y, scale));
    }
}

struct Rank {
    const Args& a;
    int me, lsa, l, ranks, nodes, node, lane, warp, workers, gw, gws;
    size_t rowb, rb, yb, ps;  // dispatch: a record's row bytes and bytes; combine: row bytes, partial bytes

    __device__ Rank(const Args& a)
        : a(a), me(a.comm.rank), lsa(a.comm.lsaSize), l(a.comm.lsaRank), ranks(a.comm.nRanks), nodes(ranks / lsa),
          node(me / lsa), lane(threadIdx.x % 32), warp(threadIdx.x / 32), workers(a.ctas - 1),
          gw(blockIdx.x * kWarps + warp), gws(workers * kWarps), rowb(size_t(a.vbytes) + align16(a.sbytes)),
          rb(rowb + meta_bytes(a.k)), yb(size_t(a.width) * 2),
          ps(a.fp8 ? align16(size_t(a.width) + 4 * ((a.width + 127) / 128)) : yb) {}

    __device__ int at(int n, int j) const { return n * lsa + j; }
    __device__ int remote(int k) const { return (node + 1 + k) % nodes; }
    __device__ char* local(size_t off) const { return static_cast<char*>(ncclGetLocalPointer(a.win, off)); }
    __device__ char* peer(size_t off, int r) const { return static_cast<char*>(ncclGetPeerPointer(a.win, off, r)); }
    __device__ uint64_t* counter(int id, int r) const {
        return reinterpret_cast<uint64_t*>(peer(a.at.counters + 8 * id, r));
    }
    __device__ unsigned* flag(int j, int n, int c, int r) const {
        const size_t i = (size_t(j) * kMaxNodes + n) * (kMaxChunks + 1) + c;
        return reinterpret_cast<unsigned*>(peer(a.at.flags + 4 * i, r));
    }
    __device__ size_t bundle(int n) const { return a.at.bundle + n * a.at.area; }
    __device__ size_t fwd(int n) const { return a.at.fwd + n * a.at.area; }
    __device__ size_t gather(int k) const { return a.at.gather + k * a.at.gather_area; }
    __device__ size_t partial(int k) const { return a.at.partial + size_t(k) * a.t * yb; }
    // Offset of chunk c in a bundle of `per` records per chunk.
    __device__ size_t chunk_at(int c, int per) const {
        return align16(sizeof(Head)) + size_t(c) * per * (rb + 2 * lsa);
    }
    __device__ uint64_t signal(int id, int c) const {
        return ncclGin(a.comm, c, NCCL_GIN_RESOURCE_SHARING_CTA).readSignal(id);
    }
    // The GIN context of chunk c of the bundle node `from` sends node `to` (c = kMaxChunks: the head), spread over
    // destinations, so that each one's proxy work goes on in parallel.
    __device__ int context(int from, int to, int c) const {
        return ((to - from - 1 + nodes) % nodes + (c == kMaxChunks ? 0 : c)) % a.contexts;
    }
    // Rows per piece of a combine block of `total` rows (at most kMaxChunks pieces); with reduce, partials per
    // piece of a block of `total` partials, whole batches.
    __device__ int piece_rows(int total) const {
        const int by = int((a.chunk + max(yb, size_t(16)) - 1) / max(yb, size_t(16)));
        return max(1, max(by, (total + kMaxChunks - 1) / kMaxChunks));
    }
    __device__ int partial_rows(int total) const { return (piece_rows(total) + kBatch - 1) / kBatch * kBatch; }

    __device__ void mark_any(int event) const {
        if (a.trace != nullptr) atomicMax(reinterpret_cast<unsigned long long*>(a.trace + event), globaltimer());
    }
    __device__ void mark(int event) const {
        if (threadIdx.x == 0) mark_any(event);
    }
    __device__ void mark_start(int event) const {
        if (a.trace != nullptr && blockIdx.x == 0 && threadIdx.x == 0) a.trace[event] = globaltimer();
    }

    // One thread: [src, src + bytes) to `peer`'s dst over GIN context c, adding `value` to signal `id` there (a
    // signal alone for 0 bytes); with system release, for data other threads or GPUs wrote.
    __device__ void put(int peer, size_t dst, size_t src, size_t bytes, int id, uint64_t value, int c) const {
        const ncclGin gin(a.comm, c, NCCL_GIN_RESOURCE_SHARING_CTA);
        const auto add = ncclGin_StrongSignalAdd{static_cast<ncclGinSignal_t>(id), value};
        if (bytes > 0)
            gin.put(ncclTeamWorld(a.comm), peer, a.win, dst, a.win, src, bytes, add, ncclGin_None{}, ncclCoopThread{},
                    ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_system);
        else
            gin.signal(ncclTeamWorld(a.comm), peer, add, ncclCoopThread{}, ncclGin_None{}, cuda::thread_scope_thread,
                       cuda::thread_scope_system);
    }
    // One thread, after the writes it publishes: flag (j, n, c) with `epoch` on every node-mate.
    __device__ void flag_node(int j, int n, int c, unsigned epoch) const {
        fence_sys();
        for (int i = 0; i < lsa; ++i) st_relaxed32(flag(j, n, c, at(node, i)), epoch);
    }

    // A warp copies `rows` rows of `segs` 16-byte segments, segment (i, g) from src(i, g) to dst(i, g), with kUnroll
    // loads in flight per lane.
    template <typename Src, typename Dst>
    __device__ void copy_rows(int rows, int segs, Src src, Dst dst) const {
        if (segs == 0) return;
        const int total = rows * segs, di = 32 / segs, dg = 32 % segs;
        int i = lane / segs, g = lane % segs;  // this lane's element (row, segment), 32 elements on per step
        for (int base = 0; base < total; base += 32 * kUnroll) {
            const int i0 = i, g0 = g;
            int4 v[kUnroll];
#pragma unroll
            for (int u = 0; u < kUnroll; ++u) {
                if (base + u * 32 + lane < total) v[u] = *reinterpret_cast<const int4*>(src(i, g));
                i += di, g += dg;
                if (g >= segs) g -= segs, ++i;
            }
            i = i0, g = g0;
#pragma unroll
            for (int u = 0; u < kUnroll; ++u) {
                if (base + u * 32 + lane < total) *reinterpret_cast<int4*>(dst(i, g)) = v[u];
                i += di, g += dg;
                if (g >= segs) g -= segs, ++i;
            }
        }
    }

    // One thread per CTA, once the CTA is past its waits: the last of all CTAs runs `advance`.
    template <typename Advance>
    __device__ void leave(Advance advance) const {
        __threadfence();
        if (atomicAdd(&a.state->exits, 1) == unsigned(a.ctas - 1)) {
            a.state->exits = 0;
            advance();
            __threadfence();
        }
    }
    template <typename Advance>
    __device__ void finish(Advance advance, int end) const {
        __syncthreads();
        mark(end);
        if (threadIdx.x == 0) leave(advance);
    }
};

// The routing, in shared memory: each token's destination GPUs and, per 32-token group and column (GPU d < ranks;
// node ranks + n; rows on node ranks + nodes + n) the ballot of the tokens bound there (the column's rows) and the
// count before the group. The worker CTAs build it together in State, then each copies it in.
struct Route {
    uint32_t* mask;
    uint32_t* ballot;
    int* base;
    int cols, groups, count[kMaxRanks + 2 * kMaxNodes];

    __device__ int pos(int t, int d) const {
        const int g = t / 32 * cols + d;
        return base[g] + __popc(ballot[g] & ((1u << t % 32) - 1));
    }
    // Node n's i-th token: in the last group with fewer before it.
    __device__ int token(const Rank& r, int n, int i) const {
        const int col = r.ranks + n;
        int lo = 0, hi = groups - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) / 2;
            if (base[mid * cols + col] <= i) lo = mid;
            else hi = mid - 1;
        }
        uint32_t b = ballot[lo * cols + col];
        for (int k = i - base[lo * cols + col]; k > 0; --k) b &= b - 1;
        return lo * 32 + __ffs(b) - 1;
    }
    // Token t's first row in node n's combine block (by token, then destination).
    __device__ int rowoff(const Rank& r, int t, int n) const {
        const int g = t / 32 * cols;
        int off = base[g + r.ranks + r.nodes + n];
        for (int j = 0; j < r.lsa; ++j) off += __popc(ballot[g + r.at(n, j)] & ((1u << t % 32) - 1));
        return off;
    }
    // The GPU's (local rank j of node n) records before chunk c of `per` records of node n's bundle.
    __device__ int first(const Rank& r, int n, int j, int c, int per) const {
        return c * per >= count[r.ranks + n] ? count[r.at(n, j)] : pos(token(r, n, c * per), r.at(n, j));
    }

    __device__ void build(const Rank& r, State& st) {
        const Args& a = r.a;
        const int rows = r.ranks + r.nodes;
        const uint32_t lsa_mask = (1u << r.lsa) - 1;
        for (int c = r.gw; c < groups; c += r.gws) {
            const int t = c * 32 + r.lane;
            int64_t ids[kMaxTopk];
            if (a.k == kMaxTopk && t < a.n) {  // the row in four loads at once
                const auto* row = reinterpret_cast<const longlong2*>(a.topk_idx + size_t(t) * kMaxTopk);
#pragma unroll
                for (int u = 0; u < kMaxTopk / 2; ++u) {
                    const longlong2 v = row[u];
                    ids[2 * u] = v.x, ids[2 * u + 1] = v.y;
                }
            } else {
#pragma unroll
                for (int i = 0; i < kMaxTopk; ++i) ids[i] = t < a.n && i < a.k ? a.topk_idx[t * a.k + i] : -1;
            }
            uint32_t m = 0;
#pragma unroll
            for (int i = 0; i < kMaxTopk; ++i)
                if (ids[i] >= 0) m |= 1u << (int(ids[i]) / a.experts_per_rank);
            if (t < a.t) st.mask[t] = m;
            const uint32_t by_gpu = transpose(m, r.lane);
            if (r.lane < r.ranks) st.ballot[c * cols + r.lane] = by_gpu;
            for (int n = 0; n < r.nodes; ++n) {
                const uint32_t dm = m >> (n * r.lsa) & lsa_mask, b = __ballot_sync(~0u, dm != 0);
                const int sum = __reduce_add_sync(~0u, __popc(dm));
                if (r.lane == n) st.ballot[c * cols + r.ranks + n] = b, st.ballot[c * cols + rows + n] = sum;
            }
        }
        // once every worker's share is in
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence();
            atomicAdd(&st.routed, 1);
            while (*reinterpret_cast<volatile unsigned*>(&st.routed) < st.routed_base + r.workers) {}
            __threadfence();
        }
        __syncthreads();
        for (int i = threadIdx.x; i < a.t; i += kThreads) mask[i] = __ldcg(&st.mask[i]);
        for (int i = threadIdx.x; i < groups * cols; i += kThreads) ballot[i] = __ldcg(&st.ballot[i]);
        __syncthreads();
        for (int d = r.warp; d < cols; d += kWarps) {
            int run = 0;
            for (int c0 = 0; c0 < groups; c0 += 32) {
                const int c = c0 + r.lane;
                const int v = c >= groups ? 0 : d < rows ? __popc(ballot[c * cols + d]) : int(ballot[c * cols + d]);
                const int incl = inclusive_scan(v, r.lane);
                if (c < groups) base[c * cols + d] = run + incl - v;
                run += __shfl_sync(~0u, incl, 31);
            }
            if (r.lane == 0) count[d] = run;
        }
        __syncthreads();
    }
};

// The relay (one warp): flags every head and chunk landing here from a remote node's rail peer in every node-mate's
// window. Lane i serves items i, i + 32, ...; item k * (kMaxChunks + 1) + c is chunk c of remote node k's bundle
// (c = kMaxChunks: its head).
__device__ void relay(const Rank& r, unsigned epoch) {
    const Args& a = r.a;
    State& st = *a.state;
    constexpr int kSpan = kMaxChunks + 1;
    const int items = (r.nodes - 1) * kSpan;
    uint32_t todo = 0;
    for (int i = 0; r.lane + 32 * i < items; ++i) todo |= 1u << i;
    const auto chunks = [&](int m) { return reinterpret_cast<const Head*>(r.local(r.fwd(m)))->chunks; };
    for (uint32_t pending = todo; pending != 0;) {
        for (uint32_t bits = pending; bits != 0; bits &= bits - 1) {
            const int i = __ffs(bits) - 1, item = r.lane + 32 * i, c = item % kSpan, m = r.remote(item / kSpan);
            if (c == kMaxChunks) {
                const int ctx = r.context(m, r.node, c);
                if (r.signal(kHeadFrom + m, ctx) < st.signal[ctx][kHeadFrom + m] + 1) continue;
                r.flag_node(r.l, m, c, epoch);
                r.mark_any(kDHeadsRelayed);
            } else {
                if (ld_acquire32(r.flag(r.l, m, kMaxChunks, r.me)) != epoch) continue;
                const int id = kChunkFrom + m * kMaxChunks + c, ctx = r.context(m, r.node, c);
                if (c < chunks(m)) {
                    if (r.signal(id, ctx) < st.signal[ctx][id] + 1) continue;
                    r.flag_node(r.l, m, c, epoch);
                    r.mark_any(kDLastRelay);
                }
            }
            pending &= ~(1u << i);
        }
    }
    for (uint32_t bits = todo; bits != 0; bits &= bits - 1) {
        const int item = r.lane + 32 * (__ffs(bits) - 1), c = item % kSpan, m = r.remote(item / kSpan);
        if (c == kMaxChunks)
            st.signal[r.context(m, r.node, c)][kHeadFrom + m] += 1;
        else if (c < chunks(m))
            st.signal[r.context(m, r.node, c)][kChunkFrom + m * kMaxChunks + c] += 1;
    }
    __syncwarp();
}

__global__ void __launch_bounds__(kThreads, 1) dispatch_kernel(const Args a) {
    const Rank r(a);
    State& st = *a.state;
    const unsigned epoch = unsigned(st.calls + 1);
    const auto advance = [&] { st.calls += 1, st.routed_base += r.workers; };
    r.mark_start(kDStart);
    if (blockIdx.x == a.ctas - 1) {
        if (r.warp == 0) {
            relay(r, epoch);
            if (r.lane == 0) r.leave(advance);
        }
        return;
    }

    extern __shared__ __align__(16) char smem[];
    __shared__ Route route;
    __shared__ int nbase[kMaxNodes + 1], warp_sums[kWarps], max_chunks;
    __shared__ int chunk_first[kMaxLsa], chunk_list[kMaxLsa + 1];
    __shared__ uint16_t slice_token[kSliceRecords], slice_row[kSliceRecords];
    __shared__ int slices_before[kMaxChunks + 1];
    if (threadIdx.x == 0) {
        const size_t groups = (a.t + 31) / 32, cols = r.ranks + 2 * r.nodes, ballots = align16(4 * groups * cols);
        route.cols = int(cols), route.groups = int(groups);
        route.mask = reinterpret_cast<uint32_t*>(smem);
        route.ballot = reinterpret_cast<uint32_t*>(smem + align16(4 * size_t(a.t)));
        route.base = reinterpret_cast<int*>(smem + align16(4 * size_t(a.t)) + ballots);
    }
    __syncthreads();
    Pull& pull = *reinterpret_cast<Pull*>(smem + route_bytes(a.t, r.ranks, r.nodes));
    char* stage = smem + route_bytes(a.t, r.ranks, r.nodes) + align16(sizeof(Pull));
    route.build(r, st);
    // each node's block of this rank's combine rows: by node; with reduce this node's rows first, then a block of
    // t partials (one per token sent there) per node from k * t rows on
    if (threadIdx.x == 0) {
        int run = 0;
        for (int n = 0; n < r.nodes; ++n) {
            nbase[n] = a.reduce ? (n == r.node ? 0 : a.k * a.t + n * a.t) : run;
            run += route.count[r.ranks + r.nodes + n];
        }
        nbase[r.nodes] = run;
    }
    __syncthreads();
    r.mark(kDRouted);
    if (blockIdx.x == 0 && threadIdx.x < kMaxNodes) {
        const int n = threadIdx.x;
        a.node_rows[n] = n >= r.nodes ? 0 : route.count[(a.reduce && n != r.node ? r.ranks : r.ranks + r.nodes) + n];
    }

    // Bundles: heads (one task per node), then chunk slices, chunk by chunk, remote nodes first; task i by worker
    // i % workers. A chunk of rc records is spc slices of sr.
    const int rc = max(1, max(int((a.chunk + r.rb - 1) / r.rb), (a.t + kMaxChunks - 1) / kMaxChunks));
    const int spc = (rc + min(min(rc, kSliceRecords), max(kWarps, int(kSliceBytes / r.rb))) - 1) /
                    min(min(rc, kSliceRecords), max(kWarps, int(kSliceBytes / r.rb)));
    const int sr = (rc + spc - 1) / spc;
    const auto node_of = [&](int k) { return (r.node + 1 + k) % r.nodes; };  // k = nodes - 1: this node
    const auto records = [&](int n) { return route.count[r.ranks + n]; };
    const auto slices_in = [&](int c, int n) {
        return c * rc >= records(n) ? 0 : (min(records(n), (c + 1) * rc) - c * rc + sr - 1) / sr;
    };
    if (threadIdx.x <= kMaxChunks) {
        int sum = 0;
        for (int c = 0; c < threadIdx.x; ++c)
            for (int n = 0; n < r.nodes; ++n) sum += slices_in(c, n);
        slices_before[threadIdx.x] = sum;
    }
    __syncthreads();
    const int tasks = r.nodes + slices_before[kMaxChunks];
    for (int task = blockIdx.x; task < tasks; task += r.workers) {
        if (task < r.nodes) {
            const int n = node_of(task), total = records(n), chunks = (total + rc - 1) / rc;
            Head* head = reinterpret_cast<Head*>(r.local(r.bundle(n)));
            if (threadIdx.x < r.lsa) head->count[threadIdx.x] = route.count[r.at(n, threadIdx.x)];
            if (threadIdx.x == 0)
                head->total = total, head->nbase = nbase[n], head->chunks = chunks, head->per_chunk = rc,
                head->row_bytes = int(r.rowb);
            for (int e = threadIdx.x; e < r.lsa * (chunks + 1); e += kThreads) {
                const int j = e / (chunks + 1), c = e % (chunks + 1);
                head->first[j][c] = route.first(r, n, j, c, rc);
            }
            __syncthreads();
            for (int c = threadIdx.x; c < chunks; c += kThreads) {
                int run = 0;
                for (int j = 0; j < r.lsa; ++j)
                    head->list[j][c] = run, run += head->first[j][c + 1] - head->first[j][c];
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                if (n != r.node)
                    r.put(r.at(n, r.l), r.fwd(r.node), r.bundle(n), sizeof(Head), kHeadFrom + r.node, 1,
                          r.context(r.node, n, kMaxChunks));
                else
                    r.flag_node(r.l, r.node, kMaxChunks, epoch);
                r.mark_any(kDHeadsSent);
            }
            continue;
        }
        int c = 0, k = 0, x = task - r.nodes;
        while (slices_before[c + 1] <= x) ++c;
        for (x -= slices_before[c]; x >= slices_in(c, node_of(k)); ++k) x -= slices_in(c, node_of(k));
        const int n = node_of(k), total = records(n), end_c = min(total, (c + 1) * rc);
        const int begin = c * rc + x * sr, end = min(end_c, begin + sr), q = (end - begin + kWarps - 1) / kWarps;
        char* chunk = r.local(r.bundle(n) + r.chunk_at(c, rc));
        if (r.warp == 1) {  // each GPU's records before and in the chunk, and where its list starts
            const int j = r.lane, f = j < r.lsa ? route.first(r, n, j, c, rc) : 0;
            const int in = j < r.lsa ? route.first(r, n, j, c + 1, rc) - f : 0, incl = inclusive_scan(in, r.lane);
            if (j < r.lsa) chunk_first[j] = f, chunk_list[j] = incl - in;
            if (j == r.lsa - 1) chunk_list[r.lsa] = incl;
        }
        // the records' tokens and first combine rows
        for (int i = threadIdx.x; i < end - begin; i += kThreads) {
            const int t = route.token(r, n, begin + i);
            slice_token[i] = uint16_t(t), slice_row[i] = uint16_t(route.rowoff(r, t, n));
        }
        __syncthreads();
        const int i0 = begin + r.warp * q, rows = max(0, min(q, end - i0)), words = a.sbytes / 4;
        const int fields = words + 2 + 2 * a.k;  // scales, then the metadata
        const char* x_rows = reinterpret_cast<const char*>(a.x);
        r.copy_rows(rows, a.vbytes / 16,
                    [&](int i, int g) { return x_rows + size_t(slice_token[i0 - begin + i]) * a.vbytes + g * 16; },
                    [&](int i, int g) { return chunk + size_t(i0 - c * rc + i) * r.rb + g * 16; });
        // the scales and metadata by the whole CTA, four loads in flight per thread
        for (int e0 = threadIdx.x; e0 < (end - begin) * fields; e0 += 4 * kThreads) {
            int v[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int e = e0 + u * kThreads, i = e / fields, f = e % fields;
                const int t = e < (end - begin) * fields ? slice_token[i] : 0;
                const int m = f - words;
                v[u] = e >= (end - begin) * fields ? 0
                       : f < words                 ? __float_as_int(a.xs[t * words + f])
                       : m == 0                    ? t
                       : m <= a.k                  ? int(a.topk_idx[t * a.k + m - 1])
                       : m <= 2 * a.k              ? __float_as_int(a.topk_weights[t * a.k + m - 1 - a.k])
                                                   : int(slice_row[i]);
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int e = e0 + u * kThreads, i = e / fields, f = e % fields;
                char* record = chunk + size_t(begin - c * rc + i) * r.rb;
                if (e < (end - begin) * fields)
                    *reinterpret_cast<int*>(f < words ? record + a.vbytes + 4 * f : record + r.rowb + 4 * (f - words)) =
                        v[u];
            }
        }
        // each GPU's list entries for these records
        __syncthreads();
        const int entries = chunk_list[r.lsa];
        uint16_t* list = reinterpret_cast<uint16_t*>(chunk + size_t(end_c - c * rc) * r.rb);
        for (int i = begin + threadIdx.x; i < end; i += kThreads) {
            const int t = slice_token[i - begin];
            for (uint32_t b = route.mask[t] >> (n * r.lsa) & ((1u << r.lsa) - 1); b != 0; b &= b - 1) {
                const int j = __ffs(b) - 1;
                list[chunk_list[j] + route.pos(t, r.at(n, j)) - chunk_first[j]] = uint16_t(i);
            }
        }
        __syncthreads();
        // the last slice's CTA sends the chunk (put and flag_node release at system scope for every slice's writes)
        if (threadIdx.x == 0) {
            __threadfence();
            const unsigned slices = (end_c - c * rc + sr - 1) / sr;
            if (atomicAdd(&st.packed[n][c], 1) == slices - 1) {
                st.packed[n][c] = 0;
                __threadfence();
                if (n != r.node)
                    r.put(r.at(n, r.l), r.fwd(r.node) + r.chunk_at(c, rc), r.bundle(n) + r.chunk_at(c, rc),
                          align16(size_t(end_c - c * rc) * r.rb + 2 * size_t(entries)),
                          kChunkFrom + r.node * kMaxChunks + c, 1, r.context(r.node, n, c));
                else
                    r.flag_node(r.l, r.node, c, epoch);
                r.mark_any(kDPacked);
            }
        }
    }
    // each token's combine rows: its node blocks by token, then destination
    for (int t = blockIdx.x * kThreads + threadIdx.x; t < a.n; t += r.workers * kThreads) {
        const uint32_t m = route.mask[t];
        int i = 0;
        const auto before = [&](int d, int n) {  // destinations there below d
            return __popc(m >> (n * r.lsa) & ((1u << (d % r.lsa)) - 1));
        };
        if (a.reduce) {
            // DeepEP's order: ptx::deduplicate keeps a key's last top-k lane, then lanes go up. Nodes by that, and this
            // node's ranks likewise. A remote node's partial once; this node's rows next to each other.
            const auto destination = [&](int l) {
                const int id = l < a.k ? int(a.topk_idx[t * a.k + l]) : -1;
                return id < 0 ? -1 : id / a.experts_per_rank;
            };
            for (int l = 0; l < a.k; ++l) {
                const int d = destination(l);
                if (d < 0) continue;
                const int n = d / r.lsa;
                bool later = false;
                for (int o = l + 1; o < a.k; ++o) later |= destination(o) >= 0 && destination(o) / r.lsa == n;
                if (later) continue;
                if (n != r.node) {
                    a.comb_rows[t * a.k + i++] = nbase[n] + route.pos(t, r.ranks + n);
                    continue;
                }
                for (int o = 0; o < a.k; ++o) {
                    const int e = destination(o);
                    if (e < 0 || e / r.lsa != n) continue;
                    bool again = false;
                    for (int q = o + 1; q < a.k; ++q) again |= destination(q) == e;
                    if (!again) a.comb_rows[t * a.k + i++] = nbase[n] + route.rowoff(r, t, n) + before(e, n);
                }
            }
        } else {
            for (uint32_t b = m; b != 0; b &= b - 1, ++i) {
                const int d = __ffs(b) - 1, n = d / r.lsa;
                a.comb_rows[t * a.k + i] = nbase[n] + route.rowoff(r, t, n) + before(d, n);
            }
        }
        for (; i < a.k; ++i) a.comb_rows[t * a.k + i] = -1;
    }

    // Every source's head (from the source on this node, else from its rail peer here).
    if (threadIdx.x < r.ranks) {
        const int s = threadIdx.x;
        while (ld_acquire32(r.flag(s % r.lsa, s / r.lsa, kMaxChunks, r.me)) != epoch) {}
    }
    __syncthreads();
    r.mark(kDSeen);
    if (threadIdx.x < r.ranks) {
        const int s = threadIdx.x, n = s / r.lsa;
        const char* src = r.peer(n == r.node ? r.bundle(r.node) : r.fwd(n), r.at(r.node, s % r.lsa));
        const Head* h = reinterpret_cast<const Head*>(src);
        int before = 0, rows = 0;
        for (int j = 0; j < r.lsa; ++j) before += j < r.l ? h->count[j] : 0, rows += h->count[j];
        pull.src[s] = src, pull.count[s] = h->count[r.l], pull.chunks[s] = h->chunks, pull.total[s] = h->total;
        pull.per_chunk[s] = h->per_chunk;
        if (blockIdx.x == 0) {
            int* head = reinterpret_cast<int*>(r.local(a.at.heads + 16 * s));
            head[0] = h->count[r.l], head[1] = n == r.node ? h->nbase : 0, head[2] = rows;
        }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < r.ranks * (kMaxChunks + 1); e += kThreads) {
        const int s = e / (kMaxChunks + 1), c = e % (kMaxChunks + 1);
        const Head* h = reinterpret_cast<const Head*>(pull.src[s]);
        if (c <= pull.chunks[s]) pull.first[s][c] = h->first[r.l][c];
        if (c < pull.chunks[s]) pull.list[s][c] = h->list[r.l][c];
    }
    if (r.warp == 0) {
        const int count = r.lane < r.ranks ? pull.count[r.lane] : 0;
        const int incl = inclusive_scan(count, r.lane);
        if (r.lane < r.ranks) pull.prev[r.lane] = incl - count;
        if (blockIdx.x == 0 && r.lane < r.ranks) a.psum[r.lane] = incl;
        const int most = __reduce_max_sync(~0u, r.lane < r.ranks ? pull.chunks[r.lane] : 0);
        if (r.lane == 0) max_chunks = most;
    }
    __syncthreads();

    // Work items of up to ir rows per (source, chunk) group, in pull order: chunk by chunk, this node's sources by
    // local rank, then the remote nodes' in the order their bundles for here go out.
    const int maxc = max_chunks, groups = r.ranks * maxc;
    const int ir = max(1, min(kRows, int(kItemBytes / r.rb)));
    const auto group = [&](int g, int& s, int& c) {
        const int i = g % r.ranks;
        c = g / r.ranks, s = r.at((r.node - i / r.lsa + r.nodes) % r.nodes, i % r.lsa);
    };
    {
        const int per = (groups + kThreads - 1) / kThreads;  // at most 4
        int items[4], sum = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int g = threadIdx.x * per + i;
            items[i] = 0;
            if (i < per && g < groups) {
                int s, c;
                group(g, s, c);
                if (c < pull.chunks[s]) items[i] = (pull.first[s][c + 1] - pull.first[s][c] + ir - 1) / ir;
            }
            sum += items[i];
        }
        const int incl = inclusive_scan(sum, r.lane);
        if (r.lane == 31) warp_sums[r.warp] = incl;
        __syncthreads();
        if (r.warp == 0) {
            const int v = r.lane < kWarps ? warp_sums[r.lane] : 0, vi = inclusive_scan(v, r.lane);
            if (r.lane < kWarps) warp_sums[r.lane] = vi - v;
        }
        __syncthreads();
        int run = warp_sums[r.warp] + incl - sum;
        for (int i = 0; i < per; ++i) {
            const int g = threadIdx.x * per + i;
            if (g < groups) pull.items[g] = run;
            run += items[i];
        }
        if (threadIdx.x == kThreads - 1) pull.items[groups] = run;
        __syncthreads();
    }

    // Each warp pulls its items' rows: records from the source's bundle, rows into recv_x, metadata via smem.
    // values into recv_x, the scales (padded to 16 bytes) and metadata into this warp's stage
    const int wsegs = a.vbytes / 16, sb = int(align16(a.sbytes)), mb = meta_bytes(a.k), words = a.sbytes / 4;
    char* meta = stage + r.warp * kRows * (sb + mb);
    int* roffs = reinterpret_cast<int*>(r.local(a.at.roffs));
    for (int item = r.gw; item < pull.items[groups]; item += r.gws) {
        int lo = 0, hi = groups;
        while (hi - lo > 1) {
            const int mid = (lo + hi) / 2;
            if (pull.items[mid] <= item) lo = mid;
            else hi = mid;
        }
        int s, c;
        group(lo, s, c);
        const int p0 = pull.first[s][c] + (item - pull.items[lo]) * ir, rows = min(ir, pull.first[s][c + 1] - p0);
        const int per = pull.per_chunk[s], records = min(per, pull.total[s] - c * per);
        if (r.lane == 0)
            while (ld_acquire32(r.flag(s % r.lsa, s / r.lsa, c, r.me)) != epoch) {}
        __syncwarp();
        const char* chunk = pull.src[s] + r.chunk_at(c, per);
        const auto* list = reinterpret_cast<const uint16_t*>(chunk + size_t(records) * r.rb) + pull.list[s][c];
        if (r.lane < rows) pull.recs[r.warp][r.lane] = list[p0 - pull.first[s][c] + r.lane] - c * per;
        __syncwarp();
        const size_t q0 = pull.prev[s] + p0;
        char* rx = reinterpret_cast<char*>(a.recv_x);
        r.copy_rows(rows, wsegs + (sb + mb) / 16,
                    [&](int i, int g) { return chunk + size_t(pull.recs[r.warp][i]) * r.rb + g * 16; },
                    [&](int i, int g) {
                        return g < wsegs ? rx + (q0 + i) * a.vbytes + g * 16 : meta + i * (sb + mb) + (g - wsegs) * 16;
                    });
        __syncwarp();
        for (int e = r.lane; e < rows * words; e += 32)
            a.recv_xs[(q0 + e / words) * words + e % words] =
                reinterpret_cast<const float*>(meta + e / words * (sb + mb))[e % words];
        if (r.lane < rows) {
            const int* m = reinterpret_cast<const int*>(meta + r.lane * (sb + mb) + sb);
            const size_t q = q0 + r.lane;
            uint32_t mates = 0;  // the token's destinations on this node
            for (int i = 0; i < a.k; ++i) {
                const int id = m[1 + i], d = id / a.experts_per_rank, e = id - r.me * a.experts_per_rank;
                if (id >= 0 && d / r.lsa == r.node) mates |= 1u << (d % r.lsa);
                a.recv_idx[q * a.k + i] = id >= 0 && e >= 0 && e < a.experts_per_rank ? e : -1;
                a.recv_weights[q * a.k + i] = __int_as_float(m[1 + a.k + i]);
            }
            a.recv_src[q] = s * a.t + m[0];
            roffs[q] = m[1 + 2 * a.k] + __popc(mates & ((1u << r.l) - 1));
        }
        __syncwarp();
    }
    r.finish(advance, kDEnd);
}

// The sender (one thread): each remote node k's source block to its source over context k % contexts, in order,
// piece by piece as they fill, the last once every node-mate CTA is done.
__device__ void send(const Rank& r) {
    const Args& a = r.a;
    State& st = *a.state;
    // the block's rows, or with reduce its partials (one per record)
    int total[kMaxNodes], per[kMaxNodes], pieces[kMaxNodes], nbase[kMaxNodes], sent[kMaxNodes], ready[kMaxNodes];
    int rows[kMaxNodes], row_per[kMaxNodes];
    for (int k = 0; k < r.nodes - 1; ++k) {
        const Head* h = reinterpret_cast<const Head*>(r.local(r.fwd(r.remote(k))));
        rows[k] = 0;
        for (int j = 0; j < r.lsa; ++j) rows[k] += h->count[j];
        row_per[k] = r.piece_rows(rows[k]);
        total[k] = a.reduce ? h->total : rows[k];
        per[k] = a.reduce ? r.partial_rows(total[k]) : r.piece_rows(total[k]);
        pieces[k] = (total[k] + per[k] - 1) / per[k];
        nbase[k] = h->nbase, sent[k] = 0, ready[k] = 0;
    }
    const auto rows_of = [&](int k, int p) { return min(per[k], total[k] - p * per[k]); };
    bool first = true;
    for (unsigned todo = (1u << (r.nodes - 1)) - 1; todo != 0;) {
        for (int k = 0; k < r.nodes - 1; ++k) {
            if (!(todo >> k & 1)) continue;
            for (; ready[k] < pieces[k]; ++ready[k]) {
                const int id = kPiece + k * kMaxChunks + ready[k];
                const uint64_t done = a.reduce ? *reinterpret_cast<volatile unsigned*>(&st.reduced[k][ready[k]])
                                               : ld_acquire(r.counter(id, r.me)) - st.counter[id];
                if (done < uint64_t(rows_of(k, ready[k]))) break;
            }
            if (a.reduce) __threadfence();
            const bool last = ready[k] == pieces[k] && ld_acquire(r.counter(kGatherDone + k, r.me)) >=
                                                           st.counter[kGatherDone + k] + uint64_t(r.lsa) * r.workers;
            if (!last && ready[k] == sent[k]) continue;
            const int begin = min(total[k], sent[k] * per[k]), rows = min(total[k], ready[k] * per[k]) - begin;
            if (first) r.mark(kCFirstPut), first = false;
            r.put(r.at(r.remote(k), r.l), a.at.comb + size_t(nbase[k]) * r.yb + size_t(begin) * r.ps,
                  (a.reduce ? r.partial(k) : r.gather(k)) + size_t(begin) * r.ps, size_t(rows) * r.ps,
                  kCombFrom + r.node, rows + (last ? 1 : 0), k % a.contexts);
            sent[k] = ready[k];
            if (last) {
                todo &= ~(1u << k);
                r.mark(kCPut + k);
            }
        }
    }
    for (int k = 0; k < r.nodes - 1; ++k) {
        for (int p = 0; p * row_per[k] < rows[k]; ++p)
            st.counter[kPiece + k * kMaxChunks + p] += min(row_per[k], rows[k] - p * row_per[k]);
        for (int p = 0; a.reduce && p < pieces[k]; ++p) st.reduced[k][p] = 0;
        st.counter[kGatherDone + k] += uint64_t(r.lsa) * r.workers;
    }
}

__global__ void __launch_bounds__(kThreads, 1) combine_kernel(const Args a) {
    const Rank r(a);
    State& st = *a.state;
    // node n's aggregator sends here over the context of this node's index among its remote nodes
    const auto context = [&](int n) { return (r.node - n - 1 + r.nodes) % r.nodes % a.contexts; };
    const auto advance = [&] {
        st.claimed = 0;
        st.counter[kCombArrive] += uint64_t(r.lsa) * r.workers;
        for (int k = 0; k < r.nodes - 1; ++k)
            for (int c = 0; c < a.contexts; ++c)
                st.signal[c][kCombFrom + r.remote(k)] = r.signal(kCombFrom + r.remote(k), c);
    };
    r.mark_start(kCStart);
    if (blockIdx.x == a.ctas - 1) {
        if (threadIdx.x == 0) {
            send(r);
            r.leave(advance);
        }
        return;
    }

    // Each received row to its source's block: at the source (on this node) or at its aggregator. Remote sources
    // first, ci rows of each in turn, in the order the aggregators send; then this node's.
    __shared__ int prev[kMaxRanks], count[kMaxRanks], base[kMaxRanks], per[kMaxRanks], rounds, local_items;
    __shared__ int offs[kWarps][kRows], landed[kWarps][kMaxNodes], filled[kWarps][kMaxNodes], nbase[kMaxNodes + 1];
    __shared__ int batch_rows[kWarps][kBatch + 1];
    const int ci = max(1, min(kRows, int(kWriteBytes / max(r.yb, size_t(16)))));  // rows per item
    if (threadIdx.x < 32) {
        const int s = r.lane;
        const int* head = reinterpret_cast<const int*>(r.local(a.at.heads + 16 * s));
        if (s < r.ranks)
            count[s] = head[0], base[s] = head[1], per[s] = r.piece_rows(head[2]), prev[s] = a.psum[s] - head[0];
        __syncwarp();
        const int remote = r.lane < (r.nodes - 1) * r.lsa ? r.at(r.remote(r.lane / r.lsa), r.lane % r.lsa) : -1;
        const int mates = r.lane < r.lsa ? (count[r.at(r.node, r.lane)] + ci - 1) / ci : 0;
        const int most = __reduce_max_sync(~0u, remote >= 0 ? (count[remote] + ci - 1) / ci : 0);
        const int sum = __reduce_add_sync(~0u, mates);
        const int rows = r.lane < r.nodes ? a.node_rows[r.lane] : 0, incl = inclusive_scan(rows, r.lane);
        if (r.lane <= r.nodes) nbase[r.lane] = incl - rows;
        if (r.lane == 0) rounds = most, local_items = sum;
    }
    if (threadIdx.x < kWarps * kMaxNodes) {
        const int w = threadIdx.x / kMaxNodes, n = threadIdx.x % kMaxNodes;
        landed[w][n] = filled[w][n] = 0;
    }
    __syncthreads();
    const int* roffs = reinterpret_cast<const int*>(r.local(a.at.roffs));
    const char* y = reinterpret_cast<const char*>(a.y);
    const int remotes = (r.nodes - 1) * r.lsa;
    // With reduce, the last kReduceWarps warps of each CTA reduce (below) while the others write.
    const int writers = a.reduce ? kWarps - kReduceWarps : kWarps;
    const auto writers_sync = [&] { asm volatile("bar.sync 1, %0;" ::"r"(writers * 32) : "memory"); };
    if (r.warp < writers) {
        const int wgw = blockIdx.x * writers + r.warp, wgws = r.workers * writers;
        for (int item = wgw; item < rounds * remotes; item += wgws) {
            const int i = item % remotes, k = i / r.lsa, s = r.at(r.remote(k), i % r.lsa), i0 = item / remotes * ci;
            if (i0 >= count[s]) continue;
            const int rows = min(ci, count[s] - i0), g = r.at(r.node, s % r.lsa);
            const char* src = y + size_t(prev[s] + i0) * r.yb;
            char* dst = r.peer(r.gather(k), g);
            if (r.lane < rows) offs[r.warp][r.lane] = roffs[prev[s] + i0 + r.lane];
            __syncwarp();
            r.copy_rows(rows, a.width / 8, [&](int i, int g) { return src + i * r.yb + g * 16; },
                        [&](int i, int g) { return dst + size_t(offs[r.warp][i]) * r.yb + g * 16; });
            __syncwarp();
            // each run of rows in one piece: its first lane adds them to the piece's counter
            const int piece = r.lane < rows ? offs[r.warp][r.lane] / per[s] : -1;
            const int before = __shfl_up_sync(~0u, piece, 1);
            const uint32_t starts = __ballot_sync(~0u, r.lane < rows && (r.lane == 0 || before != piece));
            if (starts >> r.lane & 1) {
                const uint32_t later = starts & ~((2u << r.lane) - 1);
                const int run = (later != 0 ? __ffs(later) - 1 : rows) - r.lane;
                fence_sys();
                add_relaxed(r.counter(kPiece + k * kMaxChunks + piece, g), run);
            }
            __syncwarp();
        }
        writers_sync();
        if (threadIdx.x == 0) {
            fence_sys();
            for (int j = 0; j < r.lsa; ++j)
                for (int k = 0; k < r.nodes - 1; ++k) add_relaxed(r.counter(kGatherDone + k, r.at(r.node, j)), 1);
        }
        for (int item = wgw; item < local_items; item += wgws) {
            int j = 0, first = 0;
            const auto items = [&](int j) { return (count[r.at(r.node, j)] + ci - 1) / ci; };
            for (; first + items(j) <= item; ++j) first += items(j);
            const int s = r.at(r.node, j), i0 = (item - first) * ci, rows = min(ci, count[s] - i0);
            const char* src = y + size_t(prev[s] + i0) * r.yb;
            char* dst = r.peer(a.at.comb + size_t(base[s]) * r.yb, s);
            if (r.lane < rows) offs[r.warp][r.lane] = roffs[prev[s] + i0 + r.lane];
            __syncwarp();
            r.copy_rows(rows, a.width / 8, [&](int i, int g) { return src + i * r.yb + g * 16; },
                        [&](int i, int g) { return dst + size_t(offs[r.warp][i]) * r.yb + g * 16; });
            __syncwarp();
        }
        writers_sync();
        if (threadIdx.x == 0) {
            fence_sys();
            for (int j = 0; j < r.lsa; ++j) add_relaxed(r.counter(kCombArrive, r.at(r.node, j)), 1);
        }
        r.mark(kCWritten);
    }
    if (r.warp >= writers || a.reduce) {
        // With reduce: the partial of each record of each remote source's block here, by batches of kBatch
        // records claimed in order, each once its rows are in (in piece order: a warp keeps how many of a block's
        // pieces it has seen filled).
        const int batches = (a.t + kBatch - 1) / kBatch;
        for (;;) {
            int item = 0;
            if (r.lane == 0) item = int(atomicAdd(&st.claimed, 1));
            item = __shfl_sync(~0u, item, 0);
            if (item >= (r.nodes - 1) * batches) break;
            const int k = item % (r.nodes - 1), i0 = item / (r.nodes - 1) * kBatch;
            const char* fwd = r.local(r.fwd(r.remote(k)));
            const Head* h = reinterpret_cast<const Head*>(fwd);
            if (i0 >= h->total) continue;
            int rows = 0;
            for (int j = 0; j < r.lsa; ++j) rows += h->count[j];
            const int records = min(kBatch, h->total - i0), per = r.piece_rows(rows);
            const size_t rb = size_t(h->row_bytes) + meta_bytes(a.k);  // the dispatch's records
            if (r.lane <= records) {  // each record's first row in the block, then the next's
                const int i = r.lane;
                const int e = i0 + i, c = e / h->per_chunk;
                const char* rec = fwd + align16(sizeof(Head)) + size_t(c) * h->per_chunk * (rb + 2 * r.lsa) +
                                  (e - c * h->per_chunk) * rb;
                batch_rows[r.warp][i] =
                    e < h->total ? reinterpret_cast<const int*>(rec + h->row_bytes)[1 + 2 * a.k] : rows;
            }
            __syncwarp();
            if (r.lane == 0)
                for (int p = filled[r.warp][k]; p <= (batch_rows[r.warp][records] - 1) / per; filled[r.warp][k] = ++p) {
                    const int id = kPiece + k * kMaxChunks + p;
                    wait_ge(r.counter(id, r.me), st.counter[id] + min(per, rows - p * per));
                }
            __syncwarp();
            const char* block = r.local(r.gather(k));
            char* out = r.local(r.partial(k) + size_t(i0) * r.ps);
            // record by record: lanes over the row's 16-byte segments, four each per pass, all of a row in flight
            const int segs = int(r.yb / 16);
            for (int i = 0; i < records; ++i) {
                const int first = batch_rows[r.warp][i], end = batch_rows[r.warp][i + 1];
                // this record's rows in DeepEP's order: each rank's last top-k lane, lanes going up
                const int e = i0 + i, c = e / h->per_chunk;
                const int* meta = reinterpret_cast<const int*>(fwd + align16(sizeof(Head)) +
                                                               size_t(c) * h->per_chunk * (rb + 2 * r.lsa) +
                                                               (e - c * h->per_chunk) * rb + h->row_bytes);
                uint32_t here = 0;
                int order[kMaxTopk], parts = 0;
                const auto rank_here = [&](int l) {
                    const int id = meta[1 + l];
                    if (id < 0) return -1;
                    const int d = id / a.experts_per_rank;
                    return d / r.lsa == r.node ? d % r.lsa : -1;
                };
#pragma unroll
                for (int l = 0; l < kMaxTopk; ++l) {
                    const int j = l < a.k ? rank_here(l) : -1;
                    if (j >= 0) here |= 1u << j;
                }
                for (int l = 0; l < a.k; ++l) {
                    const int j = rank_here(l);
                    if (j < 0) continue;
                    bool later = false;
                    for (int o = l + 1; o < a.k; ++o) later |= rank_here(o) == j;
                    if (!later) order[parts++] = __popc(here & ((1u << j) - 1));
                }
                const auto read = [&](int p, int g) {
                    return *reinterpret_cast<const int4*>(block + (first + order[p]) * r.yb + g * 16);
                };
                for (int g0 = 0; g0 < segs; g0 += 4 * 32) {
                    int4 part[4] = {};
                    if (parts <= 2) {
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const int g = g0 + u * 32 + r.lane;
                            if (g < segs && parts > 0)
                                part[u] = add_bf16(read(0, g), parts > 1 ? read(1, g) : int4{0, 0, 0, 0});
                        }
                    } else {
                        float acc[4][8] = {};
                        for (int p = 0; p < parts; ++p) {
                            int4 v[4];
#pragma unroll
                            for (int u = 0; u < 4; ++u) {
                                const int g = g0 + u * 32 + r.lane;
                                if (g < segs) v[u] = read(p, g);
                            }
#pragma unroll
                            for (int u = 0; u < 4; ++u)
                                if (g0 + u * 32 + r.lane < segs) accumulate(acc[u], v[u]);
                        }
#pragma unroll
                        for (int u = 0; u < 4; ++u) part[u] = to_bf16(acc[u]);
                    }
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        const int g = g0 + u * 32 + r.lane;
                        if (a.fp8)  // half a warp per 128 values (16 segments)
                            quantize(out + i * r.ps, g, a.width, part[u], g < segs);
                        else if (g < segs)
                            *reinterpret_cast<int4*>(out + i * r.yb + g * 16) = part[u];
                    }
                }
            }
            __syncwarp();
            if (r.lane == 0) {
                __threadfence();
                atomicAdd(&st.reduced[k][i0 / r.partial_rows(h->total)], records);
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) wait_ge(r.counter(kCombArrive, r.me), st.counter[kCombArrive] + uint64_t(r.lsa) * r.workers);
    __syncthreads();

    // Sum each token's rows, in destination order, in fp32, once they are in; one 512-byte chunk of a token per
    // warp, tokens in order. A remote node's rows land in order: lane 0 waits until they cover the token's.
    const char* comb = r.local(a.at.comb);
    const int chunks = int((r.yb + 511) / 512), own_rows = a.k * a.t;
    // a comb row's node and index in the node's block (see the dispatch's nbase), and where its bytes are
    const auto locate = [&](int v, int& n, int& i) {
        if (!a.reduce) {
            for (n = 0; v >= nbase[n + 1]; ++n) {}
            i = v - nbase[n];
        } else {
            n = v < own_rows ? r.node : (v - own_rows) / a.t, i = v < own_rows ? v : v - own_rows - n * a.t;
        }
    };
    const auto at = [&](int v, int n, int i) {
        return a.reduce && n != r.node ? comb + size_t(own_rows + n * a.t) * r.yb + size_t(i) * r.ps
                                       : comb + size_t(v) * r.yb;
    };
    for (int item = r.gw; item < a.n * chunks; item += r.gws) {
        const int t = item / chunks;
        const size_t off = size_t(item % chunks) * 512 + r.lane * 16;
        int rows[kMaxTopk], nodes[kMaxTopk], index[kMaxTopk];
        int4 v[kMaxTopk];
        float scale[kMaxTopk];
#pragma unroll
        for (int i = 0; i < kMaxTopk; ++i) {
            rows[i] = i < a.k ? a.comb_rows[t * a.k + i] : -1;
            if (rows[i] >= 0) locate(rows[i], nodes[i], index[i]);
        }
        if (r.lane == 0) {
#pragma unroll
            for (int i = 0; i < kMaxTopk; ++i) {
                if (rows[i] < 0) break;
                const int n = nodes[i], id = kCombFrom + n, c = context(n);
                while (n != r.node && landed[r.warp][n] <= index[i])
                    landed[r.warp][n] = int(min(r.signal(id, c) - st.signal[c][id], uint64_t(a.node_rows[n])));
            }
        }
        __syncwarp();
#pragma unroll
        for (int i = 0; i < kMaxTopk; ++i) {
            if (rows[i] < 0 || off >= r.yb) continue;
            const char* row = at(rows[i], nodes[i], index[i]);
            if (a.fp8 && nodes[i] != r.node) {  // 8 FP8 values and their block's scale
                const int2 q = *reinterpret_cast<const int2*>(row + off / 2);
                v[i].x = q.x, v[i].y = q.y, scale[i] = reinterpret_cast<const float*>(row + a.width)[off / 2 / 128];
            } else {
                v[i] = *reinterpret_cast<const int4*>(row + off);
            }
        }
        float acc[8] = {};
        int4 slot[2] = {};
        int slots = 0;
        if (!a.reduce) {
#pragma unroll
            for (int i = 0; i < kMaxTopk; ++i) {
                if (rows[i] < 0) break;
                accumulate(acc, v[i]);
            }
        } else {  // DeepEP's reduce, node by node in comb_rows' order: at most two parts by a bf16 add, more in fp32
            for (int i = 0; i < kMaxTopk && rows[i] >= 0; ++i) {
                int4 part = v[i];
                if (nodes[i] == r.node) {  // this node's rows into its partial first
                    int end = i + 1;
                    while (end < kMaxTopk && rows[end] >= 0 && nodes[end] == r.node) ++end;
                    if (end - i <= 2) {
                        part = add_bf16(v[i], end - i == 2 ? v[i + 1] : int4{0, 0, 0, 0});
                    } else {
                        float sum[8] = {};
                        for (int j = i; j < end; ++j) accumulate(sum, v[j]);
                        part = to_bf16(sum);
                    }
                    i = end - 1;
                    accumulate(acc, part);
                } else if (a.fp8) {
                    dequantize_add(acc, v[i], scale[i]);
                } else {
                    accumulate(acc, part);
                }
                if (slots < 2) slot[slots] = part;
                ++slots;
            }
        }
        if (off < r.yb)
            *reinterpret_cast<int4*>(reinterpret_cast<char*>(a.out + size_t(t) * a.width) + off) =
                a.reduce && !a.fp8 && slots <= 2 ? add_bf16(slot[0], slot[1]) : to_bf16(acc);
    }
    // every aggregator's last put (the one after all its node-mates were done)
    if (threadIdx.x == 0) {
        for (int k = 0; k < r.nodes - 1; ++k) {
            const int n = r.remote(k), id = kCombFrom + n, c = context(n);
            while (r.signal(id, c) < st.signal[c][id] + uint64_t(a.node_rows[n]) + 1) {}
        }
    }
    r.mark(kCArrived);
    r.finish(advance, kCEnd);
}

struct Stream {
    ncclComm_t comm;
    ncclDevComm dev;
    ncclWindow_t win;
    void* buf;
    Layout at;
    State* state;
    int max_t, max_width, k, experts_per_rank, contexts, ranks, nodes;
};

Args args(const Stream& s, int t, int width, int ctas, int64_t chunk) {
    if (t > s.max_t || width > s.max_width || width % 8 != 0 || ctas < 2 || chunk < 1)
        throw std::runtime_error("rail_stream: bad call shape (T, width, CTAs or chunk)");
    Args a{};
    a.comm = s.dev, a.win = s.win, a.at = s.at;
    a.t = t, a.k = s.k, a.width = width, a.experts_per_rank = s.experts_per_rank, a.ctas = ctas;
    a.contexts = s.contexts, a.chunk = size_t(chunk), a.state = s.state, a.reduce = 1;
    return a;
}

// Cooperative, so every CTA is resident: they wait on each other's and other GPUs' progress.
void launch(void (*kernel)(Args), const Args& a, size_t smem, cudaStream_t stream) {
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(a.ctas), config.blockDim = dim3(kThreads), config.stream = stream;
    config.dynamicSmemBytes = smem;
    cudaLaunchAttribute cooperative;
    cooperative.id = cudaLaunchAttributeCooperative, cooperative.val.cooperative = 1;
    config.attrs = &cooperative, config.numAttrs = 1;
    CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, a));
}

}  // namespace

int64_t stream_create(int64_t comm, int max_t, int max_width, int k, int experts, int contexts) {
    auto* s = new Stream{};
    s->comm = reinterpret_cast<ncclComm_t>(comm);
    ncclCommProperties props = NCCL_COMM_PROPERTIES_INITIALIZER;
    NCCL_CHECK(ncclCommQueryProperties(s->comm, &props));
    if (props.nRanks > kMaxRanks || max_t > kMaxT || k > kMaxTopk || contexts < 1 || contexts > kMaxContexts ||
        max_width % 8 != 0)
        throw std::runtime_error("rail_stream takes at most 32 ranks, 4096 tokens per rank, top-8 and 8 GIN contexts");

    // As rail_moe.cu: only rail peers connected, exclusive contexts, 1024-deep queues, traffic class 3.
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_CUSTOM_STRIDE;
    reqs.ginCustomStride = ncclTeamLsa(s->comm).nRanks;
    reqs.ginContextCount = contexts;
    reqs.ginSignalCount = kNumSignals;
    reqs.ginExclusiveContexts = true;
    reqs.ginQueueDepth = 1024;
    reqs.ginTrafficClass = 3;
    NCCL_CHECK(ncclDevCommCreate(s->comm, &reqs, &s->dev));
    const int ranks = props.nRanks, lsa = s->dev.lsaSize, nodes = ranks / lsa;
    if (nodes > kMaxNodes || lsa > kMaxLsa) throw std::runtime_error("rail_stream takes at most 8 nodes of 8 GPUs");
    if (int(s->dev.ginContextCount) < contexts)
        throw std::runtime_error("NCCL created " + std::to_string(s->dev.ginContextCount) + " GIN contexts");

    const size_t yb = size_t(max_width) * 2;
    size_t bytes = 0;
    const auto region = [&](size_t size) {
        const size_t off = bytes;
        bytes += (size + 4095) / 4096 * 4096;
        return off;
    };
    Layout& at = s->at;
    at.area = align16(area_bytes(max_t, max_width, k, lsa));
    at.gather_area = size_t(lsa < k ? lsa : k) * max_t * yb;
    at.flags = region(4 * size_t(kMaxLsa) * kMaxNodes * (kMaxChunks + 1));
    at.heads = region(16 * kMaxRanks);
    at.roffs = region(4 * size_t(ranks) * max_t);
    at.bundle = region(nodes * at.area);
    at.fwd = region(nodes * at.area);
    at.gather = region((nodes - 1) * at.gather_area);
    at.comb = region(size_t(k + nodes) * max_t * yb);
    at.partial = region(size_t(nodes - 1) * max_t * yb);
    at.counters = region(8 * kNumCounters);
    at.bytes = bytes;
    NCCL_CHECK(ncclMemAlloc(&s->buf, bytes));
    CUDA_CHECK(cudaMemset(s->buf, 0, bytes));
    NCCL_CHECK(ncclCommWindowRegister(s->comm, s->buf, bytes, &s->win, NCCL_WIN_STRICT_ORDERING));
    CUDA_CHECK(cudaMalloc(&s->state, sizeof(State)));
    CUDA_CHECK(cudaMemset(s->state, 0, sizeof(State)));
    // the kernel's limit is per process: the largest any buffer here needs
    static size_t smem = 0;
    smem = std::max(smem, smem_bytes(max_t, ranks, nodes, scale_bytes(max_width)));
    CUDA_CHECK(cudaFuncSetAttribute(dispatch_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)));
    s->max_t = max_t, s->max_width = max_width, s->k = k, s->experts_per_rank = experts / ranks;
    s->contexts = contexts, s->ranks = ranks, s->nodes = nodes;
    return reinterpret_cast<int64_t>(s);
}

void stream_dispatch(int64_t stream, const void* x, const float* xs, const int64_t* topk_idx,
                     const float* topk_weights, int n, int vbytes, int sbytes, int t, int ctas, int64_t chunk,
                     void* recv_x, float* recv_xs, int64_t* recv_idx, float* recv_weights, int* recv_src, int* psum,
                     int* comb_rows, int* node_rows, cudaStream_t cs) {
    const auto& s = *reinterpret_cast<Stream*>(stream);
    if (vbytes % 16 != 0 || align16(sbytes) > scale_bytes(s.max_width) || sbytes % 4 != 0)
        throw std::runtime_error("rail_stream: rows of 16-byte multiples, with at most a scale per 128 of width_max");
    Args a = args(s, t, int(vbytes + align16(sbytes)) / 2, ctas, chunk);
    a.vbytes = vbytes, a.sbytes = sbytes, a.xs = xs, a.recv_xs = recv_xs;
    a.n = n, a.x = x, a.topk_idx = topk_idx, a.topk_weights = topk_weights;
    a.recv_x = recv_x, a.recv_idx = recv_idx, a.recv_weights = recv_weights;
    a.recv_src = recv_src, a.psum = psum, a.comb_rows = comb_rows, a.node_rows = node_rows;
    launch(dispatch_kernel, a, smem_bytes(t, s.ranks, s.nodes, align16(sbytes)), cs);
}

void stream_combine(int64_t stream, const void* y, int* psum, int* comb_rows, int* node_rows, int n, int width, int t,
                    int ctas, int64_t chunk, bool fp8, void* out, cudaStream_t cs) {
    const auto& s = *reinterpret_cast<Stream*>(stream);
    if (fp8 && width % 128 != 0) throw std::runtime_error("rail_stream: an FP8 combine takes rows of 128-value blocks");
    Args a = args(s, t, width, ctas, chunk);
    a.fp8 = fp8;
    a.n = n, a.y = static_cast<const __nv_bfloat16*>(y), a.psum = psum, a.comb_rows = comb_rows;
    a.node_rows = node_rows, a.out = static_cast<__nv_bfloat16*>(out);
    launch(combine_kernel, a, 0, cs);
}

void stream_destroy(int64_t stream) {
    auto* s = reinterpret_cast<Stream*>(stream);
    NCCL_CHECK(ncclCommWindowDeregister(s->comm, s->win));
    NCCL_CHECK(ncclMemFree(s->buf));
    NCCL_CHECK(ncclDevCommDestroy(s->comm, &s->dev));
    CUDA_CHECK(cudaFree(s->state));
    delete s;
}
