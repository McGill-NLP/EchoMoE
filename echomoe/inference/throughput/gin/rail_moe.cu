// MoE dispatch and combine, all-gather and reduce-scatter at EP over GIN rails (NCCL device API): per op, one fused
// put+signal per remote node from each GPU, to its rail peer there (the same local rank), split into chunks over
// up to `contexts` GIN contexts; everything else moves over NVLink. See rails.py for the outputs.
//
// Dispatch, at source s = (node, local rank l):
//   s finds each token's destination GPUs (the owners of its experts, each once). One group of CTAs per remote node
//   writes that node's bundle, in parallel: its per-destination counts, then the records (the row, then the token
//   index, top-k ids and weights) in destination order; the group's last CTA to finish sends it to s's rail peer
//   there. Then s writes its on-node records straight into each destination's slot for s over NVLink, and forwards
//   each bundle that lands here (from its rail peers) into its destinations' slots for those sources. Whoever fills
//   a slot writes its header: the record count and where the destination's combine rows for the source go. A
//   destination packs its slots by source rank once every source has arrived.
// Combine reverses it: each received row goes to its source over NVLink if on-node, else into the source's block at
//   the source's rail peer here (the aggregator), which sends the 8 gathered blocks in one transfer straight into
//   the source's receive area; the source sums each token's rows node by node, as DeepEP's bits (with `fp8`, as
//   deepep_fp8c's).
// All-gather: each rank copies its block into its slot, sends it to its rail peers, writes it to its node-mates and
//   forwards the blocks its rail peers send to its node-mates; then copies the gathered slots out.
// Reduce-scatter: each rank writes its block for every destination to the destination's rail peer on this node;
//   that rank sums the 8 blocks per destination node in fp32 and sends each remote node its partial, and each
//   destination sums the nodes' partials (its own node's from the blocks) in node order.
// NVLink arrivals are published with one system fence per CTA and relaxed adds. Grids are `ctas` cooperative CTAs.
// Signals and counters only grow: a kernel waits for its call's arrivals above `bases`, which the last CTA to finish
// advances by them; the collectives alternate between two buffers by call parity, so back-to-back calls are safe.
// The trace stores (Trace events, globaltimer) stay compiled in so that the kernels are the measured ones, and the
// host never enables them. The same holds for the dispatch's non-reduce branch: every combine sums as DeepEP's.
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

constexpr int kThreads = 512, kWarps = kThreads / 32, kMaxT = 512, kMaxRanks = 32, kMaxNodes = 8, kMaxTopk = 8;
constexpr int kMaxLsa = 8, kMaxContexts = 8, kUnroll = 8;  // kUnroll: 16-byte loads in flight per lane in a copy
constexpr size_t kChunk = 128 << 10;          // bytes per put of a split transfer
// GIN signals (per context) and window counters share these ids, in separate storage; *From and *For are per node.
enum Id {
    kDispArrive,  // counter: a writer CTA filled its slots here (nodes per CTA)
    kCombArrive,  // counter: an on-node destination CTA returned its rows here
    kAgArrive,    // counter: a node-mate CTA wrote its block and forwarded blocks here (nodes per CTA)
    kRsArrive,    // counter: a node-mate CTA wrote its blocks for this rank's destinations here
    kAgBuilt,     // counter: a CTA copied its share of this rank's all-gather block
    kAgCalls,     // base only: all-gather calls so far (the buffer parity)
    kRsCalls,     // base only: reduce-scatter calls so far
    kDispFrom,                              // signal: a dispatch bundle from that node landed here
    kCombFrom = kDispFrom + kMaxNodes,      // signal: combine rows from that node's aggregator landed here
    kAgFrom = kCombFrom + kMaxNodes,        // signal: that node's rail peer's all-gather block landed here
    kRsFrom = kAgFrom + kMaxNodes,          // signal: that node's partial landed here
    kGatherFrom = kRsFrom + kMaxNodes,      // counter: a destination CTA returned rows for that node's source here
    kBuiltFor = kGatherFrom + kMaxNodes,    // counter: a CTA finished its share of the bundle for that node
    kSummedFor = kBuiltFor + kMaxNodes,     // counter: a CTA finished its share of the partial for that node
    kReducedFor = kSummedFor + kMaxNodes,   // counter: a CTA finished its share of the FP8 partials for that node
    kNumIds = kReducedFor + kMaxNodes,
};
// Trace slots, per kernel; k counts remote nodes from this one's next.
enum Trace {
    kDStart, kDRouted, kDPut, kDRecords = kDPut + kMaxNodes - 1, kDSeen, kDForwarded = kDSeen + kMaxNodes - 1,
    kDSignaled = kDForwarded + kMaxNodes - 1, kDArrived, kDEnd, kTraceSlots = 32,
    kCStart = 0, kCSent, kCPut, kCArrived = kCPut + kMaxNodes - 1, kCEnd,
};

// Window byte offsets. heads: 16 bytes per source; stage: one slot of t records per source; fwd and bundle: one
// bundle per node, node_bundle apart; gather and comb: lsa * t rows per node; ag: two buffers ag_buffer apart of
// one t-row block per rank; rs_stage: two buffers of one block per (local rank, node); rs_send and rs_recv: two
// buffers of one fp32 partial per node. Slots are packed for the call's t and widths.
struct Layout {
    size_t heads, stage, fwd, bundle, node_bundle, gather, comb, partial, ag, ag_buffer, rs_stage, rs_stage_buffer,
        rs_send, rs_recv, rs_partial_buffer, counters;
};

struct Args {
    ncclDevComm comm;
    ncclWindow_t win;
    Layout at;
    int t, n, k, width, experts_per_rank, ctas, contexts;
    int vbytes, sbytes;         // dispatch: each row's value bytes (2 * width in bf16) and scale bytes (FP8: a
                                //   float32 per 128 values; 0 in bf16)
    int fp8;                    // the combine's partials between nodes in FP8 (fp8_combine)
    int reduce;                 // the combine as DeepEP's: comb_rows in its slot order; in bf16, Slots at both levels;
                                //   with fp8, as deepep_fp8c's: Slots in a node, fp32 between nodes
    size_t row;                 // all-gather row bytes
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
    int* node_rows;             // [kMaxNodes]: rows each node returns in combine (with fp8, partials: tokens)
    const __nv_bfloat16* y;     // combine: [ranks * t, width]
    __nv_bfloat16* out;         // [n, width]
    const char* in;             // all-gather [t, row] bytes; reduce-scatter [ranks * t, width] bf16
    char* dst;                  // all-gather [ranks * t, row] bytes; reduce-scatter [t, width] bf16
    uint64_t* bases;            // [kMaxContexts][kNumIds]
    unsigned* exits;
    int* last_rowb;             // the last dispatch's record row bytes
    uint64_t* trace;            // [kTraceSlots] or null
};

__host__ __device__ int meta_bytes(int k) { return (4 * (1 + 2 * k) + 15) / 16 * 16; }
__host__ __device__ size_t align16(size_t v) { return (v + 15) / 16 * 16; }
__host__ __device__ int head_bytes(int lsa) { return (4 * lsa + 15) / 16 * 16; }

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
__device__ void wait_ge(const uint64_t* p, uint64_t v) {
    while (ld_acquire(p) < v) {}
}
// After fence_sys, relaxed adds publish everything this CTA wrote before its last __syncthreads.
__device__ void fence_sys() { asm volatile("fence.acq_rel.sys;" ::: "memory"); }
__device__ void add_relaxed(uint64_t* p, uint64_t v) {
    asm volatile("red.relaxed.sys.global.add.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

__device__ int inclusive_scan(int v, int lane) {
    for (int o = 1; o < 32; o <<= 1) {
        const int u = __shfl_up_sync(~0u, v, o);
        if (lane >= o) v += u;
    }
    return v;
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
__device__ int4 to_bf16(const float* acc) {
    int4 o;
    auto* h = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
    for (int e = 0; e < 4; ++e) h[e] = __floats2bfloat162_rn(acc[2 * e], acc[2 * e + 1]);
    return o;
}
// One level of DeepEP's combine_reduce (combine_utils.cuh) over 8 bf16 values per slot, slots added in order: up to two
// add in native bf16 (one: to +0, none: +0), three or more in fp32 from 0, rounded once.
struct Slots {
    int4 first = {}, second = {};
    float acc[8] = {};
    int count = 0;

    __device__ void add(const int4& v) {
        if (count == 0) {
            first = v;
        } else if (count == 1) {
            second = v;
        } else {
            if (count == 2) accumulate(acc, first), accumulate(acc, second);
            accumulate(acc, v);
        }
        ++count;
    }
    __device__ int4 sum() const {
        if (count > 2) return to_bf16(acc);
        int4 o;
        auto* h = reinterpret_cast<__nv_bfloat162*>(&o);
        const auto* a = reinterpret_cast<const __nv_bfloat162*>(&first);
        const auto* b = reinterpret_cast<const __nv_bfloat162*>(&second);
#pragma unroll
        for (int e = 0; e < 4; ++e) h[e] = __hadd2(a[e], b[e]);
        return o;
    }
};
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

// The first k in `todo` that is ready, polling until one is.
template <typename Ready>
__device__ int pick(unsigned todo, Ready ready) {
    for (;;)
        for (int k = 0; k < kMaxNodes; ++k)
            if ((todo >> k & 1) && ready(k)) return k;
}

struct Rank {
    const Args& a;
    int me, lsa, l, ranks, nodes, node, lane, warp, gw, gws;
    size_t rowb, rb, yb, ps;  // dispatch: a record's row bytes and bytes; row bytes (combine, reduce-scatter);
                              // an FP8 partial's bytes (combine)

    __device__ Rank(const Args& a)
        : a(a), me(a.comm.rank), lsa(a.comm.lsaSize), l(a.comm.lsaRank), ranks(a.comm.nRanks), nodes(ranks / lsa),
          node(me / lsa), lane(threadIdx.x % 32), warp(threadIdx.x / 32), gw(blockIdx.x * kWarps + warp),
          gws(a.ctas * kWarps), rowb(size_t(a.vbytes) + align16(a.sbytes)), rb(rowb + meta_bytes(a.k)),
          yb(size_t(a.width) * 2), ps(align16(size_t(a.width) + 4 * ((a.width + 127) / 128))) {}

    __device__ int at(int n, int j) const { return n * lsa + j; }
    // Local rank j's place among this node's ranks a token goes to, in DeepEP's slot order (each rank's highest top-k
    // lane, lanes going up); ids: the token's top-k expert ids.
    __device__ int slot_here(const int* ids, int j) const {
        int last[kMaxLsa], place = 0;
        for (int e = 0; e < lsa; ++e) last[e] = -1;
        for (int l = 0; l < a.k; ++l) {
            const int d = ids[l] >= 0 ? ids[l] / a.experts_per_rank : -1;
            if (d >= 0 && d / lsa == node) last[d % lsa] = l;
        }
        for (int e = 0; e < lsa; ++e) place += last[e] >= 0 && last[e] < last[j];
        return place;
    }
    __device__ int remote(int k) const { return (node + 1 + k) % nodes; }
    __device__ unsigned remotes() const { return (1u << (nodes - 1)) - 1; }
    __device__ char* local(size_t off) const { return static_cast<char*>(ncclGetLocalPointer(a.win, off)); }
    __device__ char* peer(size_t off, int r) const { return static_cast<char*>(ncclGetPeerPointer(a.win, off, r)); }
    __device__ uint64_t* counter(int id, int r) const {
        return reinterpret_cast<uint64_t*>(peer(a.at.counters + 8 * id, r));
    }
    __device__ uint64_t& base(int id, int c = 0) const { return a.bases[c * kNumIds + id]; }
    __device__ size_t head(int s) const { return a.at.heads + 16 * s; }
    __device__ size_t slot(int s, int i) const { return a.at.stage + (size_t(s) * a.t + i) * rb; }
    __device__ size_t bundle(int n) const { return a.at.bundle + n * a.at.node_bundle; }
    __device__ size_t fwd(int n) const { return a.at.fwd + n * a.at.node_bundle; }
    __device__ int fwd_rows(int n) const {  // records in the bundle that landed here from node n
        const int* counts = reinterpret_cast<const int*>(local(fwd(n)));
        int total = 0;
        for (int j = 0; j < lsa; ++j) total += counts[j];
        return total;
    }
    __device__ size_t bundle_bytes(int n) const { return head_bytes(lsa) + fwd_rows(n) * rb; }
    __device__ size_t gather(int n) const { return a.at.gather + size_t(n) * lsa * a.t * yb; }
    __device__ size_t comb(int n) const { return a.at.comb + size_t(n) * lsa * a.t * yb; }
    __device__ size_t partial(int k) const { return a.at.partial + size_t(k) * a.t * yb; }

    // CTA groups that each build one remote node's bundle or partial: remote node k by the CTAs c with
    // c % groups() == k % groups(), members(k) of them, this one as member blockIdx.x / groups().
    __device__ int groups() const { return max(1, min(a.ctas, nodes - 1)); }
    __device__ int members(int k) const { return (a.ctas - k % groups() + groups() - 1) / groups(); }

    __device__ void mark(int event) const {
        if (a.trace != nullptr && threadIdx.x == 0)
            atomicMax(reinterpret_cast<unsigned long long*>(a.trace + event), globaltimer());
    }
    __device__ void mark_start(int event) const {
        if (a.trace != nullptr && blockIdx.x == 0 && threadIdx.x == 0) a.trace[event] = globaltimer();
    }

    // Call with the whole CTA once its share of a local buffer is written: whether this CTA is the last of
    // `arrivals` to finish it (then every share is visible to the NIC).
    __device__ bool complete(int id, int arrivals) const {
        __shared__ int last;
        __syncthreads();
        if (threadIdx.x == 0) {
            fence_sys();
            last = atomicAdd(reinterpret_cast<unsigned long long*>(counter(id, me)), 1) + 1 == base(id) + arrivals;
            if (last) fence_sys();
        }
        __syncthreads();
        return last;
    }
    // Split transfers: chunks of kChunk bytes, at most one per context, each adding 1 to the signal there.
    __device__ int chunks(size_t bytes) const {
        const size_t n = (bytes + kChunk - 1) / kChunk;
        return int(n < 1 ? 1 : n > size_t(a.contexts) ? a.contexts : n);
    }
    // Called by a warp: lane c sends chunk c of [src, src + bytes) to `peer`'s dst over context c, adding 1 to signal
    // `id` there (a signal alone for 0 bytes); with system release, for data other GPUs or CTAs wrote.
    __device__ void put(int peer, size_t dst, size_t src, size_t bytes, int id) const {
        const int n = chunks(bytes);
        if (lane >= n) return;
        const size_t part = ((bytes + n - 1) / n + 15) / 16 * 16, begin = min(bytes, lane * part);
        const size_t size = min(bytes - begin, part);
        const ncclGin gin(a.comm, lane, NCCL_GIN_RESOURCE_SHARING_CTA);
        const auto add = ncclGin_StrongSignalAdd{static_cast<ncclGinSignal_t>(id), 1};
        if (size > 0)
            gin.put(ncclTeamWorld(a.comm), peer, a.win, dst + begin, a.win, src + begin, size, add, ncclGin_None{},
                    ncclCoopThread{}, ncclGin_None{}, cuda::thread_scope_thread, cuda::thread_scope_system);
        else
            gin.signal(ncclTeamWorld(a.comm), peer, add);
    }
    // Whether every chunk of a `bytes` transfer to signal `id` has landed.
    __device__ bool landed(int id, size_t bytes) const {
        for (int c = 0; c < chunks(bytes); ++c)
            if (ncclGin(a.comm, c, NCCL_GIN_RESOURCE_SHARING_CTA).readSignal(id) < base(id, c) + 1) return false;
        return true;
    }
    __device__ void advance_landed(int id, size_t bytes) const {
        for (int c = 0; c < chunks(bytes); ++c) base(id, c) += 1;
    }

    // The last CTA out advances the bases, once every CTA is past its waits.
    template <typename Advance>
    __device__ void finish(Advance advance, int end = -1) const {
        __syncthreads();
        if (end >= 0) mark(end);
        if (threadIdx.x == 0) {
            __threadfence();
            if (atomicAdd(a.exits, 1) == unsigned(a.ctas - 1)) {
                *a.exits = 0;
                advance();
                __threadfence();
            }
        }
    }

    // A multiple of 16 bytes, every lane with kUnroll loads in flight.
    __device__ void copy(char* to, const char* from, size_t bytes) const {
        for (size_t base = 0; base < bytes; base += 32 * 16 * kUnroll) {
            int4 v[kUnroll];
#pragma unroll
            for (int u = 0; u < kUnroll; ++u) {
                const size_t i = base + (u * 32 + lane) * 16;
                if (i < bytes) v[u] = *reinterpret_cast<const int4*>(from + i);
            }
#pragma unroll
            for (int u = 0; u < kUnroll; ++u) {
                const size_t i = base + (u * 32 + lane) * 16;
                if (i < bytes) *reinterpret_cast<int4*>(to + i) = v[u];
            }
        }
    }
    // token index, top-k ids (int32), top-k weights
    __device__ void write_meta(char* to, int t) const {
        int* m = reinterpret_cast<int*>(to);
        if (lane == 0)
            m[0] = t;
        else if (lane <= a.k)
            m[lane] = int(a.topk_idx[t * a.k + lane - 1]);
        else if (lane <= 2 * a.k)
            m[lane] = __float_as_int(a.topk_weights[t * a.k + lane - 1 - a.k]);
    }
    // The source whose rows (prefix sums `psum`, in lanes) hold row q.
    __device__ int source_of(const int* psum, int q) const {
        return __popc(__ballot_sync(~0u, lane < ranks && psum[lane] <= q));
    }
    // This CTA publishes its NVLink writes, then adds `v` to counter `id` on each node-mate (but itself unless
    // `self`).
    __device__ void signal_node(int id, uint64_t v, bool self) const {
        __syncthreads();
        if (threadIdx.x == 0) {
            fence_sys();
            for (int j = 0; j < lsa; ++j)
                if (self || j != l) add_relaxed(counter(id, at(node, j)), v);
        }
    }
};

// Each token's destination GPUs, and its position among the tokens sent to each, in token order.
struct Routing {
    uint32_t mask[kMaxT], ballot[kMaxT / 32][kMaxRanks], nballot[kMaxT / 32][kMaxNodes];
    int base[kMaxT / 32][kMaxRanks], count[kMaxRanks], pre[kMaxRanks];  // pre: counts of earlier on-node peers
    int nbase[kMaxT / 32][kMaxNodes], ncount[kMaxNodes];                // the same per node, for the tokens

    __device__ void build(const Rank& r) {
        const Args& a = r.a;
        for (int c = r.warp; c * 32 < a.t; c += kWarps) {
            const int t = c * 32 + r.lane;
            uint32_t m = 0;
            for (int i = 0; t < a.n && i < a.k; ++i) {
                const int64_t id = a.topk_idx[t * a.k + i];
                if (id >= 0) m |= 1u << (id / a.experts_per_rank);
            }
            if (t < a.t) mask[t] = m;
            for (int d = 0; d < r.ranks; ++d) {
                const uint32_t b = __ballot_sync(~0u, m >> d & 1);
                if (r.lane == d) ballot[c][d] = b;
            }
            for (int n = 0; n < r.nodes; ++n) {
                const uint32_t b = __ballot_sync(~0u, (m >> (n * r.lsa) & ((1u << r.lsa) - 1)) != 0);
                if (r.lane == n) nballot[c][n] = b;
            }
        }
        __syncthreads();
        if (threadIdx.x < r.ranks) {
            const int d = threadIdx.x;
            int run = 0;
            for (int c = 0; c * 32 < a.t; ++c) base[c][d] = run, run += __popc(ballot[c][d]);
            count[d] = run;
        }
        if (threadIdx.x >= 32 && threadIdx.x < 32 + r.nodes) {
            const int n = threadIdx.x - 32;
            int run = 0;
            for (int c = 0; c * 32 < a.t; ++c) nbase[c][n] = run, run += __popc(nballot[c][n]);
            ncount[n] = run;
        }
        __syncthreads();
        if (threadIdx.x < r.ranks) {
            const int d = threadIdx.x;
            int sum = 0;
            for (int e = d - d % r.lsa; e < d; ++e) sum += count[e];
            pre[d] = sum;
        }
        __syncthreads();
    }
    __device__ int pos(int t, int d) const {
        return base[t / 32][d] + __popc(ballot[t / 32][d] & ((1u << t % 32) - 1));
    }
    __device__ int npos(int t, int n) const {  // among the tokens sent to node n
        return nbase[t / 32][n] + __popc(nballot[t / 32][n] & ((1u << t % 32) - 1));
    }
    // Token t's rows in comb in DeepEP's slot order, a node's rows together: ptx::deduplicate keeps the highest top-k
    // lane of each node (combine_reduce_epilogue.cuh) and, within a node, of each rank (hybrid_combine.cuh);
    // compute_topk_slots takes the kept lanes in ascending order.
    __device__ void write_slots(const Rank& r, int t) const {
        const Args& a = r.a;
        int dst[kMaxTopk], i = 0;
        for (int j = 0; j < a.k; ++j) {
            const int64_t id = a.topk_idx[t * a.k + j];
            dst[j] = id >= 0 ? int(id / a.experts_per_rank) : -1;
        }
        for (int j = 0; j < a.k; ++j) {
            if (dst[j] < 0) continue;
            const int n = dst[j] / r.lsa;
            bool node_last = true;
            for (int u = j + 1; u < a.k; ++u) node_last &= dst[u] < 0 || dst[u] / r.lsa != n;
            if (!node_last) continue;
            if (a.fp8 && n != r.node) {  // another node's one FP8 partial
                a.comb_rows[t * a.k + i++] = n * r.lsa * a.t + npos(t, n);
                continue;
            }
            for (int q = 0; q <= j; ++q) {
                if (dst[q] < 0 || dst[q] / r.lsa != n) continue;
                bool rank_last = true;
                for (int u = q + 1; u <= j; ++u) rank_last &= dst[u] != dst[q];
                if (rank_last) a.comb_rows[t * a.k + i++] = n * r.lsa * a.t + pre[dst[q]] + pos(t, dst[q]);
            }
        }
        for (; i < a.k; ++i) a.comb_rows[t * a.k + i] = -1;
    }
    // This warp's records of token t for node n: into the local bundle, or (this node) straight into the slots.
    __device__ void write(const Rank& r, int t, int n) const {
        for (uint32_t bits = mask[t] >> (n * r.lsa) & ((1u << r.lsa) - 1); bits != 0; bits &= bits - 1) {
            const int d = r.at(n, __ffs(bits) - 1), p = pos(t, d);
            char* to = n == r.node ? r.peer(r.slot(r.me, p), d)
                                   : r.local(r.bundle(n) + head_bytes(r.lsa) + (pre[d] + p) * r.rb);
            r.copy(to, static_cast<const char*>(r.a.x) + size_t(t) * r.a.vbytes, r.a.vbytes);
            for (int w = r.lane; w < r.a.sbytes / 4; w += 32)
                reinterpret_cast<float*>(to + r.a.vbytes)[w] = r.a.xs[t * (r.a.sbytes / 4) + w];
            r.write_meta(to + r.rowb, t);
        }
    }
};

__global__ void __launch_bounds__(kThreads, 1) dispatch_kernel(const Args a) {
    const Rank r(a);
    r.mark_start(kDStart);
    __shared__ Routing route;
    __shared__ int next, psum[kMaxRanks];
    route.build(r);
    r.mark(kDRouted);
    if (blockIdx.x == 0 && threadIdx.x < r.ranks && threadIdx.x / r.lsa == r.node) {
        int* h = reinterpret_cast<int*>(r.peer(r.head(r.me), threadIdx.x));
        h[0] = route.count[threadIdx.x], h[1] = route.pre[threadIdx.x];
    }
    if (blockIdx.x == 0 && threadIdx.x < r.nodes) {  // with fp8, other nodes return a partial per token
        int rows = 0;
        for (int j = 0; j < r.lsa; ++j) rows += route.count[r.at(threadIdx.x, j)];
        a.node_rows[threadIdx.x] = a.fp8 && threadIdx.x != r.node ? route.ncount[threadIdx.x] : rows;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) *a.last_rowb = int(r.rowb);

    // Each remote node's bundle by its CTA group, sent by the group's last CTA to finish.
    for (int k = blockIdx.x % r.groups(); k < r.nodes - 1; k += r.groups()) {
        const int n = r.remote(k), member = blockIdx.x / r.groups();
        if (member == 0 && threadIdx.x < r.lsa)
            reinterpret_cast<int*>(r.local(r.bundle(n)))[threadIdx.x] = route.count[r.at(n, threadIdx.x)];
        for (int t = member * kWarps + r.warp; t < a.n; t += r.members(k) * kWarps) route.write(r, t, n);
        if (r.complete(kBuiltFor + n, r.members(k)) && r.warp == 0) {
            int total = 0;
            for (int j = 0; j < r.lsa; ++j) total += route.count[r.at(n, j)];
            r.put(r.at(n, r.l), r.fwd(r.node), r.bundle(n), head_bytes(r.lsa) + total * r.rb, kDispFrom + r.node);
            r.mark(kDPut + k);
        }
    }
    for (int t = r.gw; t < a.n; t += r.gws) {
        route.write(r, t, r.node);
        if (r.lane == 0 && a.reduce) {
            route.write_slots(r, t);
        } else if (r.lane == 0) {
            int i = 0;
            for (uint32_t m = route.mask[t]; m != 0; m &= m - 1, ++i) {
                const int d = __ffs(m) - 1, n = d / r.lsa;
                a.comb_rows[t * a.k + i] = n * r.lsa * a.t + (a.fp8 && n != r.node ? route.npos(t, n)
                                                                                    : route.pre[d] + route.pos(t, d));
            }
            for (; i < a.k; ++i) a.comb_rows[t * a.k + i] = -1;
        }
    }
    r.mark(kDRecords);

    // Forward each bundle from a rail peer as it lands.
    for (unsigned todo = r.remotes(); todo != 0;) {
        if (threadIdx.x == 0) {
            next = pick(todo, [&](int k) {
                const int id = kDispFrom + r.remote(k);
                return ncclGin(a.comm, 0, NCCL_GIN_RESOURCE_SHARING_CTA).readSignal(id) >= r.base(id) + 1 &&
                       r.landed(id, r.bundle_bytes(r.remote(k)));  // the counts came with chunk 0
            });
        }
        __syncthreads();
        const int k = next, n = r.remote(k), src = r.at(n, r.l);
        todo &= ~(1u << k);
        r.mark(kDSeen + k);
        // lanes < lsa: each destination's count and inclusive prefix
        const int count = r.lane < r.lsa ? reinterpret_cast<const int*>(r.local(r.fwd(n)))[r.lane] : 0;
        const int incl = inclusive_scan(count, r.lane), total = __shfl_sync(~0u, incl, r.lsa - 1);
        if (blockIdx.x == 0 && threadIdx.x < r.lsa) {
            int* h = reinterpret_cast<int*>(r.peer(r.head(src), r.at(r.node, threadIdx.x)));
            h[0] = count, h[1] = incl - count;
        }
        for (int q = r.gw; q < total; q += r.gws) {
            const int j = __popc(__ballot_sync(~0u, r.lane < r.lsa && incl <= q));
            const int first = __shfl_sync(~0u, incl - count, j);
            r.copy(r.peer(r.slot(src, q - first), r.at(r.node, j)), r.local(r.fwd(n) + head_bytes(r.lsa) + q * r.rb),
                   r.rb);
        }
        __syncthreads();
        r.mark(kDForwarded + k);
    }

    // This CTA's direct and forwarded slots, one arrival per source node, to every on-node destination.
    r.signal_node(kDispArrive, r.nodes, true);
    r.mark(kDSignaled);

    // Pack the slots by source rank once every source's writers are done.
    if (threadIdx.x == 0) wait_ge(r.counter(kDispArrive, r.me), r.base(kDispArrive) + r.ranks * a.ctas);
    __syncthreads();
    r.mark(kDArrived);
    if (threadIdx.x < 32) {
        const int count = r.lane < r.ranks ? reinterpret_cast<const int*>(r.local(r.head(r.lane)))[0] : 0;
        psum[r.lane] = inclusive_scan(count, r.lane);
        if (blockIdx.x == 0 && r.lane < r.ranks) a.psum[r.lane] = psum[r.lane];
    }
    __syncthreads();
    for (int q = r.gw; q < psum[r.ranks - 1]; q += r.gws) {
        const int s = r.source_of(psum, q), i = q - (s > 0 ? psum[s - 1] : 0);
        const char* record = r.local(r.slot(s, i));
        const int* m = reinterpret_cast<const int*>(record + r.rowb);
        r.copy(static_cast<char*>(a.recv_x) + size_t(q) * a.vbytes, record, a.vbytes);
        for (int w = r.lane; w < a.sbytes / 4; w += 32)
            a.recv_xs[size_t(q) * (a.sbytes / 4) + w] = reinterpret_cast<const float*>(record + a.vbytes)[w];
        if (r.lane < a.k) {
            const int id = m[1 + r.lane], e = id - r.me * a.experts_per_rank;
            a.recv_idx[size_t(q) * a.k + r.lane] = id >= 0 && e >= 0 && e < a.experts_per_rank ? e : -1;
            a.recv_weights[size_t(q) * a.k + r.lane] = __int_as_float(m[1 + a.k + r.lane]);
        }
        if (r.lane == 0) a.recv_src[q] = s * a.t + m[0];
    }
    r.finish(
        [&] {
            for (int k = 0; k < r.nodes - 1; ++k) {
                r.base(kBuiltFor + r.remote(k)) += r.members(k);
                r.advance_landed(kDispFrom + r.remote(k), r.bundle_bytes(r.remote(k)));
            }
            r.base(kDispArrive) += r.ranks * a.ctas;
        },
        kDEnd);
}

// kFp8 (a.fp8) as a template argument: the bf16 combine compiles as without FP8 (registers, shared memory). The sums
// are DeepEP's; in bf16 every row still crosses InfiniBand, only the rounding changes.
template <bool kFp8>
__global__ void __launch_bounds__(kThreads, 1) combine_kernel(const Args a) {
    const Rank r(a);
    r.mark_start(kCStart);
    __shared__ int next, psum[kMaxRanks];
    if (threadIdx.x < 32) psum[r.lane] = r.lane < r.ranks ? a.psum[r.lane] : 0;
    __syncthreads();
    for (int q = r.gw; q < psum[r.ranks - 1]; q += r.gws) {
        const int s = r.source_of(psum, q), n = s / r.lsa;
        const int row = reinterpret_cast<const int*>(r.local(r.head(s)))[1] + q - (s > 0 ? psum[s - 1] : 0);
        char* to = n == r.node ? r.peer(r.comb(n) + row * r.yb, s)
                               : r.peer(r.gather(n) + row * r.yb, r.at(r.node, s % r.lsa));
        r.copy(to, reinterpret_cast<const char*>(a.y + size_t(q) * a.width), r.yb);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        fence_sys();
        for (int s = 0; s < r.ranks; ++s) {
            const int n = s / r.lsa;
            add_relaxed(n == r.node ? r.counter(kCombArrive, s) : r.counter(kGatherFrom + n, r.at(r.node, s % r.lsa)),
                        1);
        }
    }
    r.mark(kCSent);

    // This CTA's share of the aggregator transfers: each remote source node's rows, once all are gathered here.
    unsigned todo = 0;
    for (int k = blockIdx.x; !kFp8 && k < r.nodes - 1; k += a.ctas) todo |= 1u << k;
    while (todo != 0) {
        if (threadIdx.x == 0) {
            next = pick(todo, [&](int k) {
                const int id = kGatherFrom + r.remote(k);
                return ld_acquire(r.counter(id, r.me)) >= r.base(id) + r.lsa * a.ctas;
            });
        }
        __syncthreads();
        const int k = next, n = r.remote(k);
        todo &= ~(1u << k);
        if (r.warp == 0) {
            r.put(r.at(n, r.l), r.comb(r.node), r.gather(n), r.fwd_rows(n) * r.yb, kCombFrom + r.node);
            r.mark(kCPut + k);
        }
        __syncthreads();
    }
    // With fp8: each remote source's tokens here, their rows summed in DeepEP's slot order by Slots into one FP8
    // partial each (in token order), by the source node's CTA group; sent by the group's last CTA to finish.
    if constexpr (kFp8) {
        // each token's row in the block from each slot's destination, or -1
        __shared__ int16_t rows_of[kMaxT][kMaxLsa];
        __shared__ int place[kMaxT], sums[kWarps];
        for (int k = blockIdx.x % r.groups(); k < r.nodes - 1; k += r.groups()) {
            const int n = r.remote(k), member = blockIdx.x / r.groups(), segs = a.width / 8;
            const int rowb = *a.last_rowb, rb = rowb + meta_bytes(a.k);  // the dispatch's records
            for (int e = threadIdx.x; e < a.t * kMaxLsa; e += kThreads) rows_of[e / kMaxLsa][e % kMaxLsa] = -1;
            if (threadIdx.x == 0) wait_ge(r.counter(kGatherFrom + n, r.me), r.base(kGatherFrom + n) + r.lsa * a.ctas);
            __syncthreads();
            // a gathered row is the forwarded record in its place: its token from the record, its destination from the
            // counts before it
            const int* counts = reinterpret_cast<const int*>(r.local(r.fwd(n)));
            for (int q = threadIdx.x; q < r.fwd_rows(n); q += kThreads) {
                int j = 0;
                for (int before = 0; before + counts[j] <= q; ++j) before += counts[j];
                const char* record = r.local(r.fwd(n) + head_bytes(r.lsa) + size_t(q) * rb);
                const int* meta = reinterpret_cast<const int*>(record + rowb);
                rows_of[meta[0]][r.slot_here(meta + 1, j)] = int16_t(q);
            }
            __syncthreads();
            // each token's place among those sent here (kThreads >= kMaxT: a thread per token)
            const int t = threadIdx.x;
            int sent = 0;
            for (int j = 0; t < a.t && j < r.lsa; ++j) sent |= rows_of[t][j] >= 0;
            const int incl = inclusive_scan(sent, r.lane);
            if (r.lane == 31) sums[r.warp] = incl;
            __syncthreads();
            int before = 0, total = 0;
            for (int w = 0; w < kWarps; ++w) before += w < r.warp ? sums[w] : 0, total += sums[w];
            if (t < a.t) place[t] = sent ? before + incl - sent : -1;
            __syncthreads();
            for (int u = member * kWarps + r.warp; u < a.t; u += r.members(k) * kWarps) {
                if (place[u] < 0) continue;
                char* out = r.local(r.partial(k) + size_t(place[u]) * r.ps);
                for (int g0 = 0; g0 < segs; g0 += 32) {
                    const int g = g0 + r.lane;
                    int4 v[kMaxLsa];
#pragma unroll
                    for (int j = 0; j < kMaxLsa; ++j)
                        if (j < r.lsa && rows_of[u][j] >= 0 && g < segs)
                            v[j] = *reinterpret_cast<const int4*>(r.local(r.gather(n) + rows_of[u][j] * r.yb) + g * 16);
                    Slots slots;
#pragma unroll
                    for (int j = 0; j < kMaxLsa; ++j)
                        if (j < r.lsa && rows_of[u][j] >= 0 && g < segs) slots.add(v[j]);
                    const int4 part = slots.sum();
                    quantize(out, g, a.width, part, g < segs);
                }
            }
            if (r.complete(kReducedFor + n, r.members(k)) && r.warp == 0) {
                r.put(r.at(n, r.l), r.comb(r.node), r.partial(k), size_t(total) * r.ps, kCombFrom + r.node);
                r.mark(kCPut + k);
            }
            __syncthreads();
        }
    }
    const size_t landing = kFp8 ? r.ps : r.yb;  // bytes of a row (partial) another node returns
    if (threadIdx.x == 0) {
        for (int k = 0; k < r.nodes - 1; ++k) {
            const int n = r.remote(k);
            while (!r.landed(kCombFrom + n, a.node_rows[n] * landing)) {}
        }
        wait_ge(r.counter(kCombArrive, r.me), r.base(kCombArrive) + r.lsa * a.ctas);
    }
    __syncthreads();
    r.mark(kCArrived);

    // Sum each token's rows as DeepEP's, one 512-byte chunk of a token per warp: comb_rows in its slot order, each
    // node's rows into a bf16 partial and the partials into the output, both by Slots. With fp8, as deepep_fp8c's
    // fp8_scaleout_reduce: this node's rows by Slots, another node's FP8 partial (its token's place there in its
    // block, at row n * lsa * t + place) dequantized, the partials in fp32 from 0 whatever their number.
    const char* comb = r.local(r.comb(0));
    const int chunks = int((r.yb + 511) / 512), block = r.lsa * a.t;
    for (int item = r.gw; item < a.n * chunks; item += r.gws) {
        const int t = item / chunks;
        const size_t off = size_t(item % chunks) * 512 + r.lane * 16;
        int rows[kMaxTopk];
        int4 v[kMaxTopk];
        float scale[kMaxTopk];
#pragma unroll
        for (int i = 0; i < kMaxTopk; ++i) rows[i] = i < a.k ? a.comb_rows[t * a.k + i] : -1;
#pragma unroll
        for (int i = 0; i < kMaxTopk; ++i) {
            if (rows[i] < 0 || off >= r.yb) continue;
            const int n = rows[i] / block;
            if (kFp8 && n != r.node) {  // 8 FP8 values and their block's scale
                const char* row = comb + size_t(n) * block * r.yb + size_t(rows[i] - n * block) * r.ps;
                const int2 q = *reinterpret_cast<const int2*>(row + off / 2);
                v[i].x = q.x, v[i].y = q.y, scale[i] = reinterpret_cast<const float*>(row + a.width)[off / 2 / 128];
            } else {
                v[i] = *reinterpret_cast<const int4*>(comb + rows[i] * r.yb + off);
            }
        }
        int4 sum;
        if constexpr (kFp8) {
            float acc[8] = {};
            Slots here;
#pragma unroll
            for (int i = 0; i < kMaxTopk; ++i) {
                if (rows[i] < 0) break;
                if (rows[i] / block != r.node) {
                    dequantize_add(acc, v[i], scale[i]);
                    continue;
                }
                here.add(v[i]);
                if (i + 1 == kMaxTopk || rows[i + 1] < 0 || rows[i + 1] / block != r.node) accumulate(acc, here.sum());
            }
            sum = to_bf16(acc);
        } else {
            Slots node, nodes;
#pragma unroll
            for (int i = 0; i < kMaxTopk; ++i) {
                if (rows[i] < 0) break;
                node.add(v[i]);
                if (i + 1 == kMaxTopk || rows[i + 1] < 0 || rows[i + 1] / block != rows[i] / block) {
                    nodes.add(node.sum());
                    node = Slots{};
                }
            }
            sum = nodes.sum();
        }
        if (off < r.yb) *reinterpret_cast<int4*>(reinterpret_cast<char*>(a.out + size_t(t) * a.width) + off) = sum;
    }
    r.finish(
        [&] {
            for (int k = 0; k < r.nodes - 1; ++k) {
                const int n = r.remote(k);
                r.base(kGatherFrom + n) += r.lsa * a.ctas;
                r.advance_landed(kCombFrom + n, a.node_rows[n] * landing);
                if (kFp8) r.base(kReducedFor + n) += r.members(k);
            }
            r.base(kCombArrive) += r.lsa * a.ctas;
        },
        kCEnd);
}

__global__ void __launch_bounds__(kThreads, 1) allgather_kernel(const Args a) {
    const Rank r(a);
    __shared__ int next;
    const size_t block = a.t * a.row, buffer = a.at.ag + (r.base(kAgCalls) & 1) * a.at.ag_buffer;
    const auto slot = [&](int s) { return buffer + s * block; };
    // this rank's block into its slot, then (the last CTA to finish its share) to every rail peer, a warp each
    for (int t = r.gw; t < a.t; t += r.gws) r.copy(r.local(slot(r.me) + t * a.row), a.in + t * a.row, a.row);
    if (r.complete(kAgBuilt, a.ctas) && r.warp < r.nodes - 1)
        r.put(r.at(r.remote(r.warp), r.l), slot(r.me), slot(r.me), block, kAgFrom + r.node);
    // to the node-mates
    for (int item = r.gw; item < (r.lsa - 1) * a.t; item += r.gws) {
        const int j = item / a.t + (item / a.t >= r.l), t = item % a.t;
        r.copy(r.peer(slot(r.me) + t * a.row, r.at(r.node, j)), a.in + t * a.row, a.row);
    }
    // each rail peer's block to the node-mates, as it lands
    for (unsigned todo = r.remotes(); todo != 0;) {
        if (threadIdx.x == 0) next = pick(todo, [&](int k) { return r.landed(kAgFrom + r.remote(k), block); });
        __syncthreads();
        const int k = next, src = r.at(r.remote(k), r.l);
        todo &= ~(1u << k);
        for (int item = r.gw; item < (r.lsa - 1) * a.t; item += r.gws) {
            const int j = item / a.t + (item / a.t >= r.l), t = item % a.t;
            r.copy(r.peer(slot(src) + t * a.row, r.at(r.node, j)), r.local(slot(src) + t * a.row), a.row);
        }
        __syncthreads();
    }
    r.signal_node(kAgArrive, r.nodes, false);
    if (threadIdx.x == 0)
        wait_ge(r.counter(kAgArrive, r.me), r.base(kAgArrive) + uint64_t(r.lsa - 1) * r.nodes * a.ctas);
    __syncthreads();
    const size_t bytes = r.ranks * block;
    for (size_t off = size_t(r.gw) * 4096; off < bytes; off += size_t(r.gws) * 4096)
        r.copy(a.dst + off, r.local(slot(0) + off), min(size_t(4096), bytes - off));
    r.finish([&] {
        r.base(kAgBuilt) += a.ctas;
        r.base(kAgArrive) += uint64_t(r.lsa - 1) * r.nodes * a.ctas;
        for (int k = 0; k < r.nodes - 1; ++k) r.advance_landed(kAgFrom + r.remote(k), block);
        r.base(kAgCalls) += 1;
    });
}

// kFp8: a node's partials cross InfiniBand in FP8 (rounded to bf16 first, as the combine's), this node's in bf16
template <bool kFp8>
__global__ void __launch_bounds__(kThreads, 1) reducescatter_kernel(const Args a) {
    const Rank r(a);
    // a partial row's bytes between nodes: fp32, or FP8 values and scales
    const size_t parity = r.base(kRsCalls) & 1, fb = kFp8 ? r.ps : size_t(a.width) * 4;
    // Node-mates' blocks for this rank's rail destinations (local rank l, every node), and fp32 partials.
    const auto stage = [&](int j, int n) {
        return a.at.rs_stage + parity * a.at.rs_stage_buffer + (size_t(j) * r.nodes + n) * a.t * r.yb;
    };
    const auto send = [&](int n) { return a.at.rs_send + parity * a.at.rs_partial_buffer + n * a.t * fb; };
    const auto recv = [&](int n) { return a.at.rs_recv + parity * a.at.rs_partial_buffer + n * a.t * fb; };
    const int chunks = int(r.yb / 16);  // 8 bf16 per lane-chunk
    // The fp32 sum over local ranks of the staged blocks for node n, at row t and 16-byte chunk c.
    const auto node_sum = [&](int n, int t, int c, float* acc) {
        for (int e = 0; e < 8; ++e) acc[e] = 0;
#pragma unroll
        for (int j = 0; j < kMaxLsa; ++j)
            if (j < r.lsa) accumulate(acc, *reinterpret_cast<const int4*>(r.local(stage(j, n) + t * r.yb) + c * 16));
    };

    // every destination's block to its rail peer on this node
    for (int item = r.gw; item < r.ranks * a.t; item += r.gws) {
        const int d = item / a.t, t = item % a.t;
        r.copy(r.peer(stage(r.l, d / r.lsa) + t * r.yb, r.at(r.node, d % r.lsa)), a.in + size_t(item) * r.yb, r.yb);
    }
    r.signal_node(kRsArrive, 1, true);
    if (threadIdx.x == 0) wait_ge(r.counter(kRsArrive, r.me), r.base(kRsArrive) + r.lsa * a.ctas);
    __syncthreads();
    // each remote node's partial by its CTA group, sent by the group's last CTA to finish
    for (int k = blockIdx.x % r.groups(); k < r.nodes - 1; k += r.groups()) {
        const int n = r.remote(k), member = blockIdx.x / r.groups();
        // whole warps (quantize's shuffles): a row's chunks are a multiple of 16, a 128-value block per half-warp
        for (int first = (member * kWarps + r.warp) * 32; first < a.t * chunks; first += r.members(k) * kThreads) {
            const int item = first + r.lane;
            const bool live = item < a.t * chunks;
            float acc[8] = {};
            if (live) node_sum(n, item / chunks, item % chunks, acc);
            if constexpr (kFp8) {
                quantize(r.local(send(n) + size_t(item / chunks) * fb), item % chunks, a.width, to_bf16(acc), live);
            } else if (live) {
                auto* to = reinterpret_cast<float4*>(r.local(send(n) + size_t(item) * 32));
                to[0] = make_float4(acc[0], acc[1], acc[2], acc[3]);
                to[1] = make_float4(acc[4], acc[5], acc[6], acc[7]);
            }
        }
        if (r.complete(kSummedFor + n, r.members(k)) && r.warp == 0)
            r.put(r.at(n, r.l), recv(r.node), send(n), a.t * fb, kRsFrom + r.node);
    }
    if (threadIdx.x == 0)
        for (int k = 0; k < r.nodes - 1; ++k)
            while (!r.landed(kRsFrom + r.remote(k), a.t * fb)) {}
    __syncthreads();
    // the nodes' partials in node order, this node's from the staged blocks
    for (int item = r.gw * 32 + r.lane; item < a.t * chunks; item += r.gws * 32) {
        float acc[8] = {}, part[8];
        for (int n = 0; n < r.nodes; ++n) {
            if (kFp8 && n != r.node) {
                const char* row = r.local(recv(n) + size_t(item / chunks) * fb);
                const int2 q = *reinterpret_cast<const int2*>(row + (item % chunks) * 8);
                dequantize_add(acc, make_int4(q.x, q.y, 0, 0),
                               reinterpret_cast<const float*>(row + a.width)[item % chunks / 16]);
            } else if (kFp8) {
                node_sum(n, item / chunks, item % chunks, part);
                accumulate(acc, to_bf16(part));
            } else {
                if (n == r.node) {
                    node_sum(n, item / chunks, item % chunks, part);
                } else {
                    const auto* from = reinterpret_cast<const float4*>(r.local(recv(n) + size_t(item) * 32));
                    const float4 p0 = from[0], p1 = from[1];
                    part[0] = p0.x, part[1] = p0.y, part[2] = p0.z, part[3] = p0.w;
                    part[4] = p1.x, part[5] = p1.y, part[6] = p1.z, part[7] = p1.w;
                }
                for (int e = 0; e < 8; ++e) acc[e] += part[e];
            }
        }
        *reinterpret_cast<int4*>(a.dst + size_t(item) * 16) = to_bf16(acc);
    }
    r.finish([&] {
        r.base(kRsArrive) += r.lsa * a.ctas;
        for (int k = 0; k < r.nodes - 1; ++k) {
            r.base(kSummedFor + r.remote(k)) += r.members(k);
            r.advance_landed(kRsFrom + r.remote(k), a.t * fb);
        }
        r.base(kRsCalls) += 1;
    });
}

struct Rail {
    ncclComm_t comm;
    ncclDevComm dev;
    ncclWindow_t win;
    void* buf;
    Layout at;
    uint64_t* bases;  // kMaxContexts * kNumIds bases, then the exit counter and the last dispatch's row bytes
    int k, experts_per_rank, contexts, max_width;
    bool fp8;
};

Args args(const Rail& rail, int t, int width, int ctas) {
    Args a{};
    a.comm = rail.dev, a.win = rail.win, a.at = rail.at;
    a.t = t, a.k = rail.k, a.width = width, a.experts_per_rank = rail.experts_per_rank, a.ctas = ctas;
    a.contexts = rail.contexts, a.fp8 = rail.fp8, a.reduce = 1, a.vbytes = 2 * width;
    a.bases = rail.bases, a.exits = reinterpret_cast<unsigned*>(rail.bases + kMaxContexts * kNumIds);
    a.last_rowb = reinterpret_cast<int*>(rail.bases + kMaxContexts * kNumIds + 1);
    return a;
}

// Cooperative, so every CTA is resident: they wait on each other's and other GPUs' progress.
void launch(void (*kernel)(Args), const Args& a, cudaStream_t stream) {
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(a.ctas), config.blockDim = dim3(kThreads), config.stream = stream;
    cudaLaunchAttribute cooperative;
    cooperative.id = cudaLaunchAttributeCooperative, cooperative.val.cooperative = 1;
    config.attrs = &cooperative, config.numAttrs = 1;
    CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, a));
}

}  // namespace

int64_t rail_create(int64_t comm, int max_t, int max_width, int k, int experts, int contexts, int gather_bytes,
                    int reduce_width, bool fp8) {
    auto* rail = new Rail{};
    rail->comm = reinterpret_cast<ncclComm_t>(comm);
    ncclCommProperties props = NCCL_COMM_PROPERTIES_INITIALIZER;
    NCCL_CHECK(ncclCommQueryProperties(rail->comm, &props));
    if (props.nRanks > kMaxRanks || max_t > kMaxT || k > kMaxTopk || contexts < 1 || contexts > kMaxContexts)
        throw std::runtime_error("rail_moe takes at most 32 ranks, 512 tokens per rank, top-8 and 8 GIN contexts");

    // As DeepEP's setup: exclusive contexts, 1024-deep queues, traffic class 3. GIN only reaches rail peers, so only
    // they are connected (a stride of the NVLink domain size, as DeepEP's hybrid mode here): NCCL's CPU proxy then
    // walks 4 QPs per context and pass instead of 32.
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_CUSTOM_STRIDE;
    reqs.ginCustomStride = ncclTeamLsa(rail->comm).nRanks;
    reqs.ginContextCount = contexts;
    reqs.ginSignalCount = kNumIds;
    reqs.ginExclusiveContexts = true;
    reqs.ginQueueDepth = 1024;
    reqs.ginTrafficClass = 3;
    NCCL_CHECK(ncclDevCommCreate(rail->comm, &reqs, &rail->dev));
    const int ranks = props.nRanks, lsa = rail->dev.lsaSize, nodes = ranks / lsa;
    if (nodes > kMaxNodes) throw std::runtime_error("rail_moe takes at most 8 nodes");
    if (lsa > kMaxLsa) throw std::runtime_error("rail_moe takes at most 8 GPUs per node");
    if (int(rail->dev.ginContextCount) < contexts)
        throw std::runtime_error("NCCL created " + std::to_string(rail->dev.ginContextCount) + " GIN contexts");

    const size_t rb = size_t(max_width) * 2 + meta_bytes(k), yb = size_t(max_width) * 2;
    const size_t cb = size_t(reduce_width) * 2;
    size_t bytes = 0;
    const auto region = [&](size_t size) {
        const size_t off = bytes;
        bytes += (size + 4095) / 4096 * 4096;
        return off;
    };
    Layout& at = rail->at;
    at.node_bundle = (head_bytes(lsa) + size_t(lsa) * max_t * rb + 15) / 16 * 16;
    at.heads = region(16 * ranks);
    at.stage = region(size_t(ranks) * max_t * rb);
    at.fwd = region(nodes * at.node_bundle);
    at.bundle = region(nodes * at.node_bundle);
    at.gather = region(size_t(nodes) * lsa * max_t * yb);
    at.comb = region(size_t(nodes) * lsa * max_t * yb);
    at.partial = region(fp8 ? size_t(nodes - 1) * max_t * yb : 0);
    at.ag_buffer = size_t(ranks) * max_t * gather_bytes;
    at.ag = region(2 * at.ag_buffer);
    at.rs_stage_buffer = size_t(lsa) * nodes * max_t * cb;
    at.rs_stage = region(2 * at.rs_stage_buffer);
    at.rs_partial_buffer = size_t(nodes) * max_t * cb * 2;
    at.rs_send = region(2 * at.rs_partial_buffer);
    at.rs_recv = region(2 * at.rs_partial_buffer);
    at.counters = region(8 * kNumIds);
    NCCL_CHECK(ncclMemAlloc(&rail->buf, bytes));
    CUDA_CHECK(cudaMemset(rail->buf, 0, bytes));
    NCCL_CHECK(ncclCommWindowRegister(rail->comm, rail->buf, bytes, &rail->win, NCCL_WIN_STRICT_ORDERING));
    CUDA_CHECK(cudaMalloc(&rail->bases, 8 * (kMaxContexts * kNumIds + 2)));
    CUDA_CHECK(cudaMemset(rail->bases, 0, 8 * (kMaxContexts * kNumIds + 2)));
    rail->k = k, rail->experts_per_rank = experts / ranks, rail->contexts = contexts, rail->max_width = max_width;
    rail->fp8 = fp8;
    return reinterpret_cast<int64_t>(rail);
}

void rail_dispatch(int64_t rail, const void* x, const float* xs, const int64_t* topk_idx, const float* topk_weights,
                   int n, int vbytes, int sbytes, int t, int ctas, void* recv_x, float* recv_xs, int64_t* recv_idx,
                   float* recv_weights, int* recv_src, int* psum, int* comb_rows, int* node_rows,
                   cudaStream_t stream) {
    const Rail& r = *reinterpret_cast<Rail*>(rail);
    if (vbytes % 16 != 0 || sbytes % 4 != 0 || vbytes + align16(sbytes) > 2 * size_t(r.max_width))
        throw std::runtime_error("rail_moe: rows of 16-byte multiples (with their scales) within width_max");
    Args a = args(r, t, 0, ctas);
    a.vbytes = vbytes, a.sbytes = sbytes, a.xs = xs, a.recv_xs = recv_xs;
    a.n = n, a.x = x, a.topk_idx = topk_idx, a.topk_weights = topk_weights;
    a.recv_x = recv_x, a.recv_idx = recv_idx, a.recv_weights = recv_weights;
    a.recv_src = recv_src, a.psum = psum, a.comb_rows = comb_rows, a.node_rows = node_rows;
    launch(dispatch_kernel, a, stream);
}

void rail_combine(int64_t rail, const void* y, int* psum, int* comb_rows, int* node_rows, int n, int width, int t,
                  int ctas, void* out, cudaStream_t stream) {
    const Rail& r = *reinterpret_cast<Rail*>(rail);
    if (r.fp8 && width % 128 != 0) throw std::runtime_error("rail_moe: an FP8 combine takes rows of 128-value blocks");
    Args a = args(r, t, width, ctas);
    a.n = n, a.y = static_cast<const __nv_bfloat16*>(y), a.psum = psum, a.comb_rows = comb_rows;
    a.node_rows = node_rows, a.out = static_cast<__nv_bfloat16*>(out);
    launch(a.fp8 ? combine_kernel<true> : combine_kernel<false>, a, stream);
}

void rail_all_gather(int64_t rail, const void* in, void* out, int t, int64_t row, int ctas, cudaStream_t stream) {
    Args a = args(*reinterpret_cast<Rail*>(rail), t, 0, ctas);
    a.row = row, a.in = static_cast<const char*>(in), a.dst = static_cast<char*>(out);
    launch(allgather_kernel, a, stream);
}

void rail_reduce_scatter(int64_t rail, const void* in, void* out, int t, int width, int ctas, bool fp8,
                         cudaStream_t stream) {
    if (fp8 && width % 128 != 0)
        throw std::runtime_error("rail_moe: an FP8 reduce-scatter takes rows of 128-value blocks");
    Args a = args(*reinterpret_cast<Rail*>(rail), t, width, ctas);
    a.in = static_cast<const char*>(in), a.dst = static_cast<char*>(out);
    launch(fp8 ? reducescatter_kernel<true> : reducescatter_kernel<false>, a, stream);
}

void rail_destroy(int64_t handle) {
    auto* rail = reinterpret_cast<Rail*>(handle);
    NCCL_CHECK(ncclCommWindowDeregister(rail->comm, rail->win));
    NCCL_CHECK(ncclMemFree(rail->buf));
    NCCL_CHECK(ncclDevCommDestroy(rail->comm, &rail->dev));
    CUDA_CHECK(cudaFree(rail->bases));
    delete rail;
}
