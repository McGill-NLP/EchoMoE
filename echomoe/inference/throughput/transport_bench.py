"""Per-call transport times at EP32 on the 4-node cluster: the tables selection.Costs reads
(configs/inference/throughput/costs-*.csv) and DeepEP's time against row width. A row times one
call shape on one backend: a MoE dispatch and combine on DeepEP v2's hybrid mode (strategy
deepep-hybrid, at an SM count), on the two-hop kernel (rail) or on the streaming kernel (stream,
both at a CTA count, rails.py), or one reference-layer collective on NCCL (CTAs 0) or the two-hop
kernel (nccl- and rail-allgather and -reducescatter, the time under dispatch_us).
Calls run as the throughput runs make them: token limits up to 512 (decode steps) replayed from
CUDA graphs, larger ones eagerly, and the collectives 10 calls to a graph. Every rank starts each
call together, a row holds the slowest rank's median of 20, and the routes are uniform random.
Rank 0 writes the rows to --out.

    torchrun --nnodes 4 --nproc-per-node 8 ... -m echomoe.inference.throughput.transport_bench \
        bf16 --out <csv> [--tokens 1,128] [--widths 512,2048]
    # the same with: fp8 --out <csv> [--tokens 1,2,4,8]
    #                larger --model deepseekv3 --precision fp8-dispatch --out <csv> [--tokens ...]
    #                deepep-scaling --out <csv> [--tokens ...]
    python -m echomoe.inference.throughput.transport_bench finish <table csv> <bench csv>...

bf16 and fp8 take `--capped 2x` or `--capped 2.5x`: each inter-node connection capped to the rate
capped.yaml sets for NCCL's, DeepEP's and each kernel's connections, through the NCCL build of
`scripts/install.sh nccl`, which the caller preloads as the throughput launcher does:

    LD_PRELOAD=third-party/nccl/build/lib/libnccl.so.2 \
        LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<CUDA 12.8>/lib torchrun ... bf16 --capped 2x --out <csv>

bf16: Qwen3-30B-A3B's BF16 calls. fp8: its FP8 calls (FP8 dispatch, FP8 between nodes in every
combine). larger: a larger model's calls in one precision. deepep-scaling: DeepEP alone at the
larger models' widths with Qwen3-30B-A3B's routing. finish joins bench outputs in order into a
table: costs-bf16 from bf16, costs-fp8 from bf16 then fp8 (run in chunks of token limits), and
costs-<model>-<precision> from larger. costs-capped-2x and costs-capped-2.5x are bf16 then fp8
with --capped 2x and 2.5x.
"""

import argparse
import csv
import os
import socket
import statistics
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from echomoe.runs import REPO

from .kernels import fp8_cast, reduce_scatter_fp8_nodes, row_layout
from .rails import RailBuffer, StreamBuffer

# the capped networks' rates and DeepEP's network contexts
CAPPED = REPO / "configs/inference/throughput/capped.yaml"
# Qwen3-30B-A3B's experts, top-k and hidden size, and its tables' token limits and row widths
EXPERTS, TOPK, HIDDEN = 128, 8, 2048
TOKENS = [1 << i for i in range(13)]
WIDTHS = [256, 512, 768, 1024, 1280, 2048]
SCALING_WIDTHS = [1024, 1792, 2816, 3584, 4608, 7168]
# name: experts, (routing groups, groups drawn per token), top-k, the widths its calls move (the
# base model's hidden size and the folded student's dispatch and combine ranks, as they travel:
# Gemma 4's and Nemotron's combine codes padded from 1408 to 1536, Nemotron's rows from 2688 to
# 2816), and its largest token limit (max_num_batched_tokens)
LARGER_MODELS = {
    "qwen235": (128, (1, 1), 8, [1024, 2048, 4096], 2048),
    "deepseekv3": (256, (8, 4), 8, [1792, 3584, 7168], 2048),
    "kimi26": (384, (1, 1), 8, [1792, 3584, 7168], 2048),
    "kanana": (128, (1, 1), 6, [512, 1024, 2048], 4096),
    "gemma4": (128, (1, 1), 8, [768, 1536, 2816], 4096),
    "moonlight": (64, (1, 1), 6, [512, 1024, 2048], 4096),
    "nemotron": (128, (1, 1), 6, [768, 1536, 2816], 4096),
}
# DeepEP's network contexts (QPs) per buffer in the larger models' runs (deepep_contexts)
LARGER_QPS = 65
# Decode token limits: captured in CUDA graphs, and the only limits at which the two-hop kernel,
# DeepEP at 2 and 8 SMs and the reference collectives are timed.
DECODE_TOKENS = 512
STREAM_CHUNK, STREAM_CONTEXTS = 256 << 10, 2
WARMUP, ITERS, GRAPH_CALLS = 3, 20, 10
COLUMNS = [
    "strategy",
    "precision",
    "sms",
    "tokens",
    "width",
    "dispatch_us",
    "combine_us",
    "chunk",
    "contexts",
]


def integers(text):
    return [int(value) for value in text.split(",")]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    tables = {
        name: commands.add_parser(name) for name in ("bf16", "fp8", "larger", "deepep-scaling")
    }
    for table in tables.values():
        table.add_argument("--out", type=Path, required=True)
        table.add_argument("--tokens", type=integers, help="token limits (default: the table's)")
    tables["bf16"].add_argument("--widths", type=integers, default=WIDTHS)
    tables["bf16"].set_defaults(precision="bf16")
    tables["fp8"].set_defaults(widths=WIDTHS, precision="fp8")
    tables["deepep-scaling"].set_defaults(widths=SCALING_WIDTHS, precision="bf16", capped=None)
    capped = yaml.safe_load(CAPPED.read_text())
    for name in ("bf16", "fp8"):
        tables[name].add_argument("--capped", choices=tuple(capped), help="a capped network")
    tables["larger"].add_argument("--model", choices=LARGER_MODELS, required=True)
    tables["larger"].add_argument(
        "--precision", choices=("bf16", "fp8", "fp8-dispatch"), required=True
    )
    finish = commands.add_parser("finish")
    finish.add_argument("out", type=Path)
    finish.add_argument("benches", type=Path, nargs="+")
    args = parser.parse_args(argv)
    if args.command == "finish":
        return args
    if args.command == "larger":
        model = LARGER_MODELS[args.model]
        args.experts, args.groups, args.topk, args.widths, args.capacity = model
        args.network, args.qps = {}, LARGER_QPS
    else:
        args.experts, args.groups, args.topk, args.capacity = EXPERTS, (1, 1), TOPK, TOKENS[-1]
        args.network = capped[args.capped] if args.capped else {}
        args.qps = args.network.get("deepep_qps", 0)  # 0: DeepEP's default
    args.tokens = args.tokens or [limit for limit in TOKENS if limit <= args.capacity]
    return args


def calls(args, auto):
    """(strategy, SMs or CTAs, token limit, width) of each row the table times, in table order,
    with `auto` DeepEP's own SM count."""
    if args.command in ("bf16", "deepep-scaling"):
        return bf16_calls(args.tokens, args.widths, auto, kernels=args.command == "bf16")
    return pair_calls(args.tokens, args.widths, auto, fp8=args.command == "fp8")


def bf16_calls(tokens, widths, auto, kernels):
    for limit in tokens:
        decode = limit <= DECODE_TOKENS
        for width in widths:
            for sms in [*((2, 8) if decode else ()), 16, 32, 64, auto]:
                yield "deepep-hybrid", sms, limit, width
            if kernels and decode:
                for ctas in (16, 32, 64):
                    yield "rail", ctas, limit, width
            if kernels and limit >= 256:
                for ctas in (33, 65):
                    yield "stream", ctas, limit, width
        if kernels and decode:
            gather = row_layout(HIDDEN, TOPK)["row_bytes"]
            for name, sizes in (("allgather", [gather]), ("reducescatter", widths)):
                for size in sizes:
                    for ctas in (0, 16, 32, 64):
                        yield ("rail-" if ctas else "nccl-") + name, ctas, limit, size


def pair_calls(tokens, widths, auto, fp8):
    suffix = "-fp8" if fp8 else ""
    for limit in tokens:
        decode = limit <= DECODE_TOKENS
        for width in widths:
            if fp8 and decode:
                yield "nccl-reducescatter-fp8", 0, limit, width
                for ctas in (32, 64):
                    yield "rail-reducescatter-fp8", ctas, limit, width
            for sms in [*((2, 8) if decode else ()), 16, 32, auto]:
                yield "deepep-hybrid", sms, limit, width
            if decode:
                for ctas in (32, 64):
                    yield "rail" + suffix, ctas, limit, width
            for ctas in (33, 65):
                yield "stream" + suffix, ctas, limit, width


def routes(tokens, experts, groups, topk, generator):
    """Uniform random top-k: distinct experts from the routing groups drawn for each token
    (DeepSeek-V3: 4 of 8), and random weights."""
    count, chosen = groups
    size = experts // count
    drawn = torch.rand(tokens, count, generator=generator, device="cuda")
    picked = torch.argsort(drawn, dim=1)[:, :chosen]
    pool = (picked[:, :, None] * size + torch.arange(size, device="cuda")).flatten(1)
    order = torch.argsort(torch.rand(pool.shape, generator=generator, device="cuda"), dim=1)
    ids = torch.gather(pool, 1, order[:, :topk]).contiguous()
    return ids, torch.rand(tokens, topk, generator=generator, device="cuda")


def pair_times(dispatch, combine, graphed, flag):
    """Median GPU times (us) of `dispatch` and `combine` run back to back, replayed from CUDA
    graphs if `graphed`, every rank starting each run together."""
    for _ in range(WARMUP):
        dispatch()
        combine()
    torch.cuda.synchronize()
    if graphed:
        graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
        for graph, call in zip(graphs, (dispatch, combine), strict=True):
            with torch.cuda.graph(graph):
                call()
        dispatch, combine = graphs[0].replay, graphs[1].replay
    times = ([], [])
    for _ in range(ITERS):
        dist.all_reduce(flag)
        events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        events[0].record()
        dispatch()
        events[1].record()
        combine()
        events[2].record()
        torch.cuda.synchronize()
        times[0].append(1000 * events[0].elapsed_time(events[1]))
        times[1].append(1000 * events[1].elapsed_time(events[2]))
    return [statistics.median(samples) for samples in times]


def call_time(call, flag, graphs):
    """Median GPU time (us) per call of `call` captured GRAPH_CALLS times in one CUDA graph, as
    decode steps hold many collectives in one graph. The graph stays in `graphs`: with each freed
    after timing, the next FP8 reduce-scatter's all-to-all over the rail group failed
    (IBV_WC_LOC_PROT_ERR)."""
    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()
    graphs.append(torch.cuda.CUDAGraph())
    with torch.cuda.graph(graphs[-1]):
        for _ in range(GRAPH_CALLS):
            call()
    times = []
    for _ in range(ITERS):
        dist.all_reduce(flag)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graphs[-1].replay()
        end.record()
        torch.cuda.synchronize()
        times.append(1000 * start.elapsed_time(end) / GRAPH_CALLS)
    return statistics.median(times)


def deepep_ops(buffer, payload, ids, weights, limit, width, experts, sms):
    """A dispatch at token limit `limit` on `sms` SMs, and the combine of its received rows as
    `width`-wide BF16 codes."""
    state = {}

    def dispatch():
        received, _, _, state["handle"], _ = buffer.dispatch(
            payload,
            topk_idx=ids,
            topk_weights=weights,
            num_experts=experts,
            num_max_tokens_per_rank=limit,
            do_expand=False,
            do_cpu_sync=False,
            num_sms=sms,
        )
        rows = (received[0] if isinstance(received, tuple) else received).shape[0]
        if "codes" not in state or state["codes"].shape[0] != rows:
            state["codes"] = torch.zeros(rows, width, device="cuda", dtype=torch.bfloat16)

    def combine():
        buffer.combine(state["codes"], handle=state["handle"], topk_weights=None)

    return dispatch, combine


def kernel_ops(buffer, payload, ids, weights, limit, width, ctas):
    """The same pair on the two-hop or the streaming kernel at `ctas` CTAs."""
    codes = torch.zeros(buffer.ranks * limit, width, device="cuda", dtype=torch.bfloat16)
    state = {}

    def dispatch():
        state["handle"] = buffer.dispatch(payload, ids, weights, limit, ctas)[3]

    def combine():
        buffer.combine(codes, state["handle"], ctas)

    return dispatch, combine


def collective(strategy, ctas, limit, width, two_hop, generator):
    """One call of a reference collective at token limit `limit`: the all-gather of `width`-byte
    rows or the reduce-scatter of `width`-wide codes, on NCCL (ctas 0) or the two-hop kernel."""
    world = dist.get_world_size()
    if strategy.endswith("allgather"):
        rows = torch.randint(
            0, 256, (limit, width), generator=generator, device="cuda", dtype=torch.uint8
        )
        out = torch.empty(world * limit, width, device="cuda", dtype=torch.uint8)
        if ctas:
            return lambda: two_hop.all_gather(out, rows, ctas)
        return lambda: dist.all_gather_into_tensor(out, rows)
    fp8 = strategy.endswith("-fp8")
    codes = torch.randn(world * limit, width, generator=generator, device="cuda").to(
        torch.bfloat16
    )
    out = torch.empty(limit, width, device="cuda", dtype=torch.bfloat16)
    if ctas:
        return lambda: two_hop.reduce_scatter(out, codes, ctas, fp8)
    if fp8:
        return lambda: reduce_scatter_fp8_nodes(codes, dist.group.WORLD, limit)
    return lambda: dist.reduce_scatter_tensor(out, codes)


def finish(out, benches):
    """A table from bench outputs in order: the streaming rows marked reduce=1 (the kernels combine
    as DeepEP does, rails.py), and the two-hop kernel's reduce-scatter priced out, since the
    reference layers' sums must keep NCCL's bits."""
    rows = []
    for path in benches:
        with open(path, newline="") as bench:
            rows += csv.DictReader(bench)
    for row in rows:
        row["reduce"] = "1" if row["strategy"].startswith("stream") else ""
        if row["strategy"].startswith("rail-reducescatter"):
            row["dispatch_us"] = str(float(row["dispatch_us"]) * 1e6)
    with open(out, "w", newline="") as bench:
        writer = csv.DictWriter(bench, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(len(rows), "rows")


def main():
    args = parse_args()
    if args.command == "finish":
        finish(args.out, args.benches)
        return
    if args.precision == "fp8":
        # The copy of DeepEP whose combine crosses nodes in FP8 (scripts/install.sh), with its own
        # kernel cache. Both must be set before DeepEP is imported.
        sys.path.insert(0, os.path.join(sys.prefix, "deepep-fp8c"))
        os.environ["EP_JIT_CACHE_DIR"] = os.path.join(
            os.environ["ECHOMOE_REPO"], "outputs/deepep-jit-cache-fp8c", socket.gethostname()
        )
    # a Triton cache per rank: ranks compiling into one on the shared filesystem race
    os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix=f"triton-rank{os.environ['RANK']}-")
    if args.network:
        # DeepEP, which checks at import that the NCCL loaded is its own, accepts the preloaded
        # build. NCCL's connections and DeepEP's take the process's rates.
        os.environ.update(
            EP_SUPPRESS_NCCL_CHECK="1",
            NCCL_IB_STATIC_RATE=str(args.network["nccl_ib_rate"]),
            NCCL_GIN_STATIC_RATE=str(args.network["deepep_gin_rate"]),
        )
    import deep_ep

    from .fairness import install_nic_rate
    from .selection import gin_rate

    if "NCCL_IB_HCA" in os.environ:
        install_nic_rate()

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank, group = dist.get_rank(), dist.group.WORLD
    fp8_combine, widest = args.precision == "fp8", max(args.widths)
    # ProcessGroupNCCL creates its communicator lazily, and DeepEP reads it at construction
    dist.all_reduce(torch.zeros(1, device="cuda"))
    deepep = deep_ep.ElasticBuffer(
        group,
        num_max_tokens_per_rank=args.capacity,
        hidden=widest,
        num_topk=args.topk,
        use_fp8_dispatch=args.precision != "bf16",
        allow_hybrid_mode=True,
        allow_multiple_reduction=True,
        prefer_overlap_with_compute=False,
        explicitly_destroy=True,
        num_allocated_qps=args.qps,
    )
    auto = deepep.get_theoretical_num_sms(num_experts=args.experts, num_topk=args.topk)
    kernels = {}
    if args.command != "deepep-scaling":
        with gin_rate(args.network.get("rail_gin_rate")):
            kernels["rail"] = RailBuffer(
                group,
                DECODE_TOKENS,
                widest,
                args.topk,
                args.experts,
                num_contexts=1,
                gather_bytes=row_layout(HIDDEN, TOPK)["row_bytes"]
                if args.command == "bf16"
                else 0,
                reduce_width=widest if args.command in ("bf16", "fp8") else 0,
                fp8_combine=fp8_combine,
            )
        with gin_rate(args.network.get("stream_gin_rate")):
            kernels["stream"] = StreamBuffer(
                group,
                args.capacity,
                widest,
                args.topk,
                args.experts,
                num_contexts=STREAM_CONTEXTS,
                chunk_bytes=STREAM_CHUNK,
                fp8_combine=fp8_combine,
            )

    flag = torch.zeros(1, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(1234 + rank)
    rows, graphs, routed = [], [], None
    for strategy, sms, limit, width in calls(args, auto):
        if limit != routed:
            ids, weights = routes(limit, args.experts, args.groups, args.topk, generator)
            routed = limit
        kernel = strategy.removesuffix("-fp8")
        if kernel in ("deepep-hybrid", "rail", "stream"):
            activations = torch.randn(limit, width, generator=generator, device="cuda").to(
                torch.bfloat16
            )
            payload = activations if args.precision == "bf16" else fp8_cast(activations)
            if kernel == "deepep-hybrid":
                ops = deepep_ops(deepep, payload, ids, weights, limit, width, args.experts, sms)
            else:
                ops = kernel_ops(kernels[kernel], payload, ids, weights, limit, width, sms)
            times = pair_times(*ops, limit <= DECODE_TOKENS, flag)
        else:
            call = collective(strategy, sms, limit, width, kernels["rail"], generator)
            times = [call_time(call, flag, graphs), 0]
        slowest = torch.tensor(times, device="cuda")
        dist.all_reduce(slowest, op=dist.ReduceOp.MAX)
        if rank == 0:
            dispatch_us, combine_us = (round(value, 1) for value in slowest.tolist())
            stream = (STREAM_CHUNK, STREAM_CONTEXTS) if kernel == "stream" else ("", "")
            rows.append(
                [strategy, args.precision, sms, limit, width, dispatch_us, combine_us, *stream]
            )
            print(*rows[-1], sep=",", flush=True)
    if rank == 0:
        with open(args.out, "w", newline="") as table:
            writer = csv.writer(table)
            writer.writerow(COLUMNS)
            writer.writerows(rows)
    dist.barrier()
    graphs.clear()  # with the captured all-to-alls alive, the teardown hung
    torch.cuda.synchronize()
    for buffer in kernels.values():
        buffer.destroy()
    deepep.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
