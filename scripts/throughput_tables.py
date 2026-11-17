#!/usr/bin/env python3
"""Build the EP32 throughput tables of the arXiv report from the measured rows, all with balanced
routing (vLLM's routing simulator). Qwen3-30B-A3B runs every configuration and workload in BF16 and
FP8, the 4096/256 (8) workload over TCP on eth0, 4096/256 (128) on the per-connection-capped
networks, the concurrency sweep, and the agent and reasoning shapes. Then the larger models.

    python scripts/throughput_tables.py outputs results/throughput > throughput_tables.md

The records are the throughput CLI's, under `outputs/<model>/throughput/<setting>/<row>/{base,folded}/
rank<r>.json`: Qwen3-30B-A3B's rows are `<config>-<workload>` under the settings `bf16`, `fp8`,
`tcp-bf16`, `tcp-fp8` and `capped-{2x,2.5x}-{bf16,fp8}`, the larger models' and the trained
Kanana-2 and Gemma 4 students' rows are `<workload>` under their own model name. Speedup and times
are the mean of the two timed runs, and spread is the larger relative gap between a row's two timed
runs. Base does not depend on the configuration, so `speedup_over_mean_base` divides the mean base
time of the rows that differ only in configuration by the row's EchoMoE time, and the TTFT and
decode TPOT speedups (latency ratios, above 1 is faster) likewise. Writes `throughput.csv` and
prints the Markdown tables.

From the profile rows (`outputs/qwen3-30b-a3b/throughput/<setting>-profile/...`, the CLI's
`--profile`) it writes where a step's time goes, in ms per step and mean over ranks, to
`breakdowns.csv`: coarsely, per operation, and the GPU idle while the CPU is inside a decode step's
cudaGraphLaunch (the launch stall), which only the profiler adds.

From the online-serving launches (serving.py, records under
`outputs/qwen3-30b-a3b/serving/<setting>/<row>/<variant>/rate-<rate>-part<client>.json`) it writes
a row per request rate to `serving.csv` (Table 35, Figures 19 to 21) and prints Table 35.
"""

import csv
import gzip
import itertools
import json
import re
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import cache, partial
from pathlib import Path

from echomoe.communication_cost import cost_table
from echomoe.inference.throughput.serving import rate_summary, read_record
from echomoe.inference.throughput.summarize import summarize, tail_tpot_ms
from echomoe.runs import REPO

FANOUT = REPO / "results" / "communication" / "routing_fanout.json"
QWEN = "qwen3-30b-a3b"
# setting: the directory its records are under
SETTINGS = {"bf16": "bf16", "FP8": "fp8"}
TCP_SETTINGS = {"TCP bf16": "tcp-bf16", "TCP FP8": "tcp-fp8"}
CAPPED_SETTINGS = {
    "bf16, ~2x slower": "capped-2x-bf16",
    "FP8, ~2x slower": "capped-2x-fp8",
    "bf16, ~2.5x slower": "capped-2.5x-bf16",
    "FP8, ~2.5x slower": "capped-2.5x-fp8",
}
# setting: (precision, network) in throughput.csv
NETWORKS = {
    "bf16": ("bf16", "rdma"),
    "FP8": ("fp8", "rdma"),
    "TCP bf16": ("bf16", "tcp"),
    "TCP FP8": ("fp8", "tcp"),
    "bf16, ~2x slower": ("bf16", "capped-2x"),
    "FP8, ~2x slower": ("fp8", "capped-2x"),
    "bf16, ~2.5x slower": ("bf16", "capped-2.5x"),
    "FP8, ~2.5x slower": ("fp8", "capped-2.5x"),
}
TCP_WORKLOAD = "i4096-o256-r8"
CAPPED_WORKLOAD = "i4096-o256"
CONFIGS = ["dres256-s8-c512", "dres256-s6-c512", "dres256-s6-c768", "dres768-s12-c512"]
CONFIGS += ["dres512-s4-c512", "dres512-s6-c768", "dres512-s4-c768", "dres512-s4-c1024"]
CONFIGS += ["dres512-s4-c1280", "dres1280-s6-c1280", "dres1280-s4-c1280"]
WORKLOADS = {
    "i4096-o256": "4096/256 x128",
    "i128-o1024": "128/1k x512 (decode)",
    "i1024-o128": "1024/128 x512",
    "i16384-o512": "16k/512 x32 (prefill)",
    "i4096-o256-r8": "4096/256 x8",
    "i128000-o512": "128k/512 x4",
}
# workload: {requests per GPU: the row's workload}, up to the workload's full batch
CONCURRENCY_SWEEP = {
    workload: {**{n: f"{workload}-r{n}" for n in requests}, full_batch: workload}
    for workload, requests, full_batch in [
        ("i4096-o256", (8, 16, 32, 64), 128),
        ("i1024-o128", (8, 16, 32, 64, 128, 256), 512),
        ("i128-o1024", (8, 16, 32, 64, 128, 256), 512),
        ("i16384-o512", (8, 16), 32),
    ]
}
SHAPES = {
    "i32000-o256-r16": "32K/256 (16), agent",
    "i64000-o256-r8": "64K/256 (8), agent",
    "i1024-o8192-r64": "1K/8K (64), reasoning",
    "i1024-o16384-r32": "1K/16K (32), reasoning",
    # an FP8 KV cache doubles the requests per GPU at agent lengths
    "i32000-o256-r16-kv8": "32K/256 (16), agent, FP8 KV",
    "i32000-o256-r32-kv8": "32K/256 (32), agent, FP8 KV",
    "i64000-o256-r8-kv8": "64K/256 (8), agent, FP8 KV",
    "i64000-o256-r16-kv8": "64K/256 (16), agent, FP8 KV",
}
SHAPE_CONFIGS = ["dres512-s4-c1280", "dres512-s4-c768", "dres256-s8-c512"]
# records directory: (model, its default setting)
LARGER_MODELS = {
    "qwen3-235b-a22b": ("Qwen3-235B-A22B", "bf16"),
    "deepseek-v3": ("DeepSeek-V3", "FP8 dispatch"),
    "kimi-k2.6": ("Kimi-K2.6", "bf16 wire"),
}
# setting directory: the setting, None for the model's default
LARGER_SETTINGS = {"default": None, "fp8": "FP8", "bf16-wire": "bf16 wire"}
# the trained Kanana-2 and Gemma 4 students: model: its records directory
ADAPTED = {"Kanana-2": "kanana-2-30b-a3b", "Gemma 4": "gemma4-26b-a4b"}
ADAPTED_WORKLOADS = ["i1024-o128", "i128-o1024", "i4096-o256", "i16384-o512", "i128000-o512"]
# online serving (serving.py): each table's launches, as (setting, the launch's row) under which
# the records are stored, and its goodput's targets (P99 TTFT, P99 TPOT, ms)
SERVING_BF16, SERVING_FP8 = "bf16-real-routing", "fp8-real-routing"
SERVING_TABLES = {
    "online": (
        [
            ("bf16", "dres512-s4-c768-ladder-fixed"),
            ("fp8", "dres512-s4-c768-ladder-fixed"),
            (SERVING_BF16, "dres512-s4-c768-ladder"),
            (SERVING_FP8, "dres512-s4-c768-ladder"),
        ],
        (2000, 150),
    ),
    "agent": (
        [
            (setting, f"dres512-s4-c768-ladder-agent{context}k")
            for setting in (SERVING_BF16, SERVING_FP8)
            for context in (16, 32)
        ],
        (5000, 300),
    ),
}
# Table 35's settings: name, setting, row
SERVING_ONLINE = [
    ("BF16, real routing", SERVING_BF16, "dres512-s4-c768-ladder"),
    ("BF16, balanced routing, fixed replies", "bf16", "dres512-s4-c768-ladder-fixed"),
    ("FP8, real routing", SERVING_FP8, "dres512-s4-c768-ladder"),
    ("FP8, balanced routing, fixed replies", "fp8", "dres512-s4-c768-ladder-fixed"),
]

# profiled workloads: steady-state prefill and decode windows (and at 8 requests per GPU only decode)
PROFILE_WORKLOADS = ["i4096-o256-steps2", "i16384-o512-steps", "i128-o1024-steps", "i4096-o256-r8"]
BREAKDOWN_CONFIGS = ["dres256-s8-c512", "dres512-s4-c768", "dres1280-s4-c1280"]
# The finer breakdowns add the lightest configuration of Figure 3(c) and the two either side of the
# kink near 66% RDMA saved. The launch stall covers the first four, in BF16 and FP8.
FINE_CONFIGS = [*BREAKDOWN_CONFIGS, "dres512-s4-c1280", "dres768-s12-c512", "dres256-s6-c768"]
STALL_CONFIGS = FINE_CONFIGS[:4]
# The rail kernels (echomoe/inference/throughput/gin) count as the DeepEP or NCCL calls they replace.
RAIL = r"_kernel(<[\w, ]+>)?\(\(anonymous namespace\)::Args\)"
# Each GPU kernel counts toward the first category whose pattern its name matches. The first eight
# are communication.
KERNEL_CATEGORIES = [
    ("dispatch", r"hybrid_dispatch_impl|dispatch_copy_epilogue"),
    ("rail dispatch and combine", r"::(dispatch|combine)" + RAIL),
    ("rail reference collectives", r"::(allgather|reducescatter)" + RAIL),
    ("combine", r"hybrid_combine_impl|combine_reduce_epilogue"),
    ("data-parallel sync", r"AllReduce_u32"),
    ("reference all-gather", r"AllGather"),
    ("reference reduce-scatter", r"ReduceScatter"),
    ("FP8 combine all-to-all", r"SendRecv"),
    ("expert matrix products", r"fused_moe_kernel"),
    (
        "expert activation, output sum and routing",
        r"act_and_mul|silu_mul_live|moe_sum|count_and_sort|moe_align_block|_compact_routes"
        r"|_routing\b|_router_input|topk|_prepare_routing",
    ),
    ("attention", r"flash|FlashAttn|attention"),
    ("other matrix products", r"nvjet|gemm|cutlass|sm90_xmma"),
    (
        "compression kernels",
        r"^_(assemble|prepare|mask_rows|publish|reference_receive_prepare|reference_send_prepare"
        r"|unpack|pack|scatter|encode|decode|wire_send|wire_receive|consumer_send|router_output"
        r"|mask)",
    ),
    ("other", r"."),
]
COMMUNICATION = [name for name, _ in KERNEL_CATEGORIES[:8]]
EXPERT = ["expert matrix products", "expert activation, output sum and routing"]
# collective calls by the kernel they run, or the rail kernel that replaces it
COLLECTIVES = {
    "dispatch": ("hybrid_dispatch_impl", "dispatch"),
    "dispatch epilogue": ("dispatch_copy_epilogue", None),
    "combine": ("hybrid_combine_impl", "combine"),
    "combine epilogue": ("combine_reduce_epilogue", None),
    "all-gather": ("AllGather", "allgather"),
    "reduce-scatter": ("ReduceScatter", "reducescatter"),
    "all-reduce": ("AllReduce", None),
    "all-to-all": ("SendRecv", None),
}
EPILOGUES = ("dispatch epilogue", "combine epilogue")
FINE_COMMUNICATION = {
    "dispatch": ("dispatch",),
    "combine": ("combine",),
    "reference all-gather/reduce-scatter": ("all-gather", "reduce-scatter"),
    "DP sync all-reduce": ("all-reduce",),
    "FP8 all-to-all": ("all-to-all",),
}
ROUTER_TOP_K = re.compile(r"topk|TopK")
NORM = re.compile(r"rms_norm")


def mean(values):
    return sum(values) / len(values)


def timed_runs(summary):
    return summary["run1_s"], summary["run2_s"]


def spread(base_runs, echomoe_runs):
    return round(max(max(runs) / min(runs) - 1 for runs in (base_runs, echomoe_runs)), 4)


def balanced_rows(root, cells, reduction):
    """A row per measured cell, given as (leading columns with the config, run label, workload)."""
    measured = []
    for head, label, workload in cells:
        records = root / label / f"{head['config']}-{workload}"
        if records.exists():
            measured.append((head, summarize(records / "base"), summarize(records / "folded")))

    # base does not depend on the configuration, so the rows that differ only in it share a base
    def base_group(head):
        return tuple(value for column, value in head.items() if column != "config")

    bases = {}
    for head, base, _ in measured:
        bases.setdefault(base_group(head), []).append(base)
    rows = []
    for head, base, echomoe in measured:
        base_runs, echomoe_runs = timed_runs(base), timed_runs(echomoe)
        base_s, echomoe_s = mean(base_runs), mean(echomoe_runs)
        group = bases[base_group(head)]
        mean_base_s = mean([mean(timed_runs(summary)) for summary in group])
        mean_base_ttft_s = mean([summary["ttft_mean_s"] for summary in group])
        mean_base_tpot_ms = mean([summary["decode_tpot_ms"] for summary in group])
        row = dict(
            reduction=round(reduction[head["config"]], 3),
            speedup=round(base_s / echomoe_s, 4),
            base_s=round(base_s, 2),
            echomoe_s=round(echomoe_s, 2),
            spread=spread(base_runs, echomoe_runs),
            base_ttft_s=base["ttft_mean_s"],
            echomoe_ttft_s=echomoe["ttft_mean_s"],
            base_tpot_ms=base["decode_tpot_ms"],
            echomoe_tpot_ms=echomoe["decode_tpot_ms"],
            mean_base_s=round(mean_base_s, 2),
            speedup_mean_base=round(mean_base_s / echomoe_s, 4),
            ttft_speedup_mean_base=round(mean_base_ttft_s / echomoe["ttft_mean_s"], 4),
            tpot_speedup_mean_base=round(mean_base_tpot_ms / echomoe["decode_tpot_ms"], 4),
        )
        rows.append(head | row)
    return rows


def larger_model_rows(root):
    cells = []
    for name, (model, default) in LARGER_MODELS.items():
        for label, setting in LARGER_SETTINGS.items():
            for workload in ("i4096-o256", "i4096-o256-r8"):
                head = dict(model=model, network="InfiniBand", setting=setting or default)
                cells.append((head | {"workload": workload}, label, name))
        head = dict(model=model, network="TCP", setting="default", workload=TCP_WORKLOAD)
        cells.append((head, "tcp", name))
    rows = []
    for head, label, name in cells:
        records = root / name / "throughput" / label / head["workload"]
        if not records.exists():
            continue
        # rank 0's timed batches, as over TCP no engine step is short enough for the summary's
        # decode step
        base, echomoe = (
            json.loads((records / variant / "rank0.json").read_text())["batch_s"]
            for variant in ("base", "folded")
        )
        base_s, echomoe_s = mean(base), mean(echomoe)
        row = dict(
            base_s=round(base_s, 2),
            echomoe_s=round(echomoe_s, 2),
            speedup=round(base_s / echomoe_s, 4),
            spread=spread(base, echomoe),
        )
        rows.append(head | row)
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as table:
        writer = csv.DictWriter(table, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def balanced_markdown(rows, reduction):
    cell = {(row["setting"], row["config"], row["workload"]): row for row in rows}
    header = ["| Config | Reduction | " + " | ".join(WORKLOADS.values()) + " |"]
    header.append("|---|---:|" + "---:|" * len(WORKLOADS))

    def section(title, value):
        lines = [f"\n### {title}\n", *header]
        for config in CONFIGS:
            cells = " | ".join(value(config, workload) for workload in WORKLOADS)
            lines.append(f"| {config} | {reduction[config]:.2f}x | {cells} |")
        return lines

    def speedup(setting):
        def value(config, workload):
            row = cell.get((setting, config, workload))
            return f"{row['speedup_mean_base']:.3f}" if row else "-"

        return value

    def fp8_over_bf16_base(config, workload):
        bf16, fp8 = cell.get(("bf16", config, workload)), cell.get(("FP8", config, workload))
        return f"{bf16['base_s'] / fp8['echomoe_s']:.3f}" if bf16 and fp8 else "-"

    tcp = [
        "\n### TCP over eth0, 4096/256 x8: per row, and over the mean base of the setting's rows\n"
    ]
    tcp.append(
        "| Config | Reduction | bf16 EchoMoE (s) | bf16 per row | bf16 mean base "
        "| FP8 EchoMoE (s) | FP8 per row | FP8 mean base |"
    )
    tcp.append("|---|---:|" + "---:|" * 6)
    mean_base = {
        setting: mean([row["base_s"] for row in rows if row["setting"] == setting])
        for setting in TCP_SETTINGS
    }
    for config in CONFIGS:
        cells = []
        for setting in TCP_SETTINGS:
            row = cell.get((setting, config, TCP_WORKLOAD))
            if row:
                cells += [f"{row['echomoe_s']:.1f}", f"{row['speedup']:.3f}"]
                cells.append(f"{mean_base[setting] / row['echomoe_s']:.3f}")
            else:
                cells += ["-", "-", "-"]
        tcp.append(f"| {config} | {reduction[config]:.2f}x | " + " | ".join(cells) + " |")
    title = (
        "Per-connection-capped networks, 4096/256 x128: speedup, mean base and EchoMoE times (s)"
    )
    capped = [f"\n### {title}\n", "| Config | Reduction | " + " | ".join(CAPPED_SETTINGS) + " |"]
    capped.append("|---|---:|" + "---:|" * len(CAPPED_SETTINGS))
    for config in CONFIGS:
        cells = []
        for setting in CAPPED_SETTINGS:
            row = cell.get((setting, config, CAPPED_WORKLOAD))
            if row:
                times = f"{row['mean_base_s']:.1f}/{row['echomoe_s']:.1f}"
                cells.append(f"{row['speedup_mean_base']:.3f} ({times})")
            else:
                cells.append("-")
        capped.append(f"| {config} | {reduction[config]:.2f}x | " + " | ".join(cells) + " |")
    title = (
        "EchoMoE speedup over base, balanced expert assignment, final kernels; speedup over the "
        "mean base of the setting and workload's rows (base does not depend on the config)"
    )
    sections = section("bf16 EchoMoE over bf16 base", speedup("bf16"))
    sections += section("FP8 EchoMoE over FP8 base", speedup("FP8"))
    sections += section("FP8 EchoMoE over bf16 base", fp8_over_bf16_base)
    return "\n".join([title, *sections, *tcp, *capped]) + "\n"


def shapes_markdown(rows):
    lines = [
        "Agent and reasoning shapes, EP32, balanced expert assignment: speedup over the mean base "
        "of the setting and shape (throughput; TTFT and decode TPOT as latency ratios, above 1 is "
        "faster)\n",
        "| Setting | Shape | Config | Reduction | Throughput | TTFT | TPOT | Base / EchoMoE (s) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['setting']} | {SHAPES[row['workload']]} | {row['config']} "
            f"| {row['reduction']:.2f}x "
            f"| {row['speedup_mean_base']:.3f} | {row['ttft_speedup_mean_base']:.3f} "
            f"| {row['tpot_speedup_mean_base']:.3f} "
            f"| {row['mean_base_s']:.1f} / {row['echomoe_s']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def larger_models_markdown(rows):
    lines = ["Larger models, balanced expert assignment, EchoMoE speedup over base\n"]
    lines.append("| Model | Network | Setting | Workload | Base (s) | EchoMoE (s) | Speedup |")
    lines.append("|---|---|---|---|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['network']} | {row['setting']} "
            f"| {WORKLOADS[row['workload']]} | "
            f"{row['base_s']:.1f} | {row['echomoe_s']:.1f} | {row['speedup']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def adapted_rows(root):
    """The Kanana-2 and Gemma 4 rows: throughput from rank 0's timed batches, TTFT over every
    timed request, and decode TPOT over the requests with the latest first token."""
    rows = []
    for setting in ("bf16", "FP8"):
        for model, name in ADAPTED.items():
            for workload in ADAPTED_WORKLOADS:
                records = root / name / "throughput" / SETTINGS[setting] / workload
                if not records.exists():
                    continue
                times = {}
                for variant in ("base", "folded"):
                    paths = sorted((records / variant).glob("rank*.json"))
                    ranks = [json.loads(path.read_text()) for path in paths]
                    ttft = [t for rank in ranks for run in rank["ttft_s"] for t in run]
                    times[variant] = (
                        ranks[0]["batch_s"],
                        statistics.fmean(ttft),
                        tail_tpot_ms(ranks),
                    )
                (base_runs, base_ttft, base_tpot), (runs, ttft, tpot) = times.values()
                base_s, echomoe_s = statistics.mean(base_runs), statistics.mean(runs)
                rows.append(
                    dict(
                        model=model,
                        setting=setting,
                        workload=workload,
                        base_s=round(base_s, 2),
                        echomoe_s=round(echomoe_s, 2),
                        speedup=round(base_s / echomoe_s, 4),
                        spread=spread(base_runs, runs),
                        base_ttft_s=round(base_ttft, 3),
                        echomoe_ttft_s=round(ttft, 3),
                        ttft_speedup=round(base_ttft / ttft, 4),
                        base_tpot_ms=round(base_tpot, 2),
                        echomoe_tpot_ms=round(tpot, 2),
                        tpot_speedup=round(base_tpot / tpot, 4),
                    )
                )
    return rows


def adapted_markdown(rows):
    lines = ["Kanana-2 and Gemma 4, balanced expert assignment, EchoMoE speedup over base\n"]
    lines.append("| Model | Setting | Workload | Throughput | TTFT | TPOT |")
    lines.append("|---|---|---|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['setting']} | {WORKLOADS[row['workload']]} | "
            f"{row['speedup']:.2f} | {row['ttft_speedup']:.2f} | {row['tpot_speedup']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def serving_rows(pool, root, launches, targets):
    """A row per launch, variant and request rate of the client records under `root`."""
    rows = []
    for label, row in launches:
        for variant in ("base", "folded"):
            rates = {}
            for path in (root / label / row / variant).glob("rate-*.json"):
                if match := re.fullmatch(r"rate-([0-9.]+)(-part\d+)?", path.stem):
                    rates.setdefault(float(match[1]), []).append(path)
            for rate, paths in sorted(rates.items()):
                records = list(pool.map(read_record, sorted(paths)))
                summary = rate_summary(records, *targets)
                rows.append(dict(setting=label, row=row, variant=variant, rate=rate, **summary))
    return rows


def serving_markdown(rows):
    cell = {(r["setting"], r["row"], r["variant"], r["rate"]): r for r in rows}
    lines = ["Online serving, uncompressed / EchoMoE (4, 512, 768)\n"]
    lines.append(
        "| Setting | Requests/s | P99 TTFT (ms) | Median TPOT (ms) | P99 TPOT (ms) "
        "| Output (k tokens/s) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, label, row in SERVING_ONLINE:
        for rate in sorted({r["rate"] for r in rows if (r["setting"], r["row"]) == (label, row)}):
            base, echomoe = (cell[label, row, variant, rate] for variant in ("base", "folded"))
            lines.append(
                f"| {name} | {rate:g} "
                f"| {base['ttft_p99_ms']:,.0f} / {echomoe['ttft_p99_ms']:,.0f} "
                f"| {base['tpot_p50_ms']:.1f} / {echomoe['tpot_p50_ms']:.1f} "
                f"| {base['tpot_p99_ms']:.1f} / {echomoe['tpot_p99_ms']:.1f} "
                f"| {base['output_tokens_per_s'] / 1000:.1f} "
                f"/ {echomoe['output_tokens_per_s'] / 1000:.1f} |"
            )
    return "\n".join(lines) + "\n"


@cache
def category(kernel):
    return next(name for name, pattern in KERNEL_CATEGORIES if re.search(pattern, kernel))


@cache
def collective(kernel):
    for kind, (name, rail) in COLLECTIVES.items():
        if name in kernel or (rail and re.search(f"::{rail}{RAIL}", kernel)):
            return kind
    return None


def read_trace(path, graph_steps):
    """One rank's GPU kernels in a profile window as (name, start, duration) in us, on the trace's
    clock, and for a window of `graph_steps` graphed steps its launch stall."""
    trace = json.loads(gzip.decompress(path.read_bytes()))
    events = [event for event in trace["traceEvents"] if event.get("ph") == "X"]
    clock = trace["baseTimeNanoseconds"] / 1000.0
    gpu = [entry for entry in events if entry.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    kernels = sorted(
        ((entry["name"], clock + entry["ts"], entry["dur"]) for entry in gpu), key=lambda k: k[1]
    )
    return kernels, graph_steps and launch_stall(gpu, events, graph_steps)


def launch_stall(gpu, events, steps):
    """A decode window's step, GPU idle inside cudaGraphLaunch, and time in cudaGraphLaunch, in ms
    per step. Under the profiler, cudaGraphLaunch takes milliseconds of CPU, and a graph that starts
    only once the call returns leaves the GPU idle for that time."""
    busy = sorted((event["ts"], event["ts"] + event["dur"]) for event in gpu)
    launches = [
        (event["ts"], event["ts"] + event["dur"])
        for event in events
        if event["name"] == "cudaGraphLaunch"
    ]
    assert len(launches) == steps, len(launches)

    def busy_within(start, end):
        total, reach = 0.0, start
        for busy_start, busy_end in busy:
            busy_start, busy_end = max(busy_start, reach), min(busy_end, end)
            if busy_end > busy_start:
                total += busy_end - busy_start
                reach = busy_end
        return total

    span = busy[-1][1] - busy[0][0]
    stall = sum((end - start) - busy_within(start, end) for start, end in launches)
    launch = sum(end - start for start, end in launches)
    return dict(step=span / steps / 1000, stall=stall / steps / 1000, launch=launch / steps / 1000)


def union_length(intervals):
    total, reach = 0.0, float("-inf")
    for start, end in sorted(intervals):
        if end > reach:
            total += end - max(start, reach)
            reach = end
    return total


def category_times(kernels, steps):
    """One rank's ms per step in each kernel category, of the step (its GPU span), of idle GPU time
    and of time two streams overlap."""
    times = {name: 0.0 for name, _ in KERNEL_CATEGORIES}
    for name, _, duration in kernels:
        times[category(name)] += duration
    span = max(start + duration for _, start, duration in kernels) - min(
        kernel[1] for kernel in kernels
    )
    busy = union_length([(start, start + duration) for _, start, duration in kernels])
    result = {name: value / steps / 1000 for name, value in times.items()}
    result["overlap"] = (sum(times.values()) - busy) / steps / 1000
    result["idle"] = (span - busy) / steps / 1000
    result["step"] = span / steps / 1000
    return result


def collective_calls(kernels):
    """Every collective call of a window, matched across ranks by its index, as (kind, duration,
    wait, post) in us, each a mean over ranks: wait from the rank's start to the latest rank's,
    post from there to the rank's end. Each node gets one clock offset: the median over the
    combines and the node's ranks of the end minus the median end over all ranks, as a combine
    ends at about the same time on every rank."""
    ranks = sorted(kernels)
    calls = {}
    for rank in ranks:
        calls[rank] = defaultdict(list)
        for name, start, duration in kernels[rank]:
            if kind := collective(name):
                calls[rank][kind].append((start, start + duration))
    kinds = [kind for kind in COLLECTIVES if calls[ranks[0]].get(kind)]
    for kind in kinds:
        assert len({len(calls[rank][kind]) for rank in ranks}) == 1, kind
    node = {rank: rank // 8 for rank in ranks}
    offset = dict.fromkeys(set(node.values()), 0.0)
    for _ in range(2):
        residuals = defaultdict(list)
        for index in range(len(calls[ranks[0]]["combine"])):
            ends = {rank: calls[rank]["combine"][index][1] - offset[node[rank]] for rank in ranks}
            middle = statistics.median(ends.values())
            for rank in ranks:
                residuals[node[rank]].append(ends[rank] - middle)
        for node_index in offset:
            offset[node_index] += statistics.median(residuals[node_index])
    result = []
    for kind in kinds:
        for index in range(len(calls[ranks[0]][kind])):
            start = {rank: calls[rank][kind][index][0] - offset[node[rank]] for rank in ranks}
            end = {rank: calls[rank][kind][index][1] - offset[node[rank]] for rank in ranks}
            latest = max(start.values())
            duration = statistics.mean(end[rank] - start[rank] for rank in ranks)
            wait = statistics.mean(latest - start[rank] for rank in ranks)
            post = statistics.mean(end[rank] - latest for rank in ranks)
            result.append((kind, duration, wait, post))
    return result


def compression_projections(kernels):
    """us of matrix products between a router's top-k and the next dispatch (compression), or
    between a combine and the next norm (decompression). The traces carry no shapes, so these
    are the compression projections by where they run."""
    total, inside = 0.0, False
    for name, _, duration in kernels:
        kind = collective(name)
        if ROUTER_TOP_K.search(name):
            inside = True
        elif kind == "dispatch":
            inside = False
        elif kind in ("combine", "combine epilogue", "reduce-scatter"):
            inside = True
        elif NORM.search(name):
            inside = False
        elif inside and category(name) == "other matrix products":
            total += duration
    return total


def fp8_casts(kernels):
    return sum(
        duration
        for name, _, duration in kernels
        if category(name) == "other" and "fp8" in name.lower()
    )


def step_parts(kernels, steps):
    """One model's step in a window, coarsely and per operation: ms per step, mean over ranks. The
    parts sum to the step."""
    ranks = list(kernels)
    per_rank = [category_times(kernels[rank], steps) for rank in ranks]
    mean_times = {name: statistics.mean(times[name] for times in per_rank) for name in per_rank[0]}
    calls = collective_calls(kernels)

    def per_step(values):
        return sum(values) / steps / 1000

    epilogues = per_step(duration for kind, duration, _, _ in calls if kind in EPILOGUES)
    waiting = per_step(wait for kind, _, wait, _ in calls if kind not in EPILOGUES)
    host_gaps = mean_times["idle"] - mean_times["overlap"]
    expert = sum(mean_times[name] for name in EXPERT)
    other = sum(
        mean_times[name]
        for name, _ in KERNEL_CATEGORIES
        if name not in COMMUNICATION and name not in EXPERT
    )
    coarse = {
        "expert compute": expert,
        "other compute": other + epilogues,
        "waiting for the busiest GPU": waiting,
        "communication": per_step(post for kind, _, _, post in calls if kind not in EPILOGUES),
        "host gaps": host_gaps,
        "step": mean_times["step"],
    }
    projections = statistics.mean(compression_projections(kernels[rank]) for rank in ranks)
    projections = projections / steps / 1000
    casts = statistics.mean(fp8_casts(kernels[rank]) for rank in ranks) / steps / 1000
    fine = {
        "expert GEMMs": mean_times["expert matrix products"],
        "MoE routing, activation, output sum": mean_times[EXPERT[1]],
        "attention": mean_times["attention"],
        "dense GEMMs (QKV/O, router, lm_head)": mean_times["other matrix products"] - projections,
        "norms, elementwise, sampling, misc": mean_times["other"] - casts,
        "codec: projection GEMMs": projections,
        "codec: pack, unpack, mask kernels": mean_times["compression kernels"],
        "FP8 quantize/dequantize": casts,
        "DeepEP epilogues (local copy/reduce)": epilogues,
        "waiting for the busiest GPU": waiting,
    }
    for name, kinds in FINE_COMMUNICATION.items():
        fine[f"communication: {name}"] = per_step(post for k, _, _, post in calls if k in kinds)
    fine["host gaps"] = host_gaps
    fine["step"] = mean_times["step"]
    return coarse, fine


def read_window(pool, records, window, graph_steps):
    """Every rank's kernels in one variant's profile window, and its launch stall, mean over ranks."""
    paths = sorted(records.glob(f"profile/rank*/{window}/*.pt.trace.json.gz"))
    traces = list(pool.map(partial(read_trace, graph_steps=graph_steps), paths))
    kernels = {
        int(path.parts[-3].removeprefix("rank")): rank_kernels
        for path, (rank_kernels, _) in zip(paths, traces)
    }
    assert len(kernels) == len(paths), records
    stalls = [stall for _, stall in traces]
    return kernels, graph_steps and {
        name: statistics.mean(stall[name] for stall in stalls) for name in stalls[0]
    }


def breakdowns(root):
    """The step breakdowns and launch stalls of every profile row under `root`."""
    coarse, fine, stall = {}, {}, {}
    settings = SETTINGS | CAPPED_SETTINGS
    cells = itertools.product(PROFILE_WORKLOADS, FINE_CONFIGS, settings)
    with ProcessPoolExecutor(8) as pool:
        for workload, config, setting in cells:
            row = root / f"{settings[setting]}-profile" / f"{config}-{workload}"
            if not row.exists():
                continue
            key = f"{workload}|{config}|{setting}"
            meta = json.loads((row / "base/meta.json").read_text())
            windows = {window: steps for window, _, steps in meta["profile_windows"]}
            fine[key] = {}
            for window in ("mid_prefill", "decode"):
                if window not in windows:
                    continue
                graph_steps = windows[window] if window == "decode" else None
                parts, stalls = {}, {}
                for model, variant in (("base", "base"), ("EchoMoE", "folded")):
                    kernels, stalls[model] = read_window(pool, row / variant, window, graph_steps)
                    parts[model] = step_parts(kernels, windows[window])
                fine[key][window] = {model: both[1] for model, both in parts.items()}
                if config in BREAKDOWN_CONFIGS:
                    coarse.setdefault(key, {})[window] = {
                        model: both[0] for model, both in parts.items()
                    }
                if graph_steps and setting in SETTINGS and config in STALL_CONFIGS:
                    stall[key] = stalls
    return coarse, fine, stall


WORKLOAD_CONFIGS = REPO / "configs" / "inference" / "throughput"
# the trained students' configurations
ADAPTED_CONFIGS = {"Kanana-2": "dres512-s4-c1024", "Gemma 4": "dres768-s4-c1408"}
# a larger model's setting: its precision in throughput.csv
LARGER_PRECISIONS = {
    "bf16": "bf16",
    "FP8": "fp8",
    "FP8 dispatch": "fp8-dispatch",
    "bf16 wire": "bf16-wire",
}
THROUGHPUT_COLUMNS = [
    "model", "precision", "network", "routing", "config", "stride", "dispatch_rank",
    "combine_rank", "input_tokens", "output_tokens", "requests_per_gpu", "kv_cache",
    "rdma_reduction", "base_s", "echomoe_s", "speedup", "speedup_over_mean_base", "mean_base_s",
    "base_ttft_s", "echomoe_ttft_s", "ttft_speedup", "ttft_speedup_over_mean_base",
    "base_tpot_ms", "echomoe_tpot_ms", "tpot_speedup", "tpot_speedup_over_mean_base", "spread",
]  # fmt: skip
SERVING_COLUMNS = [
    "model", "setting", "routing", "precision", "config", "variant", "rate_per_s", "requests",
    "failed", "arrival_s", "offered_per_s", "steady_requests", "output_tokens_per_s",
    "mean_output_tokens", "ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "tpot_p99_ms",
    "goodput_per_s", "stalled_requests", "max_gap_ms",
]  # fmt: skip


def config_columns(config):
    """A configuration's reference stride, dispatch rank and combine rank."""
    match = re.fullmatch(r"dres(\d+)-s(\d+)-c(\d+)", config)
    return dict(stride=match[2], dispatch_rank=match[1], combine_rank=match[3])


def workload_columns(workload):
    """A workload's prompt and generated tokens, requests per GPU and KV cache precision."""
    match = re.fullmatch(r"i(\d+)-o(\d+)(?:-r(\d+))?(-kv8)?(?:-steps2?)?", workload)
    requests = match[3]
    if requests is None:
        config = WORKLOAD_CONFIGS / f"i{match[1]}-o{match[2]}.yaml"
        requests = re.search(r"^batch_size: (\d+)$", config.read_text(), re.M)[1]
    return dict(
        input_tokens=match[1],
        output_tokens=match[2],
        requests_per_gpu=requests,
        kv_cache="fp8" if match[4] else "bf16",
    )


def throughput_table(balanced, sweep, shapes, larger, adapted):
    """Every offline row in one table, a measured cell once."""
    rows = {}
    for row in [*balanced, *sweep, *shapes]:
        workload = row.get("workload") or CONCURRENCY_SWEEP[row["stem"]][row["requests_per_gpu"]]
        precision, network = NETWORKS[row["setting"]]
        cell = dict(model=QWEN, precision=precision, network=network, routing="balanced")
        cell |= dict(config=row["config"], **config_columns(row["config"]))
        cell |= workload_columns(workload)
        cell |= dict(
            rdma_reduction=row["reduction"],
            base_s=row["base_s"],
            echomoe_s=row["echomoe_s"],
            speedup=row["speedup"],
            speedup_over_mean_base=row["speedup_mean_base"],
            mean_base_s=row["mean_base_s"],
            base_ttft_s=row["base_ttft_s"],
            echomoe_ttft_s=row["echomoe_ttft_s"],
            ttft_speedup_over_mean_base=row["ttft_speedup_mean_base"],
            base_tpot_ms=row["base_tpot_ms"],
            echomoe_tpot_ms=row["echomoe_tpot_ms"],
            tpot_speedup_over_mean_base=row["tpot_speedup_mean_base"],
            spread=row["spread"],
        )
        rows.setdefault(tuple(cell[column] for column in THROUGHPUT_COLUMNS[:12]), cell)
    names = {model: name for name, (model, _) in LARGER_MODELS.items()}
    for row in larger:
        cell = dict(
            model=names[row["model"]], routing="balanced", **workload_columns(row["workload"])
        )
        default = LARGER_MODELS[names[row["model"]]][1]
        setting = default if row["setting"] == "default" else row["setting"]
        network = "tcp" if row["network"] == "TCP" else "rdma"
        cell |= dict(precision=LARGER_PRECISIONS[setting], network=network)
        cell |= {key: row[key] for key in ("base_s", "echomoe_s", "speedup", "spread")}
        rows[len(rows)] = cell
    for row in adapted:
        cell = dict(model=ADAPTED[row["model"]], routing="balanced", network="rdma")
        cell |= dict(precision=NETWORKS[row["setting"]][0], config=ADAPTED_CONFIGS[row["model"]])
        cell |= config_columns(cell["config"]) | workload_columns(row["workload"])
        cell |= {key: value for key, value in row.items() if key in THROUGHPUT_COLUMNS[13:]}
        rows[len(rows)] = cell
    return [
        {column: cell.get(column, "") for column in THROUGHPUT_COLUMNS} for cell in rows.values()
    ]


def serving_table(rows):
    """The online-serving rows with their setting spelled out."""
    settings = {"ladder": "online", "ladder-fixed": "online-fixed-replies"}
    settings |= {f"ladder-agent{context}k": f"agent-{context}k" for context in (16, 32)}
    cells = []
    for row in rows:
        config, _, launch = row["row"].partition("-ladder")
        cell = row | dict(model=QWEN, setting=settings["ladder" + launch], config=config)
        cell |= dict(routing="real" if row["setting"].endswith("-real-routing") else "balanced")
        cell |= dict(
            precision=row["setting"].removesuffix("-real-routing"), rate_per_s=row["rate"]
        )
        cells.append({column: cell[column] for column in SERVING_COLUMNS})
    return cells


def breakdown_table(coarse, fine, stall):
    """The step breakdowns and launch stalls as one row per profile row, window, model and part."""
    rows = []
    for level, result in (("coarse", coarse), ("fine", fine), ("launch_stall", stall)):
        for key, windows in result.items():
            workload, config, setting = key.split("|")
            precision, network = NETWORKS[setting]
            head = dict(model=QWEN, precision=precision, network=network, config=config)
            head |= config_columns(config) | workload_columns(workload)
            head["workload"] = workload
            if level == "launch_stall":
                windows = {"decode": windows}
            for window, models in windows.items():
                for model, parts in models.items():
                    for part, ms in parts.items():
                        variant = "base" if model == "base" else "echomoe"
                        rows.append(head | dict(level=level, window=window, variant=variant))
                        rows[-1] |= dict(part=part, ms=ms)
    return rows


if __name__ == "__main__":
    outputs, output = Path(sys.argv[1]), Path(sys.argv[2])
    root = outputs / QWEN / "throughput"
    reduction = {r["config_id"]: r["reduction"] for r in cost_table(FANOUT) if r["nodes"] == 4}
    settings = [(setting, label, list(WORKLOADS)) for setting, label in SETTINGS.items()]
    settings += [(setting, label, [TCP_WORKLOAD]) for setting, label in TCP_SETTINGS.items()]
    settings += [(setting, label, [CAPPED_WORKLOAD]) for setting, label in CAPPED_SETTINGS.items()]
    cells = [
        (dict(setting=setting, config=config, workload=workload), label, workload)
        for setting, label, workloads in settings
        for workload in workloads
        for config in CONFIGS
    ]
    balanced = balanced_rows(root, cells, reduction)
    print(balanced_markdown(balanced, reduction), end="")
    cells = [
        (
            dict(setting=setting, config=config, stem=stem, requests_per_gpu=requests),
            label,
            workload,
        )
        for setting, label in SETTINGS.items()
        for stem, points in CONCURRENCY_SWEEP.items()
        for requests, workload in points.items()
        for config in CONFIGS
    ]
    sweep = balanced_rows(root, cells, reduction)
    cells = [
        (dict(setting=setting, config=config, workload=workload), label, workload)
        for setting, label in SETTINGS.items()
        for workload in SHAPES
        for config in SHAPE_CONFIGS
    ]
    shapes = balanced_rows(root, cells, reduction)
    print(shapes_markdown(shapes), end="")
    larger = larger_model_rows(outputs)
    print(larger_models_markdown(larger), end="")
    adapted = adapted_rows(outputs)
    print(adapted_markdown(adapted), end="")
    write_csv(
        output / "throughput.csv", throughput_table(balanced, sweep, shapes, larger, adapted)
    )
    serving = []
    with ProcessPoolExecutor(8) as pool:
        for name, (launches, targets) in SERVING_TABLES.items():
            rows = serving_rows(pool, outputs / QWEN / "serving", launches, targets)
            serving += rows
            if name == "online":
                print(serving_markdown(rows), end="")
    write_csv(output / "serving.csv", serving_table(serving))
    write_csv(output / "breakdowns.csv", breakdown_table(*breakdowns(root)))
