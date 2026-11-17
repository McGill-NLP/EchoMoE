"""Reduce one throughput row's records (both variants) to one CSV row: each variant's mean TTFT,
timed batches and decode TPOT, or for a larger model (a row run with `--model-config`) its
compression, formats and median decode step. `summarize` is what scripts/throughput_tables.py
reads the records with.

    python -m echomoe.inference.throughput.summarize outputs/qwen3-30b-a3b/throughput/<row> --config <config>
"""

import argparse
import csv
import itertools
import json
import statistics
import sys
from pathlib import Path

import yaml

from .checkpoint import codec_recipe, model_profile

VARIANTS = {"base": "base", "folded": "echomoe"}
# the decode TPOT is that of the requests with the latest first token, which decode almost only
# in pure-decode steps
TAIL = 0.1
# a larger model's decode step: an engine step that emitted tokens and took less than this (s).
# steps that carry prefill take 0.6-1.2 s
DECODE_STEP = 0.4


def summarize(directory: Path) -> dict[str, float]:
    """Mean TTFT over every timed request of every rank, rank 0's timed batches, and the decode
    speed: per timed batch the tail requests' mean TPOT (ms), averaged over the batches, or for a
    larger model the median of rank 0's decode steps (s) over the batches."""
    records = [json.loads(path.read_text()) for path in sorted(directory.glob("rank*.json"))]
    first = json.loads((directory / "rank0.json").read_text())
    runs = first["batch_s"]
    row = {
        "ttft_mean_s": statistics.fmean(
            ttft for record in records for run in record["ttft_s"] for ttft in run
        ),
        "run1_s": runs[0],
        "run2_s": runs[1],
    }
    if "step_s" in first:
        durations = [
            end - start
            for steps in first["step_s"]
            for (start, _), (end, emitted) in itertools.pairwise(steps)
            if end - start < DECODE_STEP and emitted > 0
        ]
        row["decode_step_s"] = statistics.median(durations)
        return row
    row["decode_tpot_ms"] = tail_tpot_ms(records)
    return row


def tail_tpot_ms(records: list[dict]) -> float:
    """Per timed batch, the mean TPOT (ms) of the requests with the latest first token, averaged
    over the batches."""
    tails = []
    for run in range(len(records[0]["batch_s"])):
        requests = sorted(
            pair
            for record in records
            for pair in zip(record["ttft_s"][run], record["tpot_s"][run])
        )
        tail = requests[-max(1, int(len(requests) * TAIL)) :]
        tails.append(1e3 * statistics.mean(tpot for _, tpot in tail))
    return statistics.mean(tails)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--config", required=True, help="the row's configuration or model, e.g. dres512-s4-c1024"
    )
    args = parser.parse_args()
    meta = json.loads((args.output / "base/meta.json").read_text())
    workload = meta["workload"]
    head = {
        "input_tokens": workload["input_len"],
        "output_tokens": workload["output_len"],
        "requests_per_gpu": workload["batch_size"],
    }
    rows = []
    if "model_config" in meta["inputs"]:
        # a larger model: its compression, storage formats and KV cache, then the measured columns
        model_config = yaml.safe_load(Path(meta["inputs"]["model_config"]).read_text())
        config = json.loads((Path(meta["inputs"]["model"]) / "config.json").read_text())
        recipe = codec_recipe(model_profile(config))
        # the dispatch's precision is the experts' unless the run set it (--fp8, --bf16-wire)
        fp8_dispatch = meta.get("fp8") or (
            model_config["expert_format"] == "fp8_block128" and not meta.get("bf16_wire")
        )
        row = dict(
            model=args.config,
            dispatch_rank=recipe.dispatch_rank,
            combine_rank=recipe.combine_rank,
            stride=recipe.stride,
            expert_format=model_config["expert_format"],
            dispatch_format="fp8" if fp8_dispatch else "bf16",
            kv_cache_gib=model_config["engine"]["kv_cache_memory_bytes"] // 2**30,
        )
        for variant, prefix in VARIANTS.items():
            for column, value in summarize(args.output / variant).items():
                row[f"{prefix}_{column}"] = value
        rows.append(dict(row, run=args.output.name))
    else:
        row = dict(config=args.config, **head)
        for variant, prefix in VARIANTS.items():
            for column, value in summarize(args.output / variant).items():
                row[f"{prefix}_{column}"] = value
        rows.append(dict(row, attempt=args.output.name))
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
