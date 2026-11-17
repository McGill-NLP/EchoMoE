#!/usr/bin/env python3
"""Write `results/evaluation/scores.csv` and `summary.csv` from the evaluation summaries under
`outputs/`: the teachers' `outputs/<model>/teacher/eval/<variant>/summary.json` and the students'
`outputs/<model>/runs/<run>/eval/step-N/<variant>/summary.json`.

    python scripts/results_tables.py outputs results/evaluation

`scores.csv` holds every task's score on its primary metric, unrounded. `summary.csv` holds the
report's rounding: each task's score in % rounded to 0.1 and each domain the mean of those, rounded
half up to 0.1, and the overall score the unrounded suite mean in % rounded to 0.1. The RDMA
reduction is the bytes per token sent between nodes at EP32 over 4 nodes against uncompressed bf16,
priced with each model's measured routing fanout.
"""

import csv
import json
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from echomoe.communication_cost import cost_table
from echomoe.inference.evaluation.suite import PREFERRED_METRIC, load_suite, suite_mean, task_score
from echomoe.runs import REPO

FANOUT = REPO / "results" / "communication" / "routing_fanout.json"
DOMAIN = {task["id"]: task["domain"] for task in load_suite()["tasks"]}
DOMAINS = ["knowledge_reasoning", "math", "code", "instruction_following"]
FAMILIES = {
    "joint": "dev-mode",
    "dispatch-only": "dispatch-only",
    "combine-only": "combine-only",
    "full-adaptation": "full-adaptation",
}
# The other models' students, priced with each model's own measured fanout (paper Table 3). The
# repository ships Qwen3-30B-A3B's fanout only.
OTHER_REDUCTIONS = {
    "kanana-2-30b-a3b": "2.01",
    "gemma4-26b-a4b": "2.01",
    "moonlight-16b-a3b": "1.99",
    "nemotron-3.5-lightning-30b-a3b": "1.91",
}


def run_columns(run):
    """A run's family and configuration: stride, dispatch rank, combine rank, decoder."""
    if run == "teacher":
        return dict(family="teacher", stride="", dispatch_rank="", combine_rank="", decoder="")
    if run.startswith("lsh-"):
        return dict(family="lsh-moe", stride="", dispatch_rank="", combine_rank="", decoder="")
    family = next(f for suffix, f in FAMILIES.items() if run.endswith(f"-{suffix}"))
    match = re.fullmatch(
        r"d(?:res(\d+)-s(\d+)|plain(\d+)|full)-c(\d+|full)(?:-(direct|bias|biasgate)(?:-q\d+)?)?-.*",
        run,
    )
    return dict(
        family=family,
        stride=match[2] or "",
        dispatch_rank=match[1] or match[3] or "",
        combine_rank="" if match[4] == "full" else match[4],
        decoder=match[5] or ("direct" if match[4] != "full" else ""),
    )


def summaries(outputs):
    """(model, run, variant, tasks) of every evaluation summary under `outputs`."""
    for path in sorted(outputs.glob("*/teacher/eval/*/summary.json")):
        yield path.parts[-5], "teacher", path.parent.name, read(path)
    for path in sorted(outputs.glob("*/runs/*/eval/step-*/*/summary.json")):
        yield path.parts[-7], path.parts[-5], path.parent.name, read(path)


def read(path):
    document = json.loads(path.read_text())
    assert document["complete"], path
    return document["tasks"]


def percent(value):
    return f"{100 * value:.1f}"


def domains(tasks):
    """The domain columns: the mean of the task scores rounded to 0.1, rounded half up."""
    rounded = {task: Decimal(f"{100 * task_score(task, m):.1f}") for task, m in tasks.items()}
    cells = {}
    for domain in DOMAINS:
        values = [value for task, value in rounded.items() if DOMAIN[task] == domain]
        cells[domain] = str((sum(values) / len(values)).quantize(Decimal("0.1"), ROUND_HALF_UP))
    return cells


def config_id(run):
    return re.sub(r"-(joint|dispatch-only|combine-only|full-adaptation)$", "", run)


def main(outputs, output):
    reduction = {r["config_id"]: r["reduction"] for r in cost_table(FANOUT) if r["nodes"] == 4}
    scores, summary = [], []
    for model, run, variant, tasks in summaries(outputs):
        head = dict(model=model, run=run, **run_columns(run), variant=variant)
        for task, metrics in sorted(tasks.items()):
            metric = next(iter(metrics)) if len(metrics) == 1 else PREFERRED_METRIC[task]
            scores.append(head | dict(task=task, domain=DOMAIN[task], metric=metric))
            scores[-1]["score"] = task_score(task, metrics)
        if run == "teacher":
            rdma = "1.00"
        elif model in OTHER_REDUCTIONS:
            rdma = OTHER_REDUCTIONS[model]
        elif config_id(run) in reduction:
            rdma = f"{reduction[config_id(run)]:.2f}"
        else:
            rdma = ""
        summary.append(head | dict(rdma_reduction=rdma, **domains(tasks)))
        summary[-1]["overall"] = percent(suite_mean(tasks))
    for name, rows in (("scores.csv", scores), ("summary.csv", summary)):
        with open(output / name, "w", newline="") as table:
            writer = csv.DictWriter(table, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
