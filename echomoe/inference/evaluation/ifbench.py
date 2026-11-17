"""IFBench: greedy chat generations graded by the official evaluator from an IFBench checkout.
`generate(prompts, seeds, task)` is the engine's seeded generation, as in code.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .suite import (
    chat_tokens,
    item_seed,
    sample_row,
    task_result,
)

# Paths inside the IFBench checkout.
SCRIPT = "run_eval.py"
DATASET = "ifbench/data/IFBench_test.jsonl"
LOOSE_RESULTS = "responses-eval_results_loose.jsonl"


def run_ifbench_task(
    task: Mapping[str, Any],
    *,
    tokenizer: Any,
    generate: Callable[..., Sequence[Any]],
    checkout: str | Path,
) -> dict[str, Any]:
    # one seeded generation per official prompt
    root = Path(checkout).resolve()
    source = root / DATASET
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    item_ids = [f"ifbench:{row['key']}" for row in rows]
    prompts = [
        chat_tokens(tokenizer, [{"role": "user", "content": row["prompt"]}]) for row in rows
    ]
    seeds = [item_seed(task, item_id, 0) for item_id in item_ids]
    outputs = generate(prompts, seeds, task)

    # grade every response with the official script
    with tempfile.TemporaryDirectory(prefix="echomoe-ifbench-") as temporary:
        workspace = Path(temporary)
        responses = workspace / "responses.jsonl"
        lines = (
            json.dumps({"prompt": row["prompt"], "response": output.text}, ensure_ascii=False)
            for row, output in zip(rows, outputs, strict=True)
        )
        responses.write_text("".join(line + "\n" for line in lines))
        output_dir = workspace / "official-output"
        with open(workspace / "official-eval.log", "w", encoding="utf-8") as log:
            command = [sys.executable, str(root / SCRIPT), f"--input_data={source}"]
            command += [f"--input_response_data={responses}", f"--output_dir={output_dir}"]
            subprocess.run(command, cwd=root, check=True, stdout=log, stderr=subprocess.STDOUT)
        graded = (output_dir / LOOSE_RESULTS).read_text().splitlines()

    # The score is the loose prompt-level accuracy (`follow_all_instructions` of the official loose
    # verifier), not the strict one.
    metric = task["metrics"]["primary"]
    item_of = {row["prompt"]: item_id for row, item_id in zip(rows, item_ids)}
    verdicts = [json.loads(line) for line in graded]
    covered = sorted(item_of[verdict["prompt"]] for verdict in verdicts)
    assert covered == sorted(item_ids), "output coverage"
    passed = {
        item_of[verdict["prompt"]]: float(verdict["follow_all_instructions"])
        for verdict in verdicts
    }
    samples = [
        sample_row(
            task["id"],
            item_id,
            0,
            seed,
            {metric: passed[item_id]},
            output=output.text,
            finish_reason=output.finish_reason,
            prompt_tokens=output.prompt_tokens,
            generated_tokens=output.generated_tokens,
            truncated=output.finish_reason == "length",
        )
        for item_id, seed, output in zip(item_ids, seeds, outputs, strict=True)
    ]
    score = sum(passed.values()) / len(passed)  # in the official output's order
    return {
        "results": task_result({metric: score}, samples),
        "samples": samples,
    }
