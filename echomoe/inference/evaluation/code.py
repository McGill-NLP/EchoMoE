"""EvalPlus and LiveCodeBench: one seeded completion per sample, graded by the official grader in a
resource-limited child process. The LiveCodeBench child runs this module:

    python -m echomoe.inference.evaluation.code <payload>

`generate(prompts, seeds, task)` is the engine's seeded generation. It returns one completion per
token-id prompt with `.text`, `.finish_reason`, `.prompt_tokens` and `.generated_tokens`.
"""

from __future__ import annotations

import contextlib
import json
import os
import resource
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import lcb_runner.lm_styles
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.provider.utility import make_raw_chat_prompt
from evalplus.sanitize import sanitize
from huggingface_hub import hf_hub_download
from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
from lcb_runner.evaluation import codegen_metrics
from lcb_runner.lm_styles import LMStyle
from lcb_runner.utils.extraction_utils import extract_code

from echomoe.runs import REPO

from .suite import (
    chat_tokens,
    item_seed,
    sample_row,
    score_samples,
    task_result,
)

# The LiveCodeBench checkout is on PYTHONPATH. lcb_runner.prompts opens its few-shot files
# relative to the checkout root when imported.
LIVECODEBENCH = Path(lcb_runner.lm_styles.__file__).resolve().parents[1]
with contextlib.chdir(LIVECODEBENCH):
    from lcb_runner.prompts.code_generation import format_prompt_generation

EVALPLUS_INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:"
)
EVALPLUS_RESPONSE_PREFIX = (
    "Below is a Python script with a self-contained function that solves the "
    "problem and passes corresponding tests:"
)
# LiveCodeBench release_v6, problems dated 2025-01-01 to 2025-04-30. Each release file adds to the
# previous ones and the window lies inside the newest, so only that file is read.
LIVECODEBENCH_FILE = "test6.jsonl"
LIVECODEBENCH_WINDOW = (datetime(2025, 1, 1), datetime(2025, 4, 30))
# The grader process tree gets 16 GiB of address space and 32768 processes (RLIMIT_NPROC counts
# every task of the uid, so this only caps fork bombs), and one wall-clock budget per pass.
MEMORY_BYTES = 17179869184
MAX_PROCESSES = 32768
EVALPLUS_TIMEOUT_SECONDS = 3600
LIVECODEBENCH_TIMEOUT_SECONDS = 10800


def _sandboxed(command: Sequence[str], timeout: int, env: Mapping[str, str] | None = None) -> None:
    def limits() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
        resource.setrlimit(resource.RLIMIT_NPROC, (MAX_PROCESSES, MAX_PROCESSES))

    completed = subprocess.run(
        list(command), preexec_fn=limits, timeout=timeout, capture_output=True, text=True, env=env
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{command[2]} exited {completed.returncode}: {completed.stderr[-300:]}"
        )


def _graded(task, keys, seeds, prompts, outputs, codes, passes) -> dict[str, Any]:
    samples = [
        sample_row(
            task["id"],
            item_id,
            index,
            seed,
            {"pass@1": float(bool(passes[item_id][index]))},
            output=output.text,
            extracted=code,
            finish_reason=output.finish_reason,
            prompt_tokens=len(prompt),
            generated_tokens=output.generated_tokens,
            truncated=output.finish_reason == "length",
            extraction_failed=not code,
        )
        for (item_id, index), seed, prompt, output, code in zip(
            keys, seeds, prompts, outputs, codes, strict=True
        )
    ]
    metrics = score_samples(task, samples)
    return {"results": task_result(metrics, samples), "samples": samples}


def run_evalplus_task(
    task: Mapping[str, Any], *, tokenizer: Any, generate: Callable[..., Sequence[Any]]
) -> dict[str, Any]:
    """HumanEval+ or MBPP+ with the official instruct prefill, graded on the Plus tests."""
    # build the official instruct prompts
    dataset = task["task"].split("_")[0]
    # The grader keys problems by the package's own ids (Mbpp/N), so the package is the one source
    # for prompts, entry points and ids.
    problems = list({"humaneval": get_human_eval_plus, "mbpp": get_mbpp_plus}[dataset]().values())
    prompts = [
        list(
            tokenizer.encode(
                make_raw_chat_prompt(
                    row["prompt"], EVALPLUS_INSTRUCTION_PREFIX, EVALPLUS_RESPONSE_PREFIX, tokenizer
                ),
                add_special_tokens=False,
            )
        )
        for row in problems
    ]
    # generate one seeded completion per problem and extract its code
    keys = [(str(row["task_id"]), 0) for row in problems]
    seeds = [item_seed(task, item_id, index) for item_id, index in keys]
    outputs = generate(prompts, seeds, task)
    codes = [
        sanitize(output.text, row.get("entry_point")) for row, output in zip(problems, outputs)
    ]

    # grade on the Plus tests in a sandboxed child
    with tempfile.TemporaryDirectory(prefix="echomoe-evalplus-") as workspace:
        solutions = Path(workspace) / "solutions.jsonl"
        lines = (
            json.dumps({"task_id": item_id, "solution": code}, ensure_ascii=False) + "\n"
            for (item_id, _), code in zip(keys, codes)
        )
        solutions.write_text("".join(lines))
        command = [sys.executable, "-m", "evalplus.evaluate", "--samples", str(solutions)]
        _sandboxed([*command, "--dataset", dataset, "--parallel", "16"], EVALPLUS_TIMEOUT_SECONDS)
        graded = json.loads((Path(workspace) / "solutions_eval_results.json").read_text())
    passes = {
        item_id: [
            record["base_status"] == "pass" and record["plus_status"] == "pass"
            for record in records
        ]
        for item_id, records in graded["eval"].items()
    }
    return _graded(task, keys, seeds, prompts, outputs, codes, passes)


def run_livecodebench_task(
    task: Mapping[str, Any],
    *,
    tokenizer: Any,
    generate: Callable[..., Sequence[Any]],
) -> dict[str, Any]:
    """LiveCodeBench with the official chat prompts, graded by its official codegen_metrics."""
    style = LMStyle.OpenAIChat

    # load the problem window from the release file at the task's pinned revision, since
    # datasets>=3 cannot run the official script-based loader
    path = hf_hub_download(
        repo_id=task["dataset_repository"],
        filename=LIVECODEBENCH_FILE,
        repo_type="dataset",
        revision=task["dataset_revision"],
    )
    start, end = LIVECODEBENCH_WINDOW
    with open(path, encoding="utf-8") as stream:
        problems = [CodeGenerationProblem(**json.loads(line)) for line in stream]
    problems = [problem for problem in problems if start <= problem.contest_date <= end]
    problems.sort(key=lambda problem: str(problem.question_id))

    # generate num_samples seeded completions per problem and extract their code
    keys, prompts = [], []
    for problem in problems:
        prompt = chat_tokens(tokenizer, format_prompt_generation(problem, style))
        for index in range(task["num_samples"]):
            keys.append((str(problem.question_id), index))
            prompts.append(prompt)
    seeds = [item_seed(task, item_id, index) for item_id, index in keys]
    outputs = generate(prompts, seeds, task)
    codes = [extract_code(output.text, style) for output in outputs]

    # grade with the official codegen_metrics in a sandboxed child
    generations: dict[str, list[str]] = {}
    for (item_id, _), code in zip(keys, codes):  # keys are in sample-index order per item
        generations.setdefault(item_id, []).append(code)
    item_ids = [str(problem.question_id) for problem in problems]
    payload = {
        "item_ids": item_ids,
        "samples": [problem.get_evaluation_sample() for problem in problems],
        "generations": [generations[item_id] for item_id in item_ids],
        "k_list": list(task["aggregation"]["k"]),
        "num_samples": task["num_samples"],
        "timeout": 6,
        "num_process_evaluate": 16,
    }
    with tempfile.TemporaryDirectory(prefix="echomoe-livecodebench-") as workspace:
        payload_path = Path(workspace) / "payload.json"
        payload_path.write_text(json.dumps(payload, ensure_ascii=False))
        # The child imports lcb_runner from the checkout and this module from the repository.
        env = {**os.environ, "PYTHONPATH": f"{LIVECODEBENCH}:{REPO}"}
        command = [sys.executable, "-m", "echomoe.inference.evaluation.code", str(payload_path)]
        _sandboxed(command, LIVECODEBENCH_TIMEOUT_SECONDS, env=env)
        passes = json.loads((Path(workspace) / "payload_passes.json").read_text())["passes"]
    return _graded(task, keys, seeds, prompts, outputs, codes, passes)


def _grade_livecodebench(path: Path) -> None:
    payload = json.loads(path.read_text())
    _metrics, results, _metadata = codegen_metrics(
        payload["samples"],
        payload["generations"],
        k_list=payload["k_list"],
        num_process_evaluate=payload["num_process_evaluate"],
        timeout=payload["timeout"],
    )
    passes = {}
    for index, item_id in enumerate(payload["item_ids"]):
        assert len(results[index]) == payload["num_samples"], item_id
        passes[item_id] = [all(score > 0 for score in generation) for generation in results[index]]
    path.with_name("payload_passes.json").write_text(json.dumps({"passes": passes}))


if __name__ == "__main__":
    _grade_livecodebench(Path(sys.argv[1]))
