"""The lm-eval tasks of the suite: run one through lm-eval 0.4.12 and score every sample. The task
YAMLs bind `boxed_math_results` and `gpqa_process_docs` by dotted path (`!function`)."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

from lm_eval import simple_evaluate
from lm_eval.tasks import TaskManager
from math_verify import parse, verify

from .suite import (
    CONFIGS,
    SeedPlan,
    protocol_identity,
    sample_row,
    score_samples,
    task_result,
)

# A generation that ends within this many tokens of the context cap reached it. Its text is
# encoded again here, which can differ from the generated ids by a few tokens.
CAP_MARGIN = 8
_BOXED = re.compile(r"\\boxed\{")


def extract_boxed(text: str) -> str | None:
    """The last balanced \\boxed{...} body, or None when absent."""
    last = None
    for match in _BOXED.finditer(text):
        depth = 1
        index = match.end()
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            last = text[match.end() : index - 1].strip()
    return last


def boxed_math_results(doc: Mapping[str, Any], results: Sequence[str]) -> dict[str, float]:
    """Score one boxed math response with math_verify, counting no answer as wrong."""
    response = results[0] if results else ""
    if "</think>" in response:
        response = response.rsplit("</think>", 1)[1]
    predicted = extract_boxed(response)
    if predicted is None:
        return {"math_verify": 0.0}
    target = str(doc.get("answer", doc.get("Answer", "")))
    gold = parse(f"${target}$")
    candidate = parse(f"${predicted}$")
    return {"math_verify": float(bool(gold) and bool(candidate) and verify(gold, candidate))}


def _gpqa_option(text: str | None) -> str:
    if text is None:
        return " "
    return text.strip().replace(" [title]", ". ").replace("  ", " ")


def gpqa_process_docs(dataset: Any) -> Any:
    """Shuffle each GPQA question's options by a permutation hashed from the question text."""
    # lm-eval's own process_docs shuffles unseeded, so variants would see different orders.

    def _process(doc: Mapping[str, Any]) -> dict[str, Any]:
        options = [
            _gpqa_option(doc["Incorrect Answer 1"]),
            _gpqa_option(doc["Incorrect Answer 2"]),
            _gpqa_option(doc["Incorrect Answer 3"]),
            _gpqa_option(doc["Correct Answer"]),
        ]
        digest = hashlib.sha256(_gpqa_option(doc["Question"]).encode()).digest()
        order = sorted(range(4), key=lambda index: digest[index])
        shuffled = [options[index] for index in order]
        correct = shuffled.index(_gpqa_option(doc["Correct Answer"]))
        return {
            "choice1": shuffled[0],
            "choice2": shuffled[1],
            "choice3": shuffled[2],
            "choice4": shuffled[3],
            "choices": shuffled,
            "answer": "ABCD"[correct],
        }

    return dataset.map(_process)


def _normalized_answer(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    # Known bug, kept: stripping every non-word character turns each BBH dyck_languages target
    # ("] ) }") into "", so that subtask always scores 0, and drops signs and decimal points, so
    # "-3" matches "3" and "1.5" matches "15".
    stripped = re.sub(r"[^\w\s]", "", text).strip().lower()
    return re.sub(r"\s+", " ", stripped)


def lm_eval_item_id(task_id: str, leaf: str, doc_id: int) -> str:
    """The item id of one lm-eval document. It keys the samples and orders their sum."""
    payload = json.dumps(
        [task_id, leaf, doc_id, "report"], ensure_ascii=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _unwrap_samples(value: Any, num_samples: int) -> Any:
    if (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], list)
        and (num_samples != 1 or len(value[0]) == 1)
    ):
        return value[0]
    return value


def _native_metrics(
    task: Mapping[str, Any], aggregate: Mapping[str, Any], leaves: Mapping[str, Any]
) -> dict[str, float]:
    # A group's score is the unweighted mean of its leaves, so MMLU 5-shot is the macro mean over
    # its 57 subjects.
    values: dict[str, list[float]] = {metric: [] for metric in task["metrics"]["report"]}
    for leaf in leaves:
        row = aggregate.get(leaf)
        if not isinstance(row, Mapping):
            continue
        for metric in values:
            for key, value in row.items():
                if key.partition(",")[0] == metric and isinstance(value, Real):
                    values[metric].append(float(value))
    return {metric: sum(found) / len(found) for metric, found in values.items()}


def run_lm_eval_task(task: Mapping[str, Any], *, lm: Any) -> dict[str, Any]:
    """`lm` is an lm-eval model whose requests take their seeds from the `seed_plan` set here."""
    identity = protocol_identity(task)
    num_samples, primary, seed = task["num_samples"], task["metrics"]["primary"], task["root_seed"]
    generates = task["sampling"]["mode"] != "none"
    plan = SeedPlan(identity, task["id"], seed) if generates else None
    lm.seed_plan = plan

    def count_tokens(text: Any) -> int:
        return len(lm.tok_encode(text)) if isinstance(text, str) and text else 0

    # load the task's leaves and make a multi-sample task repeat every request
    manager = TaskManager(include_path=str(CONFIGS / "tasks"))
    leaves = dict(manager.load([task["task"]])["tasks"])
    for leaf in leaves.values():
        if num_samples > 1:
            leaf.set_config("repeats", num_samples)

    # run lm-eval with the protocol's seeds and generation settings
    arguments: dict[str, Any] = {
        "model": lm,
        # The leaf objects, not the task name: a name is reloaded and loses its `repeats`.
        "tasks": list(leaves.values()),
        "num_fewshot": task["num_fewshot"],
        "batch_size": "auto",
        "apply_chat_template": task["apply_chat_template"],
        "fewshot_as_multiturn": False,
        "log_samples": True,
        "task_manager": manager,
        "random_seed": seed,
        "numpy_random_seed": seed,
        "torch_random_seed": seed,
        "fewshot_random_seed": seed,
    }
    if generates:
        arguments["gen_kwargs"] = {
            **{name: value for name, value in task["sampling"].items() if name != "mode"},
            "max_gen_toks": task["max_generation"],
            "until": list(task["stop"]),
        }
    # Known bug, kept: the task's dataset_revision is not passed on, so lm-eval reads the hub's
    # current snapshot.
    result = simple_evaluate(**arguments)

    # turn every logged sample of the reported filter into a samples.jsonl row
    rows_out = []
    for leaf, rows in sorted(result["samples"].items()):
        for row in rows:
            # lm-eval logs one row per filter. The protocol names the reported one.
            if (
                task["metric_filter"] is not None
                and str(row.get("filter")) != task["metric_filter"]
            ):
                continue
            item_id = lm_eval_item_id(task["id"], leaf, row["doc_id"])
            prompt = row["arguments"][0][0]
            if not generates:
                native = {
                    metric: float(row[metric])
                    for metric in task["metrics"]["report"]
                    if metric in row
                }
                rows_out.append(
                    sample_row(
                        task["id"],
                        item_id,
                        0,
                        seed,
                        native,
                        finish_reason=None,
                        prompt_tokens=count_tokens(prompt),
                    )
                )
                continue

            # `repeats` puts every sample inside one request, so both lists arrive nested.
            responses = _unwrap_samples(row["resps"], num_samples)
            filtered = _unwrap_samples(row["filtered_resps"], num_samples)
            raw_target = row.get("target")
            # Known bug, kept: IFEval's target is the integer 0, so its per-sample exact_match is
            # always 0. Its score is the native metric.
            target = _normalized_answer(raw_target)
            # Known bug, kept: the recorded seed is keyed on the prompt text, while the backend
            # seeds the request from its own payload.
            request_key = SeedPlan.request_key(prompt)
            for index in range(num_samples):
                output = responses[index]
                if primary == "math_verify":
                    # Known bug, kept: only a \boxed{} answer is extracted, also on
                    # minerva_math500, whose few-shot prompt asks for "Final Answer: ...".
                    body = output if isinstance(output, str) else ""
                    if "</think>" in body:
                        body = body.rsplit("</think>", 1)[1]
                    extracted = extract_boxed(body)
                    failed = extracted is None
                    correct = (
                        0.0
                        if failed
                        else boxed_math_results({"answer": raw_target}, [body])["math_verify"]
                    )
                else:
                    extracted = filtered[index]
                    # "[invalid]" is lm-eval's marker for a filter that matched nothing.
                    failed = not isinstance(extracted, str) or extracted in ("", "[invalid]")
                    correct = float(
                        not failed and _normalized_answer(extracted) == target and target != ""
                    )
                generated_tokens = count_tokens(output)
                prompt_tokens = count_tokens(prompt)
                rows_out.append(
                    sample_row(
                        task["id"],
                        item_id,
                        index,
                        plan.seed_for(request_key, index),
                        {primary: correct},
                        output=output if isinstance(output, str) else None,
                        extracted=extracted if isinstance(extracted, str) else None,
                        prompt_tokens=prompt_tokens,
                        generated_tokens=generated_tokens,
                        extraction_failed=failed,
                        # lm-eval drops the finish reason: a generation at its budget, or with its
                        # prompt at the context cap, is truncated.
                        truncated=generated_tokens >= task["max_generation"]
                        or (
                            lm.context_cap is not None
                            and prompt_tokens + generated_tokens >= lm.context_cap - CAP_MARGIN
                        ),
                    )
                )

    # the task score: lm-eval's own aggregate, or our mean or pass@k over the rows
    if task["aggregation"]["method"] == "native":
        metrics = _native_metrics(task, result["results"], leaves)
    else:
        metrics = score_samples(task, rows_out)
    generated = len(rows_out) if generates else 0
    return {"results": task_result(metrics, rows_out, generated), "samples": rows_out}
