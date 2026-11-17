"""The report-core protocol: task documents, per-sample seeds, sample rows, and the suite mean.

A task's resolved document is hashed into `protocol_identity`, which seeds every lm-eval
generation, so any change to it resamples those tasks. The reported number is `suite_mean`.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from echomoe.runs import REPO, atomic_write_bytes

CONFIGS = REPO / "configs" / "inference" / "evaluation"
# Two tasks report two metrics. The suite mean reads the one named here.
PREFERRED_METRIC = {"ifeval_chat": "prompt_level_strict_acc", "livecodebench_chat": "pass@1"}


def load_suite(
    path: str | Path = CONFIGS / "report-core.yaml",
    sampling: str | Path = CONFIGS / "sampling.yaml",
) -> dict[str, Any]:
    """The suite id and the resolved document of each task, in suite order."""
    profiles = yaml.safe_load(Path(sampling).read_text())["profiles"]
    document = yaml.safe_load(Path(path).read_text())
    tasks = []
    for task in document["tasks"]:
        # Retyped (numbers as floats, an integer top_k, a mean's k as a list) because the JSON
        # spelling of every value is hashed into the seeds.
        values = {
            name: value if name in ("mode", "do_sample", "top_k") else float(value)
            for name, value in profiles[task["sampling_profile"]].items()
        }
        method, k = task["aggregation"]["method"], task["aggregation"]["k"]
        aggregation = {"method": method, "k": [k] if type(k) is int else k}
        tasks.append({**task, "sampling": values, "aggregation": aggregation})
    return {"suite": document["suite"], "tasks": tasks}


def protocol_identity(task: Mapping[str, Any]) -> str:
    """The sha256 of one resolved task document."""
    payload = json.dumps(task, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def sampler_kwargs(task: Mapping[str, Any]) -> dict[str, Any]:
    """A task's sampling values as direct vLLM request arguments (vLLM rejects `do_sample`)."""
    return {
        name: value
        for name, value in task["sampling"].items()
        if name not in ("mode", "do_sample")
    }


def chat_tokens(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> list[int]:
    """The prompt token ids of a chat, rendered without thinking."""
    rendered = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        tools=None,
    )
    token_ids = rendered.get("input_ids") if isinstance(rendered, Mapping) else rendered
    return [int(token) for token in token_ids]


def derive_seed(
    protocol_identity: str, task_id: str, item_id: str, sample_index: int, root_seed: int
) -> int:
    """The seed of one sample: sha256 of the length-prefixed components, low 63 bits."""
    payload = b"".join(
        len(part.encode()).to_bytes(8, "big") + part.encode()
        for part in (protocol_identity, task_id, item_id, str(sample_index), str(root_seed))
    )
    # vLLM types the seed as a signed 64-bit integer.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (2**63 - 1)


def item_seed(task: Mapping[str, Any], item_id: str, sample_index: int) -> int:
    """The seed of an EvalPlus, LiveCodeBench or IFBench sample, keyed on the task id."""
    return derive_seed(task["id"], task["id"], item_id, sample_index, task["root_seed"])


@dataclass
class SeedPlan:
    """Seeds for one lm-eval task, whose repeats arrive as anonymous requests. Each request is
    keyed by a digest of its content and its repeats are counted for the sample index."""

    protocol_identity: str
    task_id: str
    root_seed: int
    _counters: dict[str, int] = field(default_factory=dict, init=False)

    @staticmethod
    def request_key(content: Any) -> str:
        return hashlib.sha256(repr(content).encode()).hexdigest()

    def seed_for(self, key: str, sample_index: int) -> int:
        return derive_seed(self.protocol_identity, self.task_id, key, sample_index, self.root_seed)

    def next_seed(self, content: Any) -> int:
        # Not capped: a replayed request (chunked batch, retry) draws the next index.
        key = self.request_key(content)
        index = self._counters.get(key, 0)
        self._counters[key] = index + 1
        return self.seed_for(key, index)


def sample_row(
    task_id: str,
    item_id: str,
    sample_index: int,
    seed: int,
    native_metrics: Mapping[str, float],
    **fields: Any,
) -> dict[str, Any]:
    """One samples.jsonl row: the generation and its scores, never the prompt or target."""
    return {
        "output": None,
        "extracted": None,
        "finish_reason": "stop",
        "prompt_tokens": 0,
        "generated_tokens": 0,
        "truncated": False,
        "extraction_failed": False,
        **fields,
        "task_id": task_id,
        "item_id": item_id,
        "sample_index": sample_index,
        "seed": seed,
        "native_metrics": dict(native_metrics),
    }


def task_result(
    metrics: Mapping[str, float],
    samples: Sequence[Mapping[str, Any]],
    generated_samples: int | None = None,
) -> dict[str, Any]:
    """The results envelope of one task. A likelihood task passes generated_samples=0."""
    total = len(samples)
    return {
        "metrics": dict(metrics),
        "examples": len({row["item_id"] for row in samples}),
        "generated_samples": total if generated_samples is None else generated_samples,
        "extraction_failed_rate": (
            sum(row["extraction_failed"] for row in samples) / total if total else 0.0
        ),
        "truncated_rate": (sum(row["truncated"] for row in samples) / total if total else 0.0),
    }


def write_samples(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically write sample rows ordered by (item, sample index)."""
    ordered = sorted(rows, key=lambda row: (row["item_id"], row["sample_index"]))
    lines = (json.dumps(row, ensure_ascii=False, sort_keys=True) for row in ordered)
    atomic_write_bytes(Path(path), ("\n".join(lines) + "\n").encode())


def load_samples(path: str | Path) -> list[dict[str, Any]]:
    # Records are "\n"-separated. splitlines() would also split a generation at U+2028 and kin.
    return [json.loads(line) for line in Path(path).read_text().split("\n") if line]


def unbiased_pass_at_k(n: int, correct: int, k: int) -> float:
    if n - correct < k:
        return 1.0
    failure = 1.0
    for index in range(k):
        failure *= (n - correct - index) / (n - index)
    return 1.0 - failure


def score_samples(task: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    # A native task reports its scorer's own metric, whose denominator is not always the item
    # count (MMLU averages subjects), so it is never recomputed here.
    method, ks = task["aggregation"]["method"], task["aggregation"]["k"]
    if method == "native":
        return {}
    metric = task["metrics"]["primary"]
    by_item: dict[str, list[float]] = {}
    for row in rows:
        by_item.setdefault(row["item_id"], []).append(float(row["native_metrics"][metric]))
    matrix = [by_item[item] for item in sorted(by_item)]
    if method == "pass":
        scores = {}
        for k in ks:
            contributions = [unbiased_pass_at_k(len(item), int(sum(item)), k) for item in matrix]
            scores[f"pass@{k}"] = sum(contributions) / len(contributions)
        return scores
    if method == "mean":
        contributions = [float(sum(item)) / len(item) for item in matrix]
    else:
        contributions = [float(item[0]) for item in matrix]
    return {metric: sum(contributions) / len(contributions)}


def task_is_complete(
    task: Mapping[str, Any], results_path: str | Path, samples_path: str | Path
) -> bool:
    """Whether a task's results exist and recompute from its samples."""
    if not (Path(results_path).is_file() and Path(samples_path).is_file()):
        return False
    results = json.loads(Path(results_path).read_text())["results"]
    for name, value in score_samples(task, load_samples(samples_path)).items():
        assert math.isclose(results["metrics"][name], value, rel_tol=1e-9, abs_tol=1e-12), name
    return True


def task_score(task_id: str, metrics: Mapping[str, float]) -> float:
    """The one score a task contributes to the suite mean."""
    if len(metrics) == 1:
        return next(iter(metrics.values()))
    return metrics[PREFERRED_METRIC[task_id]]


def suite_mean(tasks: Mapping[str, Mapping[str, float] | None]) -> float:
    """The unweighted mean of the task scores. An incomplete task (None) is absent, not zero."""
    # Summed in task-id order, so the mean is the same float whatever order the tasks finish in.
    scores = [task_score(name, tasks[name]) for name in sorted(tasks) if tasks[name] is not None]
    return sum(scores) / len(scores)


def suite_summary(suite_id: str, tasks: Mapping[str, Mapping[str, float] | None]) -> dict:
    complete = all(metrics is not None for metrics in tasks.values())
    return {
        "suite": suite_id,
        "tasks": dict(tasks),
        "mean": suite_mean(tasks),
        "complete": complete,
    }
