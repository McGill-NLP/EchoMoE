"""The evaluation engine: one `vllm serve` per model variant, the lm-eval client that seeds every
sample, and the task runs written to `eval/step-N/<variant>/`.

The variants:

    folded           a folded export, bf16
    folded-fp8       a folded export, simulated FP8 dispatch and combine
    base-vllm        the base model through vLLM's own fused MoE
    base-blocks      the base model through the folded student's blocks
    base-blocks-fp8  base-blocks with simulated FP8
    lsh              an LSH-MoE export through the full-width blocks, without expert parallelism

A variant serves on its engine profile, or on the one `--engine-profile` names in its place.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sysconfig
import tempfile
import time
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import yaml
from lm_eval.models.openai_completions import LocalCompletionsAPI

from echomoe.runs import atomic_write_json

from .code import run_evalplus_task, run_livecodebench_task
from .ifbench import run_ifbench_task
from .model import (
    DEEPSEEK_V3_FOLDED_ARCHITECTURE,
    FOLDED_ARCHITECTURE,
    FULL_WIDTH_ARCHITECTURE,
    GEMMA4_FOLDED_ARCHITECTURE,
    LSH_KEY,
    NEMOTRON_H_FOLDED_ARCHITECTURE,
    PRECISION_KEY,
)
from .suite import (
    CONFIGS,
    SeedPlan,
    sampler_kwargs,
    suite_summary,
    task_is_complete,
    write_samples,
)
from .tasks import run_lm_eval_task

FOLDED_PROFILE = "folded/expert-parallel-fast-start.yaml"
BASE_PROFILE = "expert-parallel-allgather.yaml"
LSH_PROFILE = "data-parallel.yaml"
# variant: (engine profile, architecture named over the model's own, communication precision)
VARIANTS = {
    "folded": (FOLDED_PROFILE, None, "bf16"),
    "folded-fp8": (FOLDED_PROFILE, None, "fp8-all"),
    "base-vllm": (BASE_PROFILE, None, "bf16"),
    "base-blocks": (BASE_PROFILE, FULL_WIDTH_ARCHITECTURE, "bf16"),
    "base-blocks-fp8": (BASE_PROFILE, FULL_WIDTH_ARCHITECTURE, "fp8-all"),
    "lsh": (LSH_PROFILE, FULL_WIDTH_ARCHITECTURE, "bf16"),
}
# The folded student's block cannot be traced (vLLM's kernel-config lookup reaches pynvml, which
# Dynamo refuses), so it and the transport run uncompiled, with whole-model graphs on decode
# batches. base-vllm keeps vLLM's defaults.
DECODE_GRAPHS = {"cudagraph_mode": "FULL_DECODE_ONLY", "mode": "NONE"}
# LSH-MoE reads each forward's real token count on the host, which a captured graph would fix.
EAGER = {"cudagraph_mode": "NONE", "mode": "NONE"}
WORKER_EXTENSION = "echomoe.inference.evaluation.model.WorkerExtension"
SERVED_MODEL = "model"
MAX_MODEL_LEN = 32768
MAX_NUM_BATCHED_TOKENS = 8192
# A profile may set its own `max_num_seqs`, `gpu_memory_utilization`, `request_timeout_seconds`,
# `cudagraph_mode` (of a variant that sets one) and `max_model_len`. A shorter `max_model_len` caps
# the context: prompts are not cut, and each generation stops where the cap leaves no room.
# Sequences per data-parallel rank: at vLLM's 256 the engine pinned at 2,048 with requests queued.
MAX_NUM_SEQS = 1024
GPU_MEMORY_UTILIZATION = 0.88
ENGINE_SEED = 42
# Requests in flight per client. Eight clients keep a task's whole request set in flight.
REQUEST_CONCURRENCY = 2048
# A folded student captures its graphs slowly, and a request generating to a task's limit can
# queue behind thousands of others.
READY_SECONDS = 1800.0
COMPLETION_SECONDS = 1800


def engine_profile(variant: str, path: Path | None) -> dict[str, Any]:
    return yaml.safe_load((path or CONFIGS / "engines" / VARIANTS[variant][0]).read_text())


def serve_command(model: Path, variant: str, port: int, profile: Mapping[str, Any]) -> list[str]:
    _, architecture, precision = VARIANTS[variant]
    command = ["vllm", "serve", str(model), "--host", "127.0.0.1", "--port", str(port)]
    command += ["--served-model-name", SERVED_MODEL]
    command += ["--tensor-parallel-size", str(profile["tensor_parallel_size"])]
    command += ["--data-parallel-size", str(profile["data_parallel_size"])]
    if profile["expert_parallel"]:
        command += ["--enable-expert-parallel", "--all2all-backend", profile["all2all_backend"]]
    command += ["--attention-backend", profile["attention_backend"]]
    command += ["--worker-extension-cls", WORKER_EXTENSION]
    command += ["--performance-mode", profile["performance_mode"]]
    if profile["prefix_caching"]:
        command += ["--enable-prefix-caching"]
    command += ["--max-num-batched-tokens", str(MAX_NUM_BATCHED_TOKENS)]
    command += ["--max-model-len", str(profile.get("max_model_len", MAX_MODEL_LEN))]
    memory = profile.get("gpu_memory_utilization", GPU_MEMORY_UTILIZATION)
    command += ["--gpu-memory-utilization", f"{memory:.2f}"]
    command += ["--max-num-seqs", str(profile.get("max_num_seqs", MAX_NUM_SEQS))]
    command += ["--seed", str(ENGINE_SEED)]
    if variant != "base-vllm":
        graphs = EAGER if variant == "lsh" else DECODE_GRAPHS
        if "cudagraph_mode" in profile:
            graphs = graphs | {"cudagraph_mode": profile["cudagraph_mode"]}
        command += ["--compilation-config", json.dumps(graphs, sort_keys=True)]
    if architecture is not None:
        command += ["--hf-overrides", json.dumps({"architectures": [architecture]})]
    if precision != "bf16":
        command += ["--additional-config", json.dumps({PRECISION_KEY: precision})]
    return command


def check_model(model: Path, variant: str) -> None:
    """Refuse a folded variant of a base model, a base variant of a folded export, and an LSH-MoE
    export served as anything but `lsh`, or the reverse."""
    config = json.loads((model / "config.json").read_text())
    folded = config["architectures"][0] in (
        FOLDED_ARCHITECTURE,
        DEEPSEEK_V3_FOLDED_ARCHITECTURE,
        GEMMA4_FOLDED_ARCHITECTURE,
        NEMOTRON_H_FOLDED_ARCHITECTURE,
    )
    assert folded == variant.startswith("folded"), f"{model} cannot be served as {variant}"
    assert (LSH_KEY in config) == (variant == "lsh"), f"{model} cannot be served as {variant}"


@contextmanager
def vllm_server(
    command: Sequence[str], log: Path, environment: Mapping[str, str]
) -> Iterator[str]:
    """Run `vllm serve` until the block ends and yield its base URL once it is healthy."""
    base = f"http://127.0.0.1:{command[command.index('--port') + 1]}"
    env = {**os.environ, **environment, "VLLM_SERVER_DEV_MODE": "1"}
    # vLLM's unquantized-MoE oracle would serve the base model through FlashInfer CUTLASS at one
    # data-parallel rank and Triton above it, while the folded student always runs Triton. Pin
    # Triton so every served model uses one kernel: on the base export CUTLASS and Triton agree on
    # 40 of 64 greedy continuations, the folded full-width block and stock Triton on 64 of 64. The
    # teacher passes (capture, targets) keep vLLM's default, as the fits and targets were made.
    env["VLLM_USE_FLASHINFER_MOE_FP16"] = "0"
    # Prompt-logprob requests allocate after the KV cache is sized. Expandable segments grow in
    # place where the default allocator fragmented and failed.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # A cold load of the export from network storage on every rank can exceed vLLM's own 600 s
    # engine-ready timeout, so wait as long as the server's readiness deadline instead.
    env.setdefault("VLLM_ENGINE_READY_TIMEOUT_S", str(int(READY_SECONDS)))
    # Node-local compile caches: shared ones on NFS failed with stale handles and lock errors.
    cache = (
        Path(tempfile.gettempdir()) / "echomoe-evaluate" / command[2].strip("/").replace("/", "_")
    )
    env["VLLM_CACHE_ROOT"] = str(cache / "vllm")
    env["TRITON_CACHE_DIR"] = str(cache / "triton")
    env["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
    env["FLASHINFER_WORKSPACE_BASE"] = str(cache / "flashinfer")
    # FlashInfer's JIT ops need libtvm_ffi's symbols. Keep what the parent already preloads (a
    # site's newer libstdc++, which those JIT ops also need).
    tvm_ffi = str(Path(sysconfig.get_paths()["purelib"]) / "tvm_ffi/lib/libtvm_ffi.so")
    env["LD_PRELOAD"] = ":".join(filter(None, (os.environ.get("LD_PRELOAD"), tvm_ffi)))
    with open(log, "w") as stream:
        process = subprocess.Popen(
            list(command), stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, env=env
        )
        try:
            deadline = time.monotonic() + READY_SECONDS
            while True:
                assert process.poll() is None, f"vLLM exited before it was ready. See {log}"
                try:
                    with urllib.request.urlopen(f"{base}/health", timeout=5) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                assert time.monotonic() < deadline, f"vLLM was not ready in time. See {log}"
                time.sleep(1.0)
            yield base
        finally:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)


def collective_rpc(base: str, method: str) -> list[Any]:
    """Every rank's result of a worker-extension method."""
    request = urllib.request.Request(
        f"{base}/collective_rpc",
        data=json.dumps({"method": method, "timeout": 300}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        # vLLM returns the first engine's result, which gathered every rank's report.
        return json.loads(response.read())["results"][0]


def check_topology(base: str, profile: Mapping[str, Any], experts: int) -> None:
    """Refuse a server whose ranks do not serve the profile's expert-parallel layout."""
    reports = collective_rpc(base, "served_topology")
    size = profile["data_parallel_size"]
    # vLLM's expert-parallel group spans the data-parallel ranks even when experts are not split
    local = experts // size if profile["expert_parallel"] else experts
    expected = {
        "data_parallel_size": size,
        "expert_parallel_size": size,
        "experts": [[local, experts]],
    }
    assert reports == [expected] * size, f"the server serves {reports}, not {expected} per rank"


def capped_max_tokens(max_tokens: int, prompt_tokens: int, context_cap: int | None) -> int:
    """A request's output budget under a context cap. vLLM refuses a request whose prompt and
    budget exceed its length, so the budget is cut to the room the cap leaves."""
    if context_cap is None:
        return max_tokens
    return max(1, min(max_tokens, context_cap - prompt_tokens))


def tokenizer_prepends_bos(tokenizer: Any) -> bool:
    """Whether the tokenizer starts plain text with its bos (Kanana-2's does, Qwen3's has none)."""
    bos = tokenizer.bos_token_id
    return bos is not None and tokenizer("a", add_special_tokens=True).input_ids[:1] == [bos]


class SeededCompletions(LocalCompletionsAPI):
    """lm-eval's local-completions client with thinking off and a per-sample seed on every
    generation request, drawn from the `seed_plan` that `tasks.run_lm_eval_task` sets. Under a
    context cap each request's budget is cut to the room its prompt leaves."""

    seed_plan: SeedPlan | None = None
    context_cap: int | None = None

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
            enable_thinking=False,
            tools=None,
        )

    def tok_encode(
        self, string, left_truncate_len=None, add_special_tokens=False, truncation=False, **kwargs
    ):
        # lm-eval encodes without special tokens, so a model whose tokenizer starts plain text
        # with bos (Kanana-2) would score plain-text prompts without it. As lm-eval's own vLLM
        # client does, bos is added unless the text (a chat template's rendering) starts with it.
        if not tokenizer_prepends_bos(self.tokenizer):
            return super().tok_encode(
                string,
                left_truncate_len=left_truncate_len,
                add_special_tokens=add_special_tokens,
                truncation=truncation,
                **kwargs,
            )
        bos = self.tokenizer.bos_token_id
        encoded = super().tok_encode(string, truncation=truncation, **kwargs)
        rows = [encoded] if isinstance(string, str) else encoded
        rows = [row if row[:1] == [bos] else [bos, *row] for row in rows]
        if left_truncate_len:
            rows = [row[-left_truncate_len:] for row in rows]
        return rows[0] if isinstance(string, str) else rows

    def _create_payload(self, messages, generate=False, gen_kwargs=None, **kwargs):
        payload = super()._create_payload(
            messages, generate=generate, gen_kwargs=gen_kwargs, **kwargs
        )
        if generate and self.seed_plan is not None:
            payload["seed"] = self.seed_plan.next_seed(messages)
        if generate and self.context_cap is not None:
            prompts = messages if messages and isinstance(messages[0], list) else [messages]
            longest = max(len(prompt) for prompt in prompts)
            payload["max_tokens"] = capped_max_tokens(
                payload["max_tokens"], longest, self.context_cap
            )
        return payload


def completions_client(
    model: Path, api_base: str, profile: Mapping[str, Any]
) -> SeededCompletions:
    client = SeededCompletions(
        model=SERVED_MODEL,
        base_url=api_base,
        tokenizer=str(model),
        tokenizer_backend="huggingface",
        tokenized_requests=True,
        batch_size=1,
        num_concurrent=REQUEST_CONCURRENCY,
        timeout=profile.get("request_timeout_seconds", COMPLETION_SECONDS),
        max_length=MAX_MODEL_LEN,
        max_gen_toks=2048,
        seed=ENGINE_SEED,
    )
    client.context_cap = profile.get("max_model_len")
    return client


class Completion(NamedTuple):
    text: str
    finish_reason: str
    prompt_tokens: int
    generated_tokens: int


def generate(
    api_base: str,
    prompts: Sequence[Sequence[int]],
    seeds: Sequence[int],
    task: Mapping[str, Any],
    *,
    timeout: int,
    context_cap: int | None,
) -> list[Completion]:
    """One seeded completion per (prompt, seed), with the task's sampling values and limits. Each
    is its own request because a request carries one seed."""
    request = {
        "model": SERVED_MODEL,
        "max_tokens": task["max_generation"],
        "stop": list(task["stop"]),
        **sampler_kwargs(task),
    }

    def one(prompt: Sequence[int], seed: int) -> Completion:
        payload = {
            **request,
            "max_tokens": capped_max_tokens(task["max_generation"], len(prompt), context_cap),
            "prompt": [int(token) for token in prompt],
            "seed": seed,
        }
        posted = urllib.request.Request(
            api_base,
            data=json.dumps(payload, allow_nan=False).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(posted, timeout=timeout) as response:
            document = json.loads(response.read())
        (choice,), usage = document["choices"], document.get("usage") or {}
        return Completion(
            text=str(choice.get("text", "")),
            finish_reason=str(choice.get("finish_reason") or "stop"),
            prompt_tokens=int(usage.get("prompt_tokens", len(prompt))),
            generated_tokens=int(usage.get("completion_tokens", 0)),
        )

    with ThreadPoolExecutor(max_workers=REQUEST_CONCURRENCY) as pool:
        return list(pool.map(one, prompts, seeds))


def task_paths(output: Path, task: Mapping[str, Any]) -> tuple[Path, Path]:
    return output / "tasks" / f"{task['id']}.json", output / "samples" / f"{task['id']}.jsonl"


def run_task(
    output: Path, task: Mapping[str, Any], *, lm: SeededCompletions, ifbench: Path | None
) -> None:
    results, samples = task_paths(output, task)
    if task_is_complete(task, results, samples):
        return
    if task["runner"] == "lm_eval":
        produced = run_lm_eval_task(task, lm=lm)
    else:
        runner = {
            "evalplus": run_evalplus_task,
            "livecodebench": run_livecodebench_task,
            "ifbench": partial(run_ifbench_task, checkout=ifbench),
        }[task["runner"]]
        produced = runner(
            task,
            tokenizer=lm.tokenizer,
            generate=partial(
                generate, lm.base_url, timeout=lm.timeout, context_cap=lm.context_cap
            ),
        )
    # The task file is written last: it marks the task complete.
    write_samples(samples, produced["samples"])
    atomic_write_json(results, {"task": task, "results": produced["results"]})


def run_suite(
    output: Path,
    tasks: Sequence[Mapping[str, Any]],
    *,
    lm: SeededCompletions,
    ifbench: Path | None,
) -> None:
    """Run every task, then fail if any did. One failure does not stop the others."""
    failed = []
    for task in tasks:
        try:
            run_task(output, task, lm=lm, ifbench=ifbench)
        except Exception:
            logging.getLogger(__name__).exception("task %s failed", task["id"])
            failed.append(task["id"])
    if failed:
        raise RuntimeError(f"tasks failed: {failed}")


def write_summary(output: Path, suite: Mapping[str, Any]) -> None:
    tasks = {}
    for task in suite["tasks"]:
        results, samples = task_paths(output, task)
        complete = task_is_complete(task, results, samples)
        tasks[task["id"]] = (
            json.loads(results.read_text())["results"]["metrics"] if complete else None
        )
    if any(metrics is not None for metrics in tasks.values()):
        atomic_write_json(output / "summary.json", suite_summary(suite["suite"], tasks))


def plan_clients(tasks: Sequence[Mapping[str, Any]], clients: int) -> list[list[str]]:
    """Deal the tasks, longest first, each to the client with the fewest decode tokens so far, so
    a second task's requests fill the drain the first ends in."""
    cost = {
        task["id"]: task["num_samples"]
        * (1 if task["sampling_profile"] == "no_generation" else task["max_generation"])
        for task in tasks
    }
    groups: list[list[str]] = [[] for _ in range(min(clients, len(tasks)))]
    totals = [0] * len(groups)
    for task_id, tokens in sorted(cost.items(), key=lambda entry: (-entry[1], entry[0])):
        index = min(range(len(groups)), key=lambda client: (totals[client], client))
        groups[index].append(task_id)
        totals[index] += tokens
    return groups
