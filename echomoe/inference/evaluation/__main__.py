"""Score one model variant on the report-core suite, on one node of 8 GPUs.

    . scripts/env.sh vllm
    python -m echomoe.inference.evaluation server \
        --model outputs/qwen3-30b-a3b/runs/<run>/export/step-N \
        --variant folded --ifbench-checkout third-party/IFBench [--tasks a,b] [--clients 8] \
        [--engine-profile configs/inference/evaluation/engines/<model>/<profile>.yaml]

`server` starts `vllm serve` for the variant (data parallel 8, expert parallel 8 but for `lsh`),
checks the layout every rank serves, runs the lm-eval clients against it and writes
`summary.json` once they finish. Each `client` it starts scores its share of the tasks.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

from echomoe.fit import moe_layout
from echomoe.runs import atomic_write_json, git_state, write_meta

from .engine import (
    VARIANTS,
    check_model,
    check_topology,
    collective_rpc,
    completions_client,
    engine_profile,
    plan_clients,
    run_suite,
    serve_command,
    vllm_server,
    write_summary,
)
from .suite import load_suite

# The server hands its model, variant and endpoint to its clients, which refuse to score anything
# else. A client that disagreed would file another variant's scores under its own name.
MODEL_VARIABLE = "ECHOMOE_SERVED_MODEL"
VARIANT_VARIABLE = "ECHOMOE_SERVED_VARIANT"
API_VARIABLE = "ECHOMOE_API_BASE"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score one model variant on report-core.")
    parser.add_argument("command", choices=("server", "client"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--output", type=Path, help="default <run>/eval/step-N/<variant>")
    parser.add_argument("--tasks", help="comma-separated task ids (default: the whole suite)")
    parser.add_argument("--ifbench-checkout", type=Path)
    parser.add_argument("--clients", type=int, default=8, help="server: lm-eval clients")
    parser.add_argument("--engine-profile", type=Path, help="in place of the variant's profile")
    args = parser.parse_args()
    profile = engine_profile(args.variant, args.engine_profile)
    model = args.model.resolve()
    if args.output is not None:
        output = args.output.resolve()
    else:
        assert model.parent.name == "export", f"give --output for {model}, not an export"
        output = model.parents[1] / "eval" / model.name / args.variant
    suite = load_suite()
    tasks = suite["tasks"]
    if args.tasks is not None:
        chosen = args.tasks.split(",")
        assert set(chosen) <= {task["id"] for task in tasks}, f"unknown tasks in {chosen}"
        tasks = [task for task in tasks if task["id"] in chosen]
    assert args.ifbench_checkout or all(task["runner"] != "ifbench" for task in tasks), (
        "IFBench needs --ifbench-checkout"
    )
    assert not git_state()["dirty"], "a reported evaluation needs a clean checkout"

    if args.command == "client":
        assert os.environ.get(MODEL_VARIABLE) == str(model), "the server serves another model"
        assert os.environ.get(VARIANT_VARIABLE) == args.variant, (
            "the server serves another variant"
        )
        lm = completions_client(model, os.environ[API_VARIABLE], profile)
        run_suite(output, tasks, lm=lm, ifbench=args.ifbench_checkout)
        return

    # one output directory holds one model, variant and suite
    check_model(model, args.variant)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "meta.json").exists():
        recorded = json.loads((output / "meta.json").read_text())
        assert recorded["inputs"]["model"] == str(model), f"{output} scores another model"
        assert recorded["variant"] == args.variant, f"{output} scores another variant"
        assert recorded["suite"] == suite, f"{output} was scored with another suite"

    # start the server and check every rank serves the profile's layout
    with socket.socket() as free:
        free.bind(("127.0.0.1", 0))
        port = free.getsockname()[1]
    command = serve_command(model, args.variant, port, profile)
    if not (output / "meta.json").exists():
        write_meta(
            output, inputs={"model": model}, variant=args.variant, server=command, suite=suite
        )
    environment = {MODEL_VARIABLE: str(model), VARIANT_VARIABLE: args.variant}
    experts = moe_layout(json.loads((model / "config.json").read_text()))[1]
    with vllm_server(command, output / "server.log", environment) as base:
        check_topology(base, profile, experts)

        # run the clients, each with its share of the tasks, then summarize what completed
        environment[API_VARIABLE] = f"{base}/v1/completions"
        client = [sys.executable, "-m", "echomoe.inference.evaluation", "client"]
        client += ["--model", str(model), "--variant", args.variant, "--output", str(output)]
        if args.ifbench_checkout:
            client += ["--ifbench-checkout", str(args.ifbench_checkout)]
        if args.engine_profile:
            client += ["--engine-profile", str(args.engine_profile)]
        running = [
            subprocess.Popen(
                [*client, "--tasks", ",".join(group)], env={**os.environ, **environment}
            )
            for group in plan_clients(tasks, args.clients)
        ]
        failed = sum(process.wait() != 0 for process in running)
        if args.variant == "lsh":
            # tokens, DeepEP rows and clusters by rank, layer and phase, which
            # `echomoe.lsh.rdma_volume` prices
            atomic_write_json(
                output / f"lsh-counts-{tasks[0]['id']}.json",
                {
                    "tasks": [task["id"] for task in tasks],
                    "ranks": collective_rpc(base, "lsh_counts"),
                },
            )
    write_summary(output, suite)
    assert not failed, f"{failed} clients failed. See the log and rerun to finish the suite"


if __name__ == "__main__":
    main()
