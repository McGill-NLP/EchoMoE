"""Online serving (Table 35, Figures 19-21): vLLM's own server (`vllm serve`, data parallel
attention and expert parallel MoE layers over 4 nodes of 8 GPUs) under Poisson arrivals of real
prompts, the uncompressed teacher against the folded student. Every server process applies the
throughput rows' patches (`install`, run by the throughput plugin when ECHOMOE_SERVING is set), so
the server runs the measured kernels with the per-call transport choice of the cost tables.

    # the prompt files, once (.venv-megatron, which reads the data recipe's split with pyarrow)
    python -m echomoe.inference.throughput.serving prompts <data dir> <dataset snapshot> \
        <tokenizer> outputs/qwen3-30b-a3b/serving/prompts
    # on each node, base first, then the same with --variant folded --model <export>
    python -m echomoe.inference.throughput.serving run --config \
        configs/inference/throughput/serving-online.yaml --variant base --model <teacher> \
        --tokenizer <teacher> --output outputs/qwen3-30b-a3b/serving/<setting> --nodes 4 \
        --node-rank $R [--fp8] [--triton-seed outputs/qwen3-30b-a3b/serving/triton-seed]
    # the Triton kernels launches compiled, merged into the seed the measured launches read
    python -m echomoe.inference.throughput.serving seed <launch output>... \
        --into outputs/qwen3-30b-a3b/serving/triton-seed
    # the balanced-routing prompts, each reply as long as the teacher's with its own routing
    # (vLLM's loader needs pandas: PYTHONPATH=.venv-vllm029/serving-client, scripts/install.sh)
    python -m echomoe.inference.throughput.serving fixed-replies <teacher's launch> \
        configs/inference/throughput/serving-online.yaml <prompts.jsonl> <tokenizer> <out> 128 256

Node 0 runs the API servers and 8 engines, waits for the server, then runs `vllm bench serve` once
per warm-up phase and once per request rate, as `clients` processes that together make one Poisson
stream. The other nodes run headless engines until node 0 is done. Each client writes
`<output>/<variant>/rate-<rate>-part<client>.json`, and `rate_summary` reduces a rate's records to
a row of `results/throughput/serving.csv` (scripts/throughput_tables.py).

EchoMoE's kernels compile a variant per reference-cache bucket and rank, and each first compile
mid-rate stalls every rank: a launch whose seed held none of them compiled 17,284 kernels and
stalled at 16 and 128 requests/s. So each setting and variant is launched once to compile, `seed`
merges the kernels every process kept under `<output>/<variant>/triton/`, and the measured launches
read that seed.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import yaml

from echomoe.runs import REPO

CONFIGS = REPO / "configs/inference/throughput"
# Qwen3-30B-A3B's expert GEMM tiles, as the CLI's
EXPERT_TILES = CONFIGS / "tiles.json"
GPUS_PER_NODE = 8
# engine.yaml's options for the offline engine only
OFFLINE_OPTIONS = (
    "disable_detokenize",
    "distributed_executor_backend",
    "num_iters_warmup",
    "num_iters",
)
SERVER_PORT, RPC_PORT, SEED = 8000, 13345, 42
# the prompt files: prompts up to 8,192 tokens with replies capped at 4,096, agent turns of 1,500
# histories with replies capped at 256
MAX_PROMPT_TOKENS, MAX_REPLY_TOKENS = 8192, 4096
AGENT_PROMPTS, AGENT_REPLY_TOKENS = 1500, 256
# the record fields a rate's summary reads (never the generated texts)
RECORD_FIELDS = ("start_times", "ttfts", "itls", "output_lens", "input_lens", "errors")


def command_line(options):
    """vLLM's command-line flags for engine options."""
    arguments = []
    for key, value in options.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            arguments.append(flag)
        elif value is False:
            arguments.append(flag.replace("--", "--no-", 1))
        else:
            arguments += [flag, json.dumps(value) if isinstance(value, dict) else str(value)]
    return arguments


# --- in every server process -------------------------------------------------------------------


def install():
    """The throughput CLI's patches for Qwen3-30B-A3B, from ECHOMOE_SERVING ({variant, fp8, costs,
    triton_seed}). The folded architecture is the plugin's own registration."""
    from vllm.v1.worker import gpu_worker

    from .fairness import (
        install_dp_sync,
        install_fp8_dispatch,
        install_live_experts,
        install_nic_rate,
        install_stock_decode_guard,
    )
    from .kernels import fp8_cast_back
    from .selection import install_selected

    settings = json.loads(os.environ["ECHOMOE_SERVING"])
    process_environment(settings)
    gpu_worker.init_worker_distributed_environment = exporting_local_rank(
        gpu_worker.init_worker_distributed_environment
    )
    install_dp_sync()
    install_live_experts(json.loads(EXPERT_TILES.read_text()))
    install_selected(settings["costs"], fp8_combine=settings["fp8"])
    if settings["fp8"]:
        from deep_ep.buffers import elastic

        assert "deepep-fp8c" in elastic.__file__, elastic.__file__
        install_fp8_dispatch(fp8_cast_back)
    if settings["variant"] == "base":
        install_stock_decode_guard(tiled=True)
    if "NCCL_IB_HCA" in os.environ:
        install_nic_rate()


def exporting_local_rank(original):
    """The two-hop kernels build once per host, on LOCAL_RANK 0, and the FP8 combine groups ranks
    by node with LOCAL_WORLD_SIZE. torchrun sets both, vLLM's workers neither: a worker's local
    rank is its GPU's index on the node, and a node runs data_parallel_size_local engines of
    world_size."""

    def initialize(vllm_config, rank, distributed_init_method=None, local_rank=-1, *args, **kw):
        parallel = vllm_config.parallel_config
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["LOCAL_WORLD_SIZE"] = str(
            parallel.data_parallel_size_local * parallel.world_size
        )
        return original(vllm_config, rank, distributed_init_method, local_rank, *args, **kw)

    return initialize


def process_environment(settings):
    """A server process's environment: DeepEP's hybrid mode with multiple reductions, as the CLI
    sets it, and its own Triton cache under the launch's `triton_cache`, kept for `seed`, read
    through to `triton_seed` if set."""
    os.makedirs(settings["triton_cache"], exist_ok=True)
    prefix = f"{os.uname().nodename}-{os.getpid()}-"
    os.environ.update(
        VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE="1",
        VLLM_DEEPEP_V2_ALLOW_MULTIPLE_REDUCTION="1",
        TRITON_CACHE_DIR=tempfile.mkdtemp(prefix=prefix, dir=settings["triton_cache"]),
    )
    if settings.get("triton_seed"):
        os.environ["TRITON_CACHE_MANAGER"] = f"{__name__}:SeededCacheManager"


def seeded_cache_manager():
    from triton.runtime.cache import FileCacheManager

    class SeededCacheManager(FileCacheManager):
        """Triton's file cache in this process's own directory, falling back to the seed. Writes
        stay private: on NFS, one process replacing a file another reads fails the read."""

        def __init__(self, key, override=False, dump=False):
            super().__init__(key, override, dump)
            seed = json.loads(os.environ["ECHOMOE_SERVING"])["triton_seed"]
            self.seed = os.path.join(seed, key)

        def get_file(self, filename):
            seeded = os.path.join(self.seed, filename)
            return super().get_file(filename) or (seeded if os.path.exists(seeded) else None)

        def get_group(self, filename):
            group = super().get_group(filename)
            seeded = os.path.join(self.seed, f"__grp__{filename}")
            if group is not None or not os.path.exists(seeded):
                return group
            with open(seeded) as f:
                children = json.load(f)["child_paths"]
            # a group file names its children where they compiled. The seed holds them beside it
            paths = {c: os.path.join(self.seed, os.path.basename(p)) for c, p in children.items()}
            return {c: p for c, p in paths.items() if os.path.exists(p)}

    return SeededCacheManager


def __getattr__(name):
    # Triton imports the cache manager by name. Its base class is imported only then
    if name == "SeededCacheManager":
        return seeded_cache_manager()
    raise AttributeError(name)


# --- the launcher, on each node ------------------------------------------------------------------


def serve_options(config, *, variant, fp8, nodes):
    """`vllm serve`'s engine options: engine.yaml's, the serving limits, and the folded loader."""
    engine = yaml.safe_load((CONFIGS / "engine.yaml").read_text())
    options = {key: value for key, value in engine.items() if key not in OFFLINE_OPTIONS}
    options.update(
        max_model_len=config["max_model_len"],
        max_num_seqs=config["max_num_seqs"],
        kv_cache_dtype=config.get("kv_cache_dtype", "auto"),
        seed=SEED,
        data_parallel_size=nodes * GPUS_PER_NODE,
        data_parallel_size_local=GPUS_PER_NODE,
        data_parallel_rpc_port=RPC_PORT,
    )
    if variant == "folded":
        from .model import ARCHITECTURE, LOAD_FORMAT

        options.update(
            load_format=LOAD_FORMAT,
            hf_overrides={"architectures": [ARCHITECTURE]},
            model_loader_extra_config={"fp8": fp8},
        )
    return options


def server_environment(config, *, variant, fp8, triton_seed, triton_cache):
    """The variables every server process starts with."""
    costs = CONFIGS / ("costs-fp8.csv" if fp8 else "costs-bf16.csv")
    settings = dict(
        variant=variant,
        fp8=fp8,
        costs=str(costs),
        triton_seed=triton_seed,
        triton_cache=str(triton_cache),
    )
    environment = dict(
        os.environ,
        ECHOMOE_SERVING=json.dumps(settings),
        VLLM_PLUGINS="echomoe_throughput",
        HF_HUB_OFFLINE="1",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        # FlashInfer's cached JIT top-k/top-p sampler needs a newer libstdc++ than the container's
        VLLM_USE_FLASHINFER_SAMPLER="0",
        # the folded student's engines take longer than vLLM's 600 s to start
        VLLM_ENGINE_READY_TIMEOUT_S=str(config["ready_seconds"]),
    )
    if config["routing"] == "balanced":
        environment["VLLM_MOE_ROUTING_SIMULATION_STRATEGY"] = "uniform_random"
    if fp8:
        # the copy of DeepEP whose combine crosses nodes in FP8 (scripts/install.sh), its own cache
        path = os.path.join(sys.prefix, "deepep-fp8c")
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(None, [path, os.environ.get("PYTHONPATH")])
        )
        environment["EP_JIT_CACHE_DIR"] = str(
            REPO / "outputs/deepep-jit-cache-fp8c" / os.uname().nodename
        )
    return environment


def client_command(config, *, model, tokenizer, name, rate, prompts, client, out):
    """One client process of a rate: a Poisson stream at rate / clients, through its own loopback
    address (an overloaded server holds more connections than one address has ports)."""
    clients = config["clients"]
    command = ["vllm", "bench", "serve", "--backend", "vllm"]
    command += ["--base-url", f"http://127.0.0.{client + 1}:{SERVER_PORT}"]
    command += ["--model", str(model), "--tokenizer", str(tokenizer)]
    command += ["--dataset-name", "custom", "--dataset-path", str(config["prompts"])]
    command += ["--custom-output-len", "-1", "--skip-chat-template"]
    command += ["--num-prompts", str(prompts // clients)]
    command += ["--request-rate", str(rate / clients), "--burstiness", "1"]
    command += ["--seed", str(SEED + client), "--temperature", str(config["temperature"])]
    command += ["--top-p", str(config["top_p"]), "--top-k", str(config["top_k"])]
    command += ["--ignore-eos"] if config.get("ignore_eos") else []
    command += ["--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,90,99"]
    command += ["--save-result", "--save-detailed", "--result-dir", str(out)]
    command += ["--result-filename", f"rate-{name}-part{client}.json", "--disable-tqdm"]
    return command


def rate_plan(config):
    """(rate name, requests per second, prompts) of each warm-up phase, then of each timed rate."""
    plan = [(f"warmup{i}", p["rate"], p["num_prompts"]) for i, p in enumerate(config["warmup"])]
    for rate in config["rates"]:
        plan.append((str(rate), rate, round(rate * config["seconds"])))
    return plan


def healthy():
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{SERVER_PORT}/health", timeout=5)
    except OSError:
        return False
    return True


def stop(server):
    """End this launcher's own server process group."""
    for signal_number, wait in ((15, 20), (9, 0)):
        try:
            os.killpg(server.pid, signal_number)
        except ProcessLookupError:
            return
        time.sleep(wait)


def run(args):
    config = yaml.safe_load(args.config.read_text())
    out = args.output / args.variant
    out.mkdir(parents=True, exist_ok=True)
    options = serve_options(config, variant=args.variant, fp8=args.fp8, nodes=args.nodes)
    environment = server_environment(
        config,
        variant=args.variant,
        fp8=args.fp8,
        triton_seed=args.triton_seed,
        triton_cache=(out / "triton").resolve(),
    )
    command = ["vllm", "serve", str(args.model), *command_line(options)]
    # vLLM binds its data-parallel sockets to the address it is given: node 0 serves on its own
    # address and publishes it for the headless nodes
    address_file, done = out / "master_ip", out / "done"
    if args.node_rank:
        while not address_file.exists():
            time.sleep(5)
        command += ["--data-parallel-address", address_file.read_text().strip(), "--headless"]
        command += ["--data-parallel-start-rank", str(args.node_rank * GPUS_PER_NODE)]
        with open(out / f"server-node{args.node_rank}.log", "w") as log:
            server = subprocess.Popen(command, env=environment, stdout=log, stderr=log,
                                      start_new_session=True)  # fmt: skip
        while not done.exists() and server.poll() is None:
            time.sleep(10)
        stop(server)
        return 0
    done.unlink(missing_ok=True)
    address = subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout.split()[0]
    command += ["--data-parallel-address", address, "--port", str(SERVER_PORT)]
    command += ["--api-server-count", str(config["api_servers"])]
    address_file.write_text(address + "\n")
    with open(out / "server-node0.log", "w") as log:
        server = subprocess.Popen(command, env=environment, stdout=log, stderr=log,
                                  start_new_session=True)  # fmt: skip
    deadline = time.time() + config["ready_seconds"]
    while not healthy() and server.poll() is None and time.time() < deadline:
        time.sleep(10)
    status = 0 if healthy() else 1
    print(f"ready={'yes' if status == 0 else 'no'} {time.strftime('%T')}", flush=True)
    # the client's dataset loader needs pandas, which the vllm029 lock lacks (scripts/install.sh)
    client_environment = dict(os.environ, VLLM_PLUGINS="")
    client_environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(args.client_packages), os.environ.get("PYTHONPATH")])
    )
    for name, rate, prompts in rate_plan(config) if status == 0 else ():
        clients = []
        for client in range(config["clients"]):
            command = client_command(config, model=args.model, tokenizer=args.tokenizer,
                                     name=name, rate=rate, prompts=prompts, client=client,
                                     out=out)  # fmt: skip
            log = open(out / f"client-{name}-part{client}.log", "w")  # noqa: SIM115
            clients.append(subprocess.Popen(["timeout", "3600", *command], env=client_environment,
                                            stdout=log, stderr=log))  # fmt: skip
        status |= any(client.wait() for client in clients)
        print(f"rate {name} exit {status} {time.strftime('%T')}", flush=True)
        if status:
            break
    done.touch()
    address_file.unlink(missing_ok=True)
    stop(server)
    return status


# --- the prompt files --------------------------------------------------------------------------


def validation_conversations(data, snapshot):
    """The data recipe's held-out validation conversations (`split/rows.parquet` and
    `source/files.jsonl` under `data`), in split order."""
    import pyarrow.parquet as pq

    columns = ["source_file_id", "source_row", "partition"]
    rows = pq.read_table(Path(data) / "split/rows.parquet", columns=columns).to_pylist()
    rows = [row for row in rows if row["partition"] == "validation"]
    paths = {}
    for line in (Path(data) / "source/files.jsonl").read_text().splitlines():
        entry = json.loads(line)
        paths[entry["source_file_id"]] = Path(snapshot) / entry["relative_path"]
    by_file = {}
    for index, row in enumerate(rows):
        by_file.setdefault(row["source_file_id"], []).append((index, row["source_row"]))
    conversations = [None] * len(rows)
    for file_id, wanted in by_file.items():
        messages = pq.read_table(paths[file_id], columns=["messages"]).column("messages")
        for index, source_row in wanted:
            conversations[index] = [
                dict(role=m["role"], content=m["content"]) for m in messages[source_row].as_py()
            ]
    return conversations


def online_prompts(conversations, tokenizer):
    """Each conversation up to its first assistant turn (any system message and the first user
    message), rendered with the chat template, up to MAX_PROMPT_TOKENS, in a seeded order."""
    prompts = []
    for messages in conversations:
        system = [m for m in messages[:1] if m["role"] == "system"]
        rest = messages[len(system) :]
        if not rest or rest[0]["role"] != "user":
            continue
        text = tokenizer.apply_chat_template(
            system + rest[:1], add_generation_prompt=True, tokenize=False
        )
        if len(tokenizer(text).input_ids) <= MAX_PROMPT_TOKENS:
            prompts.append(dict(prompt=text, output_tokens=MAX_REPLY_TOKENS))
    order = list(range(len(prompts)))
    random.Random(SEED).shuffle(order)
    return [prompts[i] for i in order]


def exchanges_of(messages):
    """A conversation's user and assistant turns, alternating from a user turn, whole exchanges
    only. None if it has none."""
    turns = []
    for message in (m for m in messages if m["role"] != "system"):
        if message["role"] != ("user" if len(turns) % 2 == 0 else "assistant"):
            break
        turns.append(message)
    return turns[: len(turns) // 2 * 2] or None


def agent_prompts(conversations, tokenizer, target):
    """Agent turns: a history of whole conversations filled to at most `target` tokens and at least
    90% of it, then another conversation's first user turn, which the model answers."""
    generator = random.Random(SEED)
    exchanges = [turns for turns in map(exchanges_of, conversations) if turns]

    def rendered(messages):
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def size(messages):
        return len(tokenizer(rendered(messages)).input_ids)

    def assemble(chosen, question):
        return [*(m for i in chosen for m in exchanges[i]), question]

    cost = [size(turns) for turns in exchanges]
    questions = [turns[0] for turns in exchanges if size(turns[:1]) <= MAX_PROMPT_TOKENS]
    prompts = []
    while len(prompts) < AGENT_PROMPTS:
        question = generator.choice(questions)
        chosen, estimate, misses = [], size([question]), 0
        while misses < 20:
            index = generator.randrange(len(exchanges))
            if estimate + cost[index] > target:
                misses += 1
                continue
            chosen.append(index)
            estimate += cost[index]
        length = size(assemble(chosen, question))
        while length > target:
            chosen.pop()
            length = size(assemble(chosen, question))
        if length >= 0.9 * target:
            prompts.append(
                dict(prompt=rendered(assemble(chosen, question)), output_tokens=AGENT_REPLY_TOKENS)
            )
    return prompts


def write_prompts(path, prompts):
    path.write_text("".join(json.dumps(prompt) + "\n" for prompt in prompts))


def seed_command(args):
    """Merges the Triton kernels every process of the given launches kept into one seed directory.
    One process copies, and a file already in the seed is kept: a kernel's key names its content."""
    copied = 0
    for cache in sorted(c for launch in args.launches for c in launch.glob("*/triton/*")):
        for source in sorted(cache.glob("*/*")):
            target = args.into / source.relative_to(cache)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
                copied += 1
    print(f"{copied} kernel files into {args.into}")


def prompts_command(args):
    from transformers import AutoTokenizer

    conversations = validation_conversations(args.data, args.snapshot)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    args.out.mkdir(parents=True, exist_ok=True)
    write_prompts(args.out / "prompts.jsonl", online_prompts(conversations, tokenizer))
    for target in (16000, 32000):
        prompts = agent_prompts(conversations, tokenizer, target)
        write_prompts(args.out / f"prompts-agent{target // 1000}k.jsonl", prompts)


def fixed_replies_command(args):
    """Each prompt with its reply length fixed to the first one recorded for it in a real-routing
    launch's `rates`, rebuilding which prompts each client sent (vLLM's custom dataset, the
    client's seed and prompt count)."""
    from transformers import AutoTokenizer
    from vllm.benchmarks.datasets.datasets import CustomDataset

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    config = yaml.safe_load(args.config.read_text())
    reply = {}
    for rate in args.rates:
        count = round(rate * config["seconds"]) // config["clients"]
        for client in range(config["clients"]):
            record = json.loads((args.launch / f"rate-{rate:g}-part{client}.json").read_text())
            requests = CustomDataset(dataset_path=str(args.prompts), random_seed=SEED + client)
            requests = requests.sample(tokenizer=tokenizer, num_requests=count, output_len=-1,
                                       skip_chat_template=True)  # fmt: skip
            assert [r.prompt_len for r in requests] == record["input_lens"], (rate, client)
            for request, tokens, error in zip(
                requests, record["output_lens"], record["errors"], strict=True
            ):
                if not error:
                    reply.setdefault(request.prompt, tokens)
    rows = [json.loads(line) for line in args.prompts.read_text().splitlines()]
    args.out.mkdir(parents=True, exist_ok=True)
    write_prompts(
        args.out / "prompts-fixed.jsonl",
        [
            dict(prompt=r["prompt"], output_tokens=reply[r["prompt"]])
            for r in rows
            if r["prompt"] in reply
        ],
    )


# --- a rate's summary --------------------------------------------------------------------------


def read_record(path):
    record = json.loads(Path(path).read_text())
    return {field: record[field] for field in RECORD_FIELDS}


def rate_summary(records, ttft_target, tpot_target):
    """One rate from its client records, after the server filled (the first third of the arrival
    window): output tokens/s over the last two thirds, TTFT of the requests arriving then, TPOT of
    those arriving in the middle third (they decode while arrivals continue), goodput (those
    meeting both targets, per second) and, as a check for kernels compiling mid-rate, how many of
    them waited over 1 s between two tokens."""
    merged = {
        field: [value for record in records for value in record[field]] for field in RECORD_FIELDS
    }
    succeeded = np.array([not error for error in merged["errors"]])
    start = np.array(merged["start_times"])
    first, window = start.min(), float(start.max() - start.min())
    filled = succeeded & (start >= first + window / 3)
    steady = filled & (start < first + 2 * window / 3)
    ttft = np.array(merged["ttfts"]) * 1000
    tokens = np.array(merged["output_lens"])
    tpot = np.array([sum(itl) / (n - 1) * 1000 if n > 1 else np.nan
                     for itl, n in zip(merged["itls"], tokens)])  # fmt: skip
    token_times = np.concatenate([
        begin + time_to_first / 1000 + np.cumsum([0.0, *itl])
        for begin, time_to_first, itl, ok in zip(start, ttft, merged["itls"], succeeded) if ok
    ])  # fmt: skip
    span = 2 * window / 3
    generated = int(((token_times >= first + window / 3) & (token_times < first + window)).sum())
    gap = np.array([max(itl, default=0.0) * 1000 for itl in merged["itls"]])
    good = steady & (ttft <= ttft_target) & ~(tpot > tpot_target)
    third = window / 3
    return dict(
        requests=len(start),
        failed=int((~succeeded).sum()),
        arrival_s=round(window, 1),
        offered_per_s=round(len(start) / window, 2),
        steady_requests=int(steady.sum()),
        output_tokens_per_s=round(generated / span, 1),
        mean_output_tokens=round(float(tokens[succeeded].mean()), 1),
        ttft_p50_ms=round(float(np.percentile(ttft[filled], 50)), 1),
        ttft_p99_ms=round(float(np.percentile(ttft[filled], 99)), 1),
        tpot_p50_ms=round(float(np.nanpercentile(tpot[steady], 50)), 2),
        tpot_p99_ms=round(float(np.nanpercentile(tpot[steady], 99)), 2),
        goodput_per_s=round(int(good.sum()) / third, 2),
        stalled_requests=int((steady & (gap > 1000)).sum()),
        max_gap_ms=round(float(gap[steady].max()), 1),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    launch = commands.add_parser("run")
    launch.add_argument("--config", type=Path, required=True)
    launch.add_argument("--variant", choices=("base", "folded"), required=True)
    launch.add_argument("--model", type=Path, required=True)
    launch.add_argument("--tokenizer", type=Path, required=True, help="the teacher's")
    launch.add_argument("--output", type=Path, required=True)
    launch.add_argument("--nodes", type=int, required=True)
    launch.add_argument("--node-rank", type=int, required=True)
    launch.add_argument("--fp8", action="store_true", help="FP8 dispatch and combine")
    launch.add_argument("--triton-seed", help="the directory `seed` merged from earlier launches")
    launch.add_argument(
        "--client-packages",
        type=Path,
        default=Path(sys.prefix) / "serving-client",
        help="pandas for the client (scripts/install.sh vllm029)",
    )
    prompts = commands.add_parser("prompts")
    for name in ("data", "snapshot", "tokenizer", "out"):
        prompts.add_argument(name, type=Path)
    fixed = commands.add_parser("fixed-replies")
    for name in ("launch", "config", "prompts", "tokenizer", "out"):
        fixed.add_argument(name, type=Path)
    fixed.add_argument("rates", type=float, nargs="+")
    seed = commands.add_parser("seed")
    seed.add_argument("launches", type=Path, nargs="+", help="launch outputs (run's --output)")
    seed.add_argument("--into", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        sys.exit(run(args))
    elif args.command == "seed":
        seed_command(args)
    elif args.command == "prompts":
        prompts_command(args)
    else:
        fixed_replies_command(args)


if __name__ == "__main__":
    main()
