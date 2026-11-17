"""Time one model variant with vLLM 0.29, one rank per GPU: data parallel attention, expert
parallel MoE layers over DeepEP v2, 1 warm-up and 2 timed batches. Qwen3-30B-A3B runs vLLM's
`vllm bench latency`. A larger model (`--model-config`: random weights from `fixtures.py`, or the
trained Kanana-2 and Gemma 4 students) runs vLLM's offline engine stepped request by request, as
it was measured.

    . scripts/env.sh vllm029
    torchrun --nnodes 4 --nproc-per-node 8 --node-rank $R --master-addr $A --master-port $P \
        -m echomoe.inference.throughput --workload configs/inference/throughput/i4096-o256.yaml \
        --variant base --model outputs/qwen3-30b-a3b/teacher/hf --output outputs/qwen3-30b-a3b/throughput/<config>-i4096-o256
    # then the same with --variant folded --model outputs/qwen3-30b-a3b/runs/<run>/export/step-N
    # a larger model: add --model-config configs/inference/throughput/<model>.yaml, with
    # --model outputs/<model>/fixtures/base, then --variant folded --model outputs/<model>/fixtures/folded
    # (Kanana-2 and Gemma 4: --model the teacher's HF export, then the student's portable export)

`base` serves the teacher (or the larger base model) with stock vLLM, `folded` the folded student.
Both run the same engine options, prompts (seeded per rank) and patches (`fairness.py`, and the
per-call transport choice of `selection.py`, by the table the model's settings name). Run base
first, then folded, on the same nodes. Each rank writes `<output>/<variant>/rank<r>.json`: every
timed request's TTFT and TPOT, the timed batch latencies of its barrier clock (rank 0's are the
reported ones) and, for a larger model, each engine step's end and output tokens. `summarize.py`
reduces a row's records to the columns of `results/throughput/`. `--routing balanced`, the
default, times both variants with vLLM's routing simulator drawing each token's top-k experts
uniformly, as every reported row was. `--routing model` keeps the model's own router. `--fp8`
sends the activations in FP8 in every dispatch and combine (a larger model's experts that take
BF16 get them cast back), and `--bf16-wire` sends DeepSeek-V3's in BF16 (its FP8 experts quantize
on the receiver).

`--tcp` runs both variants over NCCL's TCP sockets, with vLLM's all-gather + reduce-scatter
transport and no InfiniBand or DeepEP (`tcp.py`). The launcher names the interface in
NCCL_SOCKET_IFNAME. `--capped 2x` or `--capped 2.5x` (Qwen3-30B-A3B) caps each inter-node
connection's rate as capped.yaml sets it, through the NCCL build of `scripts/install.sh nccl`,
which the launcher preloads with the CUDA 12.8 runtime it links:

    LD_PRELOAD=third-party/nccl/build/lib/libnccl.so.2 \
        LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<CUDA 12.8>/lib torchrun ... --capped 2x

`--tokens` (workload tokens.yaml, with `--routing model`) writes instead each prompt's greedy
tokens and log-probabilities, the check of the kernels against DeepEP's outputs.

`--profile` records instead a torch profile of each window of engine steps the workload names,
after a warm-up batch, into `<output>/<variant>/profile/rank<r>/`. `scripts/throughput_tables.py`
reduces those of the step-breakdown workloads (i4096-o256-steps2, i16384-o512-steps,
i128-o1024-steps, i4096-o256-r8) to the breakdowns of `results/throughput/`.
"""

import os
import socket
import sys
import tempfile

# Read when vLLM, torch and NCCL load. The engine runs in this process (external launcher). DeepEP
# v2 runs its hybrid mode with multiple reductions (vLLM's default is off). Each rank compiles its
# Triton kernels into its own fresh cache, as the measured runs did.
os.environ.update(
    VLLM_ENABLE_V1_MULTIPROCESSING="0",
    VLLM_PLUGINS="",
    HF_HUB_OFFLINE="1",
    VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE="1",
    VLLM_DEEPEP_V2_ALLOW_MULTIPLE_REDUCTION="1",
    OMP_NUM_THREADS="2",
    MKL_NUM_THREADS="2",
    TRITON_CACHE_DIR=tempfile.mkdtemp(prefix=f"triton-rank{os.environ.get('RANK', '0')}-"),
)
# --fp8 combines through the copy of DeepEP whose combine crosses nodes in FP8 (scripts/install.sh).
# Its kernels need their own cache: the cache's keys do not cover DeepEP's headers. Both must be
# set before anything imports DeepEP.
if "--fp8" in sys.argv:
    sys.path.insert(0, os.path.join(sys.prefix, "deepep-fp8c"))
    os.environ["EP_JIT_CACHE_DIR"] = os.path.join(
        os.environ["ECHOMOE_REPO"], "outputs/deepep-jit-cache-fp8c", socket.gethostname()
    )
# --capped: vLLM loads the preloaded NCCL build, and DeepEP, which checks at import that the NCCL
# loaded is its own, accepts it.
if "--capped" in sys.argv:
    nccl = next(path for path in os.environ["LD_PRELOAD"].split(":") if "libnccl" in path)
    os.environ.update(VLLM_NCCL_SO_PATH=nccl, EP_SUPPRESS_NCCL_CHECK="1")

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from deep_ep.buffers import elastic
from vllm import LLM, SamplingParams
from vllm.benchmarks import latency
from vllm.distributed import get_dp_group, get_ep_group
from vllm.distributed.device_communicators.all2all import DeepEPV2All2AllManager
from vllm.sampling_params import RequestOutputKind
from vllm.utils.argparse_utils import FlexibleArgumentParser

from echomoe.runs import REPO, write_meta

from . import portable
from .fairness import (
    install_bf16_dispatch,
    install_dp_sync,
    install_fp8_dispatch,
    install_live_activation,
    install_live_experts,
    install_nic_rate,
    install_padding_route_mask,
    install_second_stream,
    install_stock_decode_guard,
)
from .kernels import fp8_cast_back
from .model import ARCHITECTURE, LOAD_FORMAT, register
from .portable_kernels import fp8_cast_back_blocks
from .selection import install_selected
from .serving import command_line
from .tcp import install_tcp

CONFIGS = REPO / "configs/inference/throughput"
ENGINE = CONFIGS / "engine.yaml"
# Qwen3-30B-A3B's expert GEMM tiles, searched per expert shape and received row count
EXPERT_TILES = CONFIGS / "tiles.json"
# its per-call backend times (selection.py), BF16 and with FP8 activations
COST_TABLES = {False: CONFIGS / "costs-bf16.csv", True: CONFIGS / "costs-fp8.csv"}
# --capped: each setting's rates, DeepEP's network contexts and per-call backend times
CAPPED = CONFIGS / "capped.yaml"
SEED = 42
# options of `vllm bench latency` that are not engine options
BENCHMARK_OPTIONS = (
    "disable_detokenize",
    "num_iters_warmup",
    "num_iters",
    "input_len",
    "output_len",
    "batch_size",
)
# --profile: the measured protocol's prompts (seeded 8000 + rank) and warm-up length
PROFILE_SEED = 8000
PROFILE_VOCAB = (0, 10000)
PROFILE_WARMUP_TOKENS = 32
# --tokens: the token check's prompts, as the measured kernels were checked against DeepEP's: per
# GPU, prompts of 16 to 3000 random tokens drawn with seed 1000 + rank
TOKEN_CHECK_SEED = 1000
TOKEN_CHECK_LENGTHS = (16, 3000)
TOKEN_CHECK_VOCAB = (1000, 150000)


def engine_options(engine, workload, *, variant, model, fp8, rank, world):
    """The engine and benchmark options of one rank."""
    options = {
        "model": str(model),
        **engine,
        **workload,
        "max_model_len": workload["input_len"] + workload["output_len"],
        "seed": SEED + rank,
        "data_parallel_size": world,
    }
    if variant == "folded":
        options["load_format"] = LOAD_FORMAT
        options["hf_overrides"] = {
            "architectures": [ARCHITECTURE],
            **workload.get("hf_overrides", {}),
        }
        options["model_loader_extra_config"] = {"fp8": fp8}
    return options


class BarrierClock:
    """`time` for vLLM's latency benchmark: every reading first waits for all data-parallel ranks,
    so rank 0's interval runs from the moment all ranks start to the moment all have finished."""

    def __init__(self):
        self.readings = []

    def perf_counter(self):
        dist.barrier(group=get_dp_group().cpu_group)
        self.readings.append(time.perf_counter())
        return self.readings[-1]


def latency_benchmark(options):
    """Qwen3-30B-A3B: vLLM's latency benchmark, timed with a barrier clock. The engine, and every
    timed request's TTFT and TPOT and the timed batches."""
    benchmark = FlexibleArgumentParser()
    latency.add_cli_args(benchmark)
    arguments = benchmark.parse_args(command_line(options))
    clock = BarrierClock()
    latency.time = clock
    calls, engines = [], []
    from_engine_args = LLM.from_engine_args

    def build(cls, *build_args, **build_kwargs):
        llm = from_engine_args(*build_args, **build_kwargs)
        generate = llm.generate

        def kept(*generate_args, **generate_kwargs):
            calls.append(generate(*generate_args, **generate_kwargs))
            return calls[-1]

        llm.generate = kept
        engines.append(llm)
        return llm

    LLM.from_engine_args = classmethod(build)
    latency.main(arguments)

    timed = calls[arguments.num_iters_warmup :]
    starts, stops = clock.readings[0::2], clock.readings[1::2]
    batches = [stop - start for start, stop in zip(starts, stops)]
    return engines[0], {
        "ttft_s": [
            [float(output.metrics.first_token_latency) for output in call] for call in timed
        ],
        "tpot_s": [
            [
                (output.metrics.last_token_ts - output.metrics.first_token_ts)
                / (output.metrics.num_generation_tokens - 1)
                for output in call
            ]
            for call in timed
        ],
        "batch_s": batches[arguments.num_iters_warmup :],
    }


def random_prompt(seed, length, vocab):
    """Token ids drawn uniformly from `vocab` ([low, high)) by a generator seeded with `seed`."""
    generator = random.Random(seed)
    return [generator.randrange(*vocab) for _ in range(length)]


def run_batch(engine, prompts, output_len, cell, counter):
    """One batch through vLLM's offline engine, all requests added at once between two barriers:
    each request's first and last token time and tokens, each step's end and tokens emitted."""
    params = SamplingParams(
        n=1,
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=output_len,
        detokenize=False,
        output_kind=RequestOutputKind.DELTA,
    )
    barrier = get_dp_group().cpu_group
    dist.barrier(group=barrier)
    start = time.perf_counter()
    ids = []
    for prompt in prompts:
        counter[0] += 1
        ids.append(f"{cell}-{counter[0]}")
        engine.add_request(ids[-1], dict(prompt_token_ids=prompt), params)
    first, last, tokens, steps = {}, {}, dict.fromkeys(ids, 0), []
    while engine.has_unfinished_requests():
        outputs = engine.step()
        now = time.perf_counter() - start
        emitted = 0
        for output in outputs:
            count = len(output.outputs[0].token_ids)
            if count:
                first.setdefault(output.request_id, now)
                last[output.request_id] = now
                tokens[output.request_id] += count
                emitted += count
        steps.append([now, emitted])
    dist.barrier(group=barrier)
    interval = time.perf_counter() - start
    return dict(
        ttft_s=[first[request] for request in ids],
        tpot_s=[
            (last[request] - first[request]) / (tokens[request] - 1)
            for request in ids
            if tokens[request] > 1
        ],
        batch_s=interval,
        step_s=steps,
    )


def engine_batches(llm_options, workload, vocab, warmups, runs, rank):
    """A larger model: warm-up and timed batches through vLLM's engine, every rank's prompts the
    same in both variants (the cell name in their seeds is the measured runs'). The engine and each
    timed batch's records."""
    cell = f"table_c{workload['input_len']}_o{workload['output_len']}_r{workload['batch_size']}"
    prompts = [
        random_prompt(f"{cell}:{rank}:{index}", workload["input_len"], vocab)
        for index in range(workload["batch_size"])
    ]
    llm = LLM(**llm_options)
    counter = [0]
    for _ in range(warmups):
        run_batch(llm.llm_engine, prompts, workload["output_len"], cell, counter)
    timed = [
        run_batch(llm.llm_engine, prompts, workload["output_len"], cell, counter)
        for _ in range(runs)
    ]
    return llm, {key: [batch[key] for batch in timed] for key in timed[0]}


def configure_profiler(wrapper, directory, steps):
    """On the worker: a fresh torch profiler that records the next `steps` engine steps."""
    worker = wrapper.worker
    torch.cuda.synchronize()
    if worker.profiler is not None:
        worker.profiler.shutdown()
    worker.profiler = None
    config = worker.profiler_config
    config.delay_iterations = 0
    config.max_iterations = steps
    config.ignore_frontend = True
    config.torch_profiler_dir = directory


def token_check(llm_options, workload, rank):
    """The workload's `batch_size` synthetic prompts through the engine, `output_len` greedy tokens
    each: the engine, and each prompt's tokens and their log-probabilities."""
    llm = LLM(**llm_options)
    generator = np.random.default_rng(TOKEN_CHECK_SEED + rank)
    lengths = generator.integers(*TOKEN_CHECK_LENGTHS, size=workload["batch_size"])
    prompts = [
        dict(prompt_token_ids=generator.integers(*TOKEN_CHECK_VOCAB, size=int(length)).tolist())
        for length in lengths
    ]
    params = SamplingParams(
        temperature=0.0,
        max_tokens=workload["output_len"],
        ignore_eos=True,
        logprobs=0,
        detokenize=False,
    )
    outputs = [output.outputs[0] for output in llm.generate(prompts, params, use_tqdm=False)]
    return llm, [
        dict(
            tokens=list(output.token_ids),
            logprobs=[
                float(step[token].logprob)
                for token, step in zip(output.token_ids, output.logprobs)
            ],
        )
        for output in outputs
    ]


def profile(llm_options, workload, windows, directory, rank):
    """Run the whole batch once (32 output tokens each), then again, recording a torch profile of
    each window of engine steps (`name`, first step, steps). Abort the batch after the last window."""
    llm = LLM(
        **llm_options,
        profiler_config=dict(
            profiler="torch",
            detailed_trace_annotation=True,
            max_iterations=windows[0][2],
            torch_profiler_dir=str(directory),
            ignore_frontend=True,
            torch_profiler_with_stack=False,
            torch_profiler_use_gzip=True,
        ),
    )
    tokens = np.random.default_rng(PROFILE_SEED + rank).integers(
        *PROFILE_VOCAB, (workload["batch_size"], workload["input_len"])
    )
    prompts = [dict(prompt_token_ids=prompt) for prompt in tokens.tolist()]
    engine = llm.llm_engine

    def params(count):
        return SamplingParams(
            temperature=0,
            ignore_eos=True,
            max_tokens=count,
            detokenize=False,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )

    for index, prompt in enumerate(prompts):
        engine.add_request(f"warm_{rank}_{index}", prompt, params(PROFILE_WARMUP_TOKENS))
    while engine.has_unfinished_requests():
        engine.step()

    ids = [
        engine.add_request(f"capture_{rank}_{index}", prompt, params(workload["output_len"]))
        for index, prompt in enumerate(prompts)
    ]
    starts = {start: (name, steps) for name, start, steps in windows}
    ends = {start + steps - 1 for _, start, steps in windows}
    step = 0
    while engine.has_unfinished_requests():
        step += 1
        if step in starts:
            name, steps = starts[step]
            window = directory / name
            window.mkdir(parents=True)
            llm.collective_rpc(configure_profiler, args=(str(window), steps))
            llm.start_profile()
            # the profiler's start must not become a collective wait inside the window
            dist.barrier(group=get_dp_group().cpu_group)
        engine.step()
        if step in ends:
            torch.cuda.synchronize()
            llm.stop_profile()
            torch.cuda.synchronize()
        if step == max(ends):
            break
    # vLLM's internal request ids, from add_request
    engine.abort_request(ids, internal=True)
    while engine.has_unfinished_requests():
        engine.step()
    return llm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--variant", choices=("base", "folded"), required=True)
    parser.add_argument(
        "--model", type=Path, required=True, help="teacher hf/, export/step-N or a fixture"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, help="a larger model's settings")
    parser.add_argument("--profile", action="store_true", help="record the workload's windows")
    parser.add_argument(
        "--tokens",
        action="store_true",
        help="the token check: greedy tokens and log-probabilities",
    )
    parser.add_argument("--routing", choices=("balanced", "model"), default="balanced")
    parser.add_argument(
        "--fp8", action="store_true", help="FP8 activations in every dispatch and combine"
    )
    parser.add_argument("--bf16-wire", action="store_true", help="DeepSeek-V3's dispatch in BF16")
    parser.add_argument(
        "--tcp", action="store_true", help="NCCL over TCP sockets, the all-gather transport"
    )
    parser.add_argument(
        "--capped",
        choices=tuple(yaml.safe_load(CAPPED.read_text())),
        help="inter-node connections capped to about 2x or 2.5x slower (Qwen3-30B-A3B)",
    )
    args = parser.parse_args()
    if args.routing == "balanced":
        # read by vLLM when the engine builds its MoE layers
        os.environ["VLLM_MOE_ROUTING_SIMULATION_STRATEGY"] = "uniform_random"
    # read by NCCL when the engine initializes it
    if args.tcp:
        os.environ.update(NCCL_IB_DISABLE="1", NCCL_NET="Socket")
    capped = yaml.safe_load(CAPPED.read_text())[args.capped] if args.capped else {}
    if capped:
        # NCCL's own connections, and DeepEP's GIN contexts
        os.environ.update(
            NCCL_IB_STATIC_RATE=str(capped["nccl_ib_rate"]),
            NCCL_GIN_STATIC_RATE=str(capped["deepep_gin_rate"]),
        )
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    output = args.output / args.variant
    output.mkdir(parents=True, exist_ok=True)
    record = output / f"rank{rank}.json"
    assert not record.exists(), f"{record} exists"

    # the engine options, from engine.yaml, the workload and a larger model's config
    engine = yaml.safe_load(ENGINE.read_text())
    if args.tcp:
        engine["all2all_backend"] = "allgather_reducescatter"
    workload = yaml.safe_load(args.workload.read_text())
    windows = workload.pop("profile_windows", None)
    larger = None if args.model_config is None else yaml.safe_load(args.model_config.read_text())
    assert larger or not args.bf16_wire, "--bf16-wire runs a larger model"
    assert not (larger and args.capped), "--capped runs Qwen3-30B-A3B"
    assert not (larger and larger.get("live_rows") and args.tcp), "live_rows runs over DeepEP"
    # a larger model's dispatch: FP8 with --fp8, else as its experts take it unless --bf16-wire
    fp8_dispatch = larger and (
        args.fp8 or (larger["expert_format"] == "fp8_block128" and not args.bf16_wire)
    )
    # (a larger model's folded variant gets its own loader and architecture below)
    options = engine_options(
        engine,
        workload,
        variant="base" if larger else args.variant,
        model=args.model.resolve(),
        fp8=args.fp8,
        rank=rank,
        world=world,
    )
    if larger:
        options.update(larger["engine"])
        if args.variant == "base":
            options.update(larger.get("base_engine", {}))
        else:
            family = json.loads((args.model / "hf/config.json").read_text())["model_type"]
            options.update(
                model=str(args.model.resolve() / "hf"),
                load_format=portable.LOAD_FORMAT,
                hf_overrides={"architectures": [portable.ARCHITECTURES[family]]},
                model_loader_extra_config={
                    "fp8": fp8_dispatch,
                    "second_stream": larger.get("second_stream", False),
                    "live_rows": larger.get("live_rows", False),
                },
            )
    llm_options = {key: value for key, value in options.items() if key not in BENCHMARK_OPTIONS}

    # seed the prompts (vLLM draws them with numpy) and install both variants' patches
    if not larger:
        random.seed(SEED + rank)
        np.random.seed(SEED + rank)  # noqa: NPY002 (vLLM draws the prompts with np.random)
        torch.manual_seed(SEED + rank)
    install_dp_sync()
    if larger:
        install_live_activation()
        precision = "fp8" if args.fp8 else "fp8-dispatch" if fp8_dispatch else "bf16"
        cost_table = CONFIGS / larger["select"][precision]
    else:
        install_live_experts(json.loads(EXPERT_TILES.read_text()))
        cost_table = CONFIGS / capped["select"] if capped else COST_TABLES[args.fp8]
    if args.tcp:
        cost_table = None
        install_tcp(args.fp8)
    else:
        install_selected(
            cost_table,
            fp8_combine=args.fp8,
            rail_gin_rate=capped.get("rail_gin_rate"),
            stream_gin_rate=capped.get("stream_gin_rate"),
        )
    if args.fp8:
        assert "deepep-fp8c" in elastic.__file__, elastic.__file__
        install_fp8_dispatch(fp8_cast_back_blocks if larger else fp8_cast_back)
    if args.bf16_wire:
        install_bf16_dispatch()
    if larger and larger.get("second_stream"):
        install_second_stream()
    if args.variant == "base":
        if larger is None or larger.get("decode_guard"):
            install_stock_decode_guard(tiled=larger is None)
        if larger and larger.get("padding_route_mask"):
            install_padding_route_mask()
    elif larger:
        portable.register()
    else:
        register()
    if "NCCL_IB_HCA" in os.environ:
        install_nic_rate()
    # A larger model's DeepEP buffers, and on the capped networks both models', allocated a fixed
    # number of network contexts (QPs).
    contexts = larger["deepep_contexts"] if larger else capped.get("deepep_qps")
    if contexts:
        make_kwargs = DeepEPV2All2AllManager._make_all2all_kwargs

        def all2all_kwargs(self, *call_args, **call_kwargs):
            return {**make_kwargs(self, *call_args, **call_kwargs), "num_allocated_qps": contexts}

        DeepEPV2All2AllManager._make_all2all_kwargs = all2all_kwargs
    if larger:
        torch.set_num_threads(1)

    # time the batches (or record the profile windows)
    if args.profile:
        llm = profile(llm_options, workload, windows, output / "profile" / f"rank{rank}", rank)
        result = {}
    elif args.tokens:
        llm, result = token_check(llm_options, workload, rank)
    elif larger:
        llm, result = engine_batches(
            llm_options,
            workload,
            larger["prompt_vocab"],
            engine["num_iters_warmup"],
            engine["num_iters"],
            rank,
        )
    else:
        llm, result = latency_benchmark(options)

    # Destroy DeepEP's buffers before vLLM's groups. Its cache holds only weak references.
    manager = get_ep_group().device_communicator.all2all_manager
    buffers = list(manager.handle_cache._cache.values())
    manager.destroy()
    llm.llm_engine.engine_core.shutdown()
    del buffers

    record.write_text(json.dumps(result) + "\n")
    if rank == 0:
        write_meta(
            output,
            inputs={
                "model": args.model,
                "workload": args.workload,
                "engine": ENGINE,
                **({"model_config": args.model_config} if larger else {}),
                **({"select": cost_table} if cost_table else {}),
            },
            variant=args.variant,
            routing=args.routing,
            fp8=args.fp8,
            bf16_wire=args.bf16_wire,
            tcp=args.tcp,
            capped=args.capped,
            workload=workload,
            world_size=world,
            engine_options=options,
            **({"profile_windows": windows} if args.profile else {}),
        )


if __name__ == "__main__":
    main()
