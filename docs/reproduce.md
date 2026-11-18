# Reproducing the paper

This page walks through every stage behind the paper's tables, from the base model to the
throughput runs. We ran every GPU stage on nodes of 8 H100s.

## Install

You need Linux, Python 3.12, [uv](https://docs.astral.sh/uv/) and CUDA 12.8 with `nvcc` and a C++
compiler on `PATH`, since Transformer Engine and other native packages build against the venv's
torch. The code uses three environments, built from committed lock files, whose pins conflict:

```bash
# .venv-megatron: torch 2.11, Megatron-Core 0.19.2, TE 2.11, DeepEP 2.1.0
scripts/install.sh megatron
# .venv-vllm: vLLM 0.21, lm-eval 0.4.12, plus pinned IFBench and LiveCodeBench
scripts/install.sh vllm
# .venv-vllm029: vLLM 0.29 on CUDA 13, DeepEP v2 with the rail patch
scripts/install.sh vllm029
# NCCL 2.31.2 with the rate cap, for the capped-network rows only
scripts/install.sh nccl
```

Select an environment with `. scripts/env.sh megatron|vllm|vllm029`. It also sets the kernel
choices that affect scores, and `ECHOMOE_SITE_ENV` may name a shell file it sources first (the
cluster's CUDA toolchain).

<details>
<summary>Prebuilt binaries and system notes</summary>

- **Prebuilt binaries.** The DeepEP wheels of `.venv-megatron` and `.venv-vllm029`, the `mamba-ssm`
  and `causal-conv1d` wheels for Nemotron, and the NCCL build are assets of the
  [GitHub release](https://github.com/McGill-NLP/EchoMoE/releases/tag/v1.0.0), all for H100 (sm_90),
  with their sha256 sums. `install.sh` uses a copy already in `third-party/` as it is, and otherwise
  downloads the asset and checks its sha256. If the download fails, it builds the binary from source,
  except the megatron DeepEP wheel.
- **DeepEP from source.** Built from source, DeepEP for `.venv-vllm029` uses the CUDA 13 compiler that
  the venv installs, so it also needs a C++20 compiler and Linux 5.6 or newer kernel headers.
- **FP8 rows.** For `--fp8`, `.venv-vllm029` also keeps a copy of DeepEP with
  `third-party/deepep/deepep_fp8_scaleout_combine.patch`, whose kernels compile at first use.
- **Older drivers.** On a kernel driver older than CUDA 13, set `ECHOMOE_CUDA_COMPAT_DIR` to NVIDIA's
  forward-compatible driver.
- **Older glibc.** On glibc older than 2.32, set `ECHOMOE_FA3_WHEEL` to a FlashAttention-3 wheel built
  for it, for example the release's optional `flash_attn_3-3.0.0` wheel (built on glibc 2.31).
- **Multi-node runs.** Set `NCCL_IB_HCA` to the compute NICs.

</details>

## Configs

| Config | Description |
| --- | --- |
| [`configs/models/`](../configs/models/) | The five base models, and the Instruct models whose chat templates they take |
| [`configs/data/`](../configs/data/) | The Dolmino and Dolci data recipes, per model |
| [`configs/fit/`](../configs/fit/) | Activation capture and teacher targets |
| [`configs/training/<model>/`](../configs/training/) | The teacher SFT and one recipe per student run |
| [`configs/inference/evaluation/`](../configs/inference/evaluation/) | The 27-task suite, its sampling settings and the vLLM engine profiles |
| [`configs/inference/throughput/`](../configs/inference/throughput/) | Workloads, serving settings, per-call cost tables and the larger models |

Each training recipe under [`configs/training/qwen3-30b-a3b/`](../configs/training/qwen3-30b-a3b/) is
one run behind the results, in one directory per table: `frozen/{dispatch_only, combine_only, joint}/`
(dev mode) and `full_adaptation/` (full-model adaptation). A recipe names its family's shared settings
with `base:` (`frozen.yaml` or `full_adaptation.yaml`) and sets only what differs.

## Stages

Run the stages in order from the repository root. Each one writes under `outputs/`. The `torchrun`
launches show the per-node arguments, and multi-node runs add `--nnodes`, `--node-rank` and a
rendezvous endpoint.

### 1. Base model with the Instruct chat template

Environment `megatron`.

```bash
python -m echomoe.training.base_model configs/models/qwen3-30b-a3b.yaml \
    outputs/qwen3-30b-a3b/base
```

### 2. Tokenize and pack Dolmino and Dolci

Environment `megatron`.

```bash
python -m echomoe.training.data.prepare --recipe configs/data/dolmino.yaml \
    --data-root outputs/qwen3-30b-a3b/data
python -m echomoe.training.data.prepare --recipe configs/data/dolci.yaml \
    --data-root outputs/qwen3-30b-a3b/data
```

### 3. Teacher SFT

Environment `megatron`.

```bash
torchrun --nproc-per-node 8 -m echomoe.training.sft \
    --config configs/training/qwen3-30b-a3b/sft.yaml
```

### 4. Teacher export

Environment `megatron`.

```bash
torchrun --standalone --nproc-per-node 8 -m echomoe.inference.evaluation.export \
    --checkpoint outputs/qwen3-30b-a3b/teacher/checkpoints/step-0002230 \
    --output outputs/qwen3-30b-a3b/teacher/hf
```

### 5. Activation capture

Environment `vllm`, for parts 0 to 23.

```bash
python -m echomoe.training.data.teacher capture --config configs/fit/dolci-capture.yaml \
    --part-index I
```

### 6. Routing fanout

Any environment.

```bash
python scripts/routing_fanout.py outputs/qwen3-30b-a3b/capture \
    results/communication/routing_fanout.json
```

### 7. Moments, then the 78 fits

Environment `megatron`.

```bash
python -m echomoe.fit moments outputs/qwen3-30b-a3b/capture outputs/qwen3-30b-a3b/fit
python -m echomoe.fit fits outputs/qwen3-30b-a3b/fit \
    --moments outputs/qwen3-30b-a3b/fit/moments.safetensors \
    --export outputs/qwen3-30b-a3b/teacher/hf \
    $(basename -s .yaml configs/training/qwen3-30b-a3b/frozen/*/*.yaml)
```

### 8. Teacher targets

Environment `vllm`, for each config's `part_count` parts. The four configs are
`configs/fit/<dolmino, dolci>-teacher-targets-<train, validation>.yaml`.

```bash
python -m echomoe.training.data.teacher targets \
    --config configs/fit/dolci-teacher-targets-train.yaml --part-index I
```

### 9. Teacher converted to Megatron, for the dev mode runs

Environment `megatron`.

```bash
torchrun --nproc-per-node 8 -m echomoe.training.model \
    --config configs/training/qwen3-30b-a3b/frozen/joint/dres512-s4-c1280-direct.yaml
```

### 10. Distill one student

Environment `megatron`.

```bash
torchrun --nproc-per-node 8 -m echomoe.training \
    --config configs/training/<family>/<config>.yaml
```

A run resumes with `--resume-from latest`. Dev mode runs end at update 1,000 and full-model
adaptation runs at 4,152.

### 11. Export and fold

Environment `megatron`.

```bash
torchrun --standalone --nproc-per-node 8 -m echomoe.inference.evaluation.export \
    --checkpoint outputs/qwen3-30b-a3b/runs/<run>/checkpoints/step-N
```

### 12. Score on the 27-task suite

Environment `vllm`.

```bash
python -m echomoe.inference.evaluation server \
    --model outputs/qwen3-30b-a3b/runs/<run>/export/step-N --variant folded \
    --ifbench-checkout third-party/IFBench
```

Variants:
- `folded`: every student.
- `folded-fp8`: Table 21's FP8 communication, full-model adaptation students.
- `base-vllm`, `base-blocks`, `base-blocks-fp8`: the uncompressed model, the baselines for Tables 3, 12,
  17–19, 41 and 42, and for Table 21 (`--model outputs/qwen3-30b-a3b/teacher/hf --output outputs/qwen3-30b-a3b/teacher/eval/<variant>`).

### 13. Throughput, one workload

Environment `vllm029`. On every node, the base model, then the folded student:

```bash
torchrun --nnodes 4 --nproc-per-node 8 ... -m echomoe.inference.throughput \
    --workload configs/inference/throughput/i4096-o256.yaml --variant base \
    --model outputs/qwen3-30b-a3b/teacher/hf \
    --output outputs/qwen3-30b-a3b/throughput/<config>-i4096-o256
torchrun --nnodes 4 --nproc-per-node 8 ... -m echomoe.inference.throughput \
    --workload configs/inference/throughput/i4096-o256.yaml --variant folded \
    --model outputs/qwen3-30b-a3b/runs/<run>/export/step-N \
    --output outputs/qwen3-30b-a3b/throughput/<config>-i4096-o256
```

Then once:

```bash
python -m echomoe.inference.throughput.summarize \
    outputs/qwen3-30b-a3b/throughput/<config>-i4096-o256 --config <config>
```

Options (`python -m echomoe.inference.throughput --help`):
- `--fp8`: FP8 activations in every dispatch and combine (Tables 21, 30, 31).
- `--tcp`: NCCL over TCP sockets with vLLM's all-gather transport (Table 34, Figure 17).
- `--capped 2x|2.5x`: inter-node connections capped to about 2x or 2.5x slower (Table 34, Figure 18).
  It needs the patched NCCL from `scripts/install.sh nccl`.
- `--model-config configs/inference/throughput/<model>.yaml`: the larger models (Tables 37, 38) and
  the adapted Kanana-2 and Gemma 4 (Table 39).
- `--tokens`: the token check, greedy tokens and log-probabilities of every prompt.
- `--routing model`: the model's own router instead of balanced expert assignment.
- `--profile`: per-step windows for the breakdowns (stage 15).

Online and agent serving (Tables 35, 36, Figures 19–21) run vLLM's own server on the same 4 nodes:
`python -m echomoe.inference.throughput.serving prompts ...` once, then `serving run --config
configs/inference/throughput/serving-<online, online-fixed, agent16k, agent32k>.yaml` on every node,
`--variant base` then `--variant folded`. Launch each setting and variant once first to compile the
Triton kernels, merge them with `serving seed <launch outputs> --into <seed>`, and pass
`--triton-seed <seed>` to the measured launches. Without a seed, EchoMoE's kernels compile mid-rate
and stall the lowest request rates. The module's docstring has the full commands.

### 14. Larger models (Tables 37, 38)

Environment `vllm029`. Random-weight fixtures, for `<model>` one of
`qwen3-235b-a22b`, `deepseek-v3` and `kimi-k2.6`:

```bash
python -m echomoe.inference.throughput.fixtures \
    --model-config configs/inference/throughput/<model>.yaml \
    --official <official snapshot> --output outputs/<model>/fixtures
```

Then stage 13 with `--model-config` and `--model outputs/<model>/fixtures/<base, folded>`.

### 15. Step breakdowns

Environment `vllm029`. Stage 13 with `--profile` and the workloads `i4096-o256-steps2`,
`i16384-o512-steps`, `i128-o1024-steps` and `i4096-o256-r8`.

### 16. Tables

Any environment.

```bash
python scripts/results_tables.py outputs results/evaluation
python scripts/throughput_tables.py outputs results/throughput
```

### 17. Similarity (Tables 2, 16) and residual spectrum (Table 4, Figure 5)

Environment `megatron`.

```bash
python scripts/analysis.py similarity-capture [--cross-token] <hf model> \
    outputs/qwen3-30b-a3b/data/dolci outputs/similarity/<model>
# Table 2
python scripts/analysis.py similarity outputs/similarity/<model>/moments.safetensors \
    outputs/analysis/similarity/<model>.json
# Table 16, from a capture with --cross-token
python scripts/analysis.py cross-token outputs/similarity/<model> \
    outputs/analysis/similarity/<model>-cross-token.json
# Table 4, Figure 5
python scripts/analysis.py spectrum outputs/qwen3-30b-a3b/fit/moments.safetensors \
    outputs/analysis/residual_spectrum.json
```

## Other models

Kanana-2-30B-A3B and Moonlight-16B-A3B (Table 3, DeepSeek-V3 models) take the same stages with
`<model>` `kanana-2-30b-a3b` or `moonlight-16b-a3b` in place of `qwen3-30b-a3b`: stage 1 with
`configs/models/<model>.yaml`, stage 2 with `configs/data/<model>/`, stages 5 and 8 with
`configs/fit/<model>/`, stage 3 with `configs/training/<model>/sft.yaml` (Kanana-2 on 4 nodes,
Moonlight on 2), stage 7 for `dres512-s4-c1024-direct` alone, stage 10 with
`configs/training/<model>/full_adaptation/dres512-s4-c1024-direct.yaml` on 4 nodes, and stage 12
with `--engine-profile configs/inference/evaluation/engines/<model>/folded.yaml` for the student and
`base.yaml` with variant `base-vllm` for the teacher. Stage 9 is not needed.

Gemma 4 26B-A4B takes the same stages with `<model>` `gemma4-26b-a4b`, its teacher SFT on 2 nodes,
and `dres768-s4-c1408-direct` in place of `dres512-s4-c1024-direct` in stages 7 and 10.
Nemotron 3.5 Lightning 30B-A3B takes them with `<model>` `nemotron-3.5-lightning-30b-a3b`, its
teacher SFT on 4 nodes, and `dres768-s4-c1408-direct` in stages 7 and 10. Its Mamba layers need
`mamba-ssm` and `causal-conv1d`, which `scripts/install.sh megatron` takes from the release or
builds into `third-party/mamba/`.

## LSH-MoE

LSH-MoE (Table 15) trains with stage 10 on `configs/training/qwen3-30b-a3b/lsh/k5-h3.yaml` and
exports with stage 11. The reported evaluation served it with variant `lsh` on four nodes, one
task group per node (`--tasks a,b,...`):
- node 0: `aime24_chat_cot,aime25_chat_cot`
- node 1: `hmmt_feb_2025_chat_cot,livecodebench_chat`
- node 2: `mmlu_pro_chat_cot,mmlu_chat_cot,mmlu_pro_5shot_cot,gpqa_diamond_chat_cot,mmlu_redux_chat_cot`
- node 3, with `--engine-profile configs/inference/evaluation/engines/data-parallel-short-tasks.yaml`:
  the other 18 tasks.

LSH-MoE clusters the tokens that share a batch, so a task's score depends on what is served with
it. Two identical runs differ by 2 to 3 points per task, so compare the suite mean.

## Recipes behind each table

| Table | Recipes | Variants |
| --- | --- | --- |
| 17, 18 dispatch only: code width x stride | `configs/training/qwen3-30b-a3b/frozen/dispatch_only/` (23) | `folded` |
| 19 combine only: decoder variants | `configs/training/qwen3-30b-a3b/frozen/combine_only/` (15) | `folded` |
| 12, 41 joint, dev mode | `configs/training/qwen3-30b-a3b/frozen/joint/` (40) | `folded` |
| 3, 12, 42 full-model adaptation | `configs/training/qwen3-30b-a3b/full_adaptation/` (19) and their joint counterparts | `folded` |
| 21 FP8 communication | `configs/training/qwen3-30b-a3b/full_adaptation/` (19) | `folded`, `folded-fp8` |
| 1, 20 bytes | stage 6's `routing_fanout.json` | none |
| 25–28, 30, 31 throughput | the 11 full-model adaptation students of Table 25, five workloads in `configs/inference/throughput/` | `base`, `folded` |
| 29 operation times | dres512-s4-c768 at 4096/256 and dres512-s4-c1024 at 128k, `--profile` | `base`, `folded` |
| 37, 38 larger models | random-weight Qwen3-235B-A22B, DeepSeek-V3 and Kimi-K2.6 (`configs/inference/throughput/<model>.yaml`) | `base`, `folded` |
