# EchoMoE

Code for *Communicate the Change: Efficient Expert Parallelism via Activation Compression*.
<!-- TODO(user): preprint link and arXiv badge -->

Expert parallelism sends every token's hidden state to the nodes that hold its experts (dispatch)
and every expert's output back (combine). EchoMoE compresses both all-to-alls with linear
compressors, fits them in closed form from the uncompressed model's activations, then distills, and
folds the dispatch decompressor and the combine compressor into the expert weights so serving runs
no extra matmuls. On Qwen3-30B-A3B it cuts the RDMA bytes per token by 2.39x at 4 nodes (58% fewer
bytes) and raises throughput by 22.6%, for a 3.8-point drop in the 27-task suite mean. Sending the
compressed activations in FP8 changes the score by less than 0.3 points.

## Method

**Dispatch.** Every `s`-th MoE layer is a *reference layer*: it publishes the token's full-width
input `x_l` once to every remote node, which caches it as the reference `z`. The `s - 1`
*non-reference layers* after it send only what the reference fails to predict, a code of width
`r_d` of the difference,

    c_d = C_d (x_l - a * z),        x_hat = D_d c_d + G z

where `C_d` and `D_d` are the dispatch compressor and decompressor, `a` is a per-dimension gain and
`G` is the reconstruction gain. In this code `G` is diagonal, `G = diag(b)`. `dplain`
configurations use plain compression and send `C_d x_l` with no reference.

**Combine.** Each expert sends `C_c y_i`, a code of width `r_c`. The token's owner decompresses the
weighted sum once, `y_hat = D_c sum_i w_i C_c y_i`, and optionally adds an owner-side correction
that sends nothing (bias and gated bias decoding, `bias` and `biasgate`).

**Initialization.** `C_d`, `D_d`, `C_c`, `D_c` and the gains come from a PCA of FP64 activation
moments captured from the uncompressed model. The student is then distilled from the teacher with
a coarsened top-1024 KL plus a 0.01-weighted LM loss, either in dev mode (backbone frozen, 1,000
updates) or with the backbone and compressors trained together (*full-model adaptation*, 4,152
updates).

**Folding.** The dispatch decompressor and the combine compressor sit next to expert matrices, so
for serving they are multiplied in. Expert `i` reads `[z, c_d]` through the folded input projection
`W_in,i = W_i [G, D_d]`, where `W_i` stacks its gate and up projections, and writes its code
through the folded output projection `W_out,i = C_c W_down,i`. The fold is computed in float64 and
checked to a relative error of 1e-6.

A configuration (s, r_d, r_c) is the run `dres{r_d}-s{s}-c{r_c}`. `dres512-s4-c1280` sends a
dispatch code of width 512 against a reference every 4 layers and a combine code of width 1280.
`dfull` / `cfull` leave a side uncompressed. The whole method is
[`echomoe/codec.py`](echomoe/codec.py), the byte model is
[`echomoe/communication_cost.py`](echomoe/communication_cost.py), and the fit is
[`echomoe/fit.py`](echomoe/fit.py). These three files need only PyTorch and run on CPU.

### Names in the code

| Report | Code |
| --- | --- |
| compressors `C_d`, `C_c` and decompressors `D_d`, `D_c` | `encoder` and `decoder` of `dispatch_compressor` and `combine_compressor` |
| gain `a`, reconstruction gain `G = diag(b)` | `encoder_gains`, `decoder_gains` (the vector `b`) |
| folded input projection `W_in,i` | `readers` |
| folded output projection `W_out,i` | `factors` |
| code widths `r_d`, `r_c` | `rank` (`dispatch_rank`, `combine_rank`) |
| non-reference layer | dispatch role `consumer` |
| dev mode (backbone frozen) | `frozen` recipes in `configs/training/qwen3-30b-a3b/frozen/` |
| configuration (s, r_d, r_c) | run ID `dres{r_d}-s{s}-c{r_c}` |

## Results

Qwen3-30B-A3B with full-model adaptation, at expert parallel 32 over 4 nodes (paper Tables 3, 21
and 25). The score is the mean of the 27-task suite in %, and the uncompressed model scores 66.9. RDMA
reduction is the bytes per token sent between nodes, against uncompressed bf16 dispatch and
combine with the measured routing fanout. Throughput speedup is for 4096-token prompts and 256
output tokens at 128 requests per GPU with balanced expert assignment, over the mean of the uncompressed
runs (`results/throughput/throughput.csv`). "FP8 change" is the score change, in percentage points,
when every side of the communication is sent in FP8 (one byte per element and a float32 scale per
128), against the same student in bf16.

| s | r_d | r_c | RDMA reduction | score (Δ pp) | throughput speedup | FP8 change |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 1280 | 1280 | 1.46x | 65.5 (−1.4) | 1.025 | −0.13 |
| 4 | 512 | 1280 | 1.84x | 64.9 (−2.0) | 1.115 | −0.04 |
| 4 | 512 | 1024 | 2.08x | 63.8 (−3.1) | 1.169 | −0.13 |
| 4 | 512 | 768 | 2.39x | 63.1 (−3.8) | 1.226 | −0.09 |
| 6 | 256 | 768 | 3.02x | 58.3 (−8.6) | 1.325 | +0.18 |
| 8 | 256 | 512 | 4.03x | 55.7 (−11.2) | 1.429 | −0.15 |

`s` is the reference stride, `r_d` the dispatch code width and `r_c` the combine code width. Every number
comes from the committed files in [`results/`](results/).

The same method on four other MoE models (paper Table 3), each distilled from its own SFT teacher:

| Model | s | r_d | r_c | RDMA reduction | score: teacher → EchoMoE |
| --- | --- | --- | --- | --- | --- |
| Gemma 4 26B-A4B | 4 | 768 | 1408 | 2.01x | 61.2 → 56.8 (−4.4) |
| Kanana-2-30B-A3B | 4 | 512 | 1024 | 2.01x | 58.5 → 56.1 (−2.4) |
| Nemotron 3.5 Lightning 30B-A3B | 4 | 768 | 1408 | 1.91x | 62.8 → 58.3 (−4.6) |
| Moonlight-16B-A3B | 4 | 512 | 1024 | 1.99x | 53.0 → 47.1 (−5.9) |

LSH-MoE, trained with the same full-model adaptation recipe, scores 57.1 against EchoMoE's 63.8 at
(4, 512, 1024) (paper Table 15).

### Models

The models are on Hugging Face in the [EchoMoE collection](https://huggingface.co/collections/McGill-NLP/echomoe-6afcbfa10b38185f87cbd0ac): the Qwen3-30B-A3B SFT
teacher with its 19 full-adaptation students, and the SFT teacher and reported student of Gemma 4,
Kanana-2, Moonlight and Nemotron. Each model card has its scores, a `vllm serve` command through this
package's plugin, and a `transformers` snippet through the remote code the students carry. The
release's scripts take a local checkpoint, so download one first and pass its directory as `--model`:

```bash
hf download McGill-NLP/EchoMoE-Qwen3-30B-A3B-s4-r512-c768 --local-dir checkpoints/EchoMoE-Qwen3-30B-A3B-s4-r512-c768
```

The Kanana-2 models are Powered by Kanana, under the Kanana License Agreement.

## Repository

```
echomoe/
├── codec.py, communication_cost.py, fit.py    the method, the byte model, the closed-form fit
├── lsh.py                                     the LSH-MoE baseline (Tables 15, 16)
├── runs.py                                    run directories and the meta.json provenance record
├── hf/                                        the folded students as transformers remote code
├── training/                                  Megatron-Core + DeepEP: SFT, distillation, data
│   └── data/                                  data preparation, activation capture, teacher targets
└── inference/
    ├── evaluation/                            HF export and fold, vLLM 0.21 serving, the 27-task suite
    └── throughput/                            vLLM 0.29 expert-parallel runtime, serving and benches (Tables 25-39)
configs/                                       models, data, fit, training recipes (one per run), evaluation, throughput
scripts/                                       env.sh, install.sh, routing_fanout.py, analysis.py, results_tables.py, throughput_tables.py
results/                                       the fanout, the byte table, suite scores, throughput tables
third-party/                                   the DeepEP FP8 and NCCL rate-cap patches, IFBench, LiveCodeBench
```

Each training recipe under [`configs/training/qwen3-30b-a3b/`](configs/training/qwen3-30b-a3b/) is one
run behind the results, in one directory per table: `frozen/{dispatch_only, combine_only, joint}/`
(dev mode) and `full_adaptation/` (full-model adaptation). A recipe names its family's shared
settings with `base:` (`frozen.yaml` or `full_adaptation.yaml`) and sets only what differs.

## Install

Linux, Python 3.12, CUDA 12.8 with `nvcc` and a C++ compiler on `PATH` (Transformer Engine and
other native packages build against the venv's torch) and [uv](https://docs.astral.sh/uv/).
There are three environments, built from committed lock files. They cannot be merged because their
pins conflict.

```bash
scripts/install.sh megatron   # .venv-megatron: torch 2.11, Megatron-Core 0.19.2, TE 2.11, DeepEP 2.1.0
scripts/install.sh vllm       # .venv-vllm: vLLM 0.21, lm-eval 0.4.12, plus pinned IFBench and LiveCodeBench
scripts/install.sh vllm029    # .venv-vllm029: vLLM 0.29 on CUDA 13, DeepEP v2 with the rail patch
scripts/install.sh nccl       # NCCL 2.31.2 with the rate cap, for the capped-network rows only
```

The prebuilt binaries, all for H100 (sm_90), are assets of the
[GitHub release](https://github.com/McGill-NLP/EchoMoE/releases/tag/v1.0.0)
with their sha256 sums. These are the DeepEP wheels of `.venv-megatron` and `.venv-vllm029`, the
`mamba-ssm` and `causal-conv1d` wheels for Nemotron, and the NCCL build. `install.sh` uses a copy
already in `third-party/` as it is, and otherwise downloads the asset and checks its sha256. If the
download fails, it builds the binary from source, except the megatron DeepEP wheel. Built from
source, DeepEP for `.venv-vllm029` uses the CUDA 13 compiler that the venv installs, so it also
needs a C++20 compiler and Linux 5.6 or newer kernel headers. For the FP8 rows (`--fp8`),
`.venv-vllm029` also keeps a copy of DeepEP with `third-party/deepep/deepep_fp8_scaleout_combine.patch`,
whose kernels compile at first use. On a kernel driver older than CUDA 13, set
`ECHOMOE_CUDA_COMPAT_DIR` to NVIDIA's forward-compatible driver. Multi-node runs set `NCCL_IB_HCA`
to the compute NICs.

On glibc older than 2.32, set `ECHOMOE_FA3_WHEEL` to a FlashAttention-3 wheel built for it, for
example the release's optional `flash_attn_3-3.0.0` wheel (built on glibc 2.31). Select an
environment with `. scripts/env.sh megatron|vllm|vllm029`. It also
sets the kernel choices that affect scores. `ECHOMOE_SITE_ENV` may name a shell file it sources
first (the cluster's CUDA toolchain).

## Reproduce

Every stage writes under `outputs/`. Run them in this order. The `torchrun` launches below show the
per-node arguments, and multi-node runs add `--nnodes`, `--node-rank` and a rendezvous endpoint.
All GPU stages ran on nodes of 8 H100s. Measured times are under [Compute](#compute).

| # | Stage | Env | Command | Hardware |
| --- | --- | --- | --- | --- |
| 1 | Base model with the Instruct chat template | megatron | `python -m echomoe.training.base_model configs/models/qwen3-30b-a3b.yaml outputs/qwen3-30b-a3b/base` | CPU |
| 2 | Tokenize and pack Dolmino and Dolci | megatron | `python -m echomoe.training.data.prepare --recipe configs/data/<dolmino, dolci>.yaml --data-root outputs/qwen3-30b-a3b/data` | CPU |
| 3 | Teacher SFT | megatron | `torchrun --nproc-per-node 8 -m echomoe.training.sft --config configs/training/qwen3-30b-a3b/sft.yaml` | 8 nodes |
| 4 | Teacher export | megatron | `torchrun --standalone --nproc-per-node 8 -m echomoe.inference.evaluation.export --checkpoint outputs/qwen3-30b-a3b/teacher/checkpoints/step-0002230 --output outputs/qwen3-30b-a3b/teacher/hf` | 1 node |
| 5 | Activation capture, parts 0–23 | vllm | `python -m echomoe.training.data.teacher capture --config configs/fit/dolci-capture.yaml --part-index I` | 1 GPU per part |
| 6 | Routing fanout | any | `python scripts/routing_fanout.py outputs/qwen3-30b-a3b/capture results/communication/routing_fanout.json` | CPU |
| 7 | Moments, then the 78 fits | megatron | `python -m echomoe.fit moments outputs/qwen3-30b-a3b/capture outputs/qwen3-30b-a3b/fit`, then `python -m echomoe.fit fits outputs/qwen3-30b-a3b/fit --moments outputs/qwen3-30b-a3b/fit/moments.safetensors --export outputs/qwen3-30b-a3b/teacher/hf $(basename -s .yaml configs/training/qwen3-30b-a3b/frozen/*/*.yaml)` | 1 GPU |
| 8 | Teacher targets, each config's `part_count` parts | vllm | `python -m echomoe.training.data.teacher targets --config configs/fit/<dolmino, dolci>-teacher-targets-<train, validation>.yaml --part-index I` | 1 GPU per part |
| 9 | Teacher converted to Megatron, for the dev mode runs | megatron | `torchrun --nproc-per-node 8 -m echomoe.training.model --config configs/training/qwen3-30b-a3b/frozen/joint/dres512-s4-c1280-direct.yaml` | 1 node |
| 10 | Distill one student | megatron | `torchrun --nproc-per-node 8 -m echomoe.training --config configs/training/<family>/<config>.yaml` | 8 nodes |
| 11 | Export and fold | megatron | `torchrun --standalone --nproc-per-node 8 -m echomoe.inference.evaluation.export --checkpoint outputs/qwen3-30b-a3b/runs/<run>/checkpoints/step-N` | 1 node |
| 12 | Score on the 27-task suite | vllm | `python -m echomoe.inference.evaluation server --model outputs/qwen3-30b-a3b/runs/<run>/export/step-N --variant folded --ifbench-checkout third-party/IFBench` | 1 node |
| 13 | Throughput, one workload | vllm029 | per node, `torchrun --nnodes 4 --nproc-per-node 8 ... -m echomoe.inference.throughput --workload configs/inference/throughput/i4096-o256.yaml --variant base --model outputs/qwen3-30b-a3b/teacher/hf --output outputs/qwen3-30b-a3b/throughput/<config>-i4096-o256`, then the same with `--variant folded --model outputs/qwen3-30b-a3b/runs/<run>/export/step-N`, then `python -m echomoe.inference.throughput.summarize outputs/qwen3-30b-a3b/throughput/<config>-i4096-o256 --config <config>` | 4 nodes |
| 14 | Larger models (Tables 37, 38): random-weight fixtures, then stage 13 with `--model-config` | vllm029 | `python -m echomoe.inference.throughput.fixtures --model-config configs/inference/throughput/<qwen3-235b-a22b, deepseek-v3, kimi-k2.6>.yaml --official <official snapshot> --output outputs/<model>/fixtures`, then stage 13 with `--model-config` and `--model outputs/<model>/fixtures/<base, folded>` | CPU, then 4 nodes |
| 15 | Step breakdowns | vllm029 | stage 13 with `--profile` and the workloads `i4096-o256-steps2`, `i16384-o512-steps`, `i128-o1024-steps` and `i4096-o256-r8` | 4 nodes |
| 16 | Tables | any | `python scripts/results_tables.py outputs results/evaluation` and `python scripts/throughput_tables.py outputs results/throughput` | CPU |
| 17 | Similarity (Tables 2, 16) and residual spectrum (Table 4, Figure 5) | megatron | `python scripts/analysis.py similarity-capture [--cross-token] <hf model> outputs/qwen3-30b-a3b/data/dolci outputs/similarity/<model>`, then `python scripts/analysis.py similarity outputs/similarity/<model>/moments.safetensors outputs/analysis/similarity/<model>.json` (Table 2) and `python scripts/analysis.py cross-token outputs/similarity/<model> outputs/analysis/similarity/<model>-cross-token.json` (Table 16, needs `--cross-token`). `python scripts/analysis.py spectrum outputs/qwen3-30b-a3b/fit/moments.safetensors outputs/analysis/residual_spectrum.json` | capture: 1 node, then CPU (cross-token: 1 GPU) |

Notes on the stages:
- Stage 10 resumes with `--resume-from latest`. Dev mode runs end at update 1,000 and full-model
  adaptation runs at 4,152.
- Stage 12 variants:
  - `folded`: every student.
  - `folded-fp8`: Table 21's FP8 communication, full-model adaptation students.
  - `base-vllm`, `base-blocks`, `base-blocks-fp8`: the uncompressed model, the baselines for Tables 3, 12,
    17–19, 41 and 42, and for Table 21 (`--model outputs/qwen3-30b-a3b/teacher/hf --output outputs/qwen3-30b-a3b/teacher/eval/<variant>`).

- Stage 13 options (`python -m echomoe.inference.throughput --help`):
  - `--fp8`: FP8 activations in every dispatch and combine (Tables 21, 30, 31).
  - `--tcp`: NCCL over TCP sockets with vLLM's all-gather transport (Table 34, Figure 17).
  - `--capped 2x|2.5x`: inter-node connections capped to about 2x or 2.5x slower (Table 34, Figure 18).
    It needs the patched NCCL from `scripts/install.sh nccl`.
  - `--model-config configs/inference/throughput/<model>.yaml`: the larger models (Tables 37, 38) and
    the adapted Kanana-2 and Gemma 4 (Table 39).
  - `--tokens`: the token check, greedy tokens and log-probabilities of every prompt.
  - `--routing model`: the model's own router instead of balanced expert assignment.
  - `--profile`: per-step windows for the breakdowns (stage 15).
- Online and agent serving (Tables 35, 36, Figures 19–21) run vLLM's own server on the same 4 nodes:
  `python -m echomoe.inference.throughput.serving prompts ...` once, then `serving run --config
  configs/inference/throughput/serving-<online, online-fixed, agent16k, agent32k>.yaml` on every node,
  `--variant base` then `--variant folded`. Launch each setting and variant once first to compile the
  Triton kernels, merge them with `serving seed <launch outputs> --into <seed>`, and pass
  `--triton-seed <seed>` to the measured launches. Without a seed, EchoMoE's kernels compile mid-rate
  and stall the lowest request rates. The module's docstring has the full commands.

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

## Compute

Measured from the logs of the runs behind `results/`. A run that restarted is left out of the per-run
figures. Stages 1, 4, 5, 6 and 9 have no timing record, and 4, 5 and 9 wrote their outputs within minutes.

| Stage | Hardware | Measured time |
| --- | --- | --- |
| 2 Data preparation | CPU, 32 workers | Dolmino: 46 min download and split, 1 h 46 min tokenize and pack. Dolci: 18 min tokenize and pack |
| 3 Teacher SFT, 2,230 updates | 8 nodes | 4.8 h busy, including one crash and resume (309 GPU-h) |
| 7 Moments, then 78 fits | 1 GPU | 12 min, then 1–8 s per fit |
| 8 Teacher targets | 2 nodes (Dolci), 8 nodes (Dolmino) | Dolci train: 7.9 h. Dolmino train: 6.5 h, plus 10.8 h when its budget was raised to 6B tokens. File times only, so lower bounds |
| 10 Dev mode student (backbone frozen), 1,000 updates | 8 nodes | median 1.12 h (0.99–1.24), 78 runs, about 5,700 GPU-h |
| 10 Full-model adaptation student, 4,152 updates | 8 nodes | median 4.06 h (3.48–4.31), 19 runs, about 5,000 GPU-h |
| 11 Export, full-model adaptation | 1 node | median 6 min (file times) |
| 12 One 27-task suite | 1 node | median 97 min (61–169), 119 suites, 198 node-hours |

In total, about 13,300 GPU-hours, 11,000 of them training.

## Known issues

The scorer keeps two bugs, which are in every score in `results/`:
- BBH `dyck_languages` always scores 0.
- The answer normaliser drops signs and decimal points.

IFEval's language detection is not seeded, so a rerun can differ slightly on that task.

Sampled tasks reproduce to within run-to-run spread, not bitwise. Every sample has a fixed seed, but
vLLM's continuous batching makes the batch a request decodes in depend on timing, and the kernels'
last bits depend on the batch, so a near-tie can sample a different token. Rescoring the
`dres512-s4-c1280` full-model adaptation export changed 115 of 1,319 GSM8K generations and moved
single-task scores by up to 0.018 (HumanEval+, 164 problems). Tasks scored by log-likelihood, such
as ARC, reproduce exactly through the release's blocks (`folded`, `base-blocks`), while vLLM's own
MoE kernel (`base-vllm`) can move them by a question.

Throughput (Tables 25–39):
- The base is the uncompressed SFT model served by stock vLLM, on the same nodes with the same
  patches.
- The benchmark measures time, not bytes: every RDMA reduction is the byte model's, priced with the
  measured routing fanout.
- TCP rows vary more than RDMA rows, because the Ethernet is shared with other jobs: the reported
  TCP base runs span 5.4% (209.6 to 221.3 s), and TCP speedups are taken over each setting's mean base.
- The per-call cost tables (`configs/inference/throughput/costs-*.csv`) are shipped as measured.
  `costs-bf16.csv` holds one slow measurement of the streaming kernel at 512 tokens and width 512
  (156 µs where repeats give about 116 µs). The reported rows used the table as it is.
- The step breakdowns count EchoMoE's expert and reference-receive kernels under "misc", as the
  reported breakdowns did.

Nemotron 3.5 Lightning:
- Its vLLM scores vary by a few items between identical runs (ARC-Challenge: 3 to 8 of 1,172), unlike
  the other models, which reproduce item for item. The likely cause is that vLLM autotunes its Mamba-2
  kernels at run time, so a run's block sizes, and with them the summation order, can differ.
- Through `transformers` (the remote code on Hugging Face), its log-probabilities differ from vLLM's
  more than the other models'. Without the `mamba-ssm` kernels, `transformers` runs the Mamba-2 layers
  in plain PyTorch: on ARC-Challenge the student scores 2.3 points and the uncompressed teacher 0.9
  points below vLLM. vLLM reproduces the reported numbers.

## Citation

```bibtex
@misc{liu2026communicate,
  title  = {Communicate the Change: Efficient Expert Parallelism via Activation Compression},
  author = {Liu, Fengyuan and Gala, Jay and Reddy, Siva and Bahdanau, Dzmitry},
  year   = {2026},
}
```
<!-- TODO(user): add eprint and url once the arXiv ID exists -->

## License

The code is under the Apache License 2.0. See [LICENSE](LICENSE). The model weights on Hugging Face
carry their base models' licenses, given on each model card.
