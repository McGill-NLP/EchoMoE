<div align="center">

# Communicate the Change: Efficient Expert Parallelism via Activation Compression

Fengyuan Liu<sup>&#42;</sup>, Jay Gala<sup>&#42;</sup>, Siva Reddy, Dzmitry Bahdanau

<sub><sup>&#42;</sup>Equal contribution. Author ordering determined by coin flip.</sub>

<!-- TODO(user): arXiv badge, e.g. [![Paper](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX) -->
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/McGill-NLP/EchoMoE)
[![Project page](https://img.shields.io/badge/Project-Page-blue)](https://mcgill-nlp.github.io/EchoMoE/)

**[Links](#links)** | **[TL;DR](#tldr)** | **[Results](#results)** | **[Quick start](#quick-start)** | **[Reproduce](#reproducing-the-paper)** | **[Citation](#citation)**

</div>

## Links

- Models: [huggingface.co/McGill-NLP/EchoMoE](https://huggingface.co/McGill-NLP/EchoMoE), also the [EchoMoE collection](https://huggingface.co/collections/McGill-NLP/echomoe-6afcbfa10b38185f87cbd0ac)
- Project page: [mcgill-nlp.github.io/EchoMoE](https://mcgill-nlp.github.io/EchoMoE/)
<!-- TODO(user): - Paper: arXiv link -->

## TL;DR

- Expert parallelism sends every token's hidden state to the nodes that hold its experts (dispatch) and every expert's output back (combine). Between nodes, these all-to-alls are a bottleneck of MoE inference.
- EchoMoE sends a compressed difference from a reference hidden state that is refreshed every few layers, and experts send compressed outputs back. The compressors are fit in closed form from the model's activations, distilled, and folded into the expert weights, so serving adds no matmuls. See [docs/method.md](docs/method.md).
- On Qwen3-30B-A3B at expert parallel 32 over 4 nodes, EchoMoE sends 58% fewer inter-node bytes and improves throughput by 22.6%, while scoring 3.8 points below the uncompressed model's average over 18 benchmarks. Sending the compressed activations in FP8 changes the score by less than 0.3 points.

## Results

Qwen3-30B-A3B with full-model adaptation, at expert parallel 32 over 4 nodes (paper Tables 3, 21 and 25). `s` is the reference stride, `r_d` the dispatch code width and `r_c` the combine code width. RDMA reduction is the bytes per token sent between nodes, against uncompressed bf16 dispatch and combine with the measured routing fanout. The score is the suite mean in %, and the uncompressed model scores 66.9. The speedup is for 4,096-token prompts and 256 output tokens at 128 requests per GPU with balanced expert assignment, over the mean of the uncompressed runs. "FP8 change" is the score change, in points, when every side of the communication is sent in FP8 (one byte per element and a float32 scale per 128).

| s | r_d | r_c | RDMA reduction | score (Δ pp) | throughput speedup | FP8 change |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 1280 | 1280 | 1.46x | 65.5 (−1.4) | 1.025 | −0.13 |
| 4 | 512 | 1280 | 1.84x | 64.9 (−2.0) | 1.115 | −0.04 |
| 4 | 512 | 1024 | 2.08x | 63.8 (−3.1) | 1.169 | −0.13 |
| 4 | 512 | 768 | 2.39x | 63.1 (−3.8) | 1.226 | −0.09 |
| 6 | 256 | 768 | 3.02x | 58.3 (−8.6) | 1.325 | +0.18 |
| 8 | 256 | 512 | 4.03x | 55.7 (−11.2) | 1.429 | −0.15 |

The same method on four other MoE models (paper Table 3), each distilled from its own SFT teacher:

| Model | s | r_d | r_c | RDMA reduction | score: teacher → EchoMoE |
| --- | --- | --- | --- | --- | --- |
| Gemma 4 26B-A4B | 4 | 768 | 1408 | 2.01x | 61.2 → 56.8 (−4.4) |
| Kanana-2-30B-A3B | 4 | 512 | 1024 | 2.01x | 58.5 → 56.1 (−2.4) |
| Nemotron 3.5 Lightning 30B-A3B | 4 | 768 | 1408 | 1.91x | 62.8 → 58.3 (−4.6) |
| Moonlight-16B-A3B | 4 | 512 | 1024 | 1.99x | 53.0 → 47.1 (−5.9) |

Every number comes from the committed files in [`results/`](results/). LSH-MoE, trained with the same recipe, scores 57.1 against EchoMoE's 63.8 at (4, 512, 1024) (paper Table 15).

## Quick start

The models are on [Hugging Face](https://huggingface.co/McGill-NLP/EchoMoE): the Qwen3-30B-A3B teacher with its 19 students, and the teacher and reported student of Gemma 4, Kanana-2, Moonlight and Nemotron. Each model card shows how to serve the model with vLLM or load it with Hugging Face Transformers. To score a model with this repository on one node of 8 H100s:

```bash
git clone https://github.com/McGill-NLP/EchoMoE && cd EchoMoE
scripts/install.sh vllm && . scripts/env.sh vllm
hf download McGill-NLP/EchoMoE-Qwen3-30B-A3B-s4-r512-c768 \
    --local-dir checkpoints/EchoMoE-Qwen3-30B-A3B-s4-r512-c768
python -m echomoe.inference.evaluation server \
    --model checkpoints/EchoMoE-Qwen3-30B-A3B-s4-r512-c768 --variant folded \
    --tasks arc_challenge_5shot --output outputs/quickstart
```

Without `--tasks`, it scores the whole 27-task suite. The Kanana-2 models are Powered by Kanana, under the Kanana License Agreement.

## Reproducing the paper

- [docs/reproduce.md](docs/reproduce.md): how to install the three environments and run the 17 stages, from the base model to every table in the paper.
- [docs/method.md](docs/method.md): how EchoMoE works, and where each part lives in the code.
- [docs/known-issues.md](docs/known-issues.md): what can differ when you rerun our evaluations and benchmarks, and why.

```
echomoe/
├── codec.py, communication_cost.py, fit.py    the method, the byte model, the closed-form fit
├── lsh.py                                     the LSH-MoE baseline (Tables 15, 16)
├── runs.py                                    run directories and the meta.json provenance record
├── hf/                                        the folded students as transformers remote code
├── training/                                  Megatron-Core + DeepEP: SFT, distillation, data
└── inference/
    ├── evaluation/                            HF export and fold, vLLM 0.21 serving, the 27-task suite
    └── throughput/                            vLLM 0.29 expert-parallel runtime, serving and benches
configs/                                       models, data, fit, training recipes (one per run), evaluation, throughput
scripts/                                       env.sh, install.sh, analysis and table scripts
results/                                       the fanout, the byte table, suite scores, throughput tables
third-party/                                   the DeepEP FP8 and NCCL rate-cap patches, IFBench, LiveCodeBench
docs/                                          method, reproduction, known issues
```

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

The code is under the Apache License 2.0. See [LICENSE](LICENSE). The model weights on Hugging Face carry their base models' licenses, given on each model card.

## Acknowledgement

EchoMoE builds on [vLLM](https://github.com/vllm-project/vllm), [Megatron-Core](https://github.com/NVIDIA/Megatron-LM), [DeepEP](https://github.com/deepseek-ai/DeepEP), [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) and [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
