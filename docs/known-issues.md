# Known issues

What you may notice when you rerun our evaluations and benchmarks, and why.

## Evaluation

### Which scorer bugs are in the reported scores?

<details>
<summary>Show answer</summary>

The scorer keeps two bugs, which are in every score in `results/`:
- BBH `dyck_languages` always scores 0.
- The answer normaliser drops signs and decimal points.

</details>

### Why does a rerun of a sampled task differ slightly?

<details>
<summary>Show answer</summary>

Sampled tasks reproduce to within run-to-run spread, not bitwise. Every sample has a fixed seed, but
vLLM's continuous batching makes the batch a request decodes in depend on timing, and the kernels'
last bits depend on the batch, so a near-tie can sample a different token. Rescoring the
`dres512-s4-c1280` full-model adaptation export changed 115 of 1,319 GSM8K generations and moved
single-task scores by up to 0.018 (HumanEval+, 164 problems).

Tasks scored by log-likelihood, such as ARC, reproduce exactly through the release's blocks
(`folded`, `base-blocks`), while vLLM's own MoE kernel (`base-vllm`) can move them by a question.
IFEval's language detection is not seeded, so a rerun can also differ slightly on that task.

</details>

### Why does Nemotron's vLLM score vary between identical runs?

<details>
<summary>Show answer</summary>

Nemotron 3.5 Lightning's vLLM scores vary by a few items between identical runs (ARC-Challenge: 3 to
8 of 1,172), unlike the other models, which reproduce item for item. vLLM autotunes its Mamba-2
kernels at run time, so a run's block sizes, and with them the summation order, can differ. With
every Triton autotuner pinned to its first configuration, two runs on different nodes matched item
for item.

</details>

### Why does Nemotron score lower through `transformers`?

<details>
<summary>Show answer</summary>

Through `transformers` (the remote code on Hugging Face), Nemotron's log-probabilities differ from
vLLM's more than the other models'. Without the `mamba-ssm` kernels, `transformers` runs the Mamba-2
layers in plain PyTorch. On ARC-Challenge the student scores 2.3 points and the uncompressed teacher
0.9 points below vLLM. vLLM reproduces the reported numbers.

</details>

## Throughput

### What is the throughput baseline?

<details>
<summary>Show answer</summary>

The base is the uncompressed SFT model served by stock vLLM, on the same nodes with the same
patches. The benchmark measures time, not bytes: every RDMA reduction is the byte model's, priced
with the measured routing fanout.

</details>

### Why do TCP rows vary more than RDMA rows?

<details>
<summary>Show answer</summary>

The Ethernet is shared with other jobs. The reported TCP base runs span 5.4% (209.6 to 221.3 s),
and TCP speedups are taken over each setting's mean base.

</details>

### Are the per-call cost tables exactly as measured?

<details>
<summary>Show answer</summary>

Yes. The tables in `configs/inference/throughput/costs-*.csv` are shipped as measured.
`costs-bf16.csv` holds one slow measurement of the streaming kernel at 512 tokens and width 512
(156 µs where repeats give about 116 µs). The reported rows used the table as it is.

</details>

### Why do the step breakdowns put some of EchoMoE's kernels under "misc"?

<details>
<summary>Show answer</summary>

[`scripts/throughput_tables.py`](../scripts/throughput_tables.py) puts each GPU kernel in the first
category whose name pattern matches (`KERNEL_CATEGORIES`), and a kernel that matches none counts as
`other`, the row "norms, elementwise, sampling, misc" of the fine breakdown. Ten of EchoMoE's kernels
match none:
- Expert path: `_align_count`, `_align_place`, `_align_scan`, `_silu_mul_sorted`, `_sum_live`,
  `_receive_routes`.
- Reference path: `_reference`, `_send_reference`, `_receive_reference`, `_residual_prepare`.

So part of EchoMoE's expert routing, activation and output sum counts as "misc", while vLLM's
equivalents for the base model (`moe_align_block`, `act_and_mul`, `moe_sum`) count as expert work.
The reported breakdowns used the same patterns, and `results/throughput/breakdowns.csv` reproduces
them.

</details>
