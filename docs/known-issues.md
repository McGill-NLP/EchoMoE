# Known issues

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
  the other models, which reproduce item for item. vLLM autotunes its Mamba-2 kernels at run time, so a
  run's block sizes, and with them the summation order, can differ. With every Triton autotuner pinned
  to its first configuration, two runs on different nodes matched item for item.
- Through `transformers` (the remote code on Hugging Face), its log-probabilities differ from vLLM's
  more than the other models'. Without the `mamba-ssm` kernels, `transformers` runs the Mamba-2 layers
  in plain PyTorch: on ARC-Challenge the student scores 2.3 points and the uncompressed teacher 0.9
  points below vLLM. vLLM reproduces the reported numbers.
