# Throughput results

Measured with vLLM 0.29 (`.venv-vllm029`) at EP32 on 4 nodes of 8 H100s, with balanced expert
assignment unless a row says otherwise. Each offline row ran both models back to back on the same
nodes with the same prompts and settings, the base first: 1 warm-up and 2 timed runs. Columns
starting `base_` belong to the uncompressed model, `echomoe_` to the compressed one.

`scripts/throughput_tables.py` rebuilds every file here except `deepep_scaling.csv` from the
measured records, laid out as the throughput CLI and `serving.py` write them under `outputs/`
(the Hugging Face artifacts of each model hold them):

    python scripts/throughput_tables.py outputs results/throughput

- `throughput.csv`: one row per measured cell. Qwen3-30B-A3B's configurations by workload,
  precision and network (the concurrency sweep and the agent and reasoning shapes included), the
  random-weight Qwen3-235B-A22B, DeepSeek-V3 and Kimi-K2.6, and the trained Kanana-2 and Gemma 4
  students. `network` is `rdma`, `tcp` (eth0), `capped-2x` or `capped-2.5x` (each inter-node
  connection capped). `config` is the codec `dres{dispatch_rank}-s{stride}-c{combine_rank}`, and
  `kv_cache` the KV cache precision. Speedup and times are the mean of the two timed runs, and
  `spread` is the larger relative gap between a row's two timed runs. Base does not depend on the
  configuration, so `speedup_over_mean_base` divides the mean base time of the rows that differ only
  in configuration (`mean_base_s`) by the row's EchoMoE time, and the TTFT and decode TPOT speedups
  over the mean base likewise (latency ratios, above 1 is faster). `ttft_speedup` and
  `tpot_speedup` compare a row with its own base. The larger models' rows time rank 0's batches only.
- `serving.csv`: online serving with Qwen3-30B-A3B (4, 512, 768), one row per setting, routing,
  precision, model and request rate. `setting` is `online` (real replies), `online-fixed-replies`
  (every reply the length the workload sets) or `agent-16k` and `agent-32k` (agent contexts).
  Goodput counts the requests within a P99 TTFT and TPOT target: 2,000 and 150 ms online, 5,000 and
  300 ms for the agent settings.
- `breakdowns.csv`: where a step's time goes in the profiled rows, in ms per step and mean over
  ranks: coarsely (`level` coarse), per operation (fine), and the GPU idle while the CPU is inside a
  decode step's cudaGraphLaunch (`launch_stall`, which only the profiler adds), by prefill or decode
  `window`.
- `deepep_scaling.csv`: DeepEP alone at the larger models' row widths with Qwen3-30B-A3B's routing,
  from `echomoe.inference.throughput.transport_bench deepep-scaling` (its `tokens` and `width`
  columns renamed `tokens_per_rank` and `row_width`).
