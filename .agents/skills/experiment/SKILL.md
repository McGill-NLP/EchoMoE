---
name: experiment
description: "Scaffold, configure, launch, profile, and evaluate ML experiments: including hyperparameter sweeps and checkpoint/resume workflows. Use whenever the user is planning, running, measuring, sweeping, or recovering an experiment."
---

# Experiment Assistant

Help the user plan, scaffold, run, measure, and recover ML experiments. The
sections below are loaded together but most sessions only touch one or two.
match the active phase of the user's work.

## When brainstorming / planning

Before jumping to implementation, think critically:

- **Challenge the hypothesis**: is this the simplest experiment that tests the claim? Is there a cheaper/faster one equally informative?
- **Apply Occam's razor**: if a simpler setup answers the same question, suggest it. Don't over-engineer.
- **Identify confounding variables**: seed, data order, hyperparams, hardware. What else could explain the result?
- **Question the metrics**: are we measuring what we think? Can the metric be gamed?
- **Consider baselines**: is the baseline fair? Apples-to-apples?
- **Push back when warranted**: if the experiment won't convincingly support or refute the hypothesis, say so and suggest alternatives.

## When setting up a new experiment

1. **Clarify the goal**: what's being tested, what's the baseline, what metrics matter?
2. **Check the existing setup**: read the repo's config system, experiment tracking, and script conventions before creating anything new.
3. **Scaffold minimally**: training/eval script (or modify existing), SLURM submission script in `scripts/`, config changes if using Hydra/YAML. For new dependencies, see `agents/cluster.md` for the `uv` mandate.
4. **Set up logging**: w&B / tensorboard / whatever the repo uses. Include run name, key hyperparams, and git commit hash.
5. **Add sanity checks**: small batch forward pass, shape verification, gradient flow check before launching full runs.

## Experiment hygiene

- **Name runs descriptively**: encode key hyperparams (e.g. `qwq32b_math500_softmax_k15_cs01`).
- **Log everything needed to reproduce**: full config, git hash, command used, random seed.
- **Save checkpoints to a path with the run name**: avoid overwriting previous experiments.
- **Separate stdout and stderr**: use `--output` and `--error` in SLURM scripts.

## Before launching

See `agents/cluster.md` for the pre-flight checks (test on a small instance,
verify paths, check GPU availability, explicit sign-off before `sbatch`).

## Profiling (perf / memory)

Distinct from *debugging* a crash (`/pytorch-debug` covers that). Profile when
something runs but is too slow or too memory-heavy.

**What to measure first:**

- **GPU utilization**. `nvidia-smi -l 1` during a run. If GPUs sit under ~80% during steady state, the bottleneck is upstream (dataloader, CPU, I/O), not compute.
- **Step time breakdown**. `torch.profiler.profile(activities=[CPU, CUDA])` with a small `schedule` over ~5 steps. Export to TensorBoard or Chrome trace.
- **Peak VRAM**. `torch.cuda.max_memory_allocated()` after a warmup step. Reset with `torch.cuda.reset_peak_memory_stats()`.
- **Dataloader stalls**: wrap `next(iter(loader))` in a timer. If it's >5% of step time, raise `num_workers`, use `persistent_workers=True`, or pre-shard into `$SLURM_TMPDIR`.

**Common bottlenecks to check:**

- Synchronous transfers (`.item()`, `.cpu()` mid-step). Force GPU stalls.
- BF16 vs FP16 vs FP32 mismatch. Implicit casts in softmax/norm promote to FP32 silently.
- `torch.autograd.set_detect_anomaly(True)` left on in a batch job. Extreme filesystem IOPS, kills throughput.
- DataLoader `num_workers=0`. Single-threaded fetch.

**Memory pressure:**

- Activation checkpointing (`torch.utils.checkpoint`) before reaching for FSDP/ZeRO.
- Gradient accumulation > batch reduction when VRAM is the limit but step count isn't.
- `del` intermediates and `torch.cuda.empty_cache()` only as a last resort. Usually a sign of a leak, not a cure.

## Checkpointing & resume

Critical on Mila's `long` partition and on Compute Canada where preemption /
time limits are routine.

**Save strategy:**

- **Frequency**: checkpoint often enough that ≤30 min of compute is at risk. On a 7-day run that's roughly every 10–30k steps depending on step time.
- **Format**: prefer `safetensors` for weights (no pickle, faster load). Keep optimizer state in `torch.save` if `safetensors` doesn't fit your trainer.
- **What to include**: model weights, optimizer state, LR scheduler state, RNG states (`torch.get_rng_state`, `numpy`, `random`, `torch.cuda` per device), step / epoch counter, dataloader position (if resumable).
- **Where**: `$SCRATCH/<exp_name>/checkpoints/step_<N>/`. Never `$HOME`. Use atomic writes (write to `.tmp`, rename) so a kill mid-write doesn't corrupt the latest.

**Resume on preemption:**

- SLURM jobs on `long` are auto-requeued after preemption. Your script must detect a checkpoint at start and load it.
- Look for `latest.pt` symlink or the highest-numbered checkpoint directory.
- Verify the RNG, scheduler, and dataloader state restore. Silent step-counter resets are the most common bug.
- For `-grace` partitions: register a SIGTERM handler that writes a final checkpoint before the kill.

**Sanity check before relying on resume:**

- Run a job for 100 steps, kill it, resume. Verify loss continues smoothly (within noise) instead of restarting from scratch.

## Evaluation

Design eval before launching training, not after. Held-out splits that get
constructed mid-experiment are the single biggest source of irreproducible
results.

**Splits:**

- Decide train / val / test before any model touches the data. Commit the split or its random seed.
- For pretraining: hold out by document, not by sample, to avoid near-duplicate leakage.
- For SFT/RLHF: hold out by prompt source/template, not just by individual prompts.
- Beware temporal leakage. If data has timestamps, the test set should be strictly later.

**Metric choice:**

- Pick **one** primary metric. Secondaries are reported but not optimized.
- For generation: use task-appropriate metrics (exact match, F1, BLEU, pass@k). Never single-sample LLM-as-judge unless calibrated.
- Couples with \[\[principles\]\] Rule 9: the metric should encode the intent, not just behavior.

**Statistical significance:**

- Always report ≥3 seeds for any claim where Δmetric \< ~2× run-to-run variance.
- For comparisons, report mean ± std and ideally a paired test (same seeds, different methods).
- A single-seed result is a data point, not a conclusion.

**Eval hygiene:**

- Save eval predictions alongside the metric. Without them you can't diagnose *why* a number changed.
- Log eval metrics to the same W&B run as training so you can correlate.
- For long evals, checkpoint mid-eval too. Mila will preempt eval jobs as readily as training.

## Hyperparameter sweeps

When you're tuning > 2 hyperparams or want to know the shape of the loss
surface, sweep. Don't hand-pick.

**Choose the runner:**

- **W&B Sweeps**: built-in Bayesian / grid / random, supports SLURM agents. Best when already using W&B.
- **Optuna**: finer control over pruning, multi-objective, parallel trials with a shared SQL backend. Heavier setup.
- **Plain grid over SLURM**. `sbatch --array=0-N` with a job-array index → config row. Best for small (\<50) grids and full reproducibility.

**Search-space hygiene:**

- Log-spaced sampling for LR, weight decay, dropout (`loguniform`). Uniform for things like depth, heads.
- Prune aggressively. Kill runs that diverge in the first 10% of training rather than burning the full budget.
- Always include the *current best known* config as a control point. If the sweep can't reproduce it, the sweep is broken.

**Practical guards:**

- Cap concurrent jobs (`--array=...%K`) to avoid stealing the cluster from yourself.
- Log the sweep id and trial id into the run name and the checkpoint path. Without this, debugging a single weird trial is brutal.
- Resume-on-preemption applies to sweeps too. Make sure each trial's resume logic uses its own trial-id checkpoint dir.

## Scope

$ARGUMENTS
