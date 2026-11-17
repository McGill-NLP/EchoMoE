# Compute cluster operations

This file covers cross-cluster rules. For Mila-specific GPU types, partitions,
storage paths, and module loading, use the `/slurm` skill. It has the full
reference tables. Use this file for the rules that always apply.

## Before running any command

1. **Verify you're on a compute node, not a login node.** Check with `hostname`
   or `squeue -u $USER`. Use `salloc` or `sbatch` to reach a compute node.

2. **Use `uv` for everything Python-environment-related.** No `pip`, no
   `conda`, no `poetry`. This repository uses `.venv-megatron` and
   `.venv-vllm`; do not create a generic `.venv`. Standard commands:

   - Create a venv: `uv venv .venv-megatron` or `uv venv .venv-vllm`.
   - Install from `pyproject.toml`: `uv sync`.
   - Add a dep: `uv add <pkg>` (tracked in `pyproject.toml` plus the repo's
     canonical lockfile workflow); remove: `uv remove <pkg>`.
   - Run a script inside the env without activating: `uv run python ...`.

   Always select the intended named environment before running `python`.
   Source `scripts/env.sh megatron` or `scripts/env.sh vllm`.

   `uv pip install <pkg>` exists as a low-level escape hatch but is **not**
   the default. It installs into the venv without updating `pyproject.toml`
   or the repo's canonical lockfiles, which silently breaks reproducibility.
   Use `uv add` for anything you want to keep. See \[\[ml-research\]\]'s
   reproducibility section.

3. **Load CUDA before importing CUDA-dependent libraries** (DeepSpeed, vLLM,
   bitsandbytes, etc.) in interactive sessions. SLURM jobs handle this via the
   submission script. Interactive shells do not. Check available versions with
   `module avail cuda`.

## Before expensive operations

Always perform these checks before submitting a real job or kicking off a long
operation:

1. **Test on a small instance first.** 1 problem, short generation, small batch,
   reduced checkpoint interval. Verify it works end-to-end.
2. **Sanity-check data paths and shapes on a small batch.**
3. **Verify GPU allocation / availability.**
4. **Dry-run or validate configs.**
5. **Check disk space** for checkpoints and logs (`disk-quota`).
6. **Verify checkpoint paths are writable and have headroom.**
7. **Get explicit sign-off before every scheduler mutation.** This includes
   `sbatch`, resubmission, and `scancel`. Show the resource
   specs, config, and key changes. Approval for one submission does NOT carry
   over to a modified resubmission, retry, replacement, or cancellation.
8. **Startup is NFS-bound, not GPU-bound.** Eight concurrent processes per
   node turn a 90-second 60 GB weight load into 20 minutes and a 19-minute
   activation capture read into an hour. Launch four per node when latency
   matters; use all eight when aggregate throughput matters and pay the
   startup once.

## Storage discipline

**Never write large outputs (logs, checkpoints, results, wandb data) to `$HOME`.**
HOME is small (~100GB on most clusters), backed up daily, and slow. Excessive I/O
on HOME degrades the shared filesystem.

- **Source code, configs, small files** → `$HOME`.
- **Large outputs, checkpoints, logs, datasets** → `$SCRATCH` (5TB, fast, no
  backup, files unused >90 days deleted on Mila).
- **Active job I/O** → `$SLURM_TMPDIR` (fastest, no quota, cleared after job).
  Copy data in at job start, copy results out at job end.
- **SLURM output files** (`-o`/`-e`) → always under `$SCRATCH/slurm-logs/`.

On Compute Canada clusters (Narval, etc.), use `$SCRATCH` and `$USER` rather
than hardcoded Mila paths.

If you ever hit `Disk quota exceeded`, **stop and ask the user**. Do not delete
files on your own.

## Job script structure

- Place submission scripts in a dedicated folder (`scripts/`, `jobs/`, or
  `slurm_scripts/`). Keep cluster-specific code separate from core logic.
- Required sbatch flags: `--job-name`, `--output`, `--error`, `--time`,
  `--mem`, `--gres` (for GPUs), `--cpus-per-task`.
- Use `set -euo pipefail` in the bash portion.
- Log hostname, GPU info (`nvidia-smi`), date, and the git commit hash at the
  start of the script.
- When using `torchrun`, set `--master_port=$((29500 + SLURM_JOB_ID % 10000))`
  to avoid port collisions on shared nodes.

## Time limits

**Never guess job duration.** Submit a small benchmark job first (e.g., 5
samples on a short partition), read the timing, then extrapolate. Wrong time
limits either waste cluster reservations or kill long runs near completion.

## Don't pipe long jobs through head / tail

The `SIGPIPE` from `head` will kill the upstream process. For long-running
output, use background tasks (`run_in_background`) and read the output file.

## Cross-references

- `/slurm`. Full GPU/partition/storage tables, debugging commands, allocation rules.
- `/experiment`. Scaffolding new experiments before they go to SLURM.
- `/pytorch-debug`. When GPU code crashes mid-job (OOM, dtype, device errors).
- \[\[ml-research\]\] for the "test on small instance first" rule's coupling with TDD.
