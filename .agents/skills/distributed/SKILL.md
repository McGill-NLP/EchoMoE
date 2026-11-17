---
name: distributed
description: "Debug and configure distributed training: DDP, FSDP, DeepSpeed, torchrun, accelerate, multi-node SLURM. Use when the user hits NCCL errors, hangs at barrier, OOM with FSDP, wrong launcher invocation, or asks how to scale a training script to multi-GPU/multi-node."
---

# Distributed Training Assistant

Help the user launch, debug, and reason about distributed training. The
failure modes here are notorious. Silent hangs, off-by-one ranks, NCCL
timeouts, uneven sharding. And the iteration loop is slow because every
test requires a real allocation. Prioritize narrowing the bug to the
smallest reproducer.

## Iterate Small First: Always

Multi-node runs cost minutes per crash. Before scaling:

1. **Single GPU, single process.** Does the model train at all? Catches
   shape/dtype bugs that have nothing to do with distribution.
2. **Single node, 2 GPUs.** Tests the launcher, NCCL, and sharding logic at
   the smallest scale that exercises them.
3. **Multi-node, 2 nodes × 2 GPUs.** Tests rendezvous, MASTER_ADDR routing,
   and inter-node NCCL.
4. **Full scale.**

Skipping step 2 is how people end up debugging FSDP on 64 GPUs.

## Launchers: pick one and stick with it

| Launcher            | When to use                                                                                            |
| ------------------- | ------------------------------------------------------------------------------------------------------ |
| `torchrun`          | Vanilla PyTorch DDP / FSDP. Most transparent. Reads `RANK`, `WORLD_SIZE`, `MASTER_ADDR/PORT` from env. |
| `accelerate launch` | HuggingFace ecosystem. Hides boilerplate. `accelerate config` writes a YAML; commit it.                |
| `deepspeed`         | When you need ZeRO-3 / pipeline parallel / 1-bit optimizers. Heavier config; reads `ds_config.json`.   |
| `srun` (SLURM)      | Wraps any of the above on Compute Canada / Mila for multi-node.                                        |

**Don't mix launchers.** `python -m torch.distributed.launch` is deprecated.
use `torchrun`. Don't run `torchrun` inside `accelerate launch`.

## Required env vars (DDP / FSDP)

`torchrun` and `accelerate launch` set these for you. If you're launching
manually or via `srun`, set them yourself:

```bash
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 10000))   # avoid collisions
export WORLD_SIZE=$((SLURM_NNODES * SLURM_GPUS_ON_NODE))
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
```

`MASTER_PORT` collision is the silent killer of shared-node setups. Always
randomize it from `SLURM_JOB_ID`.

## Common Issues

### NCCL Timeout / Hang at `init_process_group` or `barrier`

- **First step:** `export NCCL_DEBUG=INFO` and rerun. The logs will name the
  rank that failed to join.
- **Symmetric reachability:** all nodes must reach each other on the
  NCCL port range, not just `MASTER_ADDR`. On clusters with multiple
  network interfaces, set `NCCL_SOCKET_IFNAME=eth0` (or the right interface;
  use `ifconfig` / `ip addr` to identify).
- **Firewall / wrong network:** on Compute Canada use the InfiniBand interface,
  not the management Ethernet. `NCCL_IB_DISABLE=0` keeps IB on.
- **Hang at `barrier()` mid-training** usually means uneven inputs across
  ranks. One rank finished an epoch early. Use `DistributedSampler` and
  `drop_last=True`, or wrap the loop in `join()` (DDP).

### DDP: Gradients Not Syncing

- Wrap the model with `DDP(model, device_ids=[local_rank])`. Forgetting this
  silently runs each rank as an independent training process. You'll see
  loss curves diverge across ranks.
- `find_unused_parameters=True` is a perf hit but necessary if some params
  don't receive gradients on every step (common in MoE / conditional
  routing). Set it explicitly; don't rely on the default.
- After `loss.backward()`, gradients are already all-reduced. Don't call
  `dist.all_reduce(grad)` manually on top. You'll double-reduce.

### FSDP: Memory Doesn't Drop / OOM

- **Sharding strategy:** `FULL_SHARD` shards params + grads + optim state;
  `SHARD_GRAD_OP` keeps params replicated. If memory looks like DDP, you
  probably picked the wrong strategy.
- **Mixed precision:** `MixedPrecision(param_dtype=bf16, reduce_dtype=fp32, buffer_dtype=fp32)` is the safe default. All bf16 loses too much numerical
  range on the reduce.
- **CPU offload:** `CPUOffload(offload_params=True)` saves GPU memory at
  major throughput cost. Use only when you'd otherwise OOM.
- **Activation checkpointing** (`apply_activation_checkpointing`) is
  orthogonal to FSDP and usually necessary for large models. Wrap at the
  transformer-block level, not the whole model.
- **`auto_wrap_policy`** must wrap at a granularity smaller than the whole
  model, or sharding does nothing. `transformer_auto_wrap_policy` keyed on
  the block class is the standard.

### DeepSpeed: Config Confusion

- `train_micro_batch_size_per_gpu` × `gradient_accumulation_steps` ×
  `world_size` = global batch size. Get this wrong and learning rate scaling
  silently breaks.
- ZeRO stage: 1 (optim state) \< 2 (+ grads) \< 3 (+ params). Stage 3 is
  roughly equivalent to FSDP FULL_SHARD.
- `bf16` and `fp16` blocks are mutually exclusive. Set exactly one.
- DeepSpeed wraps the optimizer; don't also `optimizer.step()` manually.

### Process Group Leaks / Zombies

After a crash, `torchrun` sometimes leaves orphan processes holding GPU
memory:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
# Then for your own processes only:
ps -u $USER -o pid,cmd | grep -E 'python|torchrun'
```

Kill your own zombies (`kill -9 <pid>`). On SLURM, `scancel $JOB_ID`
should clean these up. If it doesn't, you have a worse problem.

### Saving / Loading FSDP Checkpoints

FSDP checkpoints are sharded by default. The footguns:

- `model.state_dict()` returns the **local shard**, not the full model.
  Use `FullStateDictConfig` or `LocalStateDictConfig` explicitly.
- Loading a full-state checkpoint into an FSDP model requires
  `FullStateDictConfig(rank0_only=True)` then broadcasting. See PyTorch
  docs. Don't `load_state_dict` blindly.
- For interop with HF `from_pretrained` saving, gather to rank 0 in a
  full state dict before saving.

## Debugging Knobs

```bash
# NCCL: verbose comms logs
export NCCL_DEBUG=INFO            # or WARN, TRACE
export NCCL_DEBUG_SUBSYS=ALL

# PyTorch: extra distributed sanity checks
export TORCH_DISTRIBUTED_DEBUG=DETAIL   # INFO, DETAIL, OFF
export TORCH_NCCL_BLOCKING_WAIT=1        # surface NCCL errors immediately
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Limit visible GPUs (single-node scale-down test)
export CUDA_VISIBLE_DEVICES=0,1
```

`TORCH_DISTRIBUTED_DEBUG=DETAIL` catches the most common DDP bugs at startup
(parameter ordering mismatches, unused params). Turn it on whenever
launching a new model architecture for the first time.

## Minimal multi-node SLURM template

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 10000))
export NCCL_DEBUG=INFO

srun bash -c '
    torchrun \
        --nnodes=$SLURM_NNODES \
        --nproc_per_node=$SLURM_GPUS_ON_NODE \
        --node_rank=$SLURM_NODEID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        train.py --config configs/big.yaml
'
```

Note: `srun bash -c '...'` (single quotes) so the env vars resolve **per-task
on the compute node**, not at submission time.

## Cross-references

- `/slurm`. Partition tables, allocation rules, debugging failed jobs.
- `/pytorch-debug`. When the bug is per-rank rather than distributed.
- `/experiment`. Profiling, checkpointing/resume, sweeps at scale.

## Scope

$ARGUMENTS
