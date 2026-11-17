"""Distributed checkpoints, which training resumes from and export reads:

    step-NNNNNNN/distributed-model/  Megatron's sharded model state. A dev-mode run keeps only the
                                     codec modules, and its backbone is the teacher's `megatron-ep8/`
    step-NNNNNNN/optim/              optimizer-rank-NNNN.pt, rng-rank-NNNN.pt, scheduler.pt, and
                                     parameter-state/ (distributed fp32 masters and Adam moments)
    step-NNNNNNN/meta.json           the optimizer step, world size, trainer progress, `codec_only`

Each step is written as `.step-NNNNNNN.staging/` and renamed once every rank has finished. Every
model state is kept and only the newest `keep_last` optimizer states.
"""

from __future__ import annotations

import json
import random
import re
import shutil
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from megatron.core import dist_checkpointing, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedTensor

from echomoe.codec import MODULE_ATTRIBUTE_NAMES

MODEL = "distributed-model"
OPTIM = "optim"
# Megatron's per-rank bucket shards of the distributed optimizer: no gather, no replica.
PARAMETER_STATE_SHARDING = {"distrib_optim_sharding_type": "dp_reshardable"}
_COMPRESSOR = re.compile(r"\.mlp\.(" + "|".join(MODULE_ATTRIBUTE_NAMES.values()) + r")\.")
_MODULE_PREFIX = re.compile(r"^(?:module\.)+")
# Set by the scheduler at every step, or the parameters themselves. The rest is configuration.
_GROUP_STATE_FIELDS = ("params", "step", "lr", "weight_decay")


def save(
    directory: Path,
    chunks: Sequence[Any],
    optimizer: Any,
    scheduler: Any,
    *,
    step: int,
    progress: dict[str, Any],
    keep_last: int,
    codec_only: bool = False,
) -> None:
    """Write one checkpoint. The caller has gathered the distributed parameters."""
    rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
    staging = directory.with_name(f".{directory.name}.staging")
    if rank == 0 and staging.exists():
        shutil.rmtree(staging)  # an earlier attempt at this step that never finished
    save_model(chunks, staging, codec_only=codec_only)
    optim = staging / OPTIM
    optim.mkdir(exist_ok=True)
    torch.save(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),  # noqa: NPY002 - the module-level state
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(),
            # Megatron's dropout draws from its own CUDA RNG tracker.
            "rng_tracker": tensor_parallel.get_cuda_rng_tracker().get_states(),
        },
        optim / f"rng-rank-{rank:04d}.pt",
    )
    if rank == 0:
        torch.save(scheduler.state_dict(), optim / "scheduler.pt")
    with _without_empty_sub_optimizers(optimizer):
        torch.save(
            {"optimizer": optimizer.state_dict(), "param_groups": _group_records(optimizer)},
            optim / f"optimizer-rank-{rank:04d}.pt",
        )
        if _sharded(optimizer):
            (optim / "parameter-state").mkdir(exist_ok=True)
            _save_sharded(
                optimizer.sharded_state_dict(
                    {}, is_loading=False, metadata=PARAMETER_STATE_SHARDING
                ),
                optim / "parameter-state",
            )
    torch.distributed.barrier()
    if rank == 0:
        state = {
            "optimizer_step": step,
            "world_size": world_size,
            "progress": progress,
            "codec_only": codec_only,
        }
        (staging / "meta.json").write_text(json.dumps(state, indent=2) + "\n")
        if directory.exists():
            shutil.rmtree(directory)  # a run resumed from an earlier step writes this step again
        staging.rename(directory)
        # drop all but the newest `keep_last` optimizer states
        with_state = [
            path for path in sorted(directory.parent.glob("step-*")) if (path / OPTIM).is_dir()
        ]
        for older in with_state[:-keep_last]:
            shutil.rmtree(older / OPTIM)
    torch.distributed.barrier()


def latest(run_directory: Path) -> Path:
    """The newest checkpoint that still holds its optimizer state."""
    candidates = [
        path
        for path in sorted((run_directory / "checkpoints").glob("step-*"))
        if (path / OPTIM).is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(f"no resumable checkpoint under {run_directory / 'checkpoints'}")
    return candidates[-1]


def step_meta(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "meta.json").read_text())


def load(directory: Path, optimizer: Any, scheduler: Any) -> dict[str, Any]:
    """Restore the optimizer, scheduler and RNG state of a checkpoint whose model is already
    loaded, and return its `meta.json`."""
    state = step_meta(directory)
    rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
    if state["world_size"] != world_size:
        raise ValueError(
            f"{directory} was written by {state['world_size']} ranks. Per-rank optimizer state "
            f"does not reshard to {world_size}"
        )
    optim = directory / OPTIM
    with _without_empty_sub_optimizers(optimizer):
        written = torch.load(
            optim / f"optimizer-rank-{rank:04d}.pt", map_location="cuda", weights_only=False
        )
        live = _group_records(optimizer)
        optimizer.load_state_dict(written["optimizer"])
        if _sharded(optimizer):
            # the fused Adam creates its moments lazily, so create them for the load to fill
            for sub_optimizer in _sub_optimizers(optimizer):
                if getattr(sub_optimizer, "init_state_fn", None) is not None:
                    sub_optimizer.init_state_fn(
                        sub_optimizer.optimizer, getattr(sub_optimizer, "config", None)
                    )
            template = optimizer.sharded_state_dict(
                {}, is_loading=True, metadata=PARAMETER_STATE_SHARDING
            )
            optimizer.load_state_dict(
                dist_checkpointing.load(
                    template, str(optim / "parameter-state"), strict="raise_all"
                )
            )
        # after both loads: each pairs groups by their identifier keys
        _restore_groups(optimizer, live, written["param_groups"])
    rng = torch.load(optim / f"rng-rank-{rank:04d}.pt", weights_only=False)
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])  # noqa: NPY002
    torch.set_rng_state(rng["torch_cpu"])
    torch.cuda.set_rng_state(rng["torch_cuda"])
    tensor_parallel.get_cuda_rng_tracker().set_states(rng["rng_tracker"])
    scheduler.load_state_dict(torch.load(optim / "scheduler.pt", weights_only=False))
    return state


# --- model state ---------------------------------------------------------------------------------


def _model_state(
    chunks: Sequence[Any], *, backbone_only: bool, codec_only: bool, prefix: str = ""
) -> dict[str, Any]:
    # Each codec tensor is stored whole under its own layer's key. The codec sits on some layers
    # only, so stacked along Megatron's layer axis it would leave holes the sharding validation
    # refuses.
    (chunk,) = chunks

    def rekey(state: Any) -> Any:
        if isinstance(state, dict):
            return {name: rekey(value) for name, value in state.items()}
        if (
            isinstance(state, ShardedTensor)
            and state.prepend_axis_num
            and _COMPRESSOR.search(state.key)
        ):
            layer = int(state.global_offset[0])
            return ShardedTensor.from_rank_offsets(
                state.key.replace(".layers.", f".layers.{layer}.", 1),
                state.data,
                replica_id=state.replica_id,
                allow_shape_mismatch=state.allow_shape_mismatch,
            )
        return state

    state = chunk.sharded_state_dict(prefix=prefix)
    if backbone_only or codec_only:
        state = {
            name: value
            for name, value in state.items()
            if bool(_COMPRESSOR.search(name)) == codec_only
        }
    return {"model": rekey(state)}


def _save_sharded(state: dict[str, Any], directory: Path) -> None:
    request = dist_checkpointing.save(
        state, str(directory), async_sharded_save=True, async_strategy="mcore"
    )
    if request is not None:
        request.execute_sync()


def save_model(
    chunks: Sequence[Any],
    directory: Path,
    *,
    backbone_only: bool = False,
    codec_only: bool = False,
) -> None:
    """Every rank writes its shards of the model to `directory/distributed-model/`. The backbone
    alone is the converted-teacher cache, and the codec alone is a dev-mode checkpoint."""
    target = directory / MODEL
    if torch.distributed.get_rank() == 0:
        target.mkdir(parents=True)
    torch.distributed.barrier()
    _save_sharded(_model_state(chunks, backbone_only=backbone_only, codec_only=codec_only), target)
    torch.distributed.barrier()


def load_model(
    chunks: Sequence[Any],
    directory: Path,
    *,
    backbone_only: bool = False,
    codec_only: bool = False,
    prefix: str = "",
) -> None:
    """Load `directory/distributed-model/`. The backbone alone keeps the fit already in the codec
    modules, and a model built without the trainer's DDP wrapper passes `module.` as `prefix`."""
    template = _model_state(
        chunks, backbone_only=backbone_only, codec_only=codec_only, prefix=prefix
    )
    loaded = dist_checkpointing.load(template, str(directory / MODEL), strict="raise_all")
    (chunk,) = chunks
    # The DDP and mixed-precision wrappers prefix every key with `module.`, so load into the model.
    target = chunk
    while isinstance(getattr(target, "module", None), torch.nn.Module):
        target = target.module
    state = {_MODULE_PREFIX.sub("", name): value for name, value in loaded["model"].items()}
    result = target.load_state_dict(state, strict=not (backbone_only or codec_only))
    # a partial load leaves out exactly the other part
    outside = [
        name for name in result.missing_keys if bool(_COMPRESSOR.search(name)) != backbone_only
    ]
    assert not result.unexpected_keys, result.unexpected_keys
    assert not outside, outside


# --- the optimizer state -------------------------------------------------------------------------


def _sub_optimizers(optimizer: Any) -> list[Any]:
    return list(getattr(optimizer, "chained_optimizers", None) or [optimizer])


def _inner_optimizers(optimizer: Any) -> list[Any]:
    return [
        sub_optimizer.optimizer
        for sub_optimizer in _sub_optimizers(optimizer)
        if getattr(sub_optimizer, "optimizer", None) is not None
    ]


@contextmanager
def _without_empty_sub_optimizers(optimizer: Any) -> Iterator[None]:
    # Megatron's state dict cannot handle a chained sub-optimizer that holds no parameter on this
    # rank (a dev mode run's experts, with the backbone frozen).
    chained = getattr(optimizer, "chained_optimizers", None)
    if chained is None:
        yield
        return
    original = list(chained)
    optimizer.chained_optimizers = [
        sub for sub in original if sub is not None and getattr(sub, "optimizer", None) is not None
    ]
    try:
        yield
    finally:
        optimizer.chained_optimizers = original


def _sharded(optimizer: Any) -> bool:
    return any(
        hasattr(sub, "load_parameter_state_from_dp_reshardable")
        for sub in _sub_optimizers(optimizer)
    )


def _plain(value: Any) -> Any:
    return value.item() if isinstance(value, torch.Tensor) and value.numel() == 1 else value


def _group_records(optimizer: Any) -> list[list[dict[str, Any]]]:
    return [
        [
            {
                "settings": {
                    name: _plain(value)
                    for name, value in group.items()
                    if name not in _GROUP_STATE_FIELDS
                },
                "parameters": len(group["params"]),
                "elements": sum(int(parameter.numel()) for parameter in group["params"]),
                "step": _plain(group.get("step")),
            }
            for group in inner.param_groups
        ]
        for inner in _inner_optimizers(optimizer)
    ]


def _restore_groups(optimizer: Any, live: list, saved: list) -> None:
    # Megatron's optimizer load pairs groups by `param_group_identifier_keys` alone. The backbone,
    # router and codec groups share those keys, so every one took the last saved group's `max_lr`,
    # `min_lr` and other settings. Put each group's own settings and saved step back. The moments
    # and fp32 masters load by bucket position and are unaffected.
    for inner, live_groups, saved_groups in zip(
        _inner_optimizers(optimizer), live, saved, strict=True
    ):
        for group, live_group, saved_group in zip(
            inner.param_groups, live_groups, saved_groups, strict=True
        ):
            for name in [
                name
                for name in group
                if name not in live_group["settings"] and name not in _GROUP_STATE_FIELDS
            ]:
                del group[name]  # a field this group never had, from the group it was paired with
            group.update(live_group["settings"])
            if saved_group["step"] is None:
                group.pop("step", None)
            else:
                group["step"] = saved_group["step"]
