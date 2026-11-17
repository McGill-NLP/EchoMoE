#!/bin/bash
# Select an environment (source it):  . scripts/env.sh megatron|vllm|vllm029
#
#   megatron   .venv-megatron   training, offline fit, export
#   vllm       .venv-vllm       benchmark evaluation, activation capture, teacher targets
#   vllm029    .venv-vllm029    throughput benchmarks
#
# Optional:
#   ECHOMOE_SITE_ENV         a file sourced first (the cluster's CUDA toolchain and compilers)
#   ECHOMOE_CUDA_COMPAT_DIR  NVIDIA's forward-compatible CUDA 13 user-space driver, for the vllm
#                            and vllm029 envs on a kernel driver older than CUDA 13
#   .env                     WANDB_API_KEY and HF_TOKEN, exported to child processes

ECHOMOE_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export ECHOMOE_REPO
case "${1:-}" in
    megatron | vllm | vllm029) ;;
    *)
        echo "usage: . scripts/env.sh megatron|vllm|vllm029" >&2
        return 2
        ;;
esac
_venv="$ECHOMOE_REPO/.venv-$1"

if [ -n "${ECHOMOE_SITE_ENV:-}" ]; then
    # shellcheck disable=SC1090
    . "$ECHOMOE_SITE_ENV"
fi
if [ -f "$ECHOMOE_REPO/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$ECHOMOE_REPO/.env"
    set +a
fi

# Transformer Engine would bind the toolchain's older libcudnn.so.9, but torch was built against
# the venv's. Preload the venv's so every cuDNN sub-library resolves to one version.
_cudnn="$_venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
if [ -f "$_cudnn/libcudnn.so.9" ]; then
    for _lib in libcudnn.so.9 libcudnn_graph.so.9 libcudnn_engines_precompiled.so.9 \
        libcudnn_engines_runtime_compiled.so.9 libcudnn_heuristic.so.9 libcudnn_ops.so.9; do
        LD_PRELOAD="$_cudnn/$_lib${LD_PRELOAD:+:$LD_PRELOAD}"
    done
    export LD_PRELOAD
fi

# shellcheck disable=SC1091
. "$_venv/bin/activate"

case "$1" in
    vllm)
        # Each vLLM environment loads only its own plugin of the two the package declares.
        export VLLM_PLUGINS=echomoe
        ;;
    vllm029)
        export VLLM_PLUGINS=echomoe_throughput
        # DeepEP and FlashInfer compile kernels at first use with $CUDA_HOME/bin/nvcc, the venv's
        # CUDA 13.0. Its CCCL headers replace any include path the site set, which a CUDA 12
        # toolkit's headers on it would break.
        export CUDA_HOME="$_venv/lib/python3.12/site-packages/nvidia/cu13"
        export CPATH="$CUDA_HOME/include/cccl"
        ;;
esac
if [ "$1" != megatron ]; then
    # Both vLLM stacks run the CUDA 13 runtime that the venv carries. On an older kernel driver the
    # compat driver must come first so its libcuda.so.1 wins.
    for _dir in "${ECHOMOE_CUDA_COMPAT_DIR:-}" "$_venv/lib/python3.12/site-packages/nvidia/cu13/lib"; do
        if [ -n "$_dir" ] && [ -d "$_dir" ]; then
            LD_LIBRARY_PATH="$_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        fi
    done
fi
# Run against the venv's own CUDA libraries (cuDNN, NCCL, cuBLAS), which match its torch build.
for _dir in "$VIRTUAL_ENV"/lib/python3.*/site-packages/nvidia/*/lib; do
    if [ -d "$_dir" ]; then
        LD_LIBRARY_PATH="$_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
done
export LD_LIBRARY_PATH

# DeepEP compiles elastic kernels at first use. Keep them off the ephemeral $HOME, one per host.
export EP_JIT_CACHE_DIR="${EP_JIT_CACHE_DIR:-$ECHOMOE_REPO/outputs/deepep-jit-cache/$(hostname)}"
case ":${PYTHONPATH:-}:" in
    *":$ECHOMOE_REPO:"*) ;;
    *) export PYTHONPATH="$ECHOMOE_REPO${PYTHONPATH:+:$PYTHONPATH}" ;;
esac
if [ "$1" = vllm ]; then
    # LiveCodeBench's grader and prompts are imported from its checkout (scripts/install.sh),
    # not installed.
    export PYTHONPATH="$ECHOMOE_REPO/third-party/LiveCodeBench:$PYTHONPATH"
fi
unset _venv _cudnn _lib _dir
