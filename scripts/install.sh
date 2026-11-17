#!/bin/bash
# Build .venv-megatron, .venv-vllm or .venv-vllm029 from its lock file, or the throughput runs'
# capped-network NCCL (optional):
#
#   scripts/install.sh megatron|vllm|vllm029|nccl
#
# Needs uv, curl, and CUDA 12.8's nvcc and a C++ compiler on PATH (Transformer Engine and the other
# native packages build against the venv's torch). Re-running converges the venv onto the lock.
set -euo pipefail

which="${1:?usage: scripts/install.sh megatron|vllm|vllm029|nccl}"
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

# The prebuilt binaries are assets of the GitHub release. A file already in place is used as it is.
# Otherwise it is downloaded and checked against its sha256, and if the download fails, the binaries
# that have a source build below are built here.
release=https://github.com/McGill-NLP/EchoMoE/releases/download/v1.0.0

# fetch <path> <sha256>: the asset named as <path>'s file, downloaded to <path> unless it is there
fetch() {
    local path=$1 digest=$2
    [ -f "$path" ] && return 0
    if ! curl -fsSL "$release/$(basename "$path")" -o "$path.part"; then
        rm -f "$path.part"
        echo "$(basename "$path"): not downloaded from $release" >&2
        return 1
    fi
    if ! echo "$digest  $path.part" | sha256sum --check --quiet; then
        rm -f "$path.part"
        echo "$(basename "$path"): the download does not match its sha256" >&2
        exit 1
    fi
    mv "$path.part" "$path"
}

# The throughput CLI's --capped: NCCL 2.31.2, the vllm029 venv's version, with a patch that sets
# the injection rate of each InfiniBand connection (NCCL_IB_STATIC_RATE) and of each GIN context
# when it is created (NCCL_GIN_STATIC_RATE), for H100 only. CUDA_HOME is CUDA 12.8's toolkit: the
# venv's nvcc 13.0 does not compile NCCL's FP8 kernels against the venv's CCCL headers. That
# toolkit has no static CUDA runtime, so the library links libcudart.so.12. Build on a GPU node: a
# build on a host without one linked but hung GIN at run time. The release's tarball holds the
# build the capped rows ran.
if [ "$which" = nccl ]; then
    tarball=third-party/nccl/nccl-2.31.2-static-rate-cuda12.8-sm90.tar.gz
    rm -rf third-party/nccl/build
    if fetch "$tarball" 88d468e3163890667730683e3db1de483bf53ffb4e8b8093d6c3586fab3f24a5; then
        mkdir -p third-party/nccl/build
        tar -xzf "$tarball" -C third-party/nccl/build
    else
        nccl=third-party/nccl/nccl
        [ -d "$nccl/.git" ] || git clone --quiet --depth 1 --branch v2.31.2-1 https://github.com/NVIDIA/nccl "$nccl"
        git -C "$nccl" checkout --quiet --force v2.31.2-1
        git -C "$nccl" apply "$PWD/third-party/nccl/nccl_static_rate.patch"
        env -u CPATH make -C "$nccl" -j"$(nproc)" src.build BUILDDIR="$PWD/third-party/nccl/build" \
            CUDA_HOME="$CUDA_HOME" CUDA_LIB="$CUDA_HOME/lib" CUDARTLIB=cudart \
            CUDA_INC="$CUDA_HOME/targets/x86_64-linux/include" \
            NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
    fi
    echo "installed third-party/nccl/build/lib/libnccl.so.2"
    exit 0
fi

lock="requirements-$which.lock"
py=".venv-$which/bin/python"
[ -x "$py" ] || uv venv ".venv-$which" --python 3.12

# The throughput environment: vLLM 0.29 on CUDA 13 and DeepEP v2. It brings its own CUDA 13.0
# compiler. A source build of DeepEP needs a C++20 compiler that CUDA 13 supports, with Linux
# headers new enough (5.6) to define SYS_pidfd_getfd.
if [ "$which" = vllm029 ]; then
    # cuobjdump was in the measured environment, and DeepEP's kernel compiler dumps SASS with it.
    uv pip install --python "$py" --overrides requirements-vllm029.overrides -r "$lock" \
        nvidia-cuda-cuobjdump==13.4.49

    # DeepEP at the measured commit, with the patch that connects hybrid mode only to rail peers
    # (the patched tree gives the version 2.1.0+local): the release's wheel, or built here for H100
    # with the venv's CUDA 13.0, NCCL and NVSHMEM. The wheels ship libcudart.so.13 without the
    # libcudart.so linker name, so -lcudart would otherwise find a CUDA 12 library, and DeepEP links
    # -l:libnccl.so.2 from the library path. A host without a GPU driver must put a libcuda.so on
    # LIBRARY_PATH.
    nvidia="$PWD/.venv-vllm029/lib/python3.12/site-packages/nvidia"
    ln -sfn libcudart.so.13 "$nvidia/cu13/lib/libcudart.so"
    wheel=third-party/deepep/deep_ep-2.1.0+local-cp312-cp312-linux_x86_64.whl
    if ! fetch "$wheel" 37e007e3cfc305db32f32d3e47caf21003d34eb75639ad5cc68ce6eaff78b4c9; then
        deepep=third-party/deepep/DeepEP
        [ -d "$deepep/.git" ] || git clone --quiet https://github.com/deepseek-ai/DeepEP "$deepep"
        git -C "$deepep" checkout --quiet --force 01dc3aaac82068020353dce2c302e38153c0bfaa
        git -C "$deepep" submodule update --quiet --init --recursive
        git -C "$deepep" apply "$PWD/third-party/deepep/deepep_hybrid_rail_stride.patch"
        # DeepEP writes EP_JIT_CACHE_DIR (set by scripts/env.sh) into the wheel when it is set
        env -u EP_JIT_CACHE_DIR CUDA_HOME="$nvidia/cu13" CPATH="$nvidia/cu13/include/cccl" \
            TORCH_CUDA_ARCH_LIST=9.0 LIBRARY_PATH="$nvidia/nccl/lib${LIBRARY_PATH:+:$LIBRARY_PATH}" \
            uv build --wheel --no-build-isolation --python "$py" --out-dir third-party/deepep "./$deepep"
    fi
    uv pip install --python "$py" --no-deps --reinstall "$wheel"

    # The throughput CLI's --fp8 combines through a copy of the package whose combine crosses nodes
    # in FP8. DeepEP compiles its kernels from these headers at first use.
    fp8c=".venv-$which/deepep-fp8c"
    rm -rf "$fp8c"
    mkdir -p "$fp8c"
    cp -r ".venv-$which/lib/python3.12/site-packages/deep_ep" "$fp8c/"
    patch --quiet -d "$fp8c" -p1 < third-party/deepep/deepep_fp8_scaleout_combine.patch

    # The serving client's dataset loader (vllm bench serve) needs pandas, which the lock lacks: the
    # versions of requirements-vllm.lock, in a directory only the client puts on its path.
    uv pip install --python "$py" --no-deps --target ".venv-$which/serving-client" \
        pandas==3.0.3 python-dateutil==2.9.0.post0

    # The package itself, for the vLLM plugin entry point.
    uv pip install --python "$py" --no-deps -e .
    echo "built .venv-$which"
    exit 0
fi

# torch==2.11.0+cu128 is only on the PyTorch index, and `uv pip` ignores [tool.uv.sources].
index=(--extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match)

# 1. The packages built without isolation import torch at build time, so torch and the build tools
#    go in first. hatchling builds third-party/fa3-te-compat.
mapfile -t prereqs < <(grep -E '^(torch|setuptools|packaging|ninja|pybind11|wheel|numpy|cmake)==' "$lock")
uv pip install --python "$py" "${index[@]}" "${prereqs[@]}" hatchling

# The lock installs DeepEP from the release's wheel (DeepEP 2.1.0 at the same commit, built for
# this venv), which has no source build here. Megatron's Mamba layers (Nemotron-H) need mamba-ssm
# and causal-conv1d, which have no wheel for torch 2.11. Unless the release's wheels are used, each
# pinned sdist (with its sha256, then the wheel's) is built against the venv's torch for H100
# alone: its setup.py keeps only sm_90's gencode pair.
if [ "$which" = megatron ]; then
    fetch third-party/deepep/deep_ep-2.1.0+01dc3aa-cp312-cp312-linux_x86_64.whl \
        309cdd9c981b6590a1595c4c01f457c62efe78cf1a56606f5b10aecdfd96ff61
    pypi=https://files.pythonhosted.org/packages
    for sdist in \
        "$pypi/64/cb/104778c728dc3d5ea3bf65a484e3a4cdbe894bdaa2586320e2f61d007b8c/causal_conv1d-1.5.3.post1.tar.gz aba1b717484472d0b2f2e40520a1c03f35fe5155555bd753d1c324afc56ba468 aeccda7309f9c32edfbb1183d498afe88062d02e1a6521361450d1f036dba9d0" \
        "$pypi/34/67/ec89aa703da194a813e35d2ea2de8f74a7ce6991a120a29f3a0c5e30d4b9/mamba_ssm-2.3.1.tar.gz 4d529477ad94753962216d583fc8f1c127c717b7d7c875d6bbb9376366d0d761 7247a701cdb3f47cb8c71ae2e554025b0c6fca85d8cf542d159cd98d8ee83132"; do
        read -r url digest wheel_digest <<< "$sdist"
        name="$(basename "$url" .tar.gz)"
        fetch "third-party/mamba/$name-cp312-cp312-linux_x86_64.whl" "$wheel_digest" && continue
        work="$(mktemp -d)"
        curl -fsSL "$url" -o "$work/$name.tar.gz"
        echo "$digest  $work/$name.tar.gz" | sha256sum --check --quiet
        tar -xzf "$work/$name.tar.gz" -C "$work"
        "$py" - "$work/$name/setup.py" <<'PY'
import re
import sys

pair = re.compile(r'( *)cc_flag\.append\("-gencode"\)\n *cc_flag\.append\("arch=compute_(\d+),code=sm_\d+"\)\n')
text = open(sys.argv[1]).read()
open(sys.argv[1], "w").write(pair.sub(lambda m: m[0] if m[2] == "90" else f"{m[1]}pass\n", text))
PY
        CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=8 \
            uv build --wheel --no-build-isolation --python "$py" --out-dir third-party/mamba "$work/$name"
        rm -rf "$work"
    done
fi

# 2. Install the lock. `uv pip sync` ignores no-build-isolation-package and has no per-package form
#    of the flag, so install with isolation off for everything...
# The megatron lock was compiled with overrides, and `uv pip install -r` resolves it again.
overrides=()
[ "$which" = megatron ] && overrides=(--overrides requirements-megatron.overrides)
uv pip install --python "$py" "${index[@]}" "${overrides[@]}" --no-build-isolation -r "$lock"
# 3. ...then sync, which removes what the lock does not list (hatchling).
uv pip sync --python "$py" "${index[@]}" --no-build-isolation "$lock"

# The pinned FlashAttention-3 wheel needs glibc 2.32 or newer. Older systems pass one built there,
# such as the release's optional flash_attn_3 wheel (built on glibc 2.31).
if [ -n "${ECHOMOE_FA3_WHEEL:-}" ]; then
    uv pip install --python "$py" --no-deps --reinstall "$ECHOMOE_FA3_WHEEL"
fi

# The CUDA 12 and CUDA 13 Transformer Engine cores ship the same library file. If the venv ever
# had the CUDA 13 one, the sync removed that file with it.
if [ "$which" = megatron ] &&
    [ ! -f ".venv-megatron/lib/python3.12/site-packages/transformer_engine/wheel_lib/libtransformer_engine.so" ]; then
    uv pip install --python "$py" "${index[@]}" --no-deps --reinstall "$(grep -E '^transformer-engine-cu12==' "$lock")"
fi

# The package itself, for the vLLM plugin entry point.
uv pip install --python "$py" --no-deps -e .

# Evaluation runs IFBench's grader and LiveCodeBench's from pinned checkouts.
if [ "$which" = vllm ]; then
    for spec in IFBench=https://github.com/allenai/IFBench@db69a6f05689830b0068b8f1529ebcfd2f3b164c \
        LiveCodeBench=https://github.com/LiveCodeBench/LiveCodeBench@28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24; do
        name="${spec%%=*}" url="${spec#*=}"
        dir="third-party/$name"
        [ -d "$dir/.git" ] || git clone --quiet "${url%@*}" "$dir"
        git -C "$dir" checkout --quiet "${url##*@}"
    done
fi
echo "built .venv-$which"
