"""Distill one compressed student. Dev mode (backbone frozen) or full-model adaptation is the
recipe's choice.

    torchrun --nnodes N --nproc-per-node 8 ... -m echomoe.training --config <recipe> \
        [--resume-from latest|<checkpoint>] [--stop-after-step K]
"""

import argparse
import os
import sys
import threading

# The CUDA allocator and Transformer Engine read these when torch loads: expandable segments, and
# no forced attention backend, so Transformer Engine picks one.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
for _name in ("NVTE_FUSED_ATTN", "NVTE_FLASH_ATTN", "NVTE_UNFUSED_ATTN"):
    os.environ.pop(_name, None)

import torch  # noqa: E402

from .config import load_config  # noqa: E402
from .loop import train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--resume-from", help='"latest" (the newest checkpoint of the run) or a checkpoint'
    )
    parser.add_argument("--stop-after-step", type=int, help="end after this update")
    args = parser.parse_args()
    train(
        load_config(
            args.config, resume_from=args.resume_from, stop_after_step=args.stop_after_step
        )
    )
    # Tear the process group down with a bounded wait and skip interpreter teardown: a completed
    # run otherwise held its GPUs until killed, or died in the NCCL watchdog after finishing.
    sys.stdout.flush()
    sys.stderr.flush()
    teardown = threading.Thread(target=torch.distributed.destroy_process_group, daemon=True)
    teardown.start()
    teardown.join(60)
    os._exit(0)


if __name__ == "__main__":
    main()
