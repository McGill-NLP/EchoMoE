"""Atomic writes, the `meta.json` provenance record, staged directories and run directories."""

from __future__ import annotations

import errno
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    Path(stream.name).replace(path)


def atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    atomic_write_bytes(
        path, json.dumps(document, allow_nan=False, indent=2, sort_keys=True).encode() + b"\n"
    )


def atomic_write_yaml(path: Path, document: Mapping[str, Any]) -> None:
    atomic_write_bytes(
        path,
        yaml.safe_dump(
            json.loads(json.dumps(document, allow_nan=False)), sort_keys=False
        ).encode(),
    )


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def git_state() -> dict[str, Any]:
    try:
        return {
            "commit": _git("rev-parse", "HEAD"),
            "dirty": len(_git("status", "--porcelain").splitlines()),
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None}


def write_meta(directory: str | Path, *, inputs: Mapping[str, Any], **fields: Any) -> None:
    # a data writer's format fields, already in meta.json, are kept
    path = Path(directory) / "meta.json"
    recorded = json.loads(path.read_text()) if path.is_file() else {}
    atomic_write_json(
        path,
        {
            **recorded,
            "command": sys.argv,
            "git": git_state(),
            "host": socket.gethostname(),
            "inputs": {name: str(value) for name, value in inputs.items()},
            **fields,
        },
    )


@contextmanager
def staged_directory(destination: str | Path) -> Iterator[Path]:
    """Build a directory beside `destination` and publish it atomically when the block succeeds."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(errno.EEXIST, "destination already exists", str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(dir=destination.parent, prefix=f".{destination.name}.staging-")
    )
    try:
        yield staging
        publish_staged_directory(staging, destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def publish_staged_directory(staging: str | Path, destination: str | Path) -> Path:
    """Rename a staged directory into place, never replacing an existing destination."""
    try:
        os.rename(staging, destination)
    except OSError as error:
        if error.errno in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(
                errno.EEXIST, "destination already exists", str(destination)
            ) from error
        raise
    return Path(destination)


@dataclass(frozen=True)
class RunDirectory:
    """One run: the resolved `config.yaml`, `meta.json` (provenance, then the results) and
    `metrics.jsonl`."""

    path: Path

    @classmethod
    def create(
        cls, path: str | Path, *, config: Mapping[str, Any], inputs: Mapping[str, Any], **meta: Any
    ) -> RunDirectory:
        directory = Path(path)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"{directory} already contains run state")
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_yaml(directory / "config.yaml", config)
        write_meta(directory, inputs=inputs, **meta)
        atomic_write_bytes(directory / "metrics.jsonl", b"")
        return cls(directory)

    @classmethod
    def resume(cls, path: str | Path, *, config: Mapping[str, Any]) -> RunDirectory:
        directory = Path(path)
        recorded = yaml.safe_load((directory / "config.yaml").read_text())
        if recorded != json.loads(json.dumps(config, allow_nan=False)):
            raise ValueError("resume config does not match config.yaml")
        return cls(directory)

    def write_results(self, results: Mapping[str, Any]) -> None:
        meta = json.loads((self.path / "meta.json").read_text())
        atomic_write_json(self.path / "meta.json", {**meta, "results": results})
