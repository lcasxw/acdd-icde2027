from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np

from .types import SubjectEntry


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def config_hash(config: dict[str, Any]) -> str:
    return stable_json_hash(config)


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_hash(root: str | Path) -> str:
    root = Path(root)
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    if not files:
        raise ValueError(f"no Python source files found under {root}")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def split_hash(*groups: Iterable[SubjectEntry]) -> str:
    payload = [sorted(entry.subject_key for entry in group) for group in groups]
    return stable_json_hash(payload)


def capture_rng_state(torch_module: Any, loader_generator: Any | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch_module.get_rng_state(),
    }
    if torch_module.cuda.is_available():
        state["torch_cuda"] = torch_module.cuda.get_rng_state_all()
    if loader_generator is not None:
        state["loader_generator"] = loader_generator.get_state()
    return state


def restore_rng_state(
    state: dict[str, Any], torch_module: Any, loader_generator: Any | None = None
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_module.set_rng_state(state["torch_cpu"].cpu())
    if torch_module.cuda.is_available() and "torch_cuda" in state:
        torch_module.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])
    if loader_generator is not None and "loader_generator" in state:
        loader_generator.set_state(state["loader_generator"].cpu())
