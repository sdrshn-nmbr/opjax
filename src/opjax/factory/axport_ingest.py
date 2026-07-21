"""Ingest adapters for axport / session dumps / fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from opjax.factory.schema import Trajectory


def iter_jsonl_paths(path: str | Path) -> Iterator[Path]:
    p = Path(path)
    if p.is_file():
        yield p
        return
    if not p.exists():
        return
    yield from sorted(p.rglob("*.jsonl"))


def load_trajectories(path: str | Path) -> list[Trajectory]:
    """Load trajectories from a JSONL file or directory of JSONL files.

    Accepted line shapes:
    - ``{"messages": [...], "trajectory_id"?: ...}``
    - ``{"conversation": {"messages": [...]}, ...}`` (axport-ish wrapper)
    """
    out: list[Trajectory] = []
    for fp in iter_jsonl_paths(path):
        for i, line in enumerate(fp.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            data = json.loads(line)
            if "conversation" in data and isinstance(data["conversation"], dict):
                inner = dict(data["conversation"])
                if "messages" not in inner and "messages" in data:
                    inner["messages"] = data["messages"]
                for k in ("trajectory_id", "project", "outcome", "source", "metadata"):
                    if k in data and k not in inner:
                        inner[k] = data[k]
                data = inner
            if "messages" not in data:
                raise ValueError(f"{fp}:{i}: missing messages")
            if not data.get("trajectory_id"):
                data["trajectory_id"] = f"{fp.stem}-{i}"
            if not data.get("source"):
                data["source"] = str(fp)
            out.append(Trajectory.from_dict(data))
    return out


def discover_axport_root(candidates: Iterable[str | Path] | None = None) -> Path | None:
    """Return first existing axport raw root, else None."""
    defaults = [
        Path("data/axport/raw"),
        Path("data/factory/axport/raw"),
        Path.home() / ".cache" / "axport" / "raw",
    ]
    for c in list(candidates or []) + defaults:
        p = Path(c)
        if p.exists():
            return p
    return None
