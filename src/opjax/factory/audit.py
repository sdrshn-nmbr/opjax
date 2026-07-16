"""Dataset audit metrics (not character-count vanity)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def estimate_assistant_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token). Not a billing meter."""
    return max(1, len(text) // 4) if text else 0


def audit_conversations_jsonl(path: str | Path) -> dict:
    path = Path(path)
    n = 0
    assistant_tokens = 0
    outcomes: Counter[str] = Counter()
    projects: Counter[str] = Counter()
    dup_hashes: Counter[str] = Counter()
    ids: list[str] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        n += 1
        tid = str(data.get("trajectory_id") or "")
        if tid:
            ids.append(tid)
        outcomes[str(data.get("outcome") or "unknown")] += 1
        projects[str(data.get("project") or "unknown")] += 1
        msgs = data.get("messages") or []
        # duplicate key: concatenation of assistant contents
        asst = "\n".join(
            str(m.get("content", "")) for m in msgs if m.get("role") == "assistant"
        )
        assistant_tokens += estimate_assistant_tokens(asst)
        dup_hashes[str(hash(asst))] += 1

    duplicate_groups = sum(1 for c in dup_hashes.values() if c > 1)
    duplicate_rows = sum(c - 1 for c in dup_hashes.values() if c > 1)
    id_dupes = len(ids) - len(set(ids))

    return {
        "path": str(path),
        "num_trajectories": n,
        "assistant_token_estimate": assistant_tokens,
        "outcomes": dict(outcomes),
        "projects": dict(projects),
        "duplicate_assistant_groups": duplicate_groups,
        "duplicate_assistant_extra_rows": duplicate_rows,
        "duplicate_trajectory_ids": id_dupes,
    }
