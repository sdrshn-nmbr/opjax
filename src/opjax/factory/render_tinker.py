"""Emit Tinker cookbook conversation JSONL."""

from __future__ import annotations

import json
from pathlib import Path

from opjax.factory.schema import Trajectory
from opjax.factory.scrub import scrub_messages


def render_conversations_jsonl(
    trajectories: list[Trajectory],
    out_path: str | Path,
    *,
    scrub: bool = True,
) -> dict:
    """Write OpenAI-style conversations JSONL for FromConversationFileBuilder."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    scrub_subs = 0
    with out_path.open("w", encoding="utf-8") as f:
        for t in trajectories:
            messages = [m.to_openai() for m in t.messages]
            if scrub:
                messages, subs = scrub_messages(messages)
                scrub_subs += subs
            row = {"messages": messages}
            # Extra keys are OK for our audit; cookbook only requires messages.
            if t.trajectory_id:
                row["trajectory_id"] = t.trajectory_id
            if t.outcome:
                row["outcome"] = t.outcome
            if t.project:
                row["project"] = t.project
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return {
        "path": str(out_path),
        "num_conversations": n,
        "scrub_substitutions": scrub_subs,
    }


def validate_conversations_jsonl(path: str | Path) -> dict:
    """Validate each line has messages with roles."""
    path = Path(path)
    n = 0
    roles: dict[str, int] = {}
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        data = json.loads(line)
        if "messages" not in data:
            raise ValueError(f"line {i}: missing messages")
        for m in data["messages"]:
            if "role" not in m or "content" not in m:
                raise ValueError(f"line {i}: message missing role/content")
            roles[m["role"]] = roles.get(m["role"], 0) + 1
        n += 1
    return {"num_conversations": n, "role_counts": roles}
