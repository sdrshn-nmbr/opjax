"""Parse axport Markdown session exports into Trajectory objects."""

from __future__ import annotations

import re
from pathlib import Path

from opjax.factory.schema import Message, Trajectory

_HEADER_RE = re.compile(r"^### \[(\d+)\] role=(\w+)\s*$", re.MULTILINE)
_UUID_RE = re.compile(
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
    re.IGNORECASE,
)

# Map axport roles → OpenAI-style chat roles for Tinker.
_ROLE_MAP = {
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
    "tool_result": "tool",
    "tool_call": "assistant",  # keep as assistant text for SFT v1
    "function": "assistant",
}


def parse_axport_markdown(
    text: str,
    *,
    trajectory_id: str,
    project: str | None = None,
    source: str = "axport-md",
    drop_system: bool = True,
    max_message_chars: int = 12000,
) -> Trajectory | None:
    """Parse `### [n] role=...` markdown into a Trajectory.

    Returns None if fewer than one user and one assistant message remain.
    """
    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        return None

    messages: list[Message] = []
    for i, match in enumerate(matches):
        role_raw = match.group(2).lower()
        role = _ROLE_MAP.get(role_raw)
        if role is None:
            continue
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if not content:
            continue
        if drop_system and role == "system":
            continue
        if len(content) > max_message_chars:
            content = content[:max_message_chars] + "\n…[truncated]"
        messages.append(Message(role=role, content=content))  # type: ignore[arg-type]

    # Collapse consecutive same-role assistant fragments
    collapsed: list[Message] = []
    for msg in messages:
        if (
            collapsed
            and collapsed[-1].role == msg.role
            and msg.role in {"assistant", "user"}
        ):
            collapsed[-1] = Message(
                role=msg.role,
                content=collapsed[-1].content + "\n\n" + msg.content,
            )
        else:
            collapsed.append(msg)

    if not any(m.role == "user" for m in collapsed):
        return None
    if not any(m.role == "assistant" for m in collapsed):
        return None

    return Trajectory(
        messages=collapsed,
        trajectory_id=trajectory_id,
        project=project,
        outcome="unknown",
        source=source,
        metadata={"format": "axport_markdown"},
    )


def uuid_from_path(path: Path) -> str | None:
    m = _UUID_RE.search(path.name)
    return m.group(1).lower() if m else None


def index_markdown_by_uuid(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in root.rglob("*.md"):
        uid = uuid_from_path(p)
        if uid:
            out[uid] = p
    return out
