"""Canonical trajectory / message types for the factory pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    role: Role
    content: str
    name: str | None = None

    def to_openai(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            out["name"] = self.name
        return out


@dataclass
class Trajectory:
    """One multi-turn demonstration."""

    messages: list[Message]
    trajectory_id: str
    project: str | None = None
    outcome: str | None = None  # success | failure | recovery | unknown
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_conversation_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "messages": [m.to_openai() for m in self.messages],
            "trajectory_id": self.trajectory_id,
        }
        if self.project:
            row["project"] = self.project
        if self.outcome:
            row["outcome"] = self.outcome
        if self.source:
            row["source"] = self.source
        if self.metadata:
            row["metadata"] = self.metadata
        return row

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trajectory:
        if "messages" not in data:
            raise ValueError("trajectory requires 'messages'")
        messages = [
            Message(
                role=m["role"],  # type: ignore[arg-type]
                content=str(m.get("content", "")),
                name=m.get("name"),
            )
            for m in data["messages"]
        ]
        return cls(
            messages=messages,
            trajectory_id=str(data.get("trajectory_id") or data.get("id") or ""),
            project=data.get("project"),
            outcome=data.get("outcome"),
            source=data.get("source"),
            metadata=dict(data.get("metadata") or {}),
        )


def trajectory_to_jsonable(t: Trajectory) -> dict[str, Any]:
    return asdict(t)
