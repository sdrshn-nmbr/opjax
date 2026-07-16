"""Canary tokens for leak detection before/after trainer uploads."""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Canary:
    id: str
    token: str
    purpose: str


@dataclass
class CanarySet:
    canaries: list[Canary] = field(default_factory=list)

    def tokens(self) -> list[str]:
        return [c.token for c in self.canaries]

    def to_dict(self) -> dict:
        return {
            "canaries": [
                {"id": c.id, "token": c.token, "purpose": c.purpose} for c in self.canaries
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> CanarySet:
        return cls(
            canaries=[
                Canary(id=c["id"], token=c["token"], purpose=c["purpose"])
                for c in data.get("canaries", [])
            ]
        )


def make_canary_set(n: int = 3, *, seed_material: str | None = None) -> CanarySet:
    """Create high-entropy canary tokens unlikely to appear in natural text."""
    canaries: list[Canary] = []
    for i in range(n):
        raw = secrets.token_hex(16)
        if seed_material:
            raw = hashlib.sha256(f"{seed_material}:{i}:{raw}".encode()).hexdigest()[:32]
        token = f"OPJAX_CANARY_{i}_{raw}"
        canaries.append(
            Canary(id=f"canary-{i}", token=token, purpose="pre-upload-leak-probe")
        )
    return CanarySet(canaries=canaries)


def embed_canaries(text: str, canary_set: CanarySet) -> str:
    """Append canaries in a comment-like block so scrub/upload paths carry them."""
    block = "\n".join(f"# {c.token}" for c in canary_set.canaries)
    return text.rstrip() + "\n\n" + block + "\n"


def find_canaries(text: str, canary_set: CanarySet) -> list[Canary]:
    return [c for c in canary_set.canaries if c.token in text]


def save_canary_set(path: Path | str, canary_set: CanarySet) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(canary_set.to_dict(), indent=2) + "\n", encoding="utf-8")


def load_canary_set(path: Path | str) -> CanarySet:
    return CanarySet.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
