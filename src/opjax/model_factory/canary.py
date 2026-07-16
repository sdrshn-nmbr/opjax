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
    """Embed canaries without breaking JSONL training files.

    For ``.jsonl`` payloads, append valid conversation records that contain the
    tokens (so ``json.loads`` per line still works for Tinker). For other text,
    append a comment block.
    """
    stripped = text.rstrip()
    if _looks_like_jsonl(stripped):
        return embed_canaries_jsonl(stripped, canary_set)
    block = "\n".join(f"# {c.token}" for c in canary_set.canaries)
    return stripped + "\n\n" + block + "\n"


def embed_canaries_jsonl(text: str, canary_set: CanarySet) -> str:
    """Append canaries as valid JSONL conversation rows (train-parseable)."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    # Validate existing lines stay intact.
    for ln in lines:
        json.loads(ln)
    extra: list[str] = []
    for c in canary_set.canaries:
        extra.append(
            json.dumps(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Canary probe row for upload leak detection. "
                                f"Token id={c.id}."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"Echo this canary token exactly once: {c.token}",
                        },
                        {"role": "assistant", "content": c.token},
                    ],
                    "opjax_canary_id": c.id,
                    "license": "canary",
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines + extra) + "\n"


def _looks_like_jsonl(text: str) -> bool:
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        return s.startswith("{") or s.startswith("[")
    return False


def find_canaries(text: str, canary_set: CanarySet) -> list[Canary]:
    return [c for c in canary_set.canaries if c.token in text]


def save_canary_set(path: Path | str, canary_set: CanarySet) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(canary_set.to_dict(), indent=2) + "\n", encoding="utf-8")


def load_canary_set(path: Path | str) -> CanarySet:
    return CanarySet.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
