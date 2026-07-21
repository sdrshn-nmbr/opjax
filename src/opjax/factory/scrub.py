"""Secret scrubbers and canary helpers."""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

# High-confidence secret patterns. Prefer false positives over misses.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "aws_access_key",
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    ),
    (
        "aws_secret_key",
        re.compile(
            r"(?i)(aws_secret_access_key|secret_access_key)\s*[=:]\s*['\"]?([A-Za-z0-9/+=]{40})['\"]?"
        ),
    ),
    (
        "openai_sk",
        re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    ),
    (
        "anthropic_key",
        re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
    ),
    (
        "hf_token",
        re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    ),
    (
        "github_pat",
        re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    ),
    (
        "github_fine_grained",
        re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    ),
    (
        "slack_token",
        re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    ),
    (
        "pem_block",
        re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----"
        ),
    ),
    (
        "bearer_header",
        re.compile(r"(?i)(authorization\s*:\s*bearer\s+)([A-Za-z0-9._\-+=/]{20,})"),
    ),
    (
        "env_secret_line",
        re.compile(
            r"(?im)^(\s*(?:TINKER_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY|HF_TOKEN|"
            r"HUGGINGFACE_TOKEN|AWS_SECRET_ACCESS_KEY|PRIME_API_KEY|WANDB_API_KEY|"
            r"TOGETHER_API_KEY|GITHUB_PAT|MODAL_TOKEN_SECRET)\s*=\s*)(.+)$"
        ),
    ),
    (
        # Inline KEY=value (e.g. inside chat JSON content, not start-of-line).
        "env_secret_inline",
        re.compile(
            r"(?i)\b((?:TINKER_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY|HF_TOKEN|"
            r"HUGGINGFACE_TOKEN|AWS_SECRET_ACCESS_KEY|PRIME_API_KEY|WANDB_API_KEY|"
            r"TOGETHER_API_KEY|GITHUB_PAT|MODAL_TOKEN_SECRET)\s*=\s*)([^\s\"']+)"
        ),
    ),
]

_REDACTION = "[REDACTED_{label}]"


@dataclass(frozen=True)
class ScrubHit:
    label: str
    count: int


@dataclass
class ScrubResult:
    text: str
    hits: list[ScrubHit]
    substitutions: int


def scrub_text(text: str) -> ScrubResult:
    """Scrub secrets from a string. Returns cleaned text + hit stats."""
    out = text
    hits: list[ScrubHit] = []
    substitutions = 0

    for label, pattern in _PATTERNS:
        keep_prefix = label in {
            "env_secret_line",
            "env_secret_inline",
            "bearer_header",
            "aws_secret_key",
        }

        def _sub(match: re.Match[str], _label: str = label, _keep: bool = keep_prefix) -> str:
            if _keep and match.lastindex and match.lastindex >= 1:
                return f"{match.group(1)}{_REDACTION.format(label=_label.upper())}"
            return _REDACTION.format(label=_label.upper())

        out, n = pattern.subn(_sub, out)
        if n:
            hits.append(ScrubHit(label=label, count=n))
            substitutions += n

    return ScrubResult(text=out, hits=hits, substitutions=substitutions)


def scrub_messages(messages: list[dict]) -> tuple[list[dict], int]:
    """Scrub content fields in OpenAI-style messages."""
    total = 0
    cleaned: list[dict] = []
    for msg in messages:
        m = dict(msg)
        content = m.get("content")
        if isinstance(content, str):
            result = scrub_text(content)
            m["content"] = result.text
            total += result.substitutions
        cleaned.append(m)
    return cleaned, total


def make_canary(prefix: str = "CANARY_FACTORY") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def write_canaries(path: str | Path, n: int = 3) -> list[str]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canaries = [make_canary() for _ in range(n)]
    # Also plant a key-shaped canary for pattern coverage in tests (not a real secret).
    canaries.append(f"sk-ant-CANARY{uuid.uuid4().hex[:24]}")
    path.write_text("\n".join(canaries) + "\n", encoding="utf-8")
    return canaries


def load_canaries(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.exists():
        return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def find_canaries(text: str, canaries: list[str]) -> list[str]:
    return [c for c in canaries if c in text]


def find_canaries_in_file(path: str | Path, canaries: list[str]) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    return find_canaries(text, canaries)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
