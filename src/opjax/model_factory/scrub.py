"""Secret / credential scrubbing for training uploads.

Regex scrubbing is necessary but not sufficient — Stage-0 also requires a
signed rights manifest and canary leak tests before any managed-trainer upload.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Patterns intentionally conservative: prefer false positives over leaking keys.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "aws_access_key_id",
        re.compile(r"(?<![A-Z0-9])AKIA[0-9A-Z]{16}(?![A-Z0-9])"),
    ),
    (
        "aws_secret_access_key",
        re.compile(
            r"(?i)(?:aws_secret_access_key|secret_access_key)\s*[:=]\s*['\"]?"
            r"([A-Za-z0-9/+=]{40})"
        ),
    ),
    (
        "github_pat",
        re.compile(r"ghp_[A-Za-z0-9]{36}"),
    ),
    (
        "github_fine_grained",
        re.compile(r"github_pat_[A-Za-z0-9_]{22,}"),
    ),
    (
        "slack_token",
        re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    ),
    (
        "openai_key",
        re.compile(r"sk-[A-Za-z0-9]{20,}"),
    ),
    (
        "anthropic_key",
        re.compile(r"sk-ant-[A-Za-z0-9\-_]{20,}"),
    ),
    (
        "hf_token",
        re.compile(r"hf_[A-Za-z0-9]{20,}"),
    ),
    (
        "tinker_key",
        re.compile(r"tml-[A-Za-z0-9]{20,}"),
    ),
    (
        "generic_bearer",
        re.compile(r"(?i)(authorization|bearer)\s*[:=]\s*['\"]?Bearer\s+[A-Za-z0-9\-._~+/]+=*"),
    ),
    (
        "private_key_block",
        re.compile(
            r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----[\s\S]*?"
            r"-----END (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"
        ),
    ),
    (
        "dotenv_assignment",
        re.compile(
            # All-caps env assignments only (avoids swallowing prose like "token=hf_…").
            r"(?m)^(?:export\s+)?"
            r"[A-Z][A-Z0-9_]*"
            r"(?:API_KEY|SECRET|TOKEN|PASSWORD|PRIVATE_KEY|ACCESS_KEY)"
            r"[A-Z0-9_]*\s*=\s*\S+.*$"
        ),
    ),
]

_REDACTION = "[REDACTED:{kind}]"


@dataclass
class ScrubHit:
    kind: str
    start: int
    end: int


@dataclass
class ScrubResult:
    text: str
    hits: list[ScrubHit] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.hits


def scrub_text(text: str) -> ScrubResult:
    """Return text with known secret patterns replaced by redaction markers."""
    hits: list[ScrubHit] = []
    out = text
    # Apply longer / block patterns first by sorting on pattern length desc via name order
    # that puts private_key_block early.
    ordered = sorted(_PATTERNS, key=lambda kv: -len(kv[1].pattern))
    for kind, pattern in ordered:
        for match in list(pattern.finditer(out)):
            hits.append(ScrubHit(kind=kind, start=match.start(), end=match.end()))
        out = pattern.sub(_REDACTION.format(kind=kind), out)
    return ScrubResult(text=out, hits=hits)


def scrub_file(path: str) -> ScrubResult:
    with open(path, encoding="utf-8", errors="replace") as f:
        return scrub_text(f.read())
