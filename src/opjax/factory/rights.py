"""Rights manifest loading and upload approval checks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RightsDecision:
    approved: bool
    provider_ok: bool
    slice_id: str | None
    reasons: list[str]
    path: str


_APPROVE_RE = re.compile(
    r"\|\s*Owner\s*\|\s*([^|]*)\|\s*([^|]*)\|\s*(APPROVE|REJECT)\s*\|",
    re.IGNORECASE,
)
_SLICE_ID_RE = re.compile(r"\|\s*Slice ID\s*\|\s*`?([^`|]+)`?\s*\|", re.IGNORECASE)


def check_provider(path: str | Path, provider: str) -> RightsDecision:
    """Ensure manifest Owner=APPROVE, slice id filled, and provider allowed."""
    p = Path(path)
    if not p.exists():
        return RightsDecision(
            approved=False,
            provider_ok=False,
            slice_id=None,
            reasons=[f"manifest missing: {p}"],
            path=str(p),
        )

    text = p.read_text(encoding="utf-8")
    reasons: list[str] = []

    slice_m = _SLICE_ID_RE.search(text)
    slice_id = slice_m.group(1).strip() if slice_m else None
    slice_ok = bool(slice_id) and "YYYYMMDD" not in slice_id and "<" not in slice_id
    if not slice_ok:
        reasons.append("slice id not filled")

    approve_m = _APPROVE_RE.search(text)
    owner_approve = bool(approve_m and approve_m.group(3).upper() == "APPROVE")
    if not owner_approve:
        reasons.append("owner decision is not APPROVE")

    provider_l = provider.strip().lower()
    provider_ok = False

    if re.search(rf"(?im)^-\s*\[x\].*\b{re.escape(provider_l)}\b", text):
        provider_ok = True

    for line in text.splitlines():
        if provider_l not in line.lower():
            continue
        if re.search(r"\b(Yes|Y|APPROVED)\b", line, re.IGNORECASE):
            provider_ok = True
            break
        if re.search(r"\b(No|N|REJECTED)\b", line, re.IGNORECASE):
            provider_ok = False

    if not provider_ok and owner_approve:
        if re.search(
            rf"\|\s*Intended provider\(s\)\s*\|\s*([^|]*\b{re.escape(provider_l)}\b[^|]*)\|",
            text,
            re.IGNORECASE,
        ):
            provider_ok = True

    if not provider_ok:
        reasons.append(f"provider '{provider}' not approved in manifest")

    approved = owner_approve and provider_ok and slice_ok
    return RightsDecision(
        approved=approved,
        provider_ok=provider_ok,
        slice_id=slice_id,
        reasons=[] if approved else reasons,
        path=str(p),
    )


def is_public_fixture_path(path: str | Path) -> bool:
    s = str(path).replace("\\", "/")
    markers = (
        "/tests/factory/fixtures/",
        "/data/factory/smoke/",
        "example_data/conversations.jsonl",
        "synthetic_coding_sessions.jsonl",
    )
    return any(m in f"/{s}" or s.endswith(m.lstrip("/")) for m in markers)
