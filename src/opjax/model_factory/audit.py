"""Axport / trajectory JSONL audit metrics (not character counts)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class AuditReport:
    path: str
    n_records: int = 0
    complete_trajectories: int = 0
    with_tool_use: int = 0
    assistant_messages: int = 0
    approx_assistant_tokens: int = 0
    outcome_counts: dict[str, int] = field(default_factory=dict)
    duplicate_hashes: int = 0
    recovery_segments: int = 0
    license_tags: dict[str, int] = field(default_factory=dict)
    missing_fields: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "n_records": self.n_records,
            "complete_trajectories": self.complete_trajectories,
            "with_tool_use": self.with_tool_use,
            "assistant_messages": self.assistant_messages,
            "approx_assistant_tokens": self.approx_assistant_tokens,
            "outcome_counts": self.outcome_counts,
            "duplicate_hashes": self.duplicate_hashes,
            "recovery_segments": self.recovery_segments,
            "license_tags": self.license_tags,
            "missing_fields": self.missing_fields,
        }


def _approx_tokens(text: str) -> int:
    # Cheap proxy: whitespace tokens. Enough for audit triage, not billing.
    return len(text.split()) if text else 0


def _iter_messages(record: dict[str, Any]) -> Iterable[dict[str, Any]]:
    msgs = record.get("messages") or record.get("conversation") or []
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict):
                yield m


def _has_tool_use(record: dict[str, Any]) -> bool:
    if record.get("tool_calls") or record.get("tools"):
        return True
    for m in _iter_messages(record):
        if m.get("tool_calls") or m.get("role") in {"tool", "function"}:
            return True
        content = m.get("content")
        if isinstance(content, str) and ("tool_call" in content or "<tool" in content):
            return True
    return False


def _is_complete(record: dict[str, Any]) -> bool:
    if record.get("complete") is True:
        return True
    msgs = list(_iter_messages(record))
    if len(msgs) < 2:
        return False
    roles = {m.get("role") for m in msgs}
    return "user" in roles and "assistant" in roles


def _record_hash(record: dict[str, Any]) -> str:
    blob = json.dumps(record, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def audit_jsonl(path: Path | str) -> AuditReport:
    path = Path(path)
    report = AuditReport(path=str(path))
    seen: set[str] = set()
    outcomes: Counter[str] = Counter()
    licenses: Counter[str] = Counter()
    missing: Counter[str] = Counter()

    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            report.n_records += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                missing["json"] += 1
                continue
            if not isinstance(record, dict):
                missing["non_object"] += 1
                continue

            h = _record_hash(record)
            if h in seen:
                report.duplicate_hashes += 1
            else:
                seen.add(h)

            if _is_complete(record):
                report.complete_trajectories += 1
            else:
                missing["complete"] += 1

            if _has_tool_use(record):
                report.with_tool_use += 1

            for m in _iter_messages(record):
                if m.get("role") == "assistant":
                    report.assistant_messages += 1
                    content = m.get("content") or ""
                    if isinstance(content, list):
                        content = " ".join(
                            str(p.get("text", p)) if isinstance(p, dict) else str(p)
                            for p in content
                        )
                    report.approx_assistant_tokens += _approx_tokens(str(content))

            outcome = record.get("outcome") or record.get("label") or "unknown"
            outcomes[str(outcome)] += 1

            if record.get("recovery") or record.get("repaired") or record.get("is_recovery"):
                report.recovery_segments += 1

            lic = record.get("license") or record.get("license_tag")
            if lic:
                licenses[str(lic)] += 1
            else:
                missing["license"] += 1

    report.outcome_counts = dict(outcomes)
    report.license_tags = dict(licenses)
    report.missing_fields = dict(missing)
    return report
