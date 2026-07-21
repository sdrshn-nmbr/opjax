"""Solution-channel scanners — Stage-6 anti-cheat before RL.

Looks for accidental leakage of sealed solutions into training artifacts,
git history mentions, caches, and package/registry path strings.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ScanFinding:
    channel: str
    severity: str  # info | warn | fail
    path: str
    detail: str


@dataclass
class ScanReport:
    ok: bool
    findings: list[ScanFinding] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "findings": [
                {
                    "channel": f.channel,
                    "severity": f.severity,
                    "path": f.path,
                    "detail": f.detail,
                }
                for f in self.findings
            ],
        }


def _sealed_solution_snippets(
    sealed_ids: list[str],
    tasks_dir: Path,
    repo_root: Path,
) -> dict[str, str]:
    """Map task_id → distinctive snippet from fixture solution (broken body)."""
    snippets: dict[str, str] = {}
    for tid in sealed_ids:
        task_path = tasks_dir / f"{tid}.json"
        if not task_path.exists():
            continue
        task = json.loads(task_path.read_text(encoding="utf-8"))
        fixture = Path(task["fixture_dir"])
        if not fixture.is_absolute():
            fixture = repo_root / fixture
        sol = fixture / "solution.py"
        if sol.exists():
            text = sol.read_text(encoding="utf-8").strip()
            # First def line is a weak fingerprint; prefer full body for JSONL scan.
            snippets[tid] = text
    return snippets


def scan_training_jsonl_for_sealed(
    jsonl_path: Path,
    sealed_snippets: dict[str, str],
    *,
    max_lines: int = 200_000,
) -> list[ScanFinding]:
    findings: list[ScanFinding] = []
    if not jsonl_path.exists():
        findings.append(
            ScanFinding(
                channel="train_jsonl",
                severity="info",
                path=str(jsonl_path),
                detail="path missing — skip (common when data/ is local-only)",
            )
        )
        return findings

    # Distinctive multi-line bodies only (avoid matching shared boilerplate).
    needles = {
        tid: body
        for tid, body in sealed_snippets.items()
        if len(body) >= 40 and "def " in body
    }
    with jsonl_path.open(encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= max_lines:
                break
            for tid, body in needles.items():
                if body in line:
                    findings.append(
                        ScanFinding(
                            channel="train_jsonl",
                            severity="fail",
                            path=f"{jsonl_path}:{i+1}",
                            detail=f"sealed fixture body for {tid} found in training JSONL",
                        )
                    )
    return findings


def scan_git_history_for_sealed_ids(
    sealed_ids: list[str],
    *,
    repo_root: Path,
    paths: list[str] | None = None,
) -> list[ScanFinding]:
    """Flag if sealed task IDs appear in recent commits under train data paths."""
    findings: list[ScanFinding] = []
    search_paths = paths or ["data/model-factory", "logs/model-factory"]
    for tid in sealed_ids:
        try:
            proc = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo_root),
                    "log",
                    "--all",
                    "--oneline",
                    "-S",
                    tid,
                    "--",
                    *search_paths,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            findings.append(
                ScanFinding(
                    channel="git_history",
                    severity="warn",
                    path=str(repo_root),
                    detail=f"git log -S failed: {exc}",
                )
            )
            return findings
        if proc.returncode == 0 and proc.stdout.strip():
            findings.append(
                ScanFinding(
                    channel="git_history",
                    severity="warn",
                    path=",".join(search_paths),
                    detail=f"{tid} appears in git history under train/log paths:\n{proc.stdout[:500]}",
                )
            )
    return findings


_CACHE_GLOBS = (
    "**/.pytest_cache/**",
    "**/__pycache__/**",
    "**/.cache/**",
    "**/wandb/**",
)


def scan_local_caches_for_solutions(
    repo_root: Path,
    sealed_snippets: dict[str, str],
) -> list[ScanFinding]:
    findings: list[ScanFinding] = []
    # Only scan small text-ish cache files; skip huge blobs.
    for pattern in ("**/*.json", "**/*.jsonl", "**/*.txt", "**/*.md"):
        for path in repo_root.glob(pattern):
            rel = str(path.relative_to(repo_root))
            if not any(
                part in rel
                for part in ("data/model-factory", "logs/model-factory", ".cache")
            ):
                continue
            try:
                if path.stat().st_size > 5_000_000:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for tid, body in sealed_snippets.items():
                if len(body) >= 40 and body in text and "fixtures" not in rel:
                    findings.append(
                        ScanFinding(
                            channel="cache",
                            severity="warn",
                            path=rel,
                            detail=f"sealed body for {tid} outside fixtures tree",
                        )
                    )
    return findings


_REGISTRY_PATTERNS = [
    re.compile(r"pypi\.org/(?:simple|project)/[^\s\"']+", re.I),
    re.compile(r"npmjs\.com/package/[^\s\"']+", re.I),
    re.compile(r"huggingface\.co/(?:datasets|models)/[^\s\"']+", re.I),
]


def scan_text_for_registry_shortcuts(text: str, *, path: str = "<text>") -> list[ScanFinding]:
    findings: list[ScanFinding] = []
    for pat in _REGISTRY_PATTERNS:
        for m in pat.finditer(text):
            findings.append(
                ScanFinding(
                    channel="package_registry",
                    severity="info",
                    path=path,
                    detail=f"registry URL in eval/train context: {m.group(0)}",
                )
            )
    return findings


def run_solution_channel_scan(
    *,
    sealed_ids: list[str],
    tasks_dir: Path,
    repo_root: Path,
    training_jsonl: Path | None = None,
    fail_on_warn: bool = False,
) -> ScanReport:
    snippets = _sealed_solution_snippets(sealed_ids, tasks_dir, repo_root)
    findings: list[ScanFinding] = []

    if training_jsonl is not None:
        findings.extend(
            scan_training_jsonl_for_sealed(training_jsonl, snippets)
        )
    else:
        default_jsonl = repo_root / "data/model-factory/audits/axport_full_v2_singleturn.scrubbed.jsonl"
        findings.extend(scan_training_jsonl_for_sealed(default_jsonl, snippets))

    findings.extend(scan_git_history_for_sealed_ids(sealed_ids, repo_root=repo_root))
    findings.extend(scan_local_caches_for_solutions(repo_root, snippets))

    has_fail = any(f.severity == "fail" for f in findings)
    has_warn = any(f.severity == "warn" for f in findings)
    ok = not has_fail and not (fail_on_warn and has_warn)
    return ScanReport(ok=ok, findings=findings)
