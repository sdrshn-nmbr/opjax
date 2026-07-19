"""Verifier FP/FN probe on a labeled set of (code, expected_pass) pairs.

Stage-6 qualification: measure whether the pytest reward oracle agrees with
human/fixture labels before funding RL.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from opjax.model_factory.reward_env import grade_solution_code


@dataclass
class ProbeCase:
    task_id: str
    code: str
    expected_pass: bool
    label: str = ""


@dataclass
class ProbeReport:
    n: int
    tp: int
    tn: int
    fp: int
    fn: int
    cases: list[dict] = field(default_factory=list)

    @property
    def fp_rate(self) -> float:
        denom = self.fp + self.tn
        return (self.fp / denom) if denom else 0.0

    @property
    def fn_rate(self) -> float:
        denom = self.fn + self.tp
        return (self.fn / denom) if denom else 0.0

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "fp_rate": self.fp_rate,
            "fn_rate": self.fn_rate,
            "cases": self.cases,
        }


def load_probe_cases(path: Path) -> list[ProbeCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = data["cases"] if isinstance(data, dict) else data
    out: list[ProbeCase] = []
    for c in cases:
        out.append(
            ProbeCase(
                task_id=c["task_id"],
                code=c["code"],
                expected_pass=bool(c["expected_pass"]),
                label=str(c.get("label", "")),
            )
        )
    return out


def build_default_probe_from_fixtures(
    *,
    task_ids: list[str],
    tasks_dir: Path,
    repo_root: Path,
    fixed_solutions: dict[str, str] | None = None,
) -> list[ProbeCase]:
    """Negative = broken fixture solution; positive = provided fixed solution."""
    cases: list[ProbeCase] = []
    fixed_solutions = fixed_solutions or {}
    for tid in task_ids:
        task = json.loads((tasks_dir / f"{tid}.json").read_text(encoding="utf-8"))
        fixture = Path(task["fixture_dir"])
        if not fixture.is_absolute():
            fixture = repo_root / fixture
        broken = (fixture / "solution.py").read_text(encoding="utf-8")
        cases.append(
            ProbeCase(
                task_id=tid,
                code=broken,
                expected_pass=False,
                label="broken_fixture",
            )
        )
        if tid in fixed_solutions:
            cases.append(
                ProbeCase(
                    task_id=tid,
                    code=fixed_solutions[tid],
                    expected_pass=True,
                    label="known_good",
                )
            )
    return cases


# Minimal known-good solutions for harden sealed tasks (probe positives only).
KNOWN_GOOD: dict[str, str] = {
    "sb-0013": '''def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    out = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out
''',
    "sb-0014": '''def roman_to_int(s):
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    prev = 0
    for ch in reversed(s):
        v = vals[ch]
        if v < prev:
            total -= v
        else:
            total += v
            prev = v
    return total
''',
    "sb-0015": '''from collections import defaultdict

def group_anagrams(words):
    buckets = defaultdict(list)
    for w in words:
        buckets[tuple(sorted(w))].append(w)
    return list(buckets.values())
''',
    "sb-0016": '''def search_insert(nums, target):
    lo, hi = 0, len(nums)
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo
''',
}


def run_verifier_probe(
    cases: list[ProbeCase],
    *,
    tasks_dir: Path,
    repo_root: Path | None = None,
) -> ProbeReport:
    tp = tn = fp = fn = 0
    details: list[dict] = []
    for case in cases:
        result = grade_solution_code(
            task_id=case.task_id,
            code=case.code,
            tasks_dir=tasks_dir,
            repo_root=repo_root,
        )
        predicted = result.passed
        if predicted and case.expected_pass:
            tp += 1
            kind = "tp"
        elif (not predicted) and (not case.expected_pass):
            tn += 1
            kind = "tn"
        elif predicted and not case.expected_pass:
            fp += 1
            kind = "fp"
        else:
            fn += 1
            kind = "fn"
        details.append(
            {
                "task_id": case.task_id,
                "label": case.label,
                "expected_pass": case.expected_pass,
                "predicted_pass": predicted,
                "kind": kind,
                "reward": result.reward,
            }
        )
    return ProbeReport(
        n=len(cases),
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        cases=details,
    )
