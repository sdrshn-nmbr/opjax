"""One-repo pytest reward environment for Stage-6 thin RL.

Binary reward: 1.0 if ``pytest -q`` passes on a fixture after writing candidate
``solution.py``; else 0.0. No learned reward model.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RewardResult:
    task_id: str
    reward: float
    passed: bool
    pytest_returncode: int
    stdout_tail: str
    stderr_tail: str


def load_task(tasks_dir: Path, task_id: str) -> dict:
    path = Path(tasks_dir) / f"{task_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def fixture_dir_for_task(task: dict, *, repo_root: Path | None = None) -> Path:
    fixture = Path(task["fixture_dir"])
    if fixture.is_absolute():
        return fixture
    root = repo_root or Path.cwd()
    return root / fixture


def grade_solution_code(
    *,
    task_id: str,
    code: str,
    tasks_dir: Path,
    repo_root: Path | None = None,
    timeout_s: float = 30.0,
) -> RewardResult:
    """Write ``code`` into a temp copy of the task fixture and run pytest."""
    task = load_task(tasks_dir, task_id)
    src = fixture_dir_for_task(task, repo_root=repo_root)
    if not src.is_dir():
        raise FileNotFoundError(f"fixture missing for {task_id}: {src}")

    with tempfile.TemporaryDirectory(prefix=f"mf-reward-{task_id}-") as tmp:
        work = Path(tmp) / task_id
        shutil.copytree(src, work)
        (work / "solution.py").write_text(code if code.endswith("\n") else code + "\n")
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(work)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        passed = proc.returncode == 0
        return RewardResult(
            task_id=task_id,
            reward=1.0 if passed else 0.0,
            passed=passed,
            pytest_returncode=proc.returncode,
            stdout_tail=(proc.stdout or "")[-500:],
            stderr_tail=(proc.stderr or "")[-500:],
        )


def grade_reference_solutions(
    *,
    task_ids: list[str],
    tasks_dir: Path,
    repo_root: Path | None = None,
) -> list[RewardResult]:
    """Grade the checked-in (usually broken) fixture ``solution.py`` as-is."""
    out: list[RewardResult] = []
    for tid in task_ids:
        task = load_task(tasks_dir, tid)
        src = fixture_dir_for_task(task, repo_root=repo_root)
        code = (src / "solution.py").read_text(encoding="utf-8")
        out.append(
            grade_solution_code(
                task_id=tid,
                code=code,
                tasks_dir=tasks_dir,
                repo_root=repo_root,
            )
        )
    return out
