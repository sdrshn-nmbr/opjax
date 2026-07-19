"""Stage-6 reward env, scanners, verifier probe, thin-rl dry-run."""

from __future__ import annotations

import json
from pathlib import Path

from opjax.model_factory.reward_env import grade_solution_code
from opjax.model_factory.solution_scanners import run_solution_channel_scan
from opjax.model_factory.splits import assert_manifest_disjoint, load_split_manifest
from opjax.model_factory.thin_rl import ThinRLConfig, plan_dict
from opjax.model_factory.verifier_probe import (
    KNOWN_GOOD,
    build_default_probe_from_fixtures,
    run_verifier_probe,
)

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "docs/model-factory/02-sealed-eval/sudarshanbench/tasks"
SPLITS = ROOT / "docs/model-factory/02-sealed-eval/sudarshanbench/splits.json"


def test_splits_v2_hardened():
    manifest = load_split_manifest(SPLITS)
    assert_manifest_disjoint(manifest)
    assert manifest.version == 2
    assert len(manifest.sealed) == 8
    for tid in ("sb-0013", "sb-0014", "sb-0015", "sb-0016"):
        assert tid in manifest.sealed
    # v1 sealed still present and never movable to train
    for tid in ("sb-0008", "sb-0009", "sb-0010", "sb-0011"):
        assert tid in manifest.sealed
        assert tid not in manifest.train


def test_reward_known_good_and_broken():
    broken = (TASKS.parent / "fixtures/sb-0016/solution.py").read_text()
    bad = grade_solution_code(
        task_id="sb-0016",
        code=broken,
        tasks_dir=TASKS,
        repo_root=ROOT,
    )
    assert bad.reward == 0.0
    good = grade_solution_code(
        task_id="sb-0016",
        code=KNOWN_GOOD["sb-0016"],
        tasks_dir=TASKS,
        repo_root=ROOT,
    )
    assert good.reward == 1.0


def test_verifier_probe_zero_error():
    cases = build_default_probe_from_fixtures(
        task_ids=["sb-0013", "sb-0014", "sb-0015", "sb-0016"],
        tasks_dir=TASKS,
        repo_root=ROOT,
        fixed_solutions=KNOWN_GOOD,
    )
    report = run_verifier_probe(cases, tasks_dir=TASKS, repo_root=ROOT)
    assert report.n == 8
    assert report.fp == 0
    assert report.fn == 0
    assert report.tp == 4
    assert report.tn == 4


def test_solution_scan_ok_without_train_jsonl(tmp_path: Path):
    missing = tmp_path / "nope.jsonl"
    manifest = load_split_manifest(SPLITS)
    report = run_solution_channel_scan(
        sealed_ids=manifest.sealed,
        tasks_dir=TASKS,
        repo_root=ROOT,
        training_jsonl=missing,
    )
    assert report.ok
    assert any(f.channel == "train_jsonl" for f in report.findings)


def test_thin_rl_plan_refuses_sealed_and_estimates():
    cfg = ThinRLConfig(max_steps=10, group_size=2, problems_per_step=2, max_tokens=100)
    plan = plan_dict(cfg, ["sb-0001", "sb-0002"])
    assert plan["est_tokens_rough"] == 10 * 2 * 2 * 100 * 2
    assert "tinker" in plan["substrate"]
