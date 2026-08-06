from __future__ import annotations

import json
from pathlib import Path

import pytest

from opjax.pallas.g6_contracts import (
    G6ContractError,
    discounted_advantages,
    feedback_from_result,
    kernel_score,
    load_g6_config,
)


REPO_ROOT = Path(__file__).parents[2]


def _verified_result(speedup: float) -> dict:
    return {
        "passed": True,
        "stage": "verified",
        "infrastructure_error": False,
        "stages": {
            "artifact_contract": True,
            "pallas_api": True,
            "tpu_compile": True,
            "full_shape_correctness": True,
            "normal_lowering": True,
            "runtime_safety": True,
            "profile": True,
        },
        "profile": {
            "speedup": speedup,
            "timing": {
                "speedup": speedup,
                "candidate_median_ms": 0.02,
                "baseline_median_ms": 0.03,
            },
        },
    }


def test_kernel_score_requires_full_profile_verification() -> None:
    assert kernel_score(_verified_result(1.5)) == pytest.approx(1.8)
    failed = _verified_result(100.0)
    failed["stages"]["normal_lowering"] = False
    assert kernel_score(failed) == 0
    assert kernel_score({"infrastructure_error": True}) == -1


def test_discounted_future_sum_is_normalized_over_all_turns() -> None:
    batch = discounted_advantages([[0, 0, 0, 2], [0, 1, 0, 0]], gamma=0.4)
    assert batch.raw_returns[0] == pytest.approx((0.128, 0.32, 0.8, 2.0))
    assert batch.raw_returns[1] == pytest.approx((0.4, 1.0, 0.0, 0.0))
    assert batch.trainable is True
    flat = [value for row in batch.advantages for value in row]
    assert sum(flat) == pytest.approx(0.0)
    assert sum(value * value for value in flat) / len(flat) == pytest.approx(1.0)


def test_constant_reward_group_is_explicitly_not_trainable() -> None:
    batch = discounted_advantages([[0, 0, 0, 0]] * 16, gamma=0.4)
    assert batch.trainable is False
    assert batch.standard_deviation == 0


def test_actionable_feedback_keeps_compiler_error_but_redacts_hidden_surfaces() -> None:
    feedback = feedback_from_result(
        {
            "stage": "tpu_compile",
            "error": (
                "ValueError: block shape (7, 129) invalid at /tmp/hidden/tests/task.py; "
                "api_key=abc123 reference solution"
            ),
            "infrastructure_error": False,
        }
    )
    assert "block shape (7, 129) invalid" in feedback
    assert "/tmp" not in feedback
    assert "abc123" not in feedback
    assert "reference" not in feedback.lower()
    assert "solution" not in feedback.lower()


def test_verified_feedback_reports_timing_and_profile_marker() -> None:
    feedback = feedback_from_result(_verified_result(1.5))
    assert "speedup=1.500000x" in feedback
    assert "candidate_median_ms=0.020000" in feedback
    assert "profile_marker=tpu_custom_call" in feedback


def test_infrastructure_failure_is_never_sent_as_model_feedback() -> None:
    with pytest.raises(G6ContractError, match="INFRASTRUCTURE_RESULT_NOT_FEEDBACK"):
        feedback_from_result({"stage": "infrastructure", "infrastructure_error": True})


def test_frozen_config_binds_both_parent_checkpoints() -> None:
    result = load_g6_config(
        config_path=REPO_ROOT / "config/pallas/g6-grpo.json",
        task_manifest_path=REPO_ROOT / "data/pallas/runs/g42-task-release/manifest.json",
        s0_manifest_path=REPO_ROOT / "data/pallas/runs/g42-training/manifest.json",
        s1_manifest_path=REPO_ROOT / "data/pallas/runs/g5-s1-training/manifest.json",
    )
    assert result["task_ids"] == json.loads(
        (REPO_ROOT / "data/pallas/runs/g42-task-release/manifest.json").read_text()
    )["training_selection"]
    assert [lane["lane_id"] for lane in result["lanes"]] == ["R0", "R1"]
