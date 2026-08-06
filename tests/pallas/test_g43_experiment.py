import pytest

from opjax.pallas.g43_experiment import (
    G43ExperimentError,
    _model_summary,
    _paired_summary,
)


def test_paired_summary_reports_task_level_transitions() -> None:
    baseline = [
        {"task_id": "a", "reward": 0},
        {"task_id": "b", "reward": 1},
        {"task_id": "c", "reward": 0},
        {"task_id": "d", "reward": 1},
    ]
    candidate = [
        {"task_id": "a", "reward": 1},
        {"task_id": "b", "reward": 1},
        {"task_id": "c", "reward": 0},
        {"task_id": "d", "reward": 0},
    ]

    summary = _paired_summary(candidate, baseline)

    assert summary == {
        "profile_verified_delta": 0,
        "pass_rate_delta": 0.0,
        "transitions": {
            "fail_to_pass": 1,
            "pass_to_pass": 1,
            "fail_to_fail": 1,
            "pass_to_fail": 1,
        },
    }


def test_paired_summary_rejects_different_task_sets() -> None:
    with pytest.raises(G43ExperimentError, match="G43_PAIRED_TASK_MISMATCH"):
        _paired_summary(
            [{"task_id": "a", "reward": 1}],
            [{"task_id": "b", "reward": 1}],
        )


def test_model_summary_preserves_stage_failure_and_speed_evidence() -> None:
    summary = _model_summary(
        [
            {
                "reward": 1,
                "speedup": 1.1,
                "family": "activation",
                "stage_fractions": {"artifact_contract": 1.0, "profile": 1.0},
                "worker_recovery_required": False,
            },
            {
                "reward": 0,
                "speedup": None,
                "family": "matmul",
                "failure_stage": "runtime_safety",
                "stage_fractions": {"artifact_contract": 1.0, "profile": 0.0},
                "worker_recovery_required": True,
            },
        ]
    )

    assert summary["stage_pass_fractions"] == {
        "artifact_contract": 1.0,
        "profile": 0.5,
    }
    assert summary["failure_stages"] == {"runtime_safety": 1}
    assert summary["failure_families"] == {"matmul": 1}
    assert summary["candidate_tpu_halts"] == 1
    assert summary["verified_speedups"] == [1.1]
