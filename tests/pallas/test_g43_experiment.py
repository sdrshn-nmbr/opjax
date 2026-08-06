import pytest

from opjax.pallas.g43_experiment import G43ExperimentError, _paired_summary


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
