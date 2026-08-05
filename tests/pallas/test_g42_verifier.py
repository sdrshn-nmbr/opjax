from opjax.pallas.g42_verifier import (
    classify_process_failure,
    requires_worker_recovery,
    sanitized_feedback,
)
from opjax.pallas.environment_runner import _is_runtime_safety_failure


def test_candidate_abort_and_timeout_are_runtime_safety_failures() -> None:
    aborted = classify_process_failure(returncode=134, stderr="Aborted", timed_out=False)
    timed_out = classify_process_failure(returncode=124, stderr="", timed_out=True)
    assert aborted["stage"] == "runtime_safety"
    assert aborted["infrastructure_error"] is False
    assert timed_out["worker_recovery_required"] is True


def test_unattributable_runner_failure_is_infrastructure() -> None:
    result = classify_process_failure(returncode=2, stderr="runner configuration missing", timed_out=False)
    assert result["stage"] == "infrastructure"
    assert result["infrastructure_error"] is True


def test_structured_dma_failure_requires_worker_recovery() -> None:
    assert requires_worker_recovery(
        returncode=2,
        stderr="",
        result={"stage": "full_shape_correctness", "error": "Accelerator device halted during dma.hbm_to_vmem"},
    ) is True
    assert requires_worker_recovery(
        returncode=2,
        stderr="",
        result={"stage": "full_shape_correctness", "error": "values differ"},
    ) is False


def test_environment_runner_routes_device_halts_to_runtime_safety() -> None:
    assert _is_runtime_safety_failure(
        RuntimeError("Core halted unexpectedly during BoundsCheck dma.hbm_to_vmem")
    ) is True
    assert _is_runtime_safety_failure(AssertionError("values differ")) is False


def test_curriculum_feedback_exposes_only_stage() -> None:
    feedback = sanitized_feedback({"stage": "full_shape_correctness", "error": "/hidden/tests.py: secret"})
    assert feedback == (
        "VERIFIER_STAGE full_shape_correctness: The submitted kernel failed a full-shape correctness case."
    )
    assert "hidden" not in feedback
