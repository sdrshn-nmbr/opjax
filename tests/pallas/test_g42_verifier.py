from opjax.pallas.g42_verifier import classify_process_failure, sanitized_feedback


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


def test_curriculum_feedback_exposes_only_stage() -> None:
    feedback = sanitized_feedback({"stage": "full_shape_correctness", "error": "/hidden/tests.py: secret"})
    assert feedback == (
        "VERIFIER_STAGE full_shape_correctness: The submitted kernel failed a full-shape correctness case."
    )
    assert "hidden" not in feedback
