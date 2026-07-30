from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from opjax.pallas.lowering import (
    LoweringEvidenceError,
    validate_calibration,
    validate_candidate_evidence,
)

RUNTIME = {
    "backend": "tpu",
    "chex": "0.1.90",
    "jax": "0.6.2",
    "jaxlib": "0.6.2",
    "libtpu": "0.0.17",
    "python": "3.10.12",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(root: Path, label: str, marker_count: int) -> dict[str, object]:
    case_dir = root / label
    trace = case_dir / "trace" / "perfetto_trace.json.gz"
    trace.parent.mkdir(parents=True)
    stablehlo = case_dir / "stablehlo.mlir"
    executable = case_dir / "executable.hlo.txt"
    stablehlo.write_text("stablehlo", encoding="utf-8")
    executable.write_text("executable", encoding="utf-8")
    trace.write_bytes(b"trace")
    return {
        "schema_version": 1,
        "label": label,
        "runtime": RUNTIME,
        "repetitions": 3,
        "correctness_verified": True,
        "compiler": {
            "stablehlo_sha256": _sha256(stablehlo),
            "executable_hlo_sha256": _sha256(executable),
            "stablehlo_markers": {"tpu_custom_call": marker_count},
            "executable_hlo_markers": {"tpu_custom_call": marker_count},
        },
        "trace": {
            "perfetto_relative_path": "trace/perfetto_trace.json.gz",
            "perfetto_sha256": _sha256(trace),
            "top_duration_event_names": [
                {"name": "tpu::System::Execute=>Done", "count": 3}
            ],
        },
    }


def _evidence_tree(tmp_path: Path) -> tuple[Path, Path]:
    calibration_root = tmp_path / "calibration"
    cases = {
        label: _case(
            calibration_root,
            label,
            1 if label == "normal_pallas" else 0,
        )
        for label in (
            "normal_pallas",
            "interpreted_pallas",
            "plain_jax",
            "dead_pallas",
        )
    }
    (calibration_root / "calibration.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pallas_lowering_calibration",
                "capture_tool_sha256": "b" * 64,
                "runtime": RUNTIME,
                "cases": cases,
            }
        ),
        encoding="utf-8",
    )
    candidate_root = tmp_path / "candidate"
    evidence = _case(candidate_root, "candidate", 1)
    (candidate_root / "candidate.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pallas_candidate_lowering",
                "capture_tool_sha256": "b" * 64,
                "kernel_sha256": "a" * 64,
                "evidence": evidence,
            }
        ),
        encoding="utf-8",
    )
    return calibration_root, candidate_root


def test_calibrated_tpu_custom_call_evidence_is_verified(tmp_path: Path) -> None:
    calibration_root, candidate_root = _evidence_tree(tmp_path)

    verdict = validate_candidate_evidence(
        calibration_root=calibration_root,
        candidate_root=candidate_root,
        expected_kernel_sha256="a" * 64,
        expected_runtime={
            name: RUNTIME[name]
            for name in ("python", "chex", "jax", "jaxlib", "libtpu")
        },
    )

    assert verdict.verified is True
    assert verdict.reasons == ()


def test_interpreted_control_with_custom_call_invalidates_calibration(
    tmp_path: Path,
) -> None:
    calibration_root, _ = _evidence_tree(tmp_path)
    path = calibration_root / "calibration.json"
    calibration = json.loads(path.read_text(encoding="utf-8"))
    calibration["cases"]["interpreted_pallas"]["compiler"][
        "stablehlo_markers"
    ]["tpu_custom_call"] = 1
    path.write_text(json.dumps(calibration), encoding="utf-8")

    with pytest.raises(
        LoweringEvidenceError,
        match="CALIBRATION_NEGATIVE_MARKER_PRESENT",
    ):
        validate_calibration(calibration_root)


def test_tampered_compiler_artifact_is_rejected(tmp_path: Path) -> None:
    calibration_root, _ = _evidence_tree(tmp_path)
    (calibration_root / "normal_pallas" / "stablehlo.mlir").write_text(
        "tampered",
        encoding="utf-8",
    )

    with pytest.raises(
        LoweringEvidenceError,
        match="EVIDENCE_ARTIFACT_HASH_MISMATCH",
    ):
        validate_calibration(calibration_root)


def test_candidate_without_custom_call_fails_closed(tmp_path: Path) -> None:
    calibration_root, candidate_root = _evidence_tree(tmp_path)
    path = candidate_root / "candidate.json"
    candidate = json.loads(path.read_text(encoding="utf-8"))
    candidate["evidence"]["compiler"]["stablehlo_markers"][
        "tpu_custom_call"
    ] = 0
    path.write_text(json.dumps(candidate), encoding="utf-8")

    verdict = validate_candidate_evidence(
        calibration_root=calibration_root,
        candidate_root=candidate_root,
        expected_kernel_sha256="a" * 64,
    )

    assert verdict.verified is False
    assert "TPU_CUSTOM_CALL_MISSING:stablehlo_markers" in verdict.reasons
