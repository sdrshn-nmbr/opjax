"""Fresh-process authoritative verifier boundary for G4.2 submissions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from opjax.pallas.g42_harness import file_sha256, write_verifier_artifacts


class G42VerifierError(RuntimeError):
    """The verifier could not produce an attributable result."""


def classify_process_failure(*, returncode: int, stderr: str, timed_out: bool) -> dict[str, Any]:
    lowered = stderr.lower()
    candidate_runtime_markers = (
        "aborted",
        "check failed",
        "dma",
        "out of bounds",
        "segmentation fault",
        "sigabrt",
    )
    candidate_failure = timed_out or returncode < 0 or returncode in {124, 134, 137, 139} or any(
        marker in lowered for marker in candidate_runtime_markers
    )
    if candidate_failure:
        return {
            "passed": False,
            "stage": "runtime_safety",
            "error": "CANDIDATE_PROCESS_TIMEOUT" if timed_out else f"CANDIDATE_PROCESS_EXIT_{returncode}",
            "infrastructure_error": False,
            "worker_recovery_required": True,
            "stages": {
                "artifact_contract": True,
                "pallas_api": True,
                "tpu_compile": True,
                "full_shape_correctness": False,
                "normal_lowering": False,
                "runtime_safety": False,
                "profile": False,
            },
        }
    return {
        "passed": False,
        "stage": "infrastructure",
        "error": f"VERIFIER_PROCESS_EXIT_{returncode}",
        "infrastructure_error": True,
        "worker_recovery_required": False,
    }


def run_fresh_verifier(
    *,
    task_path: Path,
    kernel_path: Path,
    output_dir: Path,
    timeout_seconds: int = 120,
    runner_command: list[str] | None = None,
) -> dict[str, Any]:
    if not task_path.is_file() or not kernel_path.is_file() or kernel_path.is_symlink():
        raise G42VerifierError("ARTIFACT_CONTRACT_INVALID")
    task = json.loads(task_path.read_text(encoding="utf-8"))
    evidence_dir = output_dir / "evidence"
    command = runner_command or [
        sys.executable,
        "-m",
        "opjax.pallas.environment_runner",
        "--task",
        str(task_path),
        "--kernel",
        str(kernel_path),
        "--evidence-dir",
        str(evidence_dir),
    ]
    timed_out = False
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        process = subprocess.CompletedProcess(
            command,
            124,
            stdout=exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout or "",
            stderr=exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr or "",
        )
    result: dict[str, Any] | None = None
    for line in reversed(process.stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            result = candidate
            break
    if result is None:
        result = classify_process_failure(
            returncode=process.returncode,
            stderr=process.stderr,
            timed_out=timed_out,
        )
    result["process"] = {
        "returncode": process.returncode,
        "timed_out": timed_out,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }
    reward = write_verifier_artifacts(
        result=result,
        output_dir=output_dir,
        task_id=task["task_id"],
        kernel_sha256=file_sha256(kernel_path),
    )
    return {"result": result, "reward": reward}


def sanitized_feedback(result: dict[str, Any]) -> str:
    stage = str(result.get("stage", "verifier"))
    allowed = {
        "artifact_contract": "The submitted module does not satisfy the output contract.",
        "pallas_api": "The submitted module does not satisfy the authentic Pallas API contract.",
        "tpu_compile": "The submitted kernel did not compile for the TPU.",
        "full_shape_correctness": "The submitted kernel failed a full-shape correctness case.",
        "normal_lowering": "The submitted kernel did not prove normal Pallas lowering.",
        "runtime_safety": "The submitted kernel failed the TPU runtime-safety check.",
        "profile": "The submitted kernel did not produce valid profile evidence.",
    }
    return f"VERIFIER_STAGE {stage}: {allowed.get(stage, 'Verification could not complete.')}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-g42-verifier")
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args(argv)
    try:
        payload = run_fresh_verifier(
            task_path=args.task,
            kernel_path=args.kernel,
            output_dir=args.output_dir,
            timeout_seconds=args.timeout_seconds,
        )
    except (G42VerifierError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"G42_VERIFIER_ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["reward"]["reward"] == 1 else 2


if __name__ == "__main__":
    sys.exit(main())
