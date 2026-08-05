"""Pinned, isolated, resumable JAXBench evaluation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from opjax.pallas.contracts import (
    ContractBundle,
    ContractError,
    git_revision,
    source_by_id,
    verify_source_checkout,
)
from opjax.pallas.lowering import (
    LoweringEvidenceError,
    validate_calibration,
    validate_candidate_evidence,
)
from opjax.pallas.scoring import (
    PromptContext,
    diagnostic_reward,
    inspect_pallas_source,
    judge,
    timing_evidence,
)


class EvaluationError(RuntimeError):
    """The evaluation cannot produce comparable evidence."""


@dataclass(frozen=True)
class SampleCandidate:
    sample_id: str
    workload: str
    seed: int
    kernel: Path
    sample: dict[str, Any]


@dataclass(frozen=True)
class ValidatedSampleRun:
    fingerprint_sha256: str
    candidates: tuple[SampleCandidate, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    _atomic_write(path, text)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationError(
                f"RESULTS_JSON_INVALID: {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise EvaluationError(f"RESULT_ROW_INVALID: {path}:{line_number}")
        rows.append(value)
    return rows


def _git_tracked_dirty(path: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode != 0 or bool(result.stdout.strip())


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def kernel_set_fingerprint(
    candidates: tuple[SampleCandidate, ...],
) -> dict[str, str]:
    return {
        candidate.sample_id: _sha256_file(candidate.kernel)
        for candidate in candidates
    }


def environment_fingerprint(
    *,
    bundle: ContractBundle,
    repo_root: Path,
    jaxbench_root: Path,
    candidates: tuple[SampleCandidate, ...],
    model_id: str,
    arm: str,
    prompt_context: PromptContext,
    runtime_hardware: dict[str, Any] | None,
    sample_fingerprint_sha256: str | None,
    lowering_calibration_sha256: str,
) -> dict[str, Any]:
    prompt_contract = {
        "prompt": bundle.experiment["prompt"],
        "sampling": bundle.experiment["sampling"],
        "context": prompt_context.value,
    }
    return {
        "contract_sha256": bundle.sha256,
        "experiment_id": bundle.experiment["experiment_id"],
        "opjax_revision": git_revision(repo_root),
        "opjax_tracked_dirty": _git_tracked_dirty(repo_root),
        "jaxbench_revision": verify_source_checkout(bundle, "jaxbench", jaxbench_root),
        "jaxbench_tracked_dirty": _git_tracked_dirty(jaxbench_root),
        "kernel_sha256": kernel_set_fingerprint(candidates),
        "model_id": model_id,
        "arm": arm,
        "prompt_context": prompt_context.value,
        "prompt_contract_sha256": _sha256_bytes(
            json.dumps(prompt_contract, sort_keys=True, separators=(",", ":")).encode()
        ),
        "sample_fingerprint_sha256": sample_fingerprint_sha256,
        "lowering_calibration_sha256": lowering_calibration_sha256,
        "target": bundle.experiment["target"],
        "runtime_hardware": runtime_hardware,
        "packages": {
            name: _package_version(name)
            for name in (
                "chex",
                "jax",
                "jaxlib",
                "libtpu",
                "tinker",
                "tinker-cookbook",
            )
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def validate_sample_run(
    *,
    bundle: ContractBundle,
    sample_run: Path,
    model_id: str,
    arm: str,
    prompt_context: PromptContext,
) -> ValidatedSampleRun:
    manifest_path = sample_run / "manifest.json"
    samples_path = sample_run / "samples.jsonl"
    if not manifest_path.is_file() or not samples_path.is_file():
        raise EvaluationError(f"SAMPLE_RUN_INCOMPLETE: {sample_run}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fingerprint = manifest.get("fingerprint")
    if manifest.get("status") != "sampled" or not isinstance(fingerprint, dict):
        raise EvaluationError(f"SAMPLE_RUN_NOT_COMPLETE: {sample_run}")
    expected_model = (
        bundle.experiment["base_model"]
        if arm == "A"
        else fingerprint.get("model_path")
    )
    required = {
        "contract_sha256": bundle.sha256,
        "jaxbench_revision": source_by_id(bundle, "jaxbench")["revision"],
        "arm": arm,
        "prompt_context": prompt_context.value,
    }
    for key, expected in required.items():
        if fingerprint.get(key) != expected:
            raise EvaluationError(
                "SAMPLE_FINGERPRINT_MISMATCH: "
                f"{key}: expected={expected!r} observed={fingerprint.get(key)!r}"
            )
    if model_id != expected_model:
        raise EvaluationError(
            f"SAMPLE_MODEL_MISMATCH: expected={expected_model!r} observed={model_id!r}"
        )
    request = fingerprint.get("request")
    if not isinstance(request, dict):
        raise EvaluationError("SAMPLE_REQUEST_MISSING")
    requested_ids = request.get("sample_ids")
    if (
        not isinstance(requested_ids, list)
        or not requested_ids
        or not all(isinstance(sample_id, str) for sample_id in requested_ids)
        or len(requested_ids) != len(set(requested_ids))
    ):
        raise EvaluationError("SAMPLE_REQUEST_INVALID")
    rows = load_jsonl(samples_path)
    sample_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str):
            raise EvaluationError("SAMPLE_ID_MISSING")
        if sample_id in sample_by_id:
            raise EvaluationError(f"SAMPLE_ID_DUPLICATE: {sample_id}")
        sample_by_id[sample_id] = row
    if set(sample_by_id) != set(requested_ids):
        raise EvaluationError(
            "SAMPLE_SET_MISMATCH: "
            f"missing={sorted(set(requested_ids) - set(sample_by_id))} "
            f"extra={sorted(set(sample_by_id) - set(requested_ids))}"
        )
    public_ids = set(bundle.splits["public_evaluation"]["task_ids"])
    contract_seeds = set(bundle.experiment["sampling"]["seeds"])
    candidates: list[SampleCandidate] = []
    sample_root = sample_run.resolve()
    for sample_id in requested_ids:
        row = sample_by_id[sample_id]
        workload = row.get("workload")
        seed = row.get("seed")
        kernel_path = row.get("kernel_path")
        if workload not in public_ids:
            raise EvaluationError(
                f"SAMPLE_WORKLOAD_NOT_PUBLIC: {sample_id}: {workload!r}"
            )
        if seed not in contract_seeds:
            raise EvaluationError(f"SAMPLE_SEED_INVALID: {sample_id}: {seed!r}")
        expected_id = f"{workload}::seed={seed}"
        if sample_id != expected_id:
            raise EvaluationError(
                f"SAMPLE_ID_MISMATCH: expected={expected_id} observed={sample_id}"
            )
        expected_path = f"kernels/seed-{seed}/{workload}.py"
        if kernel_path != expected_path:
            raise EvaluationError(
                "SAMPLE_KERNEL_PATH_MISMATCH: "
                f"{sample_id}: expected={expected_path} observed={kernel_path!r}"
            )
        kernel = (sample_run / kernel_path).resolve()
        if not kernel.is_relative_to(sample_root) or not kernel.is_file():
            raise EvaluationError(f"SAMPLE_KERNEL_MISSING: {sample_id}: {kernel}")
        expected_hash = row.get("code_sha256")
        observed_hash = _sha256_file(kernel)
        if expected_hash != observed_hash:
            raise EvaluationError(
                "SAMPLE_KERNEL_HASH_MISMATCH: "
                f"{sample_id}: expected={expected_hash} observed={observed_hash}"
            )
        candidates.append(
            SampleCandidate(
                sample_id=sample_id,
                workload=workload,
                seed=seed,
                kernel=kernel,
                sample=row,
            )
        )
    sha256 = fingerprint.get("sha256")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise EvaluationError("SAMPLE_FINGERPRINT_INVALID")
    return ValidatedSampleRun(
        fingerprint_sha256=sha256,
        candidates=tuple(candidates),
    )


def probe_runtime_hardware(timeout_seconds: float = 60) -> dict[str, Any]:
    source = (
        "import chex, json, jax; "
        "chex.assert_devices_available(1, 'tpu', not_less_than=True); "
        "devices=jax.devices(); "
        "print(json.dumps({"
        "'platforms': sorted({d.platform for d in devices}),"
        "'device_kinds': sorted({getattr(d, 'device_kind', 'unknown') for d in devices}),"
        "'device_count': len(devices),"
        "'process_count': jax.process_count(),"
        "'process_index': jax.process_index()"
        "}))"
    )
    try:
        process = subprocess.run(
            [sys.executable, "-c", source],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise EvaluationError(f"HARDWARE_PROBE_TIMEOUT: {exc}") from exc
    parsed = _parse_json_output(process.stdout)
    if process.returncode != 0 or parsed is None:
        raise EvaluationError(
            "HARDWARE_PROBE_FAILED: "
            f"returncode={process.returncode} "
            f"stderr={process.stderr[-1000:]}"
        )
    return parsed


def _assert_tpu_runtime(
    observed: dict[str, Any],
    target: dict[str, Any],
) -> None:
    device_count = observed.get("device_count")
    process_count = observed.get("process_count")
    process_index = observed.get("process_index")
    if (
        not isinstance(device_count, int)
        or isinstance(device_count, bool)
        or device_count < 1
        or not isinstance(process_count, int)
        or isinstance(process_count, bool)
        or process_count < 1
        or not isinstance(process_index, int)
        or isinstance(process_index, bool)
        or not 0 <= process_index < process_count
    ):
        raise EvaluationError(f"HARDWARE_PROBE_INVALID: {observed}")
    if observed.get("platforms") != ["tpu"]:
        raise EvaluationError(
            f"HARDWARE_TARGET_MISMATCH: expected=tpu observed={observed.get('platforms')}"
        )
    hardware = str(target["hardware"]).lower()
    device_kinds = observed.get("device_kinds")
    if (
        not isinstance(device_kinds, list)
        or not device_kinds
        or not all(isinstance(kind, str) and kind for kind in device_kinds)
    ):
        raise EvaluationError(f"HARDWARE_PROBE_INVALID: {observed}")
    expected_generation = hardware.removeprefix("v").removesuffix("e")
    expected_kind = re.compile(
        rf"^tpu\s+v{re.escape(expected_generation)}(?:e|\s+lite)(?:\s|$)"
    )
    if not any(expected_kind.search(kind.lower()) for kind in device_kinds):
        raise EvaluationError(
            "HARDWARE_TARGET_MISMATCH: "
            f"expected={target['hardware']} observed={observed.get('device_kinds')}"
        )


def _assert_evaluation_runtime(
    fingerprint: dict[str, Any],
    expected: dict[str, str],
) -> None:
    observed = {
        "python": fingerprint.get("python"),
        **{
            name: fingerprint.get("packages", {}).get(name)
            for name in ("chex", "jax", "jaxlib", "libtpu")
        },
    }
    if observed != expected:
        raise EvaluationError(
            f"EVALUATION_RUNTIME_MISMATCH: expected={expected} observed={observed}"
        )


def _assert_public_workloads_match(bundle: ContractBundle, jaxbench_root: Path) -> None:
    benchmark = jaxbench_root / "JAXBench" / "benchmark"
    if not benchmark.is_dir():
        raise EvaluationError(f"JAXBENCH_LAYOUT_INVALID: {benchmark}")
    observed = {
        path.name
        for path in benchmark.iterdir()
        if path.is_dir() and (path / "baseline.py").is_file()
    }
    expected = set(bundle.splits["public_evaluation"]["task_ids"])
    if observed != expected:
        raise EvaluationError(
            "JAXBENCH_WORKLOAD_SET_MISMATCH: "
            f"missing={sorted(expected - observed)} extra={sorted(observed - expected)}"
        )


def _load_or_create_manifest(
    *,
    out_dir: Path,
    fingerprint: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not resume:
            raise EvaluationError(
                f"RUN_ALREADY_EXISTS: {out_dir}; pass --resume to continue"
            )
        if manifest.get("fingerprint") != fingerprint:
            raise EvaluationError("RESUME_FINGERPRINT_MISMATCH")
        return manifest
    manifest = {
        "schema_version": 2,
        "created_at": _utc_now(),
        "status": "running",
        "fingerprint": fingerprint,
        "generator": {"argv": list(sys.argv)},
    }
    _write_json(manifest_path, manifest)
    return manifest


def _parse_json_output(stdout: str) -> dict[str, Any] | None:
    stripped = stdout.strip()
    if not stripped:
        return None
    try:
        value = json.loads(stripped)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    for line in reversed(stripped.splitlines()):
        try:
            value = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _run_jaxbench_once(
    *,
    jaxbench_root: Path,
    workload: str,
    kernel: Path,
    tpu: str,
    num_warmup: int,
    num_iters: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "JAXBench",
        "evaluate",
        "--workload",
        workload,
        "--kernel",
        str(kernel.resolve()),
        "--tpu",
        tpu,
        "--num-warmup",
        str(num_warmup),
        "--num-iters",
        str(num_iters),
        "--json",
    ]
    environment = os.environ.copy()
    python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{jaxbench_root}{os.pathsep}{python_path}" if python_path else str(jaxbench_root)
    )
    deadline = time.monotonic() + timeout_seconds
    transient_attempts: list[dict[str, Any]] = []
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {
                "returncode": None,
                "result": {
                    "workload": workload,
                    "status": "error",
                    "error_code": "TPU_LOCK_RETRY_TIMEOUT",
                    "error": "TPU runtime lock did not clear before the deadline",
                },
                "stdout_tail": "",
                "stderr_tail": "",
                "transient_attempts": transient_attempts,
            }
        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                env=environment,
                timeout=remaining,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "returncode": None,
                "result": {
                    "workload": workload,
                    "status": "error",
                    "error_code": "EVALUATION_TIMEOUT",
                    "error": str(exc),
                },
                "stdout_tail": "",
                "stderr_tail": "",
                "transient_attempts": transient_attempts,
            }
        parsed = _parse_json_output(process.stdout)
        if parsed is None:
            parsed = {
                "workload": workload,
                "status": "error",
                "error_code": "JAXBENCH_JSON_MISSING",
                "error": (process.stderr or process.stdout or "no output")[-1000:],
            }
        combined_output = f"{process.stdout}\n{process.stderr}"
        if "TPU is already in use by process" not in combined_output:
            return {
                "returncode": process.returncode,
                "result": parsed,
                "stdout_tail": process.stdout[-2000:],
                "stderr_tail": process.stderr[-2000:],
                "transient_attempts": transient_attempts,
            }
        transient_attempts.append(
            {
                "returncode": process.returncode,
                "result": parsed,
                "stdout_tail": process.stdout[-2000:],
                "stderr_tail": process.stderr[-2000:],
            }
        )
        print(
            "PALLAS_EVAL_TPU_LOCK_RETRY "
            f"workload={workload} attempt={len(transient_attempts)}",
            flush=True,
        )
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


def _result_correct(value: dict[str, Any]) -> bool:
    correctness = value.get("correctness")
    if isinstance(correctness, dict):
        return correctness.get("correct") is True
    return value.get("correct") is True


def _result_compiled(value: dict[str, Any]) -> bool:
    return value.get("status") in {"incorrect", "correct"}


def _capture_lowering_evidence(
    *,
    bundle: ContractBundle,
    jaxbench_root: Path,
    candidate: SampleCandidate,
    calibration_root: Path,
    evidence_root: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    expected_runtime = bundle.eval_policy["runtime"]
    candidate_root = (
        evidence_root / f"seed-{candidate.seed}" / candidate.workload
    )
    command_result: dict[str, Any] | None = None
    if not (candidate_root / "candidate.json").is_file():
        command = [
            sys.executable,
            "-m",
            "opjax.pallas.lowering",
            "capture-candidate",
            "--jaxbench-root",
            str(jaxbench_root),
            "--workload",
            candidate.workload,
            "--kernel",
            str(candidate.kernel),
            "--out-dir",
            str(candidate_root),
            "--repetitions",
            str(bundle.eval_policy["authenticity"]["profile_repetitions"]),
        ]
        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "attempted": True,
                "verified": False,
                "error": f"LOWERING_CAPTURE_TIMEOUT: {exc}",
            }
        command_result = {
            "returncode": process.returncode,
            "stdout_tail": process.stdout[-2000:],
            "stderr_tail": process.stderr[-2000:],
        }
        if process.returncode != 0:
            return {
                "attempted": True,
                "verified": False,
                "error": "LOWERING_CAPTURE_FAILED",
                "command": command_result,
            }
    try:
        verdict = validate_candidate_evidence(
            calibration_root=calibration_root,
            candidate_root=candidate_root,
            expected_kernel_sha256=_sha256_file(candidate.kernel),
            expected_runtime=expected_runtime,
        )
    except LoweringEvidenceError as exc:
        return {
            "attempted": True,
            "verified": False,
            "error": str(exc),
            "command": command_result,
        }
    return {
        "attempted": True,
        "verified": verdict.verified,
        "calibration_sha256": verdict.calibration_sha256,
        "candidate_sha256": verdict.candidate_sha256,
        "kernel_sha256": verdict.kernel_sha256,
        "runtime": verdict.runtime,
        "reasons": list(verdict.reasons),
        "artifact_root": str(candidate_root.relative_to(evidence_root.parent)),
        "command": command_result,
    }


def _evaluate_workload(
    *,
    bundle: ContractBundle,
    jaxbench_root: Path,
    candidate: SampleCandidate,
    prompt_context: PromptContext,
    lowering_calibration: Path,
    lowering_evidence_root: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    workload = candidate.workload
    kernel = candidate.kernel
    policy = bundle.eval_policy
    timing_policy = policy["timing"]
    source = kernel.read_text(encoding="utf-8")
    preflight = inspect_pallas_source(source)
    execution_status = (
        "EXECUTED"
        if preflight.parses and preflight.has_workload
        else "STATIC_REJECTED"
    )
    raw_runs = (
        [
            _run_jaxbench_once(
                jaxbench_root=jaxbench_root,
                workload=workload,
                kernel=kernel,
                tpu=bundle.experiment["target"]["hardware"],
                num_warmup=timing_policy["num_warmup"],
                num_iters=timing_policy["num_iters"],
                timeout_seconds=timeout_seconds,
            )
            for _ in range(timing_policy["min_repeated_runs"])
        ]
        if execution_status == "EXECUTED"
        else []
    )
    results = [run["result"] for run in raw_runs]
    compiled = bool(results) and all(_result_compiled(result) for result in results)
    correct = compiled and all(_result_correct(result) for result in results)
    lowering_evidence: dict[str, Any] = {
        "attempted": False,
        "verified": None,
    }
    if correct and preflight.authentic:
        lowering_evidence = _capture_lowering_evidence(
            bundle=bundle,
            jaxbench_root=jaxbench_root,
            candidate=candidate,
            calibration_root=lowering_calibration,
            evidence_root=lowering_evidence_root,
            timeout_seconds=timeout_seconds,
        )
    baseline_medians = [
        result["baseline"]["median_ms"]
        for result in results
        if isinstance(result.get("baseline"), dict)
        and isinstance(result["baseline"].get("median_ms"), (int, float))
    ]
    kernel_medians = [
        result["kernel"]["median_ms"]
        for result in results
        if isinstance(result.get("kernel"), dict)
        and isinstance(result["kernel"].get("median_ms"), (int, float))
    ]
    baseline_timing = timing_evidence(
        baseline_medians,
        min_runs=timing_policy["min_repeated_runs"],
        max_coefficient_of_variation=timing_policy["max_coefficient_of_variation"],
    )
    kernel_timing = timing_evidence(
        kernel_medians,
        min_runs=timing_policy["min_repeated_runs"],
        max_coefficient_of_variation=timing_policy["max_coefficient_of_variation"],
    )
    baseline_median = baseline_timing.median_ms
    kernel_median = kernel_timing.median_ms
    timing_stable = baseline_timing.stable and kernel_timing.stable
    speedup = (
        baseline_median / kernel_median
        if baseline_median is not None and kernel_median is not None and kernel_median > 0
        else None
    )
    baseline_path = (
        jaxbench_root / "JAXBench" / "benchmark" / workload / "baseline.py"
    )
    baseline_source = baseline_path.read_text(encoding="utf-8")
    verdict = judge(
        workload=workload,
        candidate_src=source,
        baseline_src=baseline_source,
        compiled=compiled,
        correct=correct,
        prompt_context=prompt_context,
        speedup=speedup,
        timing_stable=timing_stable,
        headline_speedup_threshold=timing_policy["headline_speedup_threshold"],
        lowering_verified=lowering_evidence["verified"],
        require_lowering_evidence=bundle.eval_policy["authenticity"][
            "require_empirical_tpu_lowering"
        ],
    )
    return {
        "schema_version": 2,
        "sample_id": candidate.sample_id,
        "workload": workload,
        "seed": candidate.seed,
        "checked_at": _utc_now(),
        "execution_status": execution_status,
        "kernel_sha256": _sha256_file(kernel),
        "compiled": compiled,
        "correct": correct,
        "inspection": asdict(verdict.inspection),
        "prompt_context": prompt_context.value,
        "similarity_to_baseline": verdict.similarity,
        "copied": verdict.copied,
        "timing": {
            "baseline": asdict(baseline_timing),
            "kernel": asdict(kernel_timing),
            "stable": timing_stable,
        },
        "baseline_median_ms": baseline_median,
        "kernel_median_ms": kernel_median,
        "speedup": speedup,
        "lowering_evidence": lowering_evidence,
        "credited": verdict.credited,
        "pallas_credited": verdict.pallas_credited,
        "headline_credited": verdict.headline_credited,
        "diagnostic_reward": diagnostic_reward(verdict),
        "no_credit_reasons": list(verdict.no_credit_reasons),
        "raw_runs": raw_runs,
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _oracle_metrics(
    *,
    candidates: tuple[SampleCandidate, ...],
    rows_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = [rows_by_id[candidate.sample_id] for candidate in candidates]
    n = len(candidates)
    n_parseable = sum(
        candidate.sample.get("status") == "sampled"
        for candidate in candidates
    )
    n_authentic_emissions = sum(
        row.get("inspection", {}).get("authentic") is True
        for row in rows
    )
    n_compiled = sum(row.get("compiled") is True for row in rows)
    n_correct = sum(row.get("correct") is True for row in rows)
    n_timing_stable = sum(
        row.get("timing", {}).get("stable") is True for row in rows
    )
    n_lowering_verified = sum(
        row.get("lowering_evidence", {}).get("verified") is True
        for row in rows
    )
    n_pallas_credited = sum(row.get("pallas_credited") is True for row in rows)
    n_headline_credited = sum(
        row.get("headline_credited") is True for row in rows
    )
    stable_speedups = [
        float(row["speedup"])
        for row in rows
        if row.get("pallas_credited") is True
        and row.get("timing", {}).get("stable") is True
        and isinstance(row.get("speedup"), (int, float))
    ]
    return {
        "n_samples": n,
        "n_sampling_attempts": sum(
            len(candidate.sample.get("attempts", []))
            for candidate in candidates
        ),
        "n_parseable": n_parseable,
        "parse_rate": _rate(n_parseable, n),
        "n_authentic_emissions": n_authentic_emissions,
        "authentic_emission_rate": _rate(n_authentic_emissions, n),
        "n_compiled": n_compiled,
        "compilation_rate": _rate(n_compiled, n),
        "n_correct": n_correct,
        "correctness_rate": _rate(n_correct, n),
        "n_timing_stable": n_timing_stable,
        "timing_stability_rate": _rate(n_timing_stable, n),
        "n_lowering_verified": n_lowering_verified,
        "lowering_verification_rate": _rate(n_lowering_verified, n),
        "n_pallas_credited": n_pallas_credited,
        "pallas_credit_rate": _rate(n_pallas_credited, n),
        "n_headline_credited": n_headline_credited,
        "headline_credit_rate": _rate(n_headline_credited, n),
        "best_stable_pallas_speedup": (
            max(stable_speedups) if stable_speedups else None
        ),
    }


def _oracle_summary(
    *,
    bundle: ContractBundle,
    candidates: tuple[SampleCandidate, ...],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rows_by_id = {row["sample_id"]: row for row in rows}
    if len(rows_by_id) != len(rows):
        raise EvaluationError("RESULT_SAMPLE_ID_DUPLICATE")
    missing = sorted(
        candidate.sample_id
        for candidate in candidates
        if candidate.sample_id not in rows_by_id
    )
    if missing:
        raise EvaluationError(f"RESULTS_INCOMPLETE: {missing}")
    overall = _oracle_metrics(candidates=candidates, rows_by_id=rows_by_id)
    by_seed: dict[str, dict[str, Any]] = {}
    for seed in bundle.experiment["sampling"]["seeds"]:
        selected = tuple(
            candidate for candidate in candidates if candidate.seed == seed
        )
        if selected:
            by_seed[str(seed)] = _oracle_metrics(
                candidates=selected,
                rows_by_id=rows_by_id,
            )
    rate_names = (
        "parse_rate",
        "authentic_emission_rate",
        "compilation_rate",
        "correctness_rate",
        "timing_stability_rate",
        "lowering_verification_rate",
        "pallas_credit_rate",
        "headline_credit_rate",
    )
    seed_rate_ranges = {
        name: round(
            max(float(metrics[name]) for metrics in by_seed.values())
            - min(float(metrics[name]) for metrics in by_seed.values()),
            6,
        )
        for name in rate_names
        if by_seed and all(metrics[name] is not None for metrics in by_seed.values())
    }
    workloads = list(
        dict.fromkeys(candidate.workload for candidate in candidates)
    )
    by_workload = {
        workload: _oracle_metrics(
            candidates=tuple(
                candidate
                for candidate in candidates
                if candidate.workload == workload
            ),
            rows_by_id=rows_by_id,
        )
        for workload in workloads
    }
    expected_seed_count = len({candidate.seed for candidate in candidates})
    seed_consistency = {
        "n_workloads": len(by_workload),
        "n_workloads_with_all_seeds_parseable": sum(
            metrics["n_parseable"] == expected_seed_count
            for metrics in by_workload.values()
        ),
        "n_workloads_with_any_authentic_emission": sum(
            metrics["n_authentic_emissions"] > 0
            for metrics in by_workload.values()
        ),
        "n_workloads_with_all_seeds_authentic": sum(
            metrics["n_authentic_emissions"] == expected_seed_count
            for metrics in by_workload.values()
        ),
        "n_workloads_with_any_correct": sum(
            metrics["n_correct"] > 0
            for metrics in by_workload.values()
        ),
        "n_workloads_with_all_seeds_correct": sum(
            metrics["n_correct"] == expected_seed_count
            for metrics in by_workload.values()
        ),
    }
    return {
        **overall,
        "by_seed": by_seed,
        "seed_rate_ranges": seed_rate_ranges,
        "seed_consistency": seed_consistency,
        "by_workload": by_workload,
        "generalization_claim_ready": False,
    }


def _rescore_result_rows(
    *,
    candidates: tuple[SampleCandidate, ...],
    rows: list[dict[str, Any]],
    jaxbench_root: Path,
    prompt_context: PromptContext,
    headline_speedup_threshold: float,
    require_lowering_evidence: bool = False,
) -> list[dict[str, Any]]:
    rows_by_id = {row.get("sample_id"): row for row in rows}
    if len(rows_by_id) != len(rows):
        raise EvaluationError("RESULT_SAMPLE_ID_DUPLICATE")
    expected_ids = {candidate.sample_id for candidate in candidates}
    if set(rows_by_id) != expected_ids:
        raise EvaluationError("RESULT_SAMPLE_SET_MISMATCH")

    rescored: list[dict[str, Any]] = []
    for candidate in candidates:
        row = dict(rows_by_id[candidate.sample_id])
        if (
            row.get("workload") != candidate.workload
            or row.get("seed") != candidate.seed
            or row.get("prompt_context") != prompt_context.value
        ):
            raise EvaluationError(
                f"RESULT_LINEAGE_MISMATCH: {candidate.sample_id}"
            )
        if row.get("kernel_sha256") != _sha256_file(candidate.kernel):
            raise EvaluationError(
                f"RESULT_KERNEL_HASH_MISMATCH: {candidate.sample_id}"
            )
        compiled = row.get("compiled")
        correct = row.get("correct")
        timing = row.get("timing")
        if not isinstance(compiled, bool) or not isinstance(correct, bool):
            raise EvaluationError(
                f"RESULT_CREDIT_INPUT_INVALID: {candidate.sample_id}"
            )
        if not isinstance(timing, dict) or not isinstance(timing.get("stable"), bool):
            raise EvaluationError(
                f"RESULT_TIMING_INPUT_INVALID: {candidate.sample_id}"
            )
        speedup = row.get("speedup")
        if (
            speedup is not None
            and (
                not isinstance(speedup, (int, float))
                or isinstance(speedup, bool)
            )
        ):
            raise EvaluationError(
                f"RESULT_SPEEDUP_INVALID: {candidate.sample_id}"
            )
        source = candidate.kernel.read_text(encoding="utf-8")
        lowering = row.get("lowering_evidence")
        lowering_verified = (
            lowering.get("verified")
            if isinstance(lowering, dict)
            and isinstance(lowering.get("verified"), bool)
            else None
        )
        baseline_source = (
            jaxbench_root
            / "JAXBench"
            / "benchmark"
            / candidate.workload
            / "baseline.py"
        ).read_text(encoding="utf-8")
        verdict = judge(
            workload=candidate.workload,
            candidate_src=source,
            baseline_src=baseline_source,
            compiled=compiled,
            correct=correct,
            prompt_context=prompt_context,
            speedup=float(speedup) if speedup is not None else None,
            timing_stable=timing["stable"],
            headline_speedup_threshold=headline_speedup_threshold,
            lowering_verified=lowering_verified,
            require_lowering_evidence=require_lowering_evidence,
        )
        row.update(
            {
                "inspection": asdict(verdict.inspection),
                "similarity_to_baseline": verdict.similarity,
                "copied": verdict.copied,
                "credited": verdict.credited,
                "pallas_credited": verdict.pallas_credited,
                "headline_credited": verdict.headline_credited,
                "diagnostic_reward": diagnostic_reward(verdict),
                "no_credit_reasons": list(verdict.no_credit_reasons),
            }
        )
        rescored.append(row)
    return rescored


def audit_evaluation(
    *,
    bundle: ContractBundle,
    repo_root: Path,
    jaxbench_root: Path,
    sample_run: Path,
    evaluation_run: Path,
    model_id: str,
    arm: str,
    prompt_context: PromptContext,
) -> dict[str, Any]:
    verify_source_checkout(bundle, "jaxbench", jaxbench_root)
    _assert_public_workloads_match(bundle, jaxbench_root)
    if _git_tracked_dirty(repo_root):
        raise EvaluationError(f"OPJAX_TRACKED_DIRTY: {repo_root}")
    if _git_tracked_dirty(jaxbench_root):
        raise EvaluationError(f"JAXBENCH_TRACKED_DIRTY: {jaxbench_root}")
    validated_sample = validate_sample_run(
        bundle=bundle,
        sample_run=sample_run,
        model_id=model_id,
        arm=arm,
        prompt_context=prompt_context,
    )

    manifest_path = evaluation_run / "manifest.json"
    results_path = evaluation_run / "tpu_results.jsonl"
    summary_path = evaluation_run / "summary.json"
    if not all(path.is_file() for path in (manifest_path, results_path, summary_path)):
        raise EvaluationError(f"EVALUATION_RUN_INCOMPLETE: {evaluation_run}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prior_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    fingerprint = manifest.get("fingerprint")
    if manifest.get("status") != "complete" or not isinstance(fingerprint, dict):
        raise EvaluationError(f"EVALUATION_RUN_NOT_COMPLETE: {evaluation_run}")
    if not isinstance(prior_summary, dict):
        raise EvaluationError(f"EVALUATION_SUMMARY_INVALID: {summary_path}")
    required_fingerprint = {
        "contract_sha256": bundle.sha256,
        "jaxbench_revision": source_by_id(bundle, "jaxbench")["revision"],
        "model_id": model_id,
        "arm": arm,
        "prompt_context": prompt_context.value,
        "sample_fingerprint_sha256": validated_sample.fingerprint_sha256,
    }
    for key, expected in required_fingerprint.items():
        if fingerprint.get(key) != expected:
            raise EvaluationError(
                "EVALUATION_FINGERPRINT_MISMATCH: "
                f"{key}: expected={expected!r} observed={fingerprint.get(key)!r}"
            )

    rows = load_jsonl(results_path)
    rescored_rows = _rescore_result_rows(
        candidates=validated_sample.candidates,
        rows=rows,
        jaxbench_root=jaxbench_root,
        prompt_context=prompt_context,
        headline_speedup_threshold=bundle.eval_policy["timing"][
            "headline_speedup_threshold"
        ],
        require_lowering_evidence=bundle.eval_policy["authenticity"][
            "require_empirical_tpu_lowering"
        ],
    )
    corrected_summary = _oracle_summary(
        bundle=bundle,
        candidates=validated_sample.candidates,
        rows=rescored_rows,
    )
    prior_rows = {row["sample_id"]: row for row in rows}
    changed_samples = [
        {
            "sample_id": row["sample_id"],
            "prior_authentic": prior_rows[row["sample_id"]]
            .get("inspection", {})
            .get("authentic"),
            "corrected_authentic": row["inspection"]["authentic"],
            "prior_pallas_credited": prior_rows[row["sample_id"]].get(
                "pallas_credited"
            ),
            "corrected_pallas_credited": row["pallas_credited"],
            "reasons": row["no_credit_reasons"],
        }
        for row in rescored_rows
        if (
            prior_rows[row["sample_id"]].get("inspection", {}).get("authentic")
            != row.get("inspection", {}).get("authentic")
            or prior_rows[row["sample_id"]].get("pallas_credited")
            != row.get("pallas_credited")
            or prior_rows[row["sample_id"]].get("headline_credited")
            != row.get("headline_credited")
        )
    ]
    audit = {
        "schema_version": 1,
        "kind": "evaluation_credit_audit",
        "contract_sha256": bundle.sha256,
        "auditor": {
            "opjax_revision": git_revision(repo_root),
            "opjax_tracked_dirty": False,
            "jaxbench_revision": source_by_id(bundle, "jaxbench")["revision"],
            "chex": _package_version("chex"),
            "jax": _package_version("jax"),
        },
        "source_artifacts": {
            "sample_manifest_sha256": _sha256_file(sample_run / "manifest.json"),
            "evaluation_manifest_sha256": _sha256_file(manifest_path),
            "tpu_results_sha256": _sha256_file(results_path),
            "prior_summary_sha256": _sha256_file(summary_path),
        },
        "prior_credit": {
            "n_authentic_emissions": prior_summary.get(
                "n_authentic_emissions"
            ),
            "n_pallas_credited": sum(
                row.get("pallas_credited") is True for row in rows
            ),
            "n_headline_credited": sum(
                row.get("headline_credited") is True for row in rows
            ),
        },
        "corrected_summary": corrected_summary,
        "changed_samples": changed_samples,
    }
    _write_json(evaluation_run / "credit_audit.json", audit)
    return audit


def audit_lowering_evidence(
    *,
    bundle: ContractBundle,
    repo_root: Path,
    jaxbench_root: Path,
    sample_run: Path,
    evaluation_run: Path,
    sample_id: str,
    calibration_root: Path,
    candidate_root: Path,
    output: Path,
    model_id: str,
    arm: str,
    prompt_context: PromptContext,
) -> dict[str, Any]:
    if _git_tracked_dirty(repo_root):
        raise EvaluationError(f"OPJAX_TRACKED_DIRTY: {repo_root}")
    sample_manifest_path = sample_run / "manifest.json"
    sample_rows_path = sample_run / "samples.jsonl"
    if not sample_manifest_path.is_file() or not sample_rows_path.is_file():
        raise EvaluationError(f"SAMPLE_RUN_INCOMPLETE: {sample_run}")
    sample_manifest = json.loads(sample_manifest_path.read_text(encoding="utf-8"))
    sample_fingerprint = sample_manifest.get("fingerprint")
    if (
        sample_manifest.get("status") != "sampled"
        or not isinstance(sample_fingerprint, dict)
        or sample_fingerprint.get("arm") != arm
        or sample_fingerprint.get("prompt_context") != prompt_context.value
    ):
        raise EvaluationError("LOWERING_AUDIT_SAMPLE_LINEAGE_INVALID")
    expected_model = (
        bundle.experiment["base_model"]
        if arm == "A"
        else sample_fingerprint.get("model_path")
    )
    if model_id != expected_model:
        raise EvaluationError("LOWERING_AUDIT_MODEL_MISMATCH")
    matching_samples = [
        row
        for row in load_jsonl(sample_rows_path)
        if row.get("sample_id") == sample_id
    ]
    if len(matching_samples) != 1:
        raise EvaluationError(f"SAMPLE_ID_NOT_FOUND: {sample_id}")
    sample = matching_samples[0]
    workload = sample.get("workload")
    seed = sample.get("seed")
    kernel_path = sample.get("kernel_path")
    if (
        not isinstance(workload, str)
        or not isinstance(seed, int)
        or isinstance(seed, bool)
        or not isinstance(kernel_path, str)
    ):
        raise EvaluationError(f"SAMPLE_ROW_INVALID: {sample_id}")
    kernel = (sample_run / kernel_path).resolve()
    if not kernel.is_relative_to(sample_run.resolve()) or not kernel.is_file():
        raise EvaluationError(f"SAMPLE_KERNEL_MISSING: {sample_id}")
    candidate = SampleCandidate(
        sample_id=sample_id,
        workload=workload,
        seed=seed,
        kernel=kernel,
        sample=sample,
    )
    rows = load_jsonl(evaluation_run / "tpu_results.jsonl")
    matching_rows = [row for row in rows if row.get("sample_id") == sample_id]
    if len(matching_rows) != 1:
        raise EvaluationError(f"RESULT_SAMPLE_ID_NOT_FOUND: {sample_id}")
    row = matching_rows[0]
    kernel_sha256 = _sha256_file(candidate.kernel)
    if (
        row.get("kernel_sha256") != kernel_sha256
        or row.get("compiled") is not True
        or row.get("correct") is not True
    ):
        raise EvaluationError(f"LOWERING_AUDIT_ROW_INELIGIBLE: {sample_id}")
    inspection = inspect_pallas_source(
        candidate.kernel.read_text(encoding="utf-8")
    )
    if not inspection.authentic:
        raise EvaluationError(
            f"LOWERING_AUDIT_STATIC_REJECTION: {sample_id}: {inspection.reasons}"
        )
    try:
        verdict = validate_candidate_evidence(
            calibration_root=calibration_root,
            candidate_root=candidate_root,
            expected_kernel_sha256=kernel_sha256,
            expected_runtime=bundle.eval_policy["runtime"],
        )
    except LoweringEvidenceError as exc:
        raise EvaluationError(str(exc)) from exc
    if not verdict.verified:
        raise EvaluationError(
            f"LOWERING_EVIDENCE_NOT_VERIFIED: {sample_id}: {verdict.reasons}"
        )
    manifest_path = evaluation_run / "manifest.json"
    summary_path = evaluation_run / "summary.json"
    required_paths = (
        sample_run / "manifest.json",
        manifest_path,
        evaluation_run / "tpu_results.jsonl",
        summary_path,
    )
    if not all(path.is_file() for path in required_paths):
        raise EvaluationError(f"EVALUATION_RUN_INCOMPLETE: {evaluation_run}")
    evaluation_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fingerprint = evaluation_manifest.get("fingerprint")
    if (
        evaluation_manifest.get("status") != "complete"
        or not isinstance(fingerprint, dict)
        or fingerprint.get("sample_fingerprint_sha256")
        != sample_fingerprint.get("sha256")
        or fingerprint.get("kernel_sha256", {}).get(sample_id) != kernel_sha256
    ):
        raise EvaluationError("LOWERING_AUDIT_EVALUATION_LINEAGE_INVALID")
    audit = {
        "schema_version": 1,
        "kind": "pallas_lowering_evidence_audit",
        "checked_at": _utc_now(),
        "source_contract_sha256": fingerprint.get("contract_sha256"),
        "auditor_contract_sha256": bundle.sha256,
        "sample_id": sample_id,
        "workload": candidate.workload,
        "seed": candidate.seed,
        "kernel_sha256": kernel_sha256,
        "static_inspection": asdict(inspection),
        "lowering_evidence": asdict(verdict),
        "raw_observation": {
            "compiled": row["compiled"],
            "correct": row["correct"],
            "speedup": row.get("speedup"),
            "timing_stable": row.get("timing", {}).get("stable"),
        },
        "auditor": {
            "opjax_revision": git_revision(repo_root),
            "jaxbench_revision": verify_source_checkout(
                bundle,
                "jaxbench",
                jaxbench_root,
            ),
        },
        "source_artifacts": {
            "sample_manifest_sha256": _sha256_file(required_paths[0]),
            "evaluation_manifest_sha256": _sha256_file(required_paths[1]),
            "tpu_results_sha256": _sha256_file(required_paths[2]),
            "evaluation_summary_sha256": _sha256_file(required_paths[3]),
        },
    }
    _write_json(output, audit)
    return audit


def evaluate_kernels(
    *,
    bundle: ContractBundle,
    repo_root: Path,
    jaxbench_root: Path,
    sample_run: Path,
    lowering_calibration: Path,
    out_dir: Path,
    model_id: str,
    arm: str,
    prompt_context: PromptContext,
    resume: bool,
    dry_run: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    verify_source_checkout(bundle, "jaxbench", jaxbench_root)
    _assert_public_workloads_match(bundle, jaxbench_root)
    validated_sample = validate_sample_run(
        bundle=bundle,
        sample_run=sample_run,
        model_id=model_id,
        arm=arm,
        prompt_context=prompt_context,
    )
    candidates = validated_sample.candidates
    try:
        validate_calibration(
            lowering_calibration,
            expected_runtime=bundle.eval_policy["runtime"],
        )
    except LoweringEvidenceError as exc:
        raise EvaluationError(str(exc)) from exc
    lowering_calibration_sha256 = _sha256_file(
        lowering_calibration / "calibration.json"
    )
    runtime_hardware = None if dry_run else probe_runtime_hardware()
    if runtime_hardware is not None:
        _assert_tpu_runtime(runtime_hardware, bundle.experiment["target"])
    fingerprint = environment_fingerprint(
        bundle=bundle,
        repo_root=repo_root,
        jaxbench_root=jaxbench_root,
        candidates=candidates,
        model_id=model_id,
        arm=arm,
        prompt_context=prompt_context,
        runtime_hardware=runtime_hardware,
        sample_fingerprint_sha256=validated_sample.fingerprint_sha256,
        lowering_calibration_sha256=lowering_calibration_sha256,
    )
    if not dry_run and fingerprint["opjax_tracked_dirty"]:
        raise EvaluationError(f"OPJAX_TRACKED_DIRTY: {repo_root}")
    if not dry_run and fingerprint["jaxbench_tracked_dirty"]:
        raise EvaluationError(f"JAXBENCH_TRACKED_DIRTY: {jaxbench_root}")
    if not dry_run:
        _assert_evaluation_runtime(
            fingerprint,
            bundle.eval_policy["runtime"],
        )
    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "n_samples": len(candidates),
            "fingerprint": fingerprint,
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_or_create_manifest(
        out_dir=out_dir,
        fingerprint=fingerprint,
        resume=resume,
    )
    results_path = out_dir / "tpu_results.jsonl"
    rows = load_jsonl(results_path)
    completed: set[str] = set()
    valid_ids = {candidate.sample_id for candidate in candidates}
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in valid_ids:
            raise EvaluationError(f"RESULT_SAMPLE_ID_INVALID: {sample_id!r}")
        if sample_id in completed:
            raise EvaluationError(f"RESULT_SAMPLE_ID_DUPLICATE: {sample_id}")
        completed.add(sample_id)
    for candidate in candidates:
        if candidate.sample_id in completed:
            print(
                f"PALLAS_EVAL_RESUME sample_id={candidate.sample_id} status=skipped",
                flush=True,
            )
            continue
        print(
            "PALLAS_EVAL_START "
            f"sample_id={candidate.sample_id} workload={candidate.workload} "
            f"seed={candidate.seed}",
            flush=True,
        )
        row = _evaluate_workload(
            bundle=bundle,
            jaxbench_root=jaxbench_root,
            candidate=candidate,
            prompt_context=prompt_context,
            lowering_calibration=lowering_calibration,
            lowering_evidence_root=out_dir / "lowering",
            timeout_seconds=timeout_seconds,
        )
        rows.append(row)
        _write_jsonl(results_path, rows)
        print(
            "PALLAS_EVAL_DONE "
            f"sample_id={candidate.sample_id} compiled={row['compiled']} "
            f"correct={row['correct']} pallas={row['inspection']['authentic']} "
            f"headline={row['headline_credited']}",
            flush=True,
        )
    summary = _oracle_summary(
        bundle=bundle,
        candidates=candidates,
        rows=rows,
    )
    summary.update(
        {
            "schema_version": 2,
            "checked_at": _utc_now(),
            "contract_sha256": bundle.sha256,
            "fingerprint": fingerprint,
            "private_evaluation_ready": bundle.splits["private_evaluation"][
                "generalization_claim_ready"
            ],
        }
    )
    _write_json(out_dir / "summary.json", summary)
    manifest.update({"status": "complete", "completed_at": _utc_now()})
    _write_json(out_dir / "manifest.json", manifest)
    return summary


def assert_checkout_ready(bundle: ContractBundle, jaxbench_root: Path) -> dict[str, Any]:
    try:
        revision = verify_source_checkout(bundle, "jaxbench", jaxbench_root)
    except ContractError as exc:
        raise EvaluationError(str(exc)) from exc
    _assert_public_workloads_match(bundle, jaxbench_root)
    return {
        "ok": True,
        "jaxbench_root": str(jaxbench_root.resolve()),
        "jaxbench_revision": revision,
        "n_public_workloads": len(bundle.splits["public_evaluation"]["task_ids"]),
    }
