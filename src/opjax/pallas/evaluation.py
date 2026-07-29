"""Pinned, isolated, resumable JAXBench evaluation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from opjax.pallas.contracts import (
    ContractBundle,
    ContractError,
    git_revision,
    verify_source_checkout,
)
from opjax.pallas.scoring import (
    PromptContext,
    diagnostic_reward,
    judge,
    summarise,
    timing_evidence,
)


class EvaluationError(RuntimeError):
    """The evaluation cannot produce comparable evidence."""


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


def kernel_set_fingerprint(kernels: list[Path]) -> dict[str, str]:
    return {path.stem: _sha256_file(path) for path in kernels}


def environment_fingerprint(
    *,
    bundle: ContractBundle,
    repo_root: Path,
    jaxbench_root: Path,
    kernels: list[Path],
    model_id: str,
    arm: str,
    prompt_context: PromptContext,
    runtime_hardware: dict[str, Any] | None,
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
        "kernel_sha256": kernel_set_fingerprint(kernels),
        "model_id": model_id,
        "arm": arm,
        "prompt_context": prompt_context.value,
        "prompt_contract_sha256": _sha256_bytes(
            json.dumps(prompt_contract, sort_keys=True, separators=(",", ":")).encode()
        ),
        "target": bundle.experiment["target"],
        "runtime_hardware": runtime_hardware,
        "packages": {
            name: _package_version(name)
            for name in ("jax", "jaxlib", "libtpu", "tinker", "tinker-cookbook")
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def probe_runtime_hardware(timeout_seconds: float = 60) -> dict[str, Any]:
    source = (
        "import json, jax; "
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
    if observed.get("platforms") != ["tpu"]:
        raise EvaluationError(
            f"HARDWARE_TARGET_MISMATCH: expected=tpu observed={observed.get('platforms')}"
        )
    hardware = str(target["hardware"]).lower()
    kinds = " ".join(str(kind).lower() for kind in observed.get("device_kinds", []))
    expected_generation = hardware.removeprefix("v").removesuffix("e")
    if expected_generation not in kinds:
        raise EvaluationError(
            "HARDWARE_TARGET_MISMATCH: "
            f"expected={target['hardware']} observed={observed.get('device_kinds')}"
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
        "schema_version": 1,
        "created_at": _utc_now(),
        "status": "running",
        "fingerprint": fingerprint,
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
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=environment,
            timeout=timeout_seconds,
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
        }
    parsed = _parse_json_output(process.stdout)
    if parsed is None:
        parsed = {
            "workload": workload,
            "status": "error",
            "error_code": "JAXBENCH_JSON_MISSING",
            "error": (process.stderr or process.stdout or "no output")[-1000:],
        }
    return {
        "returncode": process.returncode,
        "result": parsed,
        "stdout_tail": process.stdout[-2000:],
        "stderr_tail": process.stderr[-2000:],
    }


def _result_correct(value: dict[str, Any]) -> bool:
    correctness = value.get("correctness")
    if isinstance(correctness, dict):
        return correctness.get("correct") is True
    return value.get("correct") is True


def _evaluate_workload(
    *,
    bundle: ContractBundle,
    jaxbench_root: Path,
    kernel: Path,
    prompt_context: PromptContext,
    timeout_seconds: float,
) -> dict[str, Any]:
    workload = kernel.stem
    policy = bundle.eval_policy
    timing_policy = policy["timing"]
    raw_runs = [
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
    results = [run["result"] for run in raw_runs]
    compiled = all(
        run["returncode"] == 0
        and run["result"].get("status") not in {"error", "compile_error"}
        and not run["result"].get("error")
        for run in raw_runs
    )
    correct = compiled and all(_result_correct(result) for result in results)
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
    source = kernel.read_text(encoding="utf-8")
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
    )
    return {
        "schema_version": 1,
        "workload": workload,
        "checked_at": _utc_now(),
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
        "credited": verdict.credited,
        "pallas_credited": verdict.pallas_credited,
        "headline_credited": verdict.headline_credited,
        "diagnostic_reward": diagnostic_reward(verdict),
        "no_credit_reasons": list(verdict.no_credit_reasons),
        "raw_runs": raw_runs,
    }


def evaluate_kernels(
    *,
    bundle: ContractBundle,
    repo_root: Path,
    jaxbench_root: Path,
    kernels_dir: Path,
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
    kernels = sorted(
        path
        for path in kernels_dir.glob("*.py")
        if path.is_file() and not path.name.startswith("._")
    )
    if not kernels:
        raise EvaluationError(f"KERNELS_EMPTY: {kernels_dir}")
    public_ids = set(bundle.splits["public_evaluation"]["task_ids"])
    unknown = sorted(path.stem for path in kernels if path.stem not in public_ids)
    if unknown:
        raise EvaluationError(f"KERNELS_NOT_PUBLIC_JAXBENCH: {unknown}")
    runtime_hardware = None if dry_run else probe_runtime_hardware()
    if runtime_hardware is not None:
        _assert_tpu_runtime(runtime_hardware, bundle.experiment["target"])
    fingerprint = environment_fingerprint(
        bundle=bundle,
        repo_root=repo_root,
        jaxbench_root=jaxbench_root,
        kernels=kernels,
        model_id=model_id,
        arm=arm,
        prompt_context=prompt_context,
        runtime_hardware=runtime_hardware,
    )
    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "n_kernels": len(kernels),
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
    completed = {row["workload"] for row in rows}
    for kernel in kernels:
        if kernel.stem in completed:
            print(f"PALLAS_EVAL_RESUME workload={kernel.stem} status=skipped", flush=True)
            continue
        print(f"PALLAS_EVAL_WORKLOAD workload={kernel.stem} status=started", flush=True)
        row = _evaluate_workload(
            bundle=bundle,
            jaxbench_root=jaxbench_root,
            kernel=kernel,
            prompt_context=prompt_context,
            timeout_seconds=timeout_seconds,
        )
        rows.append(row)
        _write_jsonl(results_path, rows)
        print(
            "PALLAS_EVAL_WORKLOAD "
            f"workload={kernel.stem} compiled={row['compiled']} "
            f"correct={row['correct']} pallas={row['inspection']['authentic']} "
            f"headline={row['headline_credited']}",
            flush=True,
        )
    verdicts = [
        judge(
            workload=row["workload"],
            candidate_src=(kernels_dir / f"{row['workload']}.py").read_text(encoding="utf-8"),
            baseline_src=(
                jaxbench_root
                / "JAXBench"
                / "benchmark"
                / row["workload"]
                / "baseline.py"
            ).read_text(encoding="utf-8"),
            compiled=bool(row["compiled"]),
            correct=bool(row["correct"]),
            prompt_context=row["prompt_context"],
            speedup=row.get("speedup"),
            timing_stable=row.get("timing", {}).get("stable"),
            headline_speedup_threshold=bundle.eval_policy["timing"][
                "headline_speedup_threshold"
            ],
        )
        for row in rows
    ]
    summary = summarise(verdicts)
    summary.update(
        {
            "schema_version": 1,
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
