"""Execute one hidden Pallas environment task on a real TPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import chex
import jax

from opjax.pallas.corpus import _generate_inputs, _semantic_oracle
from opjax.pallas.environment import verify_static
from opjax.pallas.lowering import capture_lowering_case
from opjax.pallas.scoring import inspect_pallas_source


class EnvironmentRunnerError(RuntimeError):
    pass


def _is_runtime_safety_failure(error: BaseException) -> bool:
    detail = f"{type(error).__name__}: {error}".lower()
    return any(
        marker in detail
        for marker in (
            "core halted",
            "device halted",
            "boundscheck",
            "out of bounds",
            "dma.hbm_to_vmem",
            "dma.vmem_to_hbm",
            "sigabrt",
            "segmentation fault",
        )
    )


def _failed(
    *,
    stage: str,
    error: str,
    hardware: dict[str, Any],
    kernel_sha256: str,
    stages: dict[str, bool],
) -> dict[str, Any]:
    return {
        "passed": False,
        "stage": stage,
        "error": error,
        "hardware": hardware,
        "kernel_sha256": kernel_sha256,
        "stages": stages,
        "authentic": stages.get("pallas_api", False),
        "correct": stages.get("full_shape_correctness", False),
        "normal_lowered": stages.get("normal_lowering", False),
        "infrastructure_error": False,
    }


def _time_compiled(compiled: Any, inputs: tuple[Any, ...], *, warmups: int = 3, iterations: int = 20) -> float:
    for _ in range(warmups):
        jax.block_until_ready(compiled(*inputs))
    started = time.perf_counter()
    for _ in range(iterations):
        jax.block_until_ready(compiled(*inputs))
    return (time.perf_counter() - started) * 1000 / iterations


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("opjax_environment_candidate", path)
    if spec is None or spec.loader is None:
        raise EnvironmentRunnerError("MODULE_LOAD_FAILED")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hardware() -> dict[str, Any]:
    chex.assert_devices_available(1, "tpu", not_less_than=True)
    devices = jax.devices()
    return {
        "platforms": sorted({device.platform for device in devices}),
        "device_kinds": sorted({getattr(device, "device_kind", "unknown") for device in devices}),
        "device_count": len(devices),
        "process_count": jax.process_count(),
        "process_index": jax.process_index(),
    }


def evaluate_task(
    *, task: dict[str, Any], kernel_path: Path, evidence_dir: Path | None = None
) -> dict[str, Any]:
    hardware = _hardware()
    kernel_sha256 = _sha256_file(kernel_path)
    stages = {
        "artifact_contract": True,
        "pallas_api": False,
        "tpu_compile": False,
        "full_shape_correctness": False,
        "normal_lowering": False,
        "runtime_safety": False,
        "profile": False,
    }
    source = kernel_path.read_text(encoding="utf-8")
    static = verify_static(f"```python\n{source}\n```")
    if not static.passed:
        stage = "artifact_contract" if static.stage == "output_contract" else static.stage
        return _failed(
            stage=stage,
            error=static.feedback,
            hardware=hardware,
            kernel_sha256=kernel_sha256,
            stages=stages,
        )
    inspection = inspect_pallas_source(source)
    if not inspection.authentic:
        return _failed(
            stage="pallas_api",
            error=",".join(inspection.reasons),
            hardware=hardware,
            kernel_sha256=kernel_sha256,
            stages=stages,
        )
    stages["pallas_api"] = True
    correctness_seeds = tuple(task.get("correctness_seeds", (0, 1, 2)))
    if correctness_seeds != (0, 1, 2):
        raise EnvironmentRunnerError(f"CORRECTNESS_SEEDS_INVALID: {correctness_seeds}")
    compiled = None
    profile_inputs = None
    profile_expected = None
    tolerance = task.get("correctness_tolerance", {"rtol": 1e-3, "atol": 1e-3})
    try:
        module = _load_module(kernel_path)
        workload = module.workload
        operation = task["operation"]
        if operation == "row_sum":
            operation = "sum"
        seed_results = []
        for seed in correctness_seeds:
            inputs = _generate_inputs(
                task["input_shapes"],
                task.get("input_dtypes"),
                task.get("input_ranges"),
                seed=seed,
            )
            expected = _semantic_oracle(operation, *inputs)
            lowered = jax.jit(workload).lower(*inputs)
            compiled = lowered.compile()
            stages["tpu_compile"] = True
            actual = compiled(*inputs)
            jax.block_until_ready(actual)
            chex.assert_trees_all_close(
                actual,
                expected,
                rtol=float(tolerance["rtol"]),
                atol=float(tolerance["atol"]),
            )
            seed_results.append({"seed": seed, "passed": True})
            if seed == correctness_seeds[0]:
                profile_inputs = inputs
                profile_expected = expected
    except Exception as exc:  # noqa: BLE001 - candidate code can raise any exception
        if _is_runtime_safety_failure(exc):
            stage = "runtime_safety"
        else:
            stage = "full_shape_correctness" if stages["tpu_compile"] else "tpu_compile"
        return _failed(
            stage=stage,
            error=f"{type(exc).__name__}: {exc}",
            hardware=hardware,
            kernel_sha256=kernel_sha256,
            stages=stages,
        )
    stages["full_shape_correctness"] = True
    assert compiled is not None and profile_inputs is not None and profile_expected is not None
    executable = compiled.as_text()
    if "tpu_custom_call" not in executable:
        return _failed(
            stage="normal_lowering",
            error="TPU_CUSTOM_CALL_MISSING",
            hardware=hardware,
            kernel_sha256=kernel_sha256,
            stages=stages,
        )
    stages["normal_lowering"] = True
    stages["runtime_safety"] = True
    if evidence_dir is None:
        return _failed(
            stage="profile",
            error="PROFILE_EVIDENCE_MISSING",
            hardware=hardware,
            kernel_sha256=kernel_sha256,
            stages=stages,
        )
    try:
        profile = capture_lowering_case(
            label="candidate",
            function=workload,
            inputs=profile_inputs,
            out_dir=evidence_dir,
            repetitions=3,
            expected_output=profile_expected,
            rtol=float(tolerance["rtol"]),
            atol=float(tolerance["atol"]),
        )
        baseline = jax.jit(lambda *values: _semantic_oracle(operation, *values)).lower(*profile_inputs).compile()
        candidate_samples = [_time_compiled(compiled, profile_inputs) for _ in range(3)]
        baseline_samples = [_time_compiled(baseline, profile_inputs) for _ in range(3)]
        candidate_median = sorted(candidate_samples)[1]
        baseline_median = sorted(baseline_samples)[1]
        profile["timing"] = {
            "candidate_ms": candidate_samples,
            "baseline_ms": baseline_samples,
            "candidate_median_ms": candidate_median,
            "baseline_median_ms": baseline_median,
            "speedup": baseline_median / candidate_median if candidate_median > 0 else None,
        }
        profile["speedup"] = profile["timing"]["speedup"]
    except Exception as exc:  # noqa: BLE001 - profiler/runtime failures are evidence
        return _failed(
            stage="profile",
            error=f"{type(exc).__name__}: {exc}",
            hardware=hardware,
            kernel_sha256=kernel_sha256,
            stages=stages,
        )
    stages["profile"] = True
    return {
        "passed": True,
        "stage": "verified",
        "error": None,
        "hardware": hardware,
        "kernel_sha256": kernel_sha256,
        "stages": stages,
        "authentic": True,
        "correct": True,
        "normal_lowered": True,
        "infrastructure_error": False,
        "seed_results": seed_results,
        "executable_tpu_custom_call": True,
        "profile": profile,
    }


def evaluate_repair_run(run_dir: Path, evidence_dir: Path | None) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results = []
    for row in rows:
        final_attempt = row["attempts"][-1]
        result = evaluate_task(
            task=row["task"],
            kernel_path=run_dir / final_attempt["kernel_path"],
            evidence_dir=(
                evidence_dir / row["task"]["task_id"]
                if evidence_dir is not None
                else None
            ),
        )
        results.append(
            {
                "task_id": row["task"]["task_id"],
                "attempt": final_attempt["attempt"],
                **result,
            }
        )
    return {
        "passed": all(result["passed"] for result in results),
        "verified": sum(result["passed"] for result in results),
        "task_count": len(results),
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-environment-runner")
    parser.add_argument("--task", type=Path)
    parser.add_argument("--kernel", type=Path)
    parser.add_argument("--repair-run", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.repair_run is not None:
            if args.task is not None or args.kernel is not None:
                raise EnvironmentRunnerError("RUNNER_ARGUMENT_CONFLICT")
            result = evaluate_repair_run(args.repair_run, args.evidence_dir)
        else:
            if args.task is None or args.kernel is None:
                raise EnvironmentRunnerError("TASK_AND_KERNEL_REQUIRED")
            task = json.loads(args.task.read_text(encoding="utf-8"))
            result = evaluate_task(
                task=task,
                kernel_path=args.kernel,
                evidence_dir=args.evidence_dir,
            )
    except Exception as exc:  # noqa: BLE001 - CLI must preserve infrastructure failures
        result = {
            "passed": False,
            "stage": "infrastructure",
            "error": f"{type(exc).__name__}: {exc}",
            "infrastructure_error": True,
        }
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
