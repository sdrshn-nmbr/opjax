"""Execute one hidden Pallas environment task on a real TPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import chex
import jax

from opjax.pallas.corpus import _generate_inputs, _semantic_oracle
from opjax.pallas.scoring import inspect_pallas_source


class EnvironmentRunnerError(RuntimeError):
    pass


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


def evaluate_task(*, task: dict[str, Any], kernel_path: Path) -> dict[str, Any]:
    hardware = _hardware()
    source = kernel_path.read_text(encoding="utf-8")
    inspection = inspect_pallas_source(source)
    if not inspection.authentic:
        return {
            "passed": False,
            "stage": "pallas_api",
            "error": ",".join(inspection.reasons),
            "hardware": hardware,
            "kernel_sha256": _sha256_file(kernel_path),
        }
    try:
        module = _load_module(kernel_path)
        workload = getattr(module, "workload")
        operation = task["operation"]
        if operation == "row_sum":
            operation = "sum"
        inputs = _generate_inputs(
            task["input_shapes"],
            task.get("input_dtypes"),
            task.get("input_ranges"),
            seed=int(task.get("seed", 0)),
        )
        expected = _semantic_oracle(operation, *inputs)
        lowered = jax.jit(workload).lower(*inputs)
        compiled = lowered.compile()
    except Exception as exc:
        return {
            "passed": False,
            "stage": "tpu_compile",
            "error": f"{type(exc).__name__}: {exc}",
            "hardware": hardware,
            "kernel_sha256": _sha256_file(kernel_path),
        }
    try:
        actual = compiled(*inputs)
        jax.block_until_ready(actual)
        tolerance = task.get("correctness_tolerance", {"rtol": 1e-3, "atol": 1e-3})
        chex.assert_trees_all_close(
            actual,
            expected,
            rtol=float(tolerance["rtol"]),
            atol=float(tolerance["atol"]),
        )
    except Exception as exc:
        return {
            "passed": False,
            "stage": "full_shape_correctness",
            "error": f"{type(exc).__name__}: {exc}",
            "hardware": hardware,
            "kernel_sha256": _sha256_file(kernel_path),
        }
    executable = compiled.as_text()
    if "tpu_custom_call" not in executable:
        return {
            "passed": False,
            "stage": "normal_lowering",
            "error": "TPU_CUSTOM_CALL_MISSING",
            "hardware": hardware,
            "kernel_sha256": _sha256_file(kernel_path),
        }
    return {
        "passed": True,
        "stage": "verified",
        "error": None,
        "hardware": hardware,
        "kernel_sha256": _sha256_file(kernel_path),
        "executable_tpu_custom_call": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-environment-runner")
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--kernel", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        task = json.loads(args.task.read_text(encoding="utf-8"))
        result = evaluate_task(task=task, kernel_path=args.kernel)
    except Exception as exc:
        result = {"passed": False, "stage": "runner", "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
