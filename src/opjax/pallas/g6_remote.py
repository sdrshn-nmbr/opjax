"""Run an immutable Gate 6 verifier batch on one TPU worker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from opjax.pallas.g42_experiment import _worker_health
from opjax.pallas.environment import verify_static
from opjax.pallas.g42_harness import (
    MANDATORY_STAGES,
    canonical_sha256,
    file_sha256,
    write_verifier_artifacts,
)
from opjax.pallas.g42_verifier import run_fresh_verifier


class G6RemoteError(RuntimeError):
    """A remote verifier batch is incomplete or corrupt."""


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise G6RemoteError(f"G6_REMOTE_JSON_OBJECT_REQUIRED: {path}")
    return value


def _static_preflight(*, task: dict[str, Any], kernel_path: Path, output: Path) -> dict[str, Any] | None:
    source = kernel_path.read_text(encoding="utf-8")
    verdict = verify_static(f"```python\n{source}\n```")
    if verdict.passed:
        return None
    stage = "artifact_contract" if verdict.stage == "output_contract" else verdict.stage
    stages = {name: False for name in MANDATORY_STAGES}
    if stage != "artifact_contract":
        stages["artifact_contract"] = True
    result = {
        "passed": False,
        "stage": stage,
        "error": verdict.feedback,
        "hardware": {"target": "tpu", "execution": "static_preflight"},
        "kernel_sha256": file_sha256(kernel_path),
        "stages": stages,
        "authentic": False,
        "correct": False,
        "normal_lowered": False,
        "infrastructure_error": False,
        "preflight_evidence": verdict.evidence,
    }
    write_verifier_artifacts(
        result=result,
        output_dir=output,
        task_id=task["task_id"],
        kernel_sha256=result["kernel_sha256"],
    )
    return result


def verify_batch(*, batch_root: Path, timeout_seconds: int) -> dict[str, Any]:
    manifest = _load(batch_root / "manifest.json")
    payload = dict(manifest)
    expected_sha = payload.pop("release_sha256", None)
    records = manifest.get("records")
    if (
        manifest.get("kind") != "pallas_g6_verifier_batch"
        or canonical_sha256(payload) != expected_sha
        or not isinstance(records, list)
        or manifest.get("counts", {}).get("units") != len(records)
    ):
        raise G6RemoteError("G6_REMOTE_BATCH_INVALID")
    results = []
    recovery_events = []
    for index, record in enumerate(records, start=1):
        unit = batch_root / "units" / record["unit_id"]
        task_path = unit / "task.json"
        kernel_path = unit / "kernel.py"
        if (
            file_sha256(task_path) != record["task_sha256"]
            or file_sha256(kernel_path) != record["kernel_sha256"]
        ):
            raise G6RemoteError(f"G6_REMOTE_UNIT_HASH_MISMATCH: {record['unit_id']}")
        output = batch_root / "results" / record["unit_id"]
        run_log = output / "run.log"
        if run_log.is_file():
            result = _load(run_log)
        else:
            result = _static_preflight(
                task=_load(task_path), kernel_path=kernel_path, output=output
            )
            if result is None:
                result = run_fresh_verifier(
                    task_path=task_path,
                    kernel_path=kernel_path,
                    output_dir=output,
                    timeout_seconds=timeout_seconds,
                )["result"]
        if result.get("worker_recovery_required") is True:
            health = _worker_health(None)
            recovery_events.append({"after_unit": record["unit_id"], **health})
            if not health["healthy"]:
                (batch_root / "recovery-events.json").write_text(
                    json.dumps(recovery_events, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                raise G6RemoteError(f"G6_REMOTE_WORKER_QUARANTINED: {record['unit_id']}")
        results.append(
            {
                "unit_id": record["unit_id"],
                "kernel_sha256": record["kernel_sha256"],
                "result": result,
                "run_log_sha256": file_sha256(run_log),
                "reward_sha256": file_sha256(output / "reward.json"),
            }
        )
        print(
            f"G6_REMOTE_VERIFY completed={index}/{len(records)} "
            f"unit={record['unit_id']} stage={result.get('stage')}",
            flush=True,
        )
    result_manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pallas_g6_verifier_batch_results",
        "input_release_sha256": manifest["release_sha256"],
        "counts": {
            "units": len(results),
            "verified": sum(row["result"].get("passed") is True for row in results),
            "infrastructure_failures": sum(
                row["result"].get("infrastructure_error") is True for row in results
            ),
            "recovery_probes": len(recovery_events),
        },
        "records": results,
        "recovery_events": recovery_events,
    }
    result_manifest["release_sha256"] = canonical_sha256(result_manifest)
    (batch_root / "results.json").write_text(
        json.dumps(result_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-g6-remote")
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args(argv)
    try:
        result = verify_batch(
            batch_root=args.batch_root.resolve(), timeout_seconds=args.timeout_seconds
        )
    except (G6RemoteError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"G6_REMOTE_ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["counts"]["infrastructure_failures"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
