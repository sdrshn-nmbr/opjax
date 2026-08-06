"""Local orchestration for the frozen Laguna XS 2.1 Pallas baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from opjax.pallas.g42_harness import canonical_sha256, file_sha256, load_task_package
from opjax.pallas.g43_corpus import validate_benchmark_release
from opjax.pallas.sglang_agent import run_sglang_agent
from opjax.remote.laguna_sglang import (
    MODEL_ID,
    MODEL_REVISION,
    PRECISION,
    SGLANG_REVISION,
    LagunaEngine,
    app,
)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def summarize_baseline(*, verifier_root: Path, out_path: Path) -> dict[str, Any]:
    manifest = json.loads((verifier_root / "manifest.json").read_text(encoding="utf-8"))
    verification = json.loads((verifier_root / "verification.json").read_text(encoding="utf-8"))
    manifest_payload = dict(manifest)
    manifest_sha = manifest_payload.pop("release_sha256")
    verification_payload = dict(verification)
    verification_sha = verification_payload.pop("release_sha256")
    if canonical_sha256(manifest_payload) != manifest_sha:
        raise RuntimeError("LAGUNA_BASELINE_MANIFEST_HASH_MISMATCH")
    if canonical_sha256(verification_payload) != verification_sha:
        raise RuntimeError("LAGUNA_BASELINE_VERIFICATION_HASH_MISMATCH")
    if verification["input_release_sha256"] != manifest_sha:
        raise RuntimeError("LAGUNA_BASELINE_RELEASE_MISMATCH")
    if verification["counts"]["infrastructure_failures"] != 0:
        raise RuntimeError("LAGUNA_BASELINE_INFRASTRUCTURE_FAILURES_PRESENT")
    verification_records = {
        record["unit_id"]: record for record in verification["records"]
    }
    if set(verification_records) != {
        record["unit_id"] for record in manifest["records"]
    }:
        raise RuntimeError("LAGUNA_BASELINE_VERIFICATION_UNITS_MISMATCH")
    records = []
    for record in manifest["records"]:
        reward_path = verifier_root / "results" / record["unit_id"] / "reward.json"
        reward = json.loads(reward_path.read_text(encoding="utf-8"))
        verified_record = verification_records[record["unit_id"]]
        reward_sha256 = file_sha256(reward_path)
        if verified_record["artifacts"]["reward.json"] != reward_sha256:
            raise RuntimeError(
                f"LAGUNA_BASELINE_REWARD_HASH_MISMATCH: {record['unit_id']}"
            )
        if verified_record["reward"] != reward["reward"]:
            raise RuntimeError(
                f"LAGUNA_BASELINE_REWARD_VALUE_MISMATCH: {record['unit_id']}"
            )
        records.append(
            {
                "task_id": record["task_id"],
                "family": record["family"],
                "reward": reward["reward"],
                "failure_stage": reward.get("failure_stage"),
                "patch_sha256": record["patch_sha256"],
                "reward_sha256": reward_sha256,
            }
        )
    failure_stages: dict[str, int] = {}
    for record in records:
        stage = record["failure_stage"] or "none"
        failure_stages[stage] = failure_stages.get(stage, 0) + 1
    result = {
        "schema_version": 1,
        "kind": "pallas_laguna_xs_21_baseline_result",
        "model": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "runtime": "sglang",
            "runtime_revision": SGLANG_REVISION,
            "precision": PRECISION,
        },
        "sampling": {
            "turn_limit": 3,
            "seed": 0,
            "max_tokens": 8192,
            "temperature": 0.2,
            "top_p": 0.95,
            "thinking": True,
        },
        "observed_h200_residency_mib": {
            "source": "modal_canary_nvidia_smi_after_cuda_graph_capture",
            "total": 143771,
            "used_after_cuda_graph_capture": 106044,
            "remaining": 37727,
        },
        "counts": {
            "tasks": len(records),
            "profile_verified": sum(record["reward"] == 1 for record in records),
            "candidate_failures": sum(record["reward"] == 0 for record in records),
            "infrastructure_failures": sum(record["reward"] == -1 for record in records),
            "nonempty_patches": sum(
                record["patch_sha256"]
                != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                for record in records
            ),
        },
        "failure_stages": dict(sorted(failure_stages.items())),
        "records": records,
        "verifier_input_release_sha256": manifest_sha,
        "verification_release_sha256": verification_sha,
    }
    result["result_sha256"] = canonical_sha256(result)
    _write(out_path, result)
    return result


@app.local_entrypoint()
def canary() -> None:
    engine = LagunaEngine()
    print(json.dumps(engine.smoke.remote(), indent=2, sort_keys=True))
    response = engine.generate.remote(
        [{"role": "user", "content": "Return exactly: READY"}],
        {"max_new_tokens": 32, "temperature": 0.0, "top_p": 1.0, "sampling_seed": 0},
    )
    print(json.dumps(response, indent=2, sort_keys=True))


@app.local_entrypoint()
def baseline(
    benchmark_root: str = "data/pallas/runs/g43-benchmark-release",
    out_dir: str = "data/pallas/runs/laguna-xs-21-baseline-samples",
    limit: int = 0,
) -> None:
    benchmark_path = Path(benchmark_root).resolve()
    output_path = Path(out_dir).resolve()
    if output_path.exists():
        raise RuntimeError(f"LAGUNA_BASELINE_OUTPUT_EXISTS: {output_path}")
    validation = validate_benchmark_release(benchmark_path)
    benchmark = json.loads((benchmark_path / "manifest.json").read_text(encoding="utf-8"))
    tasks = [load_task_package(benchmark_path / relative) for relative in benchmark["tasks"]]
    if limit > 0:
        tasks = tasks[:limit]
    engine = LagunaEngine()

    def generate(messages: list[dict[str, str]], sampling: dict[str, Any]) -> dict[str, Any]:
        return engine.generate.remote(messages, sampling)

    records = []
    for index, task in enumerate(tasks, start=1):
        run_id = f"laguna-xs-21-base--{task.task_id}--seed-0"
        run_root = output_path / "runs" / run_id
        run_sglang_agent(
            task_dir=task.root,
            output_dir=run_root,
            generate=generate,
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            runtime_revision=SGLANG_REVISION,
            precision=PRECISION,
            seed=0,
            max_tokens=8192,
            temperature=0.2,
            top_p=0.95,
            turn_limit=3,
            snapshot_turns=(3,),
        )
        records.append(
            {
                "model_id": "laguna-xs-21-base",
                "checkpoint": MODEL_ID,
                "group": "external_baseline",
                "trajectory_count": None,
                "training_seed": None,
                "task_id": task.task_id,
                "task_sha256": task.task_sha256,
                "family": task.family,
                "run_path": f"runs/{run_id}",
                "trajectory_sha256": file_sha256(run_root / "trajectory.json"),
            }
        )
        print(f"LAGUNA_BASELINE_SAMPLE completed={index}/{len(tasks)} task={task.task_id}", flush=True)
    manifest = {
        "schema_version": 1,
        "kind": "pallas_g43_sample_matrix",
        "evaluation_config_sha256": None,
        "benchmark_release_sha256": validation["release_sha256"],
        "model": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "runtime": "sglang",
            "runtime_revision": SGLANG_REVISION,
            "precision": PRECISION,
        },
        "sampling": {
            "turn_limit": 3,
            "seed": 0,
            "max_tokens": 8192,
            "temperature": 0.2,
            "top_p": 0.95,
            "thinking": True,
        },
        "counts": {"runs": len(records), "snapshots": len(records)},
        "records": records,
    }
    manifest["release_sha256"] = canonical_sha256(manifest)
    _write(output_path / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
