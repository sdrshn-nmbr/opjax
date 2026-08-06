"""Build and summarize the matched Gate 6 GRPO evaluation."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

from opjax.pallas.g42_harness import canonical_sha256
from opjax.pallas.g43_corpus import validate_benchmark_release
from opjax.pallas.g43_experiment import (
    _model_summary,
    _paired_summary,
    _tree_hashes,
    prepare_verifier_release,
    verify_release,
)
from opjax.pallas.g6_verifier_backend import (
    RemoteTPUPoolVerifier,
    VerifierBackend,
    VerifierCandidate,
)


class G6ExperimentError(RuntimeError):
    """Gate 6 evaluation evidence is incomplete or causally mismatched."""


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise G6ExperimentError(f"G6_EXPERIMENT_JSON_OBJECT_REQUIRED: {path}")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sampler(manifest: Mapping[str, Any], lane_id: str) -> str:
    sampler = manifest.get("sampler_weights")
    path = sampler.get("path") if isinstance(sampler, Mapping) else None
    if (
        manifest.get("kind") != "pallas_g6_grpo_run"
        or manifest.get("status") != "completed"
        or manifest.get("lane_id") != lane_id
        or not isinstance(path, str)
        or not path.startswith("tinker://")
    ):
        raise G6ExperimentError(f"G6_RUN_MANIFEST_INVALID: {lane_id}")
    return path


def build_evaluation_config(
    *,
    config_path: Path,
    benchmark_root: Path,
    r0_manifest_path: Path,
    r1_manifest_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    config = _load(config_path)
    benchmark = validate_benchmark_release(benchmark_root)
    if benchmark["release_sha256"] != config.get("benchmark_release_sha256"):
        raise G6ExperimentError("G6_BENCHMARK_RELEASE_MISMATCH")
    r0 = _load(r0_manifest_path)
    r1 = _load(r1_manifest_path)
    if (
        r0.get("config_sha256") != r1.get("config_sha256")
        or r0.get("task_release_sha256") != r1.get("task_release_sha256")
        or r0.get("completed_steps") != r1.get("completed_steps")
        or r0.get("total_steps") != r1.get("total_steps")
    ):
        raise G6ExperimentError("G6_MATCHED_TRAINING_CONTRACT_MISMATCH")
    evaluation: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pallas_g6_evaluation_config",
        "experiment_id": config["experiment_id"],
        "benchmark_release_sha256": benchmark["release_sha256"],
        "evaluation_seed": config["evaluation"]["seed"],
        "turn_limit": config["evaluation"]["turn_limit"],
        "snapshot_turns": config["evaluation"]["snapshot_turns"],
        "max_sampling_workers": config["evaluation"]["max_sampling_workers"],
        "models": [
            {
                "model_id": "g6-r0",
                "checkpoint": _sampler(r0, "R0"),
                "group": "grpo",
                "trajectory_count": 512,
                "training_seed": config["rollout"]["sampling_seed"],
                "checkpoint_run_sha256": r0["run_sha256"],
            },
            {
                "model_id": "g6-r1",
                "checkpoint": _sampler(r1, "R1"),
                "group": "dapt_sft_grpo",
                "trajectory_count": 512,
                "training_seed": config["rollout"]["sampling_seed"],
                "checkpoint_run_sha256": r1["run_sha256"],
            },
        ],
        "lineage": {
            "r0_parent_id": r0["parent_id"],
            "r0_parent_run_sha256": r0["parent_run_sha256"],
            "r1_parent_id": r1["parent_id"],
            "r1_parent_run_sha256": r1["parent_run_sha256"],
            "training_config_sha256": r0["config_sha256"],
        },
    }
    evaluation["config_sha256"] = canonical_sha256(evaluation)
    _write(out_path, evaluation)
    return evaluation


def _verified_rows(verifier_root: Path) -> list[dict[str, Any]]:
    manifest = _load(verifier_root / "manifest.json")
    verification = _load(verifier_root / "verification.json")
    verification_payload = dict(verification)
    expected = verification_payload.pop("release_sha256", None)
    if (
        verification.get("input_release_sha256") != manifest.get("release_sha256")
        or canonical_sha256(verification_payload) != expected
        or verification.get("counts", {}).get("infrastructure_failures") != 0
    ):
        raise G6ExperimentError("G6_VERIFICATION_RELEASE_INVALID")
    verified = {record["unit_id"]: record for record in verification["records"]}
    rows = []
    for record in manifest["records"]:
        output = verifier_root / "results" / record["unit_id"]
        if _tree_hashes(output) != verified[record["unit_id"]]["artifacts"]:
            raise G6ExperimentError(f"G6_RESULT_HASH_MISMATCH: {record['unit_id']}")
        rows.append({**record, **_load(output / "reward.json")})
    return rows


def _training_receipt(run_root: Path) -> dict[str, Any]:
    manifest = _load(run_root / "manifest.json")
    events = [
        json.loads(line)
        for line in (run_root / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(events) != manifest.get("total_steps"):
        raise G6ExperimentError(f"G6_TRAINING_EVENT_COUNT_INVALID: {run_root}")
    return {
        "steps": len(events),
        "updates": sum(len(event["updates"]) for event in events),
        "turn_samples": sum(event["counts"]["turn_samples"] for event in events),
        "profile_verified_samples": sum(
            event["counts"]["profile_verified"] for event in events
        ),
        "trainable_task_groups": sum(
            event["counts"]["trainable_task_groups"] for event in events
        ),
        "trainable_datums": sum(
            event["counts"]["trainable_datums"] for event in events
        ),
        "reward_curve": [event["reward"]["mean_score"] for event in events],
    }


def verify_release_on_pool(
    *,
    verifier_root: Path,
    workers: list[str],
    zone: str,
    timeout_seconds: int = 180,
    verifier: VerifierBackend | None = None,
) -> dict[str, Any]:
    manifest = _load(verifier_root / "manifest.json")
    payload = dict(manifest)
    expected = payload.pop("release_sha256", None)
    records = manifest.get("records")
    if (
        manifest.get("kind") != "pallas_g43_verifier_input_release"
        or canonical_sha256(payload) != expected
        or not isinstance(records, list)
        or manifest.get("counts", {}).get("units") != len(records)
        or not workers
    ):
        raise G6ExperimentError("G6_VERIFIER_RELEASE_INVALID")
    batch_root = verifier_root / "remote-batches"
    candidates = [
        VerifierCandidate(
            unit_id=record["unit_id"],
            task_path=verifier_root / "units" / record["unit_id"] / "task.json",
            kernel_path=verifier_root / "units" / record["unit_id"] / "kernel.py",
        )
        for record in records
    ]
    backend = verifier or RemoteTPUPoolVerifier(
        workers=workers, zone=zone, timeout_seconds=timeout_seconds
    )
    results = backend.verify(candidates=candidates, batch_root=batch_root)
    result_records = []
    recovery_events = []
    for worker_result in sorted(batch_root.glob("worker-*/results.json")):
        recovery_events.extend(_load(worker_result).get("recovery_events", []))
    for index, record in enumerate(records, start=1):
        unit_id = record["unit_id"]
        sources = list(batch_root.glob(f"worker-*/results/{unit_id}"))
        if len(sources) != 1 or unit_id not in results:
            raise G6ExperimentError(f"G6_REMOTE_RESULT_MISSING: {unit_id}")
        output = verifier_root / "results" / unit_id
        shutil.copytree(sources[0], output)
        reward = _load(output / "reward.json")
        result = results[unit_id]
        if (
            reward.get("task_id") != record["task_id"]
            or reward.get("kernel_sha256") != record["kernel_sha256"]
            or result.get("kernel_sha256") != record["kernel_sha256"]
        ):
            raise G6ExperimentError(f"G6_REMOTE_RESULT_MISMATCH: {unit_id}")
        result_records.append(
            {
                "unit_id": unit_id,
                "reward": reward["reward"],
                "artifacts": _tree_hashes(output),
            }
        )
        print(
            f"G6_VERIFY completed={index}/{len(records)} reward={reward['reward']}",
            flush=True,
        )
    verification: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pallas_g43_verification_release",
        "input_release_sha256": manifest["release_sha256"],
        "counts": {
            "units": len(result_records),
            "verified": sum(record["reward"] == 1 for record in result_records),
            "candidate_failures": sum(
                record["reward"] == 0 for record in result_records
            ),
            "infrastructure_failures": sum(
                record["reward"] == -1 for record in result_records
            ),
            "recovery_probes": len(recovery_events),
        },
        "records": result_records,
        "recovery_events": recovery_events,
    }
    verification["release_sha256"] = canonical_sha256(verification)
    _write(verifier_root / "verification.json", verification)
    return verification


def summarize_results(
    *,
    evaluation_config_path: Path,
    verifier_root: Path,
    g5_results_path: Path,
    r0_root: Path,
    r1_root: Path,
    out_path: Path,
) -> dict[str, Any]:
    evaluation = _load(evaluation_config_path)
    payload = dict(evaluation)
    expected = payload.pop("config_sha256", None)
    if canonical_sha256(payload) != expected:
        raise G6ExperimentError("G6_EVALUATION_CONFIG_HASH_MISMATCH")
    new_rows = _verified_rows(verifier_root)
    g5 = _load(g5_results_path)
    g5_payload = dict(g5)
    g5_sha = g5_payload.pop("result_sha256", None)
    if canonical_sha256(g5_payload) != g5_sha:
        raise G6ExperimentError("G6_G5_RESULT_HASH_MISMATCH")
    controls = [
        row
        for row in g5["rows"]
        if row["model_id"] in {"g42-repair-sft", "g5-s1"}
    ]
    rows = [*controls, *new_rows]
    model_ids = {"g42-repair-sft", "g5-s1", "g6-r0", "g6-r1"}
    if (
        {row["model_id"] for row in rows} != model_ids
        or len(rows) != 64
        or any(row["seed"] != 0 or row["turn"] != 3 for row in rows)
    ):
        raise G6ExperimentError("G6_RESULT_MATRIX_INCOMPLETE")
    task_sets = {
        model_id: {(row["task_id"], row["task_sha256"]) for row in rows if row["model_id"] == model_id}
        for model_id in model_ids
    }
    if len({frozenset(tasks) for tasks in task_sets.values()}) != 1:
        raise G6ExperimentError("G6_MATCHED_TASK_IDENTITY_MISMATCH")
    by_model = {
        model_id: _model_summary([row for row in rows if row["model_id"] == model_id])
        for model_id in sorted(model_ids)
    }
    comparisons = {
        "r0_vs_s0": _paired_summary(
            [row for row in rows if row["model_id"] == "g6-r0"],
            [row for row in rows if row["model_id"] == "g42-repair-sft"],
        ),
        "r1_vs_s1": _paired_summary(
            [row for row in rows if row["model_id"] == "g6-r1"],
            [row for row in rows if row["model_id"] == "g5-s1"],
        ),
        "r1_vs_r0": _paired_summary(
            [row for row in rows if row["model_id"] == "g6-r1"],
            [row for row in rows if row["model_id"] == "g6-r0"],
        ),
    }
    family_comparison = {
        family: {
            model_id: sum(
                row["reward"] == 1
                for row in rows
                if row["model_id"] == model_id and row["family"] == family
            )
            for model_id in sorted(model_ids)
        }
        for family in sorted({row["family"] for row in rows})
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pallas_g6_grpo_results",
        "evaluation_config_sha256": evaluation["config_sha256"],
        "g5_control_result_sha256": g5["result_sha256"],
        "rows": rows,
        "summary": {
            "models": by_model,
            "paired_comparisons": comparisons,
            "family_comparison": family_comparison,
            "training": {
                "R0": _training_receipt(r0_root),
                "R1": _training_receipt(r1_root),
            },
        },
        "decision": {
            "r0_improves_s0": comparisons["r0_vs_s0"]["profile_verified_delta"] > 0,
            "r1_improves_s1": comparisons["r1_vs_s1"]["profile_verified_delta"] > 0,
            "dapt_interaction_positive": comparisons["r1_vs_r0"]["profile_verified_delta"] > 0,
            "gate_complete": True,
            "advance_to_g7": True,
        },
    }
    result["result_sha256"] = canonical_sha256(result)
    _write(out_path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-g6-experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-evaluation")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--benchmark-root", type=Path, required=True)
    build.add_argument("--r0-manifest", type=Path, required=True)
    build.add_argument("--r1-manifest", type=Path, required=True)
    build.add_argument("--out-path", type=Path, required=True)
    prepare = commands.add_parser("prepare-verifier")
    prepare.add_argument("--sample-root", type=Path, required=True)
    prepare.add_argument("--benchmark-root", type=Path, required=True)
    prepare.add_argument("--out-dir", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--verifier-root", type=Path, required=True)
    verify.add_argument("--timeout-seconds", type=int, default=180)
    verify.add_argument("--workers")
    verify.add_argument("--zone", default="us-west4-a")
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--evaluation-config", type=Path, required=True)
    summarize.add_argument("--verifier-root", type=Path, required=True)
    summarize.add_argument("--g5-results", type=Path, required=True)
    summarize.add_argument("--r0-root", type=Path, required=True)
    summarize.add_argument("--r1-root", type=Path, required=True)
    summarize.add_argument("--out-path", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "build-evaluation":
            result = build_evaluation_config(
                config_path=args.config,
                benchmark_root=args.benchmark_root,
                r0_manifest_path=args.r0_manifest,
                r1_manifest_path=args.r1_manifest,
                out_path=args.out_path,
            )
        elif args.command == "prepare-verifier":
            result = prepare_verifier_release(
                sample_root=args.sample_root,
                benchmark_root=args.benchmark_root,
                out_dir=args.out_dir,
            )
        elif args.command == "verify":
            if args.workers:
                result = verify_release_on_pool(
                    verifier_root=args.verifier_root,
                    workers=[worker for worker in args.workers.split(",") if worker],
                    zone=args.zone,
                    timeout_seconds=args.timeout_seconds,
                )
            else:
                result = verify_release(
                    verifier_root=args.verifier_root,
                    timeout_seconds=args.timeout_seconds,
                )
        else:
            result = summarize_results(
                evaluation_config_path=args.evaluation_config,
                verifier_root=args.verifier_root,
                g5_results_path=args.g5_results,
                r0_root=args.r0_root,
                r1_root=args.r1_root,
                out_path=args.out_path,
            )
    except (G6ExperimentError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"G6_EXPERIMENT_ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
