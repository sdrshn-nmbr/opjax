"""Run and assemble the matched G4.2 k=3 versus k=6 experiment."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from opjax.pallas.contracts import git_revision
from opjax.pallas.g42_curriculum import validate_benchmark_release
from opjax.pallas.g42_harness import (
    AGENT_IMAGE,
    canonical_sha256,
    file_sha256,
    load_task_package,
    materialize_submission,
    summarize_horizons,
)
from opjax.pallas.g42_verifier import run_fresh_verifier


class G42ExperimentError(RuntimeError):
    """The matched experiment is incomplete or violates its frozen pairing contract."""


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise G42ExperimentError(f"JSON_OBJECT_REQUIRED: {path}")
    return value


def _tree_file_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _tracked_dirty(repo_root: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
    )
    return result.returncode != 0 or bool(result.stdout.strip())


def _docker_image_id(image: str) -> str:
    result = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip().startswith("sha256:"):
        raise G42ExperimentError(f"AGENT_IMAGE_UNAVAILABLE: {image}")
    return result.stdout.strip()


def _validate_config(config: dict[str, Any], benchmark_root: Path) -> list[dict[str, Any]]:
    validation = validate_benchmark_release(benchmark_root)
    if config.get("schema_version") != 1 or validation["release_sha256"] != config.get(
        "benchmark_release_sha256"
    ):
        raise G42ExperimentError("EVALUATION_CONFIG_INVALID")
    if config.get("seeds") != [0, 1, 2] or config.get("snapshot_turns") != [3, 6]:
        raise G42ExperimentError("EVALUATION_PAIRING_INVALID")
    models = config.get("models")
    if not isinstance(models, list) or [model.get("model_id") for model in models] != [
        "inkling-small-base",
        "g41-sft",
        "g42-repair-sft",
    ]:
        raise G42ExperimentError("EVALUATION_MODELS_INVALID")
    return models


def sample_experiment(
    *,
    config_path: Path,
    config_root: Path,
    benchmark_root: Path,
    repo_root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    from opjax.pallas.g42_agent import run_tinker_agent

    if _tracked_dirty(repo_root):
        raise G42ExperimentError(f"OPJAX_TRACKED_DIRTY: {repo_root}")
    config = _load(config_path)
    models = _validate_config(config, benchmark_root)
    benchmark_manifest = _load(benchmark_root / "manifest.json")
    tasks = [load_task_package(benchmark_root / relative) for relative in benchmark_manifest["tasks"]]
    jobs = [
        (model, task, seed)
        for model in models
        for task in tasks
        for seed in config["seeds"]
    ]
    out_dir.mkdir(parents=True, exist_ok=True)

    def completed_record(model: dict[str, Any], task: Any, seed: int) -> dict[str, Any] | None:
        run_root = out_dir / "runs" / model["model_id"] / task.task_id / f"seed-{seed}"
        manifest_path = run_root / "manifest.json"
        if not manifest_path.is_file():
            if run_root.exists():
                failed_root = out_dir / "failed-attempts"
                failed_root.mkdir(parents=True, exist_ok=True)
                attempt = failed_root / f"{model['model_id']}--{task.task_id}--seed-{seed}"
                suffix = 1
                while attempt.exists():
                    suffix += 1
                    attempt = failed_root / f"{model['model_id']}--{task.task_id}--seed-{seed}--{suffix}"
                run_root.rename(attempt)
            return None
        manifest = _load(manifest_path)
        if (
            manifest.get("task_sha256") != task.task_sha256
            or manifest.get("checkpoint") != model["checkpoint"]
            or manifest.get("seed") != seed
            or sorted(int(turn) for turn in manifest.get("snapshots", {})) != [3, 6]
        ):
            raise G42ExperimentError(f"COMPLETED_RUN_INVALID: {run_root}")
        return {
            "model_id": model["model_id"],
            "checkpoint": model["checkpoint"],
            "task_id": task.task_id,
            "task_sha256": task.task_sha256,
            "seed": seed,
            "run_path": str(run_root.relative_to(out_dir)),
            "submitted": manifest["submitted"],
            "snapshots": manifest["snapshots"],
        }

    def run(job: tuple[dict[str, Any], Any, int]) -> dict[str, Any]:
        model, task, seed = job
        run_root = out_dir / "runs" / model["model_id"] / task.task_id / f"seed-{seed}"
        manifest = asyncio.run(
            run_tinker_agent(
                config_root=config_root,
                task_dir=task.root,
                output_dir=run_root,
                checkpoint=model["checkpoint"],
                seed=seed,
                turn_limit=config["turn_limit"],
                snapshot_turns=tuple(config["snapshot_turns"]),
            )
        )
        return {
            "model_id": model["model_id"],
            "checkpoint": model["checkpoint"],
            "task_id": task.task_id,
            "task_sha256": task.task_sha256,
            "seed": seed,
            "run_path": str(run_root.relative_to(out_dir)),
            "submitted": manifest["submitted"],
            "snapshots": manifest["snapshots"],
        }

    records = []
    pending = []
    for job in jobs:
        previous = completed_record(*job)
        if previous is None:
            pending.append(job)
        else:
            records.append(previous)
    print(f"G42_SAMPLE resumed={len(records)} pending={len(pending)}", flush=True)
    with ThreadPoolExecutor(max_workers=config["max_workers"]) as executor:
        futures = {executor.submit(run, job): job for job in pending}
        for future in as_completed(futures):
            records.append(future.result())
            print(f"G42_SAMPLE completed={len(records)}/{len(jobs)}", flush=True)
    records.sort(key=lambda row: (row["model_id"], row["task_id"], row["seed"]))
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pallas_g42_sample_matrix",
        "experiment_id": config["experiment_id"],
        "config_sha256": file_sha256(config_path),
        "benchmark_release_sha256": config["benchmark_release_sha256"],
        "opjax_revision": git_revision(repo_root),
        "agent_environment": {"image": AGENT_IMAGE, "image_id": _docker_image_id(AGENT_IMAGE)},
        "counts": {"runs": len(records), "snapshots": len(records) * 2},
        "records": records,
    }
    manifest["release_sha256"] = canonical_sha256(manifest)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def prepare_verifier_release(
    *, sample_root: Path, benchmark_root: Path, out_dir: Path
) -> dict[str, Any]:
    if out_dir.exists():
        raise G42ExperimentError(f"OUTPUT_EXISTS: {out_dir}")
    sample_manifest = _load(sample_root / "manifest.json")
    benchmark_manifest = _load(benchmark_root / "manifest.json")
    sample_payload = dict(sample_manifest)
    expected_sample_sha = sample_payload.pop("release_sha256", None)
    if sample_manifest.get("kind") != "pallas_g42_sample_matrix" or canonical_sha256(
        sample_payload
    ) != expected_sample_sha:
        raise G42ExperimentError("SAMPLE_RELEASE_INVALID")
    if validate_benchmark_release(benchmark_root)["release_sha256"] != benchmark_manifest.get(
        "release_sha256"
    ):
        raise G42ExperimentError("BENCHMARK_RELEASE_INVALID")
    tasks = {
        package.task_id: package
        for package in (load_task_package(benchmark_root / relative) for relative in benchmark_manifest["tasks"])
    }
    records = []
    for run in sample_manifest["records"]:
        task = tasks[run["task_id"]]
        run_root = sample_root / run["run_path"]
        for turn in (3, 6):
            unit_id = f"{run['model_id']}--{task.task_id}--seed-{run['seed']}--turn-{turn}"
            unit_root = out_dir / "units" / unit_id
            materialized = materialize_submission(
                task=task,
                patch_path=run_root / "snapshots" / f"turn-{turn}.patch",
                destination=unit_root / "workspace",
            )
            (unit_root / "task.json").parent.mkdir(parents=True, exist_ok=True)
            (unit_root / "task.json").write_bytes((task.root / "tests" / "task.json").read_bytes())
            (unit_root / "kernel.py").write_bytes(Path(materialized["kernel_path"]).read_bytes())
            (unit_root / "model.patch").write_bytes(
                (run_root / "snapshots" / f"turn-{turn}.patch").read_bytes()
            )
            metadata = {
                "unit_id": unit_id,
                "model_id": run["model_id"],
                "checkpoint": run["checkpoint"],
                "task_id": task.task_id,
                "family": task.family,
                "task_sha256": task.task_sha256,
                "seed": run["seed"],
                "turn": turn,
                "patch_sha256": materialized["patch_sha256"],
                "kernel_sha256": materialized["kernel_sha256"],
            }
            (unit_root / "trajectory.json").write_bytes((run_root / "trajectory.json").read_bytes())
            metadata["trajectory_sha256"] = file_sha256(unit_root / "trajectory.json")
            (unit_root / "metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            records.append(metadata)
            (unit_root / "workspace").rename(unit_root / "materialized-workspace")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pallas_g42_verifier_input_release",
        "sample_release_sha256": sample_manifest["release_sha256"],
        "benchmark_release_sha256": benchmark_manifest["release_sha256"],
        "counts": {"units": len(records)},
        "records": records,
    }
    manifest["release_sha256"] = canonical_sha256(manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _worker_health(command: list[str] | None) -> dict[str, Any]:
    command = command or [
        sys.executable,
        "-c",
        (
            "import jax, jax.numpy as jnp; "
            "x=jax.jit(lambda y:y+1)(jnp.asarray(1)); "
            "x.block_until_ready(); print(int(x))"
        ),
    ]
    process = subprocess.run(command, capture_output=True, text=True, timeout=120)
    return {
        "command": command,
        "returncode": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
        "healthy": process.returncode == 0,
    }


def verify_release(
    *,
    verifier_root: Path,
    timeout_seconds: int = 120,
    runner_command: list[str] | None = None,
    health_command: list[str] | None = None,
) -> dict[str, Any]:
    """Grade every immutable unit sequentially and prove health after TPU poison events."""
    manifest = _load(verifier_root / "manifest.json")
    if manifest.get("kind") != "pallas_g42_verifier_input_release":
        raise G42ExperimentError("VERIFIER_RELEASE_KIND_INVALID")
    manifest_payload = dict(manifest)
    expected_manifest_sha = manifest_payload.pop("release_sha256", None)
    if canonical_sha256(manifest_payload) != expected_manifest_sha:
        raise G42ExperimentError("VERIFIER_RELEASE_HASH_MISMATCH")
    records = manifest.get("records", [])
    if manifest.get("counts", {}).get("units") != len(records):
        raise G42ExperimentError("VERIFIER_UNIT_COUNT_INVALID")
    unit_ids = [record.get("unit_id") for record in records]
    if None in unit_ids or len(set(unit_ids)) != len(unit_ids):
        raise G42ExperimentError("VERIFIER_UNIT_IDS_INVALID")
    results_root = verifier_root / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    result_records: list[dict[str, Any]] = []
    recovery_events: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        unit_root = verifier_root / "units" / record["unit_id"]
        task_path = unit_root / "task.json"
        kernel_path = unit_root / "kernel.py"
        trajectory_path = unit_root / "trajectory.json"
        if not all(path.is_file() for path in (task_path, kernel_path, trajectory_path, unit_root / "model.patch")):
            raise G42ExperimentError(f"VERIFIER_UNIT_ARTIFACT_MISSING: {record['unit_id']}")
        if file_sha256(kernel_path) != record["kernel_sha256"]:
            raise G42ExperimentError(f"VERIFIER_KERNEL_HASH_MISMATCH: {record['unit_id']}")
        if file_sha256(unit_root / "model.patch") != record["patch_sha256"]:
            raise G42ExperimentError(f"VERIFIER_PATCH_HASH_MISMATCH: {record['unit_id']}")
        if file_sha256(trajectory_path) != record["trajectory_sha256"]:
            raise G42ExperimentError(f"VERIFIER_TRAJECTORY_HASH_MISMATCH: {record['unit_id']}")
        output_dir = results_root / record["unit_id"]
        reward_path = output_dir / "reward.json"
        if reward_path.is_file():
            reward = _load(reward_path)
            if reward.get("task_id") != record["task_id"] or reward.get("kernel_sha256") != record["kernel_sha256"]:
                raise G42ExperimentError(f"VERIFIER_RESULT_MISMATCH: {record['unit_id']}")
            result = _load(output_dir / "run.log")
        else:
            payload = run_fresh_verifier(
                task_path=task_path,
                kernel_path=kernel_path,
                output_dir=output_dir,
                timeout_seconds=timeout_seconds,
                runner_command=runner_command,
            )
            result = payload["result"]
            reward = payload["reward"]
        if result.get("worker_recovery_required") is True:
            health = _worker_health(health_command)
            event = {"after_unit": record["unit_id"], **health}
            recovery_events.append(event)
            if not health["healthy"]:
                (verifier_root / "recovery-events.json").write_text(
                    json.dumps(recovery_events, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                raise G42ExperimentError(f"WORKER_QUARANTINED: {record['unit_id']}")
        result_records.append(
            {
                "unit_id": record["unit_id"],
                "reward": reward["reward"],
                "reward_sha256": file_sha256(reward_path),
                "ctrf_sha256": file_sha256(output_dir / "ctrf.json"),
                "artifacts": _tree_file_hashes(output_dir),
            }
        )
        print(f"G42_VERIFY completed={index}/{len(records)} reward={reward['reward']}", flush=True)
    verification: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pallas_g42_verification_release",
        "input_release_sha256": manifest["release_sha256"],
        "counts": {
            "units": len(result_records),
            "verified": sum(record["reward"] == 1 for record in result_records),
            "candidate_failures": sum(record["reward"] == 0 for record in result_records),
            "infrastructure_failures": sum(record["reward"] == -1 for record in result_records),
            "recovery_probes": len(recovery_events),
        },
        "records": result_records,
        "recovery_events": recovery_events,
    }
    verification["release_sha256"] = canonical_sha256(verification)
    (verifier_root / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return verification


def _paired_model_deltas(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cell: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (row["task_id"], int(row["seed"]), int(row["turn"]))
        by_cell.setdefault(key, {})[row["model_id"]] = row
    model_ids = ("inkling-small-base", "g41-sft", "g42-repair-sft")
    for key, models in by_cell.items():
        if set(models) != set(model_ids):
            raise G42ExperimentError(f"MODEL_PAIR_INCOMPLETE: {key}")
    comparisons = {}
    for candidate, baseline in (
        ("g42-repair-sft", "inkling-small-base"),
        ("g42-repair-sft", "g41-sft"),
        ("g41-sft", "inkling-small-base"),
    ):
        deltas = [
            models[candidate]["reward"] - models[baseline]["reward"]
            for models in by_cell.values()
        ]
        comparisons[f"{candidate}_vs_{baseline}"] = {
            "cells": len(deltas),
            "wins": sum(delta > 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
            "losses": sum(delta < 0 for delta in deltas),
            "verified_delta": sum(deltas),
            "mean_reward_delta": statistics.fmean(deltas),
        }
    return comparisons


def _family_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary = [row for row in rows if int(row["turn"]) == 6]
    families = sorted({row["family"] for row in primary})
    by_family: dict[str, dict[str, int]] = {}
    regressions = []
    for family in families:
        counts = {
            model_id: sum(
                row["reward"] == 1
                for row in primary
                if row["family"] == family and row["model_id"] == model_id
            )
            for model_id in ("inkling-small-base", "g41-sft", "g42-repair-sft")
        }
        by_family[family] = counts
        if counts["g42-repair-sft"] < counts["inkling-small-base"]:
            regressions.append(family)
    return {"horizon": 6, "families": by_family, "regressions_vs_base": regressions}


def _stratified_stage_fractions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model_id in sorted({row["model_id"] for row in rows}):
        result[model_id] = {}
        for turn in (3, 6):
            subset = [
                row for row in rows if row["model_id"] == model_id and int(row["turn"]) == turn
            ]
            if not subset:
                raise G42ExperimentError(f"STAGE_STRATUM_EMPTY: {model_id}:turn={turn}")
            result[model_id][f"k{turn}"] = {
                stage: sum(row["stage_fractions"].get(stage, 0.0) for row in subset)
                / len(subset)
                for stage in subset[0]["stage_fractions"]
            }
    return result


def _verified_speedups(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model_id in sorted({row["model_id"] for row in rows}):
        result[model_id] = {}
        for turn in (3, 6):
            values = sorted(
                float(row["speedup"])
                for row in rows
                if row["model_id"] == model_id
                and int(row["turn"]) == turn
                and row["reward"] == 1
                and isinstance(row.get("speedup"), (int, float))
            )
            result[model_id][f"k{turn}"] = {
                "count": len(values),
                "values": values,
                "median": statistics.median(values) if values else None,
            }
    return result


def summarize_results(*, verifier_root: Path, out_path: Path) -> dict[str, Any]:
    manifest = _load(verifier_root / "manifest.json")
    verification = _load(verifier_root / "verification.json")
    if verification.get("input_release_sha256") != manifest.get("release_sha256"):
        raise G42ExperimentError("VERIFICATION_RELEASE_MISMATCH")
    verification_payload = dict(verification)
    expected_verification_sha = verification_payload.pop("release_sha256", None)
    if canonical_sha256(verification_payload) != expected_verification_sha:
        raise G42ExperimentError("VERIFICATION_RELEASE_HASH_MISMATCH")
    verification_by_unit = {record["unit_id"]: record for record in verification.get("records", [])}
    if set(verification_by_unit) != {record["unit_id"] for record in manifest.get("records", [])}:
        raise G42ExperimentError("VERIFICATION_RECORD_SET_MISMATCH")
    rows = []
    failures: dict[str, int] = {}
    halts = 0
    speedups = []
    for record in manifest["records"]:
        reward = _load(verifier_root / "results" / record["unit_id"] / "reward.json")
        result_root = verifier_root / "results" / record["unit_id"]
        if _tree_file_hashes(result_root) != verification_by_unit[record["unit_id"]].get("artifacts"):
            raise G42ExperimentError(f"RESULT_ARTIFACT_HASH_MISMATCH: {record['unit_id']}")
        if reward.get("kernel_sha256") != record["kernel_sha256"] or reward.get("task_id") != record["task_id"]:
            raise G42ExperimentError(f"RESULT_RECORD_MISMATCH: {record['unit_id']}")
        row = {**record, **reward}
        rows.append(row)
        if reward["reward"] != 1:
            stage = reward["failure_stage"]
            failures[stage] = failures.get(stage, 0) + 1
            halts += int(reward.get("worker_recovery_required") is True)
        elif isinstance(reward.get("speedup"), (int, float)):
            speedups.append(reward["speedup"])
    if not rows:
        raise G42ExperimentError("RESULT_ROWS_EMPTY")
    horizon = summarize_horizons(rows)
    stage_fractions = {
        stage: sum(row["stage_fractions"].get(stage, 0.0) for row in rows) / len(rows)
        for stage in rows[0]["stage_fractions"]
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pallas_g42_matched_results",
        "verifier_release_sha256": manifest["release_sha256"],
        "rows": rows,
        "summary": {
            **horizon,
            "stage_fractions": stage_fractions,
            "stage_fractions_by_model_horizon": _stratified_stage_fractions(rows),
            "paired_model_deltas": _paired_model_deltas(rows),
            "family_gate": _family_gate(rows),
            "failure_stages": dict(sorted(failures.items())),
            "candidate_attributable_tpu_halts": halts,
            "verified_speedups": sorted(speedups),
            "verified_speedups_by_model_horizon": _verified_speedups(rows),
        },
    }
    base_verified = sum(row["reward"] == 1 for row in rows if row["model_id"] == "inkling-small-base")
    g42_verified = sum(row["reward"] == 1 for row in rows if row["model_id"] == "g42-repair-sft")
    result["gate"] = {
        "base_profile_verified": base_verified,
        "g42_profile_verified": g42_verified,
        "positive_paired_delta": g42_verified > base_verified,
        "no_task_family_regression": not result["summary"]["family_gate"]["regressions_vs_base"],
    }
    result["gate"]["capability_passed"] = (
        result["gate"]["positive_paired_delta"] and result["gate"]["no_task_family_regression"]
    )
    result["result_sha256"] = canonical_sha256(result)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-g42-experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample")
    sample.add_argument("--config", type=Path, default=Path("config/pallas/g42-evaluation.json"))
    sample.add_argument("--config-root", type=Path, default=Path("config/pallas"))
    sample.add_argument("--benchmark-root", type=Path, required=True)
    sample.add_argument("--repo-root", type=Path, default=Path("."))
    sample.add_argument("--out-dir", type=Path, required=True)
    prepare = commands.add_parser("prepare-verifier")
    prepare.add_argument("--sample-root", type=Path, required=True)
    prepare.add_argument("--benchmark-root", type=Path, required=True)
    prepare.add_argument("--out-dir", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--verifier-root", type=Path, required=True)
    verify.add_argument("--timeout-seconds", type=int, default=120)
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--verifier-root", type=Path, required=True)
    summarize.add_argument("--out-path", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "sample":
            result = sample_experiment(
                config_path=args.config,
                config_root=args.config_root,
                benchmark_root=args.benchmark_root,
                repo_root=args.repo_root,
                out_dir=args.out_dir,
            )
        elif args.command == "prepare-verifier":
            result = prepare_verifier_release(
                sample_root=args.sample_root,
                benchmark_root=args.benchmark_root,
                out_dir=args.out_dir,
            )
        elif args.command == "verify":
            result = verify_release(
                verifier_root=args.verifier_root,
                timeout_seconds=args.timeout_seconds,
            )
        else:
            result = summarize_results(verifier_root=args.verifier_root, out_path=args.out_path)
    except (G42ExperimentError, OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print(f"G42_EXPERIMENT_ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
