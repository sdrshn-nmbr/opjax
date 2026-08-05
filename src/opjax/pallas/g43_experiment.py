"""Run G4.3 benchmark admission, matched sampling, verification, and analysis."""

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
from opjax.pallas.g42_agent import run_tinker_agent
from opjax.pallas.g42_experiment import _worker_health
from opjax.pallas.g42_harness import (
    AGENT_IMAGE,
    canonical_sha256,
    file_sha256,
    load_task_package,
    materialize_submission,
)
from opjax.pallas.g42_verifier import run_fresh_verifier
from opjax.pallas.g43_corpus import (
    FAMILIES,
    TRAINING_SEEDS,
    TRAJECTORY_COUNTS,
    validate_benchmark_release,
    validate_learning_curve_release,
)


class G43ExperimentError(RuntimeError):
    """G4.3 cannot produce a causal result from incomplete or invalid evidence."""


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise G43ExperimentError(f"JSON_OBJECT_REQUIRED: {path}")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _tracked_dirty(repo_root: Path) -> bool:
    process = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
    )
    return process.returncode != 0 or bool(process.stdout.strip())


def admit_benchmark(
    *, benchmark_root: Path, out_dir: Path, timeout_seconds: int = 180
) -> dict[str, Any]:
    if out_dir.exists():
        raise G43ExperimentError(f"OUTPUT_EXISTS: {out_dir}")
    validation = validate_benchmark_release(benchmark_root)
    benchmark = _load(benchmark_root / "manifest.json")
    records = []
    for index, relative in enumerate(benchmark["tasks"], start=1):
        package = load_task_package(benchmark_root / relative)
        task_path = package.root / "tests" / "task.json"
        starter = package.root / "environment" / "starter" / "kernel.py"
        solution = package.root / "solution" / "kernel.py"
        starter_result = run_fresh_verifier(
            task_path=task_path,
            kernel_path=starter,
            output_dir=out_dir / "tasks" / package.task_id / "starter",
            timeout_seconds=timeout_seconds,
        )
        solution_result = run_fresh_verifier(
            task_path=task_path,
            kernel_path=solution,
            output_dir=out_dir / "tasks" / package.task_id / "solution",
            timeout_seconds=timeout_seconds,
        )
        starter_reward = starter_result["reward"]
        solution_reward = solution_result["reward"]
        if starter_reward["reward"] != 0:
            raise G43ExperimentError(f"G43_STARTER_ADMISSION_FAILED: {package.task_id}")
        if solution_reward["reward"] != 1:
            raise G43ExperimentError(
                f"G43_REFERENCE_ADMISSION_FAILED: {package.task_id}:{solution_reward['failure_stage']}"
            )
        speedup = solution_reward.get("speedup")
        records.append(
            {
                "task_id": package.task_id,
                "task_sha256": package.task_sha256,
                "family": package.family,
                "starter_failure_stage": starter_reward["failure_stage"],
                "solution_reward_sha256": file_sha256(
                    out_dir / "tasks" / package.task_id / "solution" / "reward.json"
                ),
                "solution_speedup": speedup,
                "performance_headroom": bool(
                    isinstance(speedup, (int, float)) and speedup >= 1.05
                ),
            }
        )
        print(f"G43_ADMISSION completed={index}/16 task={package.task_id}", flush=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pallas_g43_benchmark_admission",
        "benchmark_release_sha256": validation["release_sha256"],
        "counts": {
            "tasks": len(records),
            "verified_references": len(records),
            "failed_starters": len(records),
            "performance_headroom_tasks": sum(
                record["performance_headroom"] for record in records
            ),
        },
        "headroom_threshold": 1.05,
        "headroom_task_ids": [
            record["task_id"] for record in records if record["performance_headroom"]
        ],
        "records": records,
    }
    manifest["release_sha256"] = canonical_sha256(manifest)
    _write(out_dir / "manifest.json", manifest)
    return manifest


def _sampler_weights(manifest: dict[str, Any]) -> str:
    if manifest.get("status") != "completed":
        raise G43ExperimentError("G43_TRAINING_RUN_INCOMPLETE")
    sampler = manifest.get("sampler_weights")
    if isinstance(sampler, dict):
        response = sampler.get("response", sampler)
        if isinstance(response, dict) and isinstance(response.get("path"), str):
            return response["path"]
    raise G43ExperimentError("G43_SAMPLER_WEIGHTS_MISSING")


def build_evaluation_config(
    *,
    config_path: Path,
    benchmark_root: Path,
    admission_root: Path,
    learning_curve_root: Path,
    training_root: Path,
    g42_training_manifest: Path,
    out_path: Path,
) -> dict[str, Any]:
    config = _load(config_path)
    benchmark = validate_benchmark_release(benchmark_root)
    admission = _load(admission_root / "manifest.json")
    curve = _load(learning_curve_root / "manifest.json")
    curve_validation = validate_learning_curve_release(learning_curve_root)
    if admission.get("benchmark_release_sha256") != benchmark["release_sha256"]:
        raise G43ExperimentError("G43_ADMISSION_BENCHMARK_MISMATCH")
    admission_payload = dict(admission)
    admission_sha = admission_payload.pop("release_sha256", None)
    if canonical_sha256(admission_payload) != admission_sha:
        raise G43ExperimentError("G43_ADMISSION_HASH_MISMATCH")
    models = [
        {
            "model_id": "inkling-small-base",
            "checkpoint": None,
            "group": "control",
            "trajectory_count": 0,
            "training_seed": None,
            "checkpoint_run_sha256": None,
        }
    ]
    g42_manifest = _load(g42_training_manifest)
    models.append(
        {
            "model_id": "g42-repair-sft",
            "checkpoint": _sampler_weights(g42_manifest),
            "group": "control",
            "trajectory_count": 32,
            "training_seed": 0,
            "checkpoint_run_sha256": g42_manifest["run_sha256"],
        }
    )
    for count in TRAJECTORY_COUNTS:
        for seed in TRAINING_SEEDS:
            manifest = _load(training_root / f"n{count}-seed{seed}" / "manifest.json")
            preparation = _load(
                training_root / f"n{count}-seed{seed}" / "preparation.json"
            )
            if (
                preparation.get("data", {}).get("trajectories") != count
                or preparation.get("training", {}).get("training_seed") != seed
            ):
                raise G43ExperimentError(f"G43_TRAINING_IDENTITY_MISMATCH: n{count}-seed{seed}")
            models.append(
                {
                    "model_id": f"g43-n{count}-seed{seed}",
                    "checkpoint": _sampler_weights(manifest),
                    "group": "learning_curve",
                    "trajectory_count": count,
                    "training_seed": seed,
                    "checkpoint_run_sha256": manifest["run_sha256"],
                }
            )
    evaluation = {
        "schema_version": 1,
        "kind": "pallas_g43_evaluation_config",
        "experiment_id": config["experiment_id"],
        "benchmark_release_sha256": benchmark["release_sha256"],
        "admission_release_sha256": admission["release_sha256"],
        "learning_curve_release_sha256": curve["release_sha256"],
        "evaluation_seed": config["evaluation_seed"],
        "turn_limit": config["turn_limit"],
        "snapshot_turns": config["snapshot_turns"],
        "max_sampling_workers": config["max_sampling_workers"],
        "headroom_task_ids": admission["headroom_task_ids"],
        "models": models,
    }
    evaluation["config_sha256"] = canonical_sha256(evaluation)
    if curve_validation["release_sha256"] != evaluation["learning_curve_release_sha256"]:
        raise G43ExperimentError("G43_LEARNING_CURVE_RELEASE_MISMATCH")
    _write(out_path, evaluation)
    return evaluation


def sample_experiment(
    *,
    evaluation_config: Path,
    config_root: Path,
    benchmark_root: Path,
    repo_root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    config = _load(evaluation_config)
    benchmark = _load(benchmark_root / "manifest.json")
    validation = validate_benchmark_release(benchmark_root)
    if config.get("benchmark_release_sha256") != validation["release_sha256"]:
        raise G43ExperimentError("G43_EVALUATION_BENCHMARK_MISMATCH")
    config_payload = dict(config)
    expected_config_sha = config_payload.pop("config_sha256", None)
    if canonical_sha256(config_payload) != expected_config_sha:
        raise G43ExperimentError("G43_EVALUATION_CONFIG_HASH_MISMATCH")
    if _tracked_dirty(repo_root):
        raise G43ExperimentError(f"OPJAX_TRACKED_DIRTY: {repo_root}")
    tasks = [load_task_package(benchmark_root / relative) for relative in benchmark["tasks"]]
    jobs = [
        (model, task)
        for model in config["models"]
        for task in tasks
    ]
    out_dir.mkdir(parents=True, exist_ok=True)

    def run(job: tuple[dict[str, Any], Any]) -> dict[str, Any]:
        model, task = job
        run_id = f"{model['model_id']}--{task.task_id}--seed-{config['evaluation_seed']}"
        run_root = out_dir / "runs" / run_id
        manifest_path = run_root / "manifest.json"
        if manifest_path.is_file():
            existing = _load(manifest_path)
            expected = {
                "task_id": task.task_id,
                "task_sha256": task.task_sha256,
                "checkpoint": model["checkpoint"],
                "seed": config["evaluation_seed"],
                "turn_limit": 3,
                "snapshot_turns": [3],
            }
            if any(existing.get(key) != value for key, value in expected.items()):
                raise G43ExperimentError(f"G43_SAMPLE_RESUME_MISMATCH: {run_id}")
            return {
                **model,
                "task_id": task.task_id,
                "task_sha256": task.task_sha256,
                "family": task.family,
                "run_path": f"runs/{run_id}",
                "trajectory_sha256": file_sha256(run_root / "trajectory.json"),
            }
        if run_root.exists():
            raise G43ExperimentError(f"G43_SAMPLE_PARTIAL_RUN: {run_id}")
        asyncio.run(
            run_tinker_agent(
                config_root=config_root,
                task_dir=task.root,
                output_dir=run_root,
                checkpoint=model["checkpoint"],
                seed=config["evaluation_seed"],
                turn_limit=3,
                snapshot_turns=(3,),
            )
        )
        return {
            **model,
            "task_id": task.task_id,
            "task_sha256": task.task_sha256,
            "family": task.family,
            "run_path": f"runs/{run_id}",
            "trajectory_sha256": file_sha256(run_root / "trajectory.json"),
        }

    records = []
    with ThreadPoolExecutor(max_workers=config["max_sampling_workers"]) as executor:
        futures = {executor.submit(run, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            records.append(future.result())
            print(f"G43_SAMPLE completed={index}/{len(jobs)}", flush=True)
    records.sort(key=lambda row: (row["model_id"], row["task_id"]))
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pallas_g43_sample_matrix",
        "evaluation_config_sha256": config["config_sha256"],
        "benchmark_release_sha256": validation["release_sha256"],
        "agent_image": AGENT_IMAGE,
        "opjax_revision": git_revision(repo_root),
        "counts": {"runs": len(records), "snapshots": len(records)},
        "records": records,
    }
    manifest["release_sha256"] = canonical_sha256(manifest)
    _write(out_dir / "manifest.json", manifest)
    return manifest


def prepare_verifier_release(
    *, sample_root: Path, benchmark_root: Path, out_dir: Path
) -> dict[str, Any]:
    if out_dir.exists():
        raise G43ExperimentError(f"OUTPUT_EXISTS: {out_dir}")
    sample = _load(sample_root / "manifest.json")
    sample_payload = dict(sample)
    expected_sample_sha = sample_payload.pop("release_sha256", None)
    if (
        sample.get("kind") != "pallas_g43_sample_matrix"
        or canonical_sha256(sample_payload) != expected_sample_sha
    ):
        raise G43ExperimentError("G43_SAMPLE_RELEASE_INVALID")
    benchmark = _load(benchmark_root / "manifest.json")
    validation = validate_benchmark_release(benchmark_root)
    tasks = {
        package.task_id: package
        for package in (
            load_task_package(benchmark_root / relative) for relative in benchmark["tasks"]
        )
    }
    records = []
    for run in sample["records"]:
        task = tasks[run["task_id"]]
        run_root = sample_root / run["run_path"]
        unit_id = f"{run['model_id']}--{task.task_id}--seed-0--turn-3"
        unit_root = out_dir / "units" / unit_id
        materialized = materialize_submission(
            task=task,
            patch_path=run_root / "snapshots" / "turn-3.patch",
            destination=unit_root / "workspace",
        )
        unit_root.mkdir(parents=True, exist_ok=True)
        (unit_root / "task.json").write_bytes((task.root / "tests" / "task.json").read_bytes())
        (unit_root / "kernel.py").write_bytes(Path(materialized["kernel_path"]).read_bytes())
        (unit_root / "model.patch").write_bytes(
            (run_root / "snapshots" / "turn-3.patch").read_bytes()
        )
        (unit_root / "trajectory.json").write_bytes((run_root / "trajectory.json").read_bytes())
        metadata = {
            "unit_id": unit_id,
            "model_id": run["model_id"],
            "checkpoint": run["checkpoint"],
            "group": run["group"],
            "trajectory_count": run["trajectory_count"],
            "training_seed": run["training_seed"],
            "task_id": task.task_id,
            "family": task.family,
            "task_sha256": task.task_sha256,
            "seed": 0,
            "turn": 3,
            "patch_sha256": materialized["patch_sha256"],
            "kernel_sha256": materialized["kernel_sha256"],
            "trajectory_sha256": file_sha256(unit_root / "trajectory.json"),
        }
        _write(unit_root / "metadata.json", metadata)
        (unit_root / "workspace").rename(unit_root / "materialized-workspace")
        records.append(metadata)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pallas_g43_verifier_input_release",
        "sample_release_sha256": sample["release_sha256"],
        "benchmark_release_sha256": validation["release_sha256"],
        "counts": {"units": len(records)},
        "records": records,
    }
    manifest["release_sha256"] = canonical_sha256(manifest)
    _write(out_dir / "manifest.json", manifest)
    return manifest


def verify_release(
    *,
    verifier_root: Path,
    timeout_seconds: int = 180,
    health_command: list[str] | None = None,
) -> dict[str, Any]:
    manifest = _load(verifier_root / "manifest.json")
    payload = dict(manifest)
    expected = payload.pop("release_sha256", None)
    if (
        manifest.get("kind") != "pallas_g43_verifier_input_release"
        or canonical_sha256(payload) != expected
    ):
        raise G43ExperimentError("G43_VERIFIER_RELEASE_INVALID")
    result_records = []
    recovery_events = []
    for index, record in enumerate(manifest["records"], start=1):
        unit_root = verifier_root / "units" / record["unit_id"]
        output = verifier_root / "results" / record["unit_id"]
        reward_path = output / "reward.json"
        if reward_path.is_file():
            reward = _load(reward_path)
        else:
            result = run_fresh_verifier(
                task_path=unit_root / "task.json",
                kernel_path=unit_root / "kernel.py",
                output_dir=output,
                timeout_seconds=timeout_seconds,
            )
            reward = result["reward"]
        if (
            reward.get("task_id") != record["task_id"]
            or reward.get("kernel_sha256") != record["kernel_sha256"]
        ):
            raise G43ExperimentError(f"G43_VERIFIER_RESULT_MISMATCH: {record['unit_id']}")
        if reward.get("worker_recovery_required") is True:
            health = _worker_health(health_command)
            event = {"after_unit": record["unit_id"], **health}
            recovery_events.append(event)
            _write(verifier_root / "recovery-events.json", recovery_events)
            if not health["healthy"]:
                raise G43ExperimentError(f"G43_WORKER_QUARANTINED: {record['unit_id']}")
        result_records.append(
            {
                "unit_id": record["unit_id"],
                "reward": reward["reward"],
                "artifacts": _tree_hashes(output),
            }
        )
        print(
            f"G43_VERIFY completed={index}/{len(manifest['records'])} reward={reward['reward']}",
            flush=True,
        )
    verification: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pallas_g43_verification_release",
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
    _write(verifier_root / "verification.json", verification)
    return verification


def _model_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    verified = sum(row["reward"] == 1 for row in rows)
    speedups = [
        float(row["speedup"])
        for row in rows
        if row["reward"] == 1 and isinstance(row.get("speedup"), (int, float))
    ]
    return {
        "tasks": len(rows),
        "profile_verified": verified,
        "pass_rate": verified / len(rows),
        "median_verified_speedup": statistics.median(speedups) if speedups else None,
        "beats_xla": sum(value > 1.0 for value in speedups),
        "meets_headroom_threshold": sum(value >= 1.05 for value in speedups),
    }


def summarize_results(
    *, verifier_root: Path, admission_root: Path, out_path: Path
) -> dict[str, Any]:
    manifest = _load(verifier_root / "manifest.json")
    verification = _load(verifier_root / "verification.json")
    if verification.get("input_release_sha256") != manifest.get("release_sha256"):
        raise G43ExperimentError("G43_VERIFICATION_RELEASE_MISMATCH")
    verification_payload = dict(verification)
    verification_sha = verification_payload.pop("release_sha256", None)
    if canonical_sha256(verification_payload) != verification_sha:
        raise G43ExperimentError("G43_VERIFICATION_HASH_MISMATCH")
    if verification.get("counts", {}).get("infrastructure_failures") != 0:
        raise G43ExperimentError("G43_INFRASTRUCTURE_FAILURES_PRESENT")
    verified_by_unit = {record["unit_id"]: record for record in verification["records"]}
    rows = []
    for record in manifest["records"]:
        output = verifier_root / "results" / record["unit_id"]
        if _tree_hashes(output) != verified_by_unit[record["unit_id"]]["artifacts"]:
            raise G43ExperimentError(f"G43_RESULT_HASH_MISMATCH: {record['unit_id']}")
        rows.append({**record, **_load(output / "reward.json")})
    expected_cells = len({row["model_id"] for row in rows}) * 16
    if len(rows) != expected_cells or any(row["seed"] != 0 or row["turn"] != 3 for row in rows):
        raise G43ExperimentError("G43_RESULT_MATRIX_INCOMPLETE")
    by_model = {
        model_id: _model_summary([row for row in rows if row["model_id"] == model_id])
        for model_id in sorted({row["model_id"] for row in rows})
    }
    learning_curve = {}
    means = []
    for count in TRAJECTORY_COUNTS:
        seed_rows = []
        for seed in TRAINING_SEEDS:
            model_id = f"g43-n{count}-seed{seed}"
            summary = by_model[model_id]
            seed_rows.append(
                {
                    "training_seed": seed,
                    "profile_verified": summary["profile_verified"],
                    "pass_rate": summary["pass_rate"],
                }
            )
        rates = [row["pass_rate"] for row in seed_rows]
        mean_rate = statistics.fmean(rates)
        means.append(mean_rate)
        learning_curve[str(count)] = {
            "training_seeds": seed_rows,
            "mean_pass_rate": mean_rate,
            "seed_population_stddev": statistics.pstdev(rates),
            "minimum_pass_rate": min(rates),
            "maximum_pass_rate": max(rates),
        }
    slope = (means[2] - means[0]) / 2.0
    monotonic = means[0] <= means[1] <= means[2]
    base_rows = [row for row in rows if row["model_id"] == "inkling-small-base"]
    family_results = {}
    regressions = []
    for family in FAMILIES:
        base_count = sum(row["reward"] == 1 for row in base_rows if row["family"] == family)
        n32_counts = [
            sum(
                row["reward"] == 1
                for row in rows
                if row["model_id"] == f"g43-n32-seed{seed}" and row["family"] == family
            )
            for seed in TRAINING_SEEDS
        ]
        mean_count = statistics.fmean(n32_counts)
        family_results[family] = {
            "base_profile_verified": base_count,
            "n32_profile_verified_by_training_seed": n32_counts,
            "n32_mean_profile_verified": mean_count,
        }
        if mean_count < base_count:
            regressions.append(family)
    admission = _load(admission_root / "manifest.json")
    headroom_ids = set(admission["headroom_task_ids"])
    performance = {}
    for model_id in by_model:
        model_headroom = [
            row for row in rows if row["model_id"] == model_id and row["task_id"] in headroom_ids
        ]
        performance[model_id] = {
            "eligible_tasks": len(model_headroom),
            "profile_verified": sum(row["reward"] == 1 for row in model_headroom),
            "meets_1_05x": sum(
                row["reward"] == 1
                and isinstance(row.get("speedup"), (int, float))
                and row["speedup"] >= 1.05
                for row in model_headroom
            ),
        }
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pallas_g43_learning_curve_results",
        "verifier_release_sha256": manifest["release_sha256"],
        "admission_release_sha256": admission["release_sha256"],
        "rows": rows,
        "summary": {
            "models": by_model,
            "learning_curve": learning_curve,
            "slope_pass_rate_per_data_doubling": slope,
            "monotonic_non_decreasing": monotonic,
            "n32_family_comparison": family_results,
            "n32_regressions_vs_base": regressions,
            "performance_headroom_task_ids": sorted(headroom_ids),
            "performance_on_headroom_tasks": performance,
            "training_seed_variance_measured": True,
            "evaluation_sampling_seed": 0,
        },
        "decision": {
            "positive_learning_curve": monotonic and slope > 0,
            "flat_or_negative": slope <= 0,
            "no_n32_family_regression": not regressions,
            "dapt_authorized_by_g43": False,
        },
    }
    result["result_sha256"] = canonical_sha256(result)
    _write(out_path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-g43-experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    admit = commands.add_parser("admit")
    admit.add_argument("--benchmark-root", type=Path, required=True)
    admit.add_argument("--out-dir", type=Path, required=True)
    evaluation = commands.add_parser("build-evaluation")
    evaluation.add_argument("--config", type=Path, required=True)
    evaluation.add_argument("--benchmark-root", type=Path, required=True)
    evaluation.add_argument("--admission-root", type=Path, required=True)
    evaluation.add_argument("--learning-curve-root", type=Path, required=True)
    evaluation.add_argument("--training-root", type=Path, required=True)
    evaluation.add_argument("--g42-training-manifest", type=Path, required=True)
    evaluation.add_argument("--out-path", type=Path, required=True)
    sample = commands.add_parser("sample")
    sample.add_argument("--evaluation-config", type=Path, required=True)
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
    verify.add_argument("--timeout-seconds", type=int, default=180)
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--verifier-root", type=Path, required=True)
    summarize.add_argument("--admission-root", type=Path, required=True)
    summarize.add_argument("--out-path", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "admit":
            result = admit_benchmark(
                benchmark_root=args.benchmark_root,
                out_dir=args.out_dir,
            )
        elif args.command == "build-evaluation":
            result = build_evaluation_config(
                config_path=args.config,
                benchmark_root=args.benchmark_root,
                admission_root=args.admission_root,
                learning_curve_root=args.learning_curve_root,
                training_root=args.training_root,
                g42_training_manifest=args.g42_training_manifest,
                out_path=args.out_path,
            )
        elif args.command == "sample":
            result = sample_experiment(
                evaluation_config=args.evaluation_config,
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
            result = summarize_results(
                verifier_root=args.verifier_root,
                admission_root=args.admission_root,
                out_path=args.out_path,
            )
    except (
        G43ExperimentError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"G43_EXPERIMENT_ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
