"""Run and assemble the matched G4.2 k=3 versus k=6 experiment."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from opjax.pallas.contracts import git_revision
from opjax.pallas.g42_agent import run_tinker_agent
from opjax.pallas.g42_curriculum import validate_benchmark_release
from opjax.pallas.g42_harness import (
    canonical_sha256,
    file_sha256,
    load_task_package,
    materialize_submission,
    summarize_horizons,
)


class G42ExperimentError(RuntimeError):
    """The matched experiment is incomplete or violates its frozen pairing contract."""


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise G42ExperimentError(f"JSON_OBJECT_REQUIRED: {path}")
    return value


def _tracked_dirty(repo_root: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
    )
    return result.returncode != 0 or bool(result.stdout.strip())


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
    if _tracked_dirty(repo_root):
        raise G42ExperimentError(f"OPJAX_TRACKED_DIRTY: {repo_root}")
    if out_dir.exists():
        raise G42ExperimentError(f"OUTPUT_EXISTS: {out_dir}")
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
    out_dir.mkdir(parents=True)

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
    with ThreadPoolExecutor(max_workers=config["max_workers"]) as executor:
        futures = {executor.submit(run, job): job for job in jobs}
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
                "task_sha256": task.task_sha256,
                "seed": run["seed"],
                "turn": turn,
                "patch_sha256": materialized["patch_sha256"],
                "kernel_sha256": materialized["kernel_sha256"],
            }
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


def summarize_results(*, verifier_root: Path, out_path: Path) -> dict[str, Any]:
    manifest = _load(verifier_root / "manifest.json")
    rows = []
    failures: dict[str, int] = {}
    halts = 0
    speedups = []
    for record in manifest["records"]:
        reward = _load(verifier_root / "results" / record["unit_id"] / "reward.json")
        row = {**record, **reward}
        rows.append(row)
        if reward["reward"] != 1:
            stage = reward["failure_stage"]
            failures[stage] = failures.get(stage, 0) + 1
            halts += int(stage == "runtime_safety")
        elif isinstance(reward.get("speedup"), (int, float)):
            speedups.append(reward["speedup"])
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
            "failure_stages": dict(sorted(failures.items())),
            "candidate_attributable_tpu_halts": halts,
            "verified_speedups": sorted(speedups),
        },
    }
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
        else:
            result = summarize_results(verifier_root=args.verifier_root, out_path=args.out_path)
    except (G42ExperimentError, OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print(f"G42_EXPERIMENT_ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
