"""Sample the frozen G4.3 matched agent matrix through Tinker."""

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
from opjax.pallas.g42_harness import (
    AGENT_IMAGE,
    canonical_sha256,
    file_sha256,
    load_task_package,
)
from opjax.pallas.g43_corpus import validate_benchmark_release


class G43SamplingError(RuntimeError):
    """The matched sample matrix is incomplete or violates its frozen contract."""


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise G43SamplingError(f"JSON_OBJECT_REQUIRED: {path}")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tracked_dirty(repo_root: Path) -> bool:
    process = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
    )
    return process.returncode != 0 or bool(process.stdout.strip())


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
        raise G43SamplingError("G43_EVALUATION_BENCHMARK_MISMATCH")
    config_payload = dict(config)
    expected_config_sha = config_payload.pop("config_sha256", None)
    if canonical_sha256(config_payload) != expected_config_sha:
        raise G43SamplingError("G43_EVALUATION_CONFIG_HASH_MISMATCH")
    if _tracked_dirty(repo_root):
        raise G43SamplingError(f"OPJAX_TRACKED_DIRTY: {repo_root}")
    tasks = [load_task_package(benchmark_root / relative) for relative in benchmark["tasks"]]
    jobs = [(model, task) for model in config["models"] for task in tasks]
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
                raise G43SamplingError(f"G43_SAMPLE_RESUME_MISMATCH: {run_id}")
            return {
                **model,
                "task_id": task.task_id,
                "task_sha256": task.task_sha256,
                "family": task.family,
                "run_path": f"runs/{run_id}",
                "trajectory_sha256": file_sha256(run_root / "trajectory.json"),
            }
        if run_root.exists():
            raise G43SamplingError(f"G43_SAMPLE_PARTIAL_RUN: {run_id}")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-g43-sample")
    parser.add_argument("--evaluation-config", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, default=Path("config/pallas"))
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = sample_experiment(
            evaluation_config=args.evaluation_config,
            config_root=args.config_root,
            benchmark_root=args.benchmark_root,
            repo_root=args.repo_root,
            out_dir=args.out_dir,
        )
    except (G43SamplingError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"G43_SAMPLING_ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
