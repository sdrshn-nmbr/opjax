"""Validate and freeze real-TPU admission evidence for G4.2 task releases."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

from opjax.pallas.environment import verify_static
from opjax.pallas.g42_harness import canonical_sha256, file_sha256, load_task_package


class G42AdmissionError(RuntimeError):
    """Admission evidence is incomplete, inconsistent, or not candidate-attributable."""


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise G42AdmissionError(f"JSON_OBJECT_REQUIRED: {path}")
    return value


def _extract_bundle(bundle: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    with tarfile.open(bundle, "r:gz") as archive:
        for member in archive.getmembers():
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                raise G42AdmissionError(f"BUNDLE_MEMBER_UNSAFE: {member.name}")
        archive.extractall(destination, filter="data")


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def build_admission_release(
    *, task_release: Path, benchmark_release: Path, bundle: Path, out_dir: Path
) -> dict[str, Any]:
    if out_dir.exists():
        raise G42AdmissionError(f"OUTPUT_EXISTS: {out_dir}")
    raw = out_dir / "raw"
    _extract_bundle(bundle, raw)
    task_manifest = _load_json(task_release / "manifest.json")
    benchmark_manifest = _load_json(benchmark_release / "manifest.json")
    stage_counts: Counter[str] = Counter()
    task_records = []
    for relative in task_manifest["tasks"]:
        package = load_task_package(task_release / relative)
        task_json = _load_json(package.root / "tests" / "task.json")
        solution = _load_json(raw / "g42-admission" / "solutions" / package.task_id / "result.json")
        starter_reward = _load_json(
            raw / "g42-admission-v2" / "starters" / package.task_id / "reward.json"
        )
        static = verify_static(
            f"```python\n{(package.root / 'solution' / 'kernel.py').read_text(encoding='utf-8')}\n```"
        )
        expected_stage = task_json["expected_initial_failure_stage"]
        if not static.passed:
            raise G42AdmissionError(f"SOLUTION_STATIC_FAILED: {package.task_id}:{static.stage}")
        if solution.get("passed") is not True or solution.get("stage") != "verified":
            raise G42AdmissionError(f"SOLUTION_NOT_VERIFIED: {package.task_id}")
        if solution.get("kernel_sha256") != task_json["reference_kernel_sha256"]:
            raise G42AdmissionError(f"SOLUTION_HASH_MISMATCH: {package.task_id}")
        if starter_reward.get("reward") != 0 or starter_reward.get("failure_stage") != expected_stage:
            raise G42AdmissionError(
                f"STARTER_FAILURE_MISMATCH: {package.task_id}:{starter_reward.get('failure_stage')}:{expected_stage}"
            )
        stage_counts[expected_stage] += 1
        task_records.append(
            {
                "task_id": package.task_id,
                "task_sha256": package.task_sha256,
                "solution_kernel_sha256": solution["kernel_sha256"],
                "starter_kernel_sha256": starter_reward["kernel_sha256"],
                "starter_failure_stage": expected_stage,
                "solution_speedup": solution["profile"]["speedup"],
            }
        )
    benchmark_records = []
    for relative in benchmark_manifest["tasks"]:
        package = load_task_package(benchmark_release / relative)
        reward = _load_json(raw / "g42-canary" / package.task_id / "reward.json")
        if reward.get("reward") != 1 or reward.get("profiled") is not True:
            raise G42AdmissionError(f"BENCHMARK_CANARY_FAILED: {package.task_id}")
        benchmark_records.append(
            {
                "task_id": package.task_id,
                "task_sha256": package.task_sha256,
                "kernel_sha256": reward["kernel_sha256"],
                "speedup": reward["speedup"],
            }
        )
    hashes = _tree_hashes(raw)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pallas_g42_admission_evidence",
        "task_release_sha256": task_manifest["release_sha256"],
        "benchmark_release_sha256": benchmark_manifest["release_sha256"],
        "counts": {
            "candidate_tasks": len(task_records),
            "verified_solutions": sum(1 for record in task_records if record["solution_kernel_sha256"]),
            "deterministic_failed_starters": sum(stage_counts.values()),
            "benchmark_canaries": len(benchmark_records),
        },
        "starter_failure_stages": dict(sorted(stage_counts.items())),
        "task_records": task_records,
        "benchmark_records": benchmark_records,
        "artifacts": hashes,
    }
    manifest["release_sha256"] = canonical_sha256(manifest)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def validate_admission_release(root: Path) -> dict[str, Any]:
    manifest = _load_json(root / "manifest.json")
    if manifest.get("kind") != "pallas_g42_admission_evidence":
        raise G42AdmissionError(f"RELEASE_KIND_INVALID: {root}")
    observed = _tree_hashes(root / "raw")
    if observed != manifest.get("artifacts"):
        raise G42AdmissionError(f"ARTIFACT_HASH_MISMATCH: {root}")
    expected_sha = manifest.pop("release_sha256", None)
    observed_sha = canonical_sha256(manifest)
    if expected_sha != observed_sha:
        raise G42AdmissionError(f"RELEASE_HASH_MISMATCH: {root}")
    return {"release_sha256": observed_sha, **manifest["counts"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-g42-admission")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--task-release", type=Path, required=True)
    build.add_argument("--benchmark-release", type=Path, required=True)
    build.add_argument("--bundle", type=Path, required=True)
    build.add_argument("--out-dir", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build_admission_release(
                task_release=args.task_release,
                benchmark_release=args.benchmark_release,
                bundle=args.bundle,
                out_dir=args.out_dir,
            )
        else:
            result = validate_admission_release(args.root)
    except (G42AdmissionError, OSError, ValueError, json.JSONDecodeError, tarfile.TarError) as exc:
        print(f"G42_ADMISSION_ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
