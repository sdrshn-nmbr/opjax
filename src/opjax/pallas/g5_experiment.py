"""Build and summarize the matched Gate 5 DAPT evaluation."""

from __future__ import annotations

import argparse
import json
import statistics
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


class G5ExperimentError(RuntimeError):
    """The Gate 5 experiment or its lineage is invalid."""


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise G5ExperimentError(f"G5_JSON_INVALID: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise G5ExperimentError(f"G5_JSON_OBJECT_REQUIRED: {path}")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sampler_path(manifest: Mapping[str, Any], *, kind: str) -> str:
    sampler = manifest.get("sampler_weights")
    path = sampler.get("path") if isinstance(sampler, dict) else None
    if (
        manifest.get("status") != "completed"
        or manifest.get("kind") != kind
        or not isinstance(path, str)
        or not path.startswith("tinker://")
        or not isinstance(manifest.get("run_sha256"), str)
    ):
        raise G5ExperimentError(f"G5_TRAINING_MANIFEST_INVALID: {kind}")
    return path


def _recipe(preparation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "base_model": preparation.get("base_model"),
        "corpus_release_sha256": preparation.get("corpus_release_sha256"),
        "dataset_sha256": preparation.get("dataset_sha256"),
        "training": preparation.get("training"),
        "row_ids": preparation.get("row_ids"),
        "data": preparation.get("data"),
    }


def _validate_result_hash(result: Mapping[str, Any]) -> None:
    payload = dict(result)
    expected = payload.pop("result_sha256", None)
    if canonical_sha256(payload) != expected:
        raise G5ExperimentError("G5_CONTROL_RESULT_HASH_MISMATCH")


def build_evaluation_config(
    *,
    config_path: Path,
    benchmark_root: Path,
    admission_root: Path,
    control_results_path: Path,
    d0_root: Path,
    s0_root: Path,
    s1_root: Path,
    out_path: Path,
) -> dict[str, Any]:
    config = _load(config_path)
    if config.get("schema_version") != 1:
        raise G5ExperimentError("G5_EVALUATION_CONFIG_INVALID")
    benchmark = validate_benchmark_release(benchmark_root)
    admission = _load(admission_root / "manifest.json")
    controls = _load(control_results_path)
    _validate_result_hash(controls)
    if (
        benchmark["release_sha256"] != config["benchmark_release_sha256"]
        or admission.get("release_sha256") != config["admission_release_sha256"]
        or admission.get("benchmark_release_sha256") != benchmark["release_sha256"]
        or controls.get("result_sha256") != config["control_result_sha256"]
        or controls.get("admission_release_sha256") != admission["release_sha256"]
    ):
        raise G5ExperimentError("G5_EVALUATION_SOURCE_RELEASE_MISMATCH")
    d0_manifest = _load(d0_root / "manifest.json")
    s0_manifest = _load(s0_root / "manifest.json")
    s1_manifest = _load(s1_root / "manifest.json")
    d0_sampler = _sampler_path(d0_manifest, kind="pallas_g5_dapt_run")
    s0_sampler = _sampler_path(s0_manifest, kind="pallas_sft_run")
    s1_sampler = _sampler_path(s1_manifest, kind="pallas_g5_s1_run")
    d0_state = d0_manifest.get("final_state", {}).get("path")
    if (
        s1_manifest.get("parent_run_sha256") != d0_manifest["run_sha256"]
        or s1_manifest.get("initial_state_path") != d0_state
    ):
        raise G5ExperimentError("G5_S1_PARENT_MISMATCH")
    s0_preparation = _load(s0_root / "preparation.json")
    s1_preparation = _load(s1_root / "preparation.json")
    s0_recipe = canonical_sha256(_recipe(s0_preparation))
    s1_recipe = canonical_sha256(_recipe(s1_preparation))
    if s0_recipe != s1_recipe:
        raise G5ExperimentError("G5_SFT_RECIPE_MISMATCH")
    evaluation: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pallas_g5_evaluation_config",
        "experiment_id": config["experiment_id"],
        "benchmark_release_sha256": benchmark["release_sha256"],
        "admission_release_sha256": admission["release_sha256"],
        "control_result_sha256": controls["result_sha256"],
        "evaluation_seed": config["evaluation_seed"],
        "turn_limit": config["turn_limit"],
        "snapshot_turns": config["snapshot_turns"],
        "max_sampling_workers": config["max_sampling_workers"],
        "headroom_task_ids": admission["headroom_task_ids"],
        "controls": {"base": "inkling-small-base", "s0": "g42-repair-sft"},
        "lineage": {
            "d0_run_sha256": d0_manifest["run_sha256"],
            "s0_run_sha256": s0_manifest["run_sha256"],
            "s1_run_sha256": s1_manifest["run_sha256"],
            "s1_parent_run_sha256": s1_manifest["parent_run_sha256"],
            "s0_recipe_sha256": s0_recipe,
            "s1_recipe_sha256": s1_recipe,
        },
        "models": [
            {
                "model_id": "g5-d0",
                "checkpoint": d0_sampler,
                "group": "dapt",
                "trajectory_count": 0,
                "training_seed": 0,
                "checkpoint_run_sha256": d0_manifest["run_sha256"],
            },
            {
                "model_id": "g5-s1",
                "checkpoint": s1_sampler,
                "group": "dapt_sft",
                "trajectory_count": 32,
                "training_seed": 0,
                "checkpoint_run_sha256": s1_manifest["run_sha256"],
            },
        ],
        "control_checkpoints": {
            "s0": s0_sampler,
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
        raise G5ExperimentError("G5_VERIFICATION_RELEASE_INVALID")
    verified_by_unit = {record["unit_id"]: record for record in verification["records"]}
    rows = []
    for record in manifest["records"]:
        output = verifier_root / "results" / record["unit_id"]
        observed = _tree_hashes(output)
        if observed != verified_by_unit[record["unit_id"]]["artifacts"]:
            raise G5ExperimentError(f"G5_RESULT_HASH_MISMATCH: {record['unit_id']}")
        rows.append({**record, **_load(output / "reward.json")})
    return rows


def _validation_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    delta = float(after["mean_nll"]) - float(before["mean_nll"])
    return {
        "before_mean_nll": before["mean_nll"],
        "after_mean_nll": after["mean_nll"],
        "absolute_delta": delta,
        "relative_percent": delta / float(before["mean_nll"]) * 100,
        "by_lane": {
            lane: {
                "before_mean_nll": before["by_lane"][lane]["mean_nll"],
                "after_mean_nll": after["by_lane"][lane]["mean_nll"],
                "absolute_delta": float(after["by_lane"][lane]["mean_nll"])
                - float(before["by_lane"][lane]["mean_nll"]),
            }
            for lane in sorted(before["by_lane"])
        },
    }


def summarize_results(
    *,
    evaluation_config: Path,
    verifier_root: Path,
    control_results_path: Path,
    d0_manifest_path: Path,
    s1_manifest_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    evaluation = _load(evaluation_config)
    config_payload = dict(evaluation)
    expected_config = config_payload.pop("config_sha256", None)
    if canonical_sha256(config_payload) != expected_config:
        raise G5ExperimentError("G5_EVALUATION_CONFIG_HASH_MISMATCH")
    new_rows = _verified_rows(verifier_root)
    controls = _load(control_results_path)
    _validate_result_hash(controls)
    control_rows = [
        row
        for row in controls["rows"]
        if row["model_id"] in {evaluation["controls"]["base"], evaluation["controls"]["s0"]}
    ]
    rows = [*control_rows, *new_rows]
    model_ids = {"inkling-small-base", "g42-repair-sft", "g5-d0", "g5-s1"}
    if (
        {row["model_id"] for row in rows} != model_ids
        or len(rows) != 64
        or any(row["seed"] != 0 or row["turn"] != 3 for row in rows)
    ):
        raise G5ExperimentError("G5_RESULT_MATRIX_INCOMPLETE")
    task_sets = {
        model_id: {(row["task_id"], row["task_sha256"]) for row in rows if row["model_id"] == model_id}
        for model_id in model_ids
    }
    if len({frozenset(tasks) for tasks in task_sets.values()}) != 1:
        raise G5ExperimentError("G5_MATCHED_TASK_IDENTITY_MISMATCH")
    by_model = {
        model_id: _model_summary([row for row in rows if row["model_id"] == model_id])
        for model_id in sorted(model_ids)
    }
    comparisons = {
        "d0_vs_base": _paired_summary(
            [row for row in rows if row["model_id"] == "g5-d0"],
            [row for row in rows if row["model_id"] == "inkling-small-base"],
        ),
        "s1_vs_s0": _paired_summary(
            [row for row in rows if row["model_id"] == "g5-s1"],
            [row for row in rows if row["model_id"] == "g42-repair-sft"],
        ),
        "s1_vs_d0": _paired_summary(
            [row for row in rows if row["model_id"] == "g5-s1"],
            [row for row in rows if row["model_id"] == "g5-d0"],
        ),
    }
    families = sorted({row["family"] for row in rows})
    family_comparison = {
        family: {
            model_id: sum(
                row["reward"] == 1
                for row in rows
                if row["model_id"] == model_id and row["family"] == family
            )
            for model_id in sorted(model_ids)
        }
        for family in families
    }
    s1_regressions = [
        family
        for family, values in family_comparison.items()
        if values["g5-s1"] < values["g42-repair-sft"]
    ]
    d0_manifest = _load(d0_manifest_path)
    s1_manifest = _load(s1_manifest_path)
    d0_validation = d0_manifest["validation"]
    s1_validation = s1_manifest["validation"]
    parity_delta = float(s1_validation["before"]["mean_nll"]) - float(
        d0_validation["after"]["mean_nll"]
    )
    headroom_ids = set(evaluation["headroom_task_ids"])
    performance = {
        model_id: [
            {
                "task_id": row["task_id"],
                "reward": row["reward"],
                "speedup": row.get("speedup"),
            }
            for row in rows
            if row["model_id"] == model_id and row["task_id"] in headroom_ids
        ]
        for model_id in sorted(model_ids)
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pallas_g5_dapt_results",
        "evaluation_config_sha256": evaluation["config_sha256"],
        "control_result_sha256": controls["result_sha256"],
        "verifier_release_sha256": _load(verifier_root / "manifest.json")["release_sha256"],
        "verification_release_sha256": _load(verifier_root / "verification.json")["release_sha256"],
        "rows": rows,
        "summary": {
            "models": by_model,
            "paired_comparisons": comparisons,
            "family_comparison": family_comparison,
            "s1_regressions_vs_s0": s1_regressions,
            "validation": {
                "d0": _validation_delta(d0_validation["before"], d0_validation["after"]),
                "s1": _validation_delta(s1_validation["before"], s1_validation["after"]),
                "d0_after_vs_s1_before_absolute_nll_delta": parity_delta,
            },
            "performance_headroom": performance,
            "median_profile_verified_passes": statistics.median(
                summary["profile_verified"] for summary in by_model.values()
            ),
        },
        "decision": {
            "dapt_only_improves_base": comparisons["d0_vs_base"]["profile_verified_delta"] > 0,
            "dapt_before_sft_improves_s0": comparisons["s1_vs_s0"]["profile_verified_delta"] > 0,
            "dapt_before_sft_no_family_regression": not s1_regressions,
            "gate_complete": True,
            "advance_to_g6": True,
        },
    }
    result["result_sha256"] = canonical_sha256(result)
    _write(out_path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-g5-experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-evaluation")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--benchmark-root", type=Path, required=True)
    build.add_argument("--admission-root", type=Path, required=True)
    build.add_argument("--control-results", type=Path, required=True)
    build.add_argument("--d0-root", type=Path, required=True)
    build.add_argument("--s0-root", type=Path, required=True)
    build.add_argument("--s1-root", type=Path, required=True)
    build.add_argument("--out-path", type=Path, required=True)
    prepare = commands.add_parser("prepare-verifier")
    prepare.add_argument("--sample-root", type=Path, required=True)
    prepare.add_argument("--benchmark-root", type=Path, required=True)
    prepare.add_argument("--out-dir", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--verifier-root", type=Path, required=True)
    verify.add_argument("--timeout-seconds", type=int, default=180)
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--evaluation-config", type=Path, required=True)
    summarize.add_argument("--verifier-root", type=Path, required=True)
    summarize.add_argument("--control-results", type=Path, required=True)
    summarize.add_argument("--d0-manifest", type=Path, required=True)
    summarize.add_argument("--s1-manifest", type=Path, required=True)
    summarize.add_argument("--out-path", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "build-evaluation":
            result = build_evaluation_config(
                config_path=args.config,
                benchmark_root=args.benchmark_root,
                admission_root=args.admission_root,
                control_results_path=args.control_results,
                d0_root=args.d0_root,
                s0_root=args.s0_root,
                s1_root=args.s1_root,
                out_path=args.out_path,
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
                evaluation_config=args.evaluation_config,
                verifier_root=args.verifier_root,
                control_results_path=args.control_results,
                d0_manifest_path=args.d0_manifest,
                s1_manifest_path=args.s1_manifest,
                out_path=args.out_path,
            )
    except (G5ExperimentError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"G5_EXPERIMENT_ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
