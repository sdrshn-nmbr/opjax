"""Validation and fingerprinting for the frozen Pallas experiment contract."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONFIG_NAMES = (
    "sources.json",
    "experiment.json",
    "splits.json",
    "eval-policy.json",
    "sft-candidates.json",
)
EXPECTED_ARMS = {
    "A": [],
    "B": ["pallas_sft"],
    "C": ["kernel_domain_adaptive_lora", "pallas_sft"],
    "D": ["kernel_domain_adaptive_lora"],
}


class ContractError(ValueError):
    """The versioned Pallas experiment contract is invalid."""


@dataclass(frozen=True)
class ContractBundle:
    root: Path
    sources: dict[str, Any]
    experiment: dict[str, Any]
    splits: dict[str, Any]
    eval_policy: dict[str, Any]
    sft_candidates: dict[str, Any]
    sha256: str


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError(f"CONTRACT_MISSING: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"CONTRACT_JSON_INVALID: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"CONTRACT_NOT_OBJECT: {path}")
    if value.get("schema_version") != 1:
        raise ContractError(f"CONTRACT_SCHEMA_UNSUPPORTED: {path}")
    return value


def _canonical_sha256(values: list[dict[str, Any]]) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise ContractError(f"{code}: {detail}")


def load_contracts(root: Path) -> ContractBundle:
    root = root.resolve()
    loaded = [_load_object(root / name) for name in CONFIG_NAMES]
    sources, experiment, splits, eval_policy, sft_candidates = loaded
    _validate_sources(sources)
    _validate_experiment(experiment)
    _validate_splits(splits, sources)
    _validate_eval_policy(eval_policy, experiment)
    _validate_sft_candidates(sft_candidates, sources)
    return ContractBundle(
        root=root,
        sources=sources,
        experiment=experiment,
        splits=splits,
        eval_policy=eval_policy,
        sft_candidates=sft_candidates,
        sha256=_canonical_sha256(loaded),
    )


def _validate_sft_candidates(
    value: dict[str, Any],
    sources: dict[str, Any],
) -> None:
    _require(
        value.get("derivation_policy") == "audited_semantic_reduction",
        "SFT_DERIVATION_POLICY_INVALID",
        repr(value.get("derivation_policy")),
    )
    groups = value.get("groups")
    _require(isinstance(groups, list) and groups, "SFT_GROUPS_EMPTY", "groups")
    trainable_sources = {
        source["id"]
        for source in sources["sources"]
        if source["training_policy"] == "allowlisted_paths_only"
    }
    group_ids: set[str] = set()
    variant_ids: set[str] = set()
    for group in groups:
        _require(isinstance(group, dict), "SFT_GROUP_INVALID", repr(group))
        group_id = group.get("id")
        _require(
            isinstance(group_id, str) and group_id not in group_ids,
            "SFT_GROUP_ID_INVALID",
            repr(group_id),
        )
        group_ids.add(group_id)
        _require(
            group.get("source_id") in trainable_sources,
            "SFT_SOURCE_NOT_TRAINABLE",
            repr(group.get("source_id")),
        )
        for name in ("source_path", "source_function", "kernel_kind"):
            _require(
                isinstance(group.get(name), str) and group[name],
                "SFT_GROUP_FIELD_INVALID",
                f"{group_id}:{name}",
            )
        variants = group.get("variants")
        _require(
            isinstance(variants, list) and len(variants) >= 3,
            "SFT_VARIANTS_INSUFFICIENT",
            group_id,
        )
        for variant in variants:
            variant_id = variant.get("id") if isinstance(variant, dict) else None
            _require(
                isinstance(variant_id, str) and variant_id not in variant_ids,
                "SFT_VARIANT_ID_INVALID",
                repr(variant_id),
            )
            variant_ids.add(variant_id)
            shape = variant.get("shape")
            _require(
                isinstance(shape, list)
                and shape
                and all(isinstance(size, int) and size > 0 for size in shape),
                "SFT_VARIANT_SHAPE_INVALID",
                variant_id,
            )
            _require(
                isinstance(variant.get("operation"), str) and variant["operation"],
                "SFT_VARIANT_OPERATION_INVALID",
                variant_id,
            )


def _validate_sources(value: dict[str, Any]) -> None:
    sources = value.get("sources")
    _require(isinstance(sources, list) and sources, "SOURCES_EMPTY", "sources")
    ids: set[str] = set()
    for source in sources:
        _require(isinstance(source, dict), "SOURCE_INVALID", repr(source))
        source_id = source.get("id")
        _require(
            isinstance(source_id, str) and source_id,
            "SOURCE_ID_INVALID",
            repr(source_id),
        )
        _require(source_id not in ids, "SOURCE_ID_DUPLICATE", source_id)
        ids.add(source_id)
        revision = source.get("revision")
        _require(
            isinstance(revision, str)
            and len(revision) == 40
            and all(c in "0123456789abcdef" for c in revision),
            "SOURCE_REVISION_INVALID",
            source_id,
        )
        kind = source.get("kind")
        _require(
            kind in {"git", "hf_dataset"},
            "SOURCE_KIND_UNSUPPORTED",
            source_id,
        )
        _require(
            source.get("training_policy")
            in {"forbidden", "allowlisted_paths_only", "discovery_only"},
            "SOURCE_TRAINING_POLICY_INVALID",
            source_id,
        )
        if kind == "git":
            paths = source.get("allowlisted_paths")
            _require(
                isinstance(paths, list)
                and all(isinstance(path, str) and path for path in paths),
                "SOURCE_PATHS_INVALID",
                source_id,
            )
        if source.get("training_policy") == "allowlisted_paths_only":
            _require(
                source.get("license") not in {None, "", "unverified"},
                "SOURCE_LICENSE_UNVERIFIED",
                source_id,
            )
    jaxbench = next((s for s in sources if s["id"] == "jaxbench"), None)
    _require(jaxbench is not None, "JAXBENCH_SOURCE_MISSING", "jaxbench")
    _require(
        jaxbench["training_policy"] == "forbidden",
        "JAXBENCH_TRAINING_FORBIDDEN",
        "jaxbench must never train",
    )
    pallasbench = next((s for s in sources if s["id"] == "pallasbench"), None)
    _require(pallasbench is not None, "PALLASBENCH_SOURCE_MISSING", "pallasbench")
    _require(
        pallasbench["training_policy"] == "forbidden",
        "PALLASBENCH_TRAINING_FORBIDDEN",
        "PallasBench is benchmark mining evidence and must never train",
    )


def _validate_experiment(value: dict[str, Any]) -> None:
    _require(
        value.get("base_model") == "thinkingmachines/Inkling",
        "BASE_MODEL_INVALID",
        "base",
    )
    target = value.get("target", {})
    _require(target.get("accelerator") == "tpu", "TARGET_INVALID", "accelerator")
    _require(target.get("hardware") == "v5e", "TARGET_INVALID", "hardware")
    prompt = value.get("prompt", {})
    _require(
        prompt.get("scored_context") == "spec",
        "SCORED_CONTEXT_INVALID",
        "spec required",
    )
    _require(
        prompt.get("diagnostic_context") == "baseline",
        "DIAGNOSTIC_CONTEXT_INVALID",
        "baseline required",
    )
    sampling = value.get("sampling", {})
    seeds = sampling.get("seeds")
    _require(
        isinstance(seeds, list)
        and len(seeds) >= 2
        and all(isinstance(seed, int) and seed >= 0 for seed in seeds)
        and len(set(seeds)) == len(seeds),
        "SAMPLING_SEEDS_INVALID",
        repr(seeds),
    )
    retry_seed_stride = sampling.get("retry_seed_stride")
    _require(
        isinstance(retry_seed_stride, int) and retry_seed_stride > max(seeds),
        "RETRY_SEED_STRIDE_INVALID",
        repr(retry_seed_stride),
    )
    _require(
        isinstance(sampling.get("max_concurrency"), int)
        and sampling["max_concurrency"] > 0,
        "SAMPLING_CONCURRENCY_INVALID",
        repr(sampling.get("max_concurrency")),
    )
    readiness = value.get("sft_readiness", {})
    integer_minima = {
        "minimum_verified_rows": 32,
        "minimum_family_count": 8,
        "minimum_source_count": 2,
        "minimum_rows_per_family": 3,
    }
    for name, lower_bound in integer_minima.items():
        observed = readiness.get(name)
        _require(
            isinstance(observed, int)
            and not isinstance(observed, bool)
            and observed >= lower_bound,
            "SFT_READINESS_MINIMUM_INVALID",
            f"{name}={observed!r}",
        )
    for name, upper_bound in (
        ("maximum_source_fraction", 0.75),
        ("maximum_family_fraction", 0.25),
    ):
        observed = readiness.get(name)
        _require(
            isinstance(observed, (int, float))
            and not isinstance(observed, bool)
            and 0 < observed <= upper_bound,
            "SFT_READINESS_CONCENTRATION_INVALID",
            f"{name}={observed!r}",
        )
    _require(
        readiness.get("required_correctness_seeds") == [0, 1, 2],
        "SFT_READINESS_SEEDS_INVALID",
        repr(readiness.get("required_correctness_seeds")),
    )
    for name in (
        "require_full_declared_shapes",
        "require_normal_tpu_lowering",
        "require_profile_evidence",
        "require_zero_holdout_contamination",
    ):
        _require(
            readiness.get(name) is True,
            "SFT_READINESS_EVIDENCE_INVALID",
            name,
        )
    arms = value.get("arms", {})
    _require(set(arms) == set(EXPECTED_ARMS), "ARMS_INVALID", repr(sorted(arms)))
    for arm, objectives in EXPECTED_ARMS.items():
        _require(
            arms[arm].get("start") == value["base_model"],
            "ARM_START_INVALID",
            arm,
        )
        _require(
            arms[arm].get("objectives") == objectives, "ARM_OBJECTIVES_INVALID", arm
        )


def _validate_splits(value: dict[str, Any], sources: dict[str, Any]) -> None:
    train = value.get("train", {})
    development = value.get("development", {})
    public = value.get("public_evaluation", {})
    private = value.get("private_evaluation", {})
    source_ids = {source["id"] for source in sources["sources"]}
    forbidden_source_ids = set(train.get("forbidden_source_ids", []))
    _require(
        forbidden_source_ids <= source_ids,
        "TRAIN_FORBIDDEN_SOURCE_UNKNOWN",
        repr(sorted(forbidden_source_ids - source_ids)),
    )
    required_forbidden = {
        source["id"]
        for source in sources["sources"]
        if source["training_policy"] in {"forbidden", "discovery_only"}
    }
    _require(
        required_forbidden <= forbidden_source_ids,
        "SOURCE_FORBIDDEN_SPLIT_MISSING",
        repr(sorted(required_forbidden - forbidden_source_ids)),
    )
    _require(
        public.get("source_id") in source_ids, "PUBLIC_SOURCE_UNKNOWN", repr(public)
    )
    _require(
        public.get("source_id") in set(train.get("forbidden_source_ids", [])),
        "PUBLIC_SOURCE_NOT_FORBIDDEN_FROM_TRAIN",
        str(public.get("source_id")),
    )
    split_ids = {
        "train": set(train.get("task_ids", [])),
        "development": set(development.get("task_ids", [])),
        "public": set(public.get("task_ids", [])),
        "private": set(private.get("task_ids", [])),
    }
    names = list(split_ids)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = split_ids[left] & split_ids[right]
            _require(
                not overlap, "TASK_SPLIT_OVERLAP", f"{left}/{right}: {sorted(overlap)}"
            )
    _require(
        len(split_ids["public"]) == 50,
        "JAXBENCH_TASK_COUNT_INVALID",
        str(len(split_ids["public"])),
    )
    private_ready = private.get("generalization_claim_ready")
    _require(
        isinstance(private_ready, bool), "PRIVATE_READY_INVALID", repr(private_ready)
    )
    if not private.get("task_ids") or not private.get("family_ids"):
        _require(
            private_ready is False,
            "PRIVATE_GENERALIZATION_NOT_READY",
            "empty private split cannot support a generalization claim",
        )


def _validate_eval_policy(value: dict[str, Any], experiment: dict[str, Any]) -> None:
    _require(
        value.get("scored_prompt_context") == experiment["prompt"]["scored_context"],
        "POLICY_PROMPT_MISMATCH",
        "scored context",
    )
    _require(
        value.get("diagnostic_prompt_contexts")
        == [experiment["prompt"]["diagnostic_context"]],
        "POLICY_PROMPT_MISMATCH",
        "diagnostic context",
    )
    timing = value.get("timing", {})
    _require(
        timing.get("min_repeated_runs", 0) >= 3, "TIMING_REPEATS_INVALID", repr(timing)
    )
    _require(timing.get("num_warmup", 0) > 0, "TIMING_WARMUP_INVALID", repr(timing))
    _require(timing.get("num_iters", 0) > 0, "TIMING_ITERS_INVALID", repr(timing))
    threshold = timing.get("headline_speedup_threshold")
    _require(
        isinstance(threshold, (int, float)) and threshold > 1,
        "SPEEDUP_THRESHOLD_INVALID",
        repr(threshold),
    )
    authenticity = value.get("authenticity", {})
    _require(
        authenticity.get("require_empirical_tpu_lowering") is True,
        "LOWERING_EVIDENCE_POLICY_INVALID",
        repr(authenticity),
    )
    _require(
        authenticity.get("reject_interpret_mode") is True,
        "INTERPRET_POLICY_INVALID",
        repr(authenticity),
    )
    repetitions = authenticity.get("profile_repetitions")
    _require(
        isinstance(repetitions, int)
        and not isinstance(repetitions, bool)
        and repetitions >= 3,
        "PROFILE_REPETITIONS_INVALID",
        repr(repetitions),
    )
    _require(
        authenticity.get("compiler_marker") == "tpu_custom_call",
        "LOWERING_MARKER_INVALID",
        repr(authenticity.get("compiler_marker")),
    )
    runtime = value.get("runtime", {})
    expected_runtime = {
        "python": "3.10.12",
        "chex": "0.1.90",
        "jax": "0.6.2",
        "jaxlib": "0.6.2",
        "libtpu": "0.0.17",
    }
    _require(
        runtime == expected_runtime,
        "EVALUATION_RUNTIME_INVALID",
        repr(runtime),
    )


def source_by_id(bundle: ContractBundle, source_id: str) -> dict[str, Any]:
    for source in bundle.sources["sources"]:
        if source["id"] == source_id:
            return source
    raise ContractError(f"SOURCE_UNKNOWN: {source_id}")


def git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise ContractError(
            f"SOURCE_NOT_GIT_CHECKOUT: {path}: {exc.output.strip()}"
        ) from exc


def verify_source_checkout(bundle: ContractBundle, source_id: str, path: Path) -> str:
    source = source_by_id(bundle, source_id)
    observed = git_revision(path)
    expected = source["revision"]
    if observed != expected:
        raise ContractError(
            f"SOURCE_REVISION_MISMATCH: {source_id}: expected={expected} observed={observed}"
        )
    return observed


def contract_report(bundle: ContractBundle) -> dict[str, Any]:
    private = bundle.splits["private_evaluation"]
    return {
        "ok": True,
        "contract_sha256": bundle.sha256,
        "experiment_id": bundle.experiment["experiment_id"],
        "target": bundle.experiment["target"],
        "source_revisions": {
            source["id"]: source["revision"] for source in bundle.sources["sources"]
        },
        "split_counts": {
            "train": len(bundle.splits["train"]["task_ids"]),
            "development": len(bundle.splits["development"]["task_ids"]),
            "public": len(bundle.splits["public_evaluation"]["task_ids"]),
            "private": len(private["task_ids"]),
        },
        "generalization_claim_ready": private["generalization_claim_ready"],
    }
