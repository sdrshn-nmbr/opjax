"""Validation and fingerprinting for the frozen Pallas experiment contract."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONFIG_NAMES = ("sources.json", "experiment.json", "splits.json", "eval-policy.json")
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
    sources, experiment, splits, eval_policy = loaded
    _validate_sources(sources)
    _validate_experiment(experiment)
    _validate_splits(splits, sources)
    _validate_eval_policy(eval_policy, experiment)
    return ContractBundle(
        root=root,
        sources=sources,
        experiment=experiment,
        splits=splits,
        eval_policy=eval_policy,
        sha256=_canonical_sha256(loaded),
    )


def _validate_sources(value: dict[str, Any]) -> None:
    sources = value.get("sources")
    _require(isinstance(sources, list) and sources, "SOURCES_EMPTY", "sources")
    ids: set[str] = set()
    for source in sources:
        _require(isinstance(source, dict), "SOURCE_INVALID", repr(source))
        source_id = source.get("id")
        _require(isinstance(source_id, str) and source_id, "SOURCE_ID_INVALID", repr(source_id))
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
        _require(source.get("kind") == "git", "SOURCE_KIND_UNSUPPORTED", source_id)
        _require(
            source.get("training_policy")
            in {"forbidden", "allowlisted_paths_only"},
            "SOURCE_TRAINING_POLICY_INVALID",
            source_id,
        )
    jaxbench = next((s for s in sources if s["id"] == "jaxbench"), None)
    _require(jaxbench is not None, "JAXBENCH_SOURCE_MISSING", "jaxbench")
    _require(
        jaxbench["training_policy"] == "forbidden",
        "JAXBENCH_TRAINING_FORBIDDEN",
        "jaxbench must never train",
    )


def _validate_experiment(value: dict[str, Any]) -> None:
    _require(value.get("base_model") == "thinkingmachines/Inkling", "BASE_MODEL_INVALID", "base")
    target = value.get("target", {})
    _require(target.get("accelerator") == "tpu", "TARGET_INVALID", "accelerator")
    _require(target.get("hardware") == "v5e", "TARGET_INVALID", "hardware")
    prompt = value.get("prompt", {})
    _require(prompt.get("scored_context") == "spec", "SCORED_CONTEXT_INVALID", "spec required")
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
        isinstance(retry_seed_stride, int)
        and retry_seed_stride > max(seeds),
        "RETRY_SEED_STRIDE_INVALID",
        repr(retry_seed_stride),
    )
    _require(
        isinstance(sampling.get("max_concurrency"), int)
        and sampling["max_concurrency"] > 0,
        "SAMPLING_CONCURRENCY_INVALID",
        repr(sampling.get("max_concurrency")),
    )
    arms = value.get("arms", {})
    _require(set(arms) == set(EXPECTED_ARMS), "ARMS_INVALID", repr(sorted(arms)))
    for arm, objectives in EXPECTED_ARMS.items():
        _require(
            arms[arm].get("start") == value["base_model"],
            "ARM_START_INVALID",
            arm,
        )
        _require(arms[arm].get("objectives") == objectives, "ARM_OBJECTIVES_INVALID", arm)


def _validate_splits(value: dict[str, Any], sources: dict[str, Any]) -> None:
    train = value.get("train", {})
    development = value.get("development", {})
    public = value.get("public_evaluation", {})
    private = value.get("private_evaluation", {})
    source_ids = {source["id"] for source in sources["sources"]}
    _require(public.get("source_id") in source_ids, "PUBLIC_SOURCE_UNKNOWN", repr(public))
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
            _require(not overlap, "TASK_SPLIT_OVERLAP", f"{left}/{right}: {sorted(overlap)}")
    _require(len(split_ids["public"]) == 50, "JAXBENCH_TASK_COUNT_INVALID", str(len(split_ids["public"])))
    private_ready = private.get("generalization_claim_ready")
    _require(isinstance(private_ready, bool), "PRIVATE_READY_INVALID", repr(private_ready))
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
    _require(timing.get("min_repeated_runs", 0) >= 3, "TIMING_REPEATS_INVALID", repr(timing))
    _require(timing.get("num_warmup", 0) > 0, "TIMING_WARMUP_INVALID", repr(timing))
    _require(timing.get("num_iters", 0) > 0, "TIMING_ITERS_INVALID", repr(timing))
    threshold = timing.get("headline_speedup_threshold")
    _require(isinstance(threshold, (int, float)) and threshold > 1, "SPEEDUP_THRESHOLD_INVALID", repr(threshold))


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
        raise ContractError(f"SOURCE_NOT_GIT_CHECKOUT: {path}: {exc.output.strip()}") from exc


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
