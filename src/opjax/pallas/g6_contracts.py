"""Frozen reward, feedback, and lineage contracts for Gate 6."""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from opjax.pallas.g42_harness import MANDATORY_STAGES, canonical_sha256


class G6ContractError(RuntimeError):
    """A Gate 6 experiment input violates its frozen causal contract."""


@dataclass(frozen=True)
class AdvantageBatch:
    raw_returns: tuple[tuple[float, ...], ...]
    advantages: tuple[tuple[float, ...], ...]
    mean: float
    standard_deviation: float
    trainable: bool


_ABSOLUTE_PATH = re.compile(r"(?<![\w.])/(?:[^\s:'\"]+/)+[^\s:'\"]*")
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|authorization|bearer|credential|password|secret|token)\b"
    r"\s*[:=]\s*[^\s,;]+"
)
_HIDDEN_NAME = re.compile(r"(?i)\b(hidden[\w.-]*|tests?|solutions?|references?)\b")


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise G6ContractError(f"G6_JSON_INVALID: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise G6ContractError(f"G6_JSON_OBJECT_REQUIRED: {path}")
    return value


def sanitize_diagnostic(value: object, *, limit: int = 4000) -> str:
    """Keep actionable compiler text while removing hidden paths and secrets."""
    text = str(value or "").replace("\x00", "")
    text = _ABSOLUTE_PATH.sub("<path>", text)
    text = _SECRET_ASSIGNMENT.sub(lambda match: f"{match.group(1)}=<redacted>", text)
    text = _HIDDEN_NAME.sub("<hidden>", text)
    text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    return text[:limit] if text else "No diagnostic text was produced."


def feedback_from_result(result: Mapping[str, Any]) -> str:
    """Render feedback visible to a non-held-out online curriculum trajectory."""
    if result.get("infrastructure_error") is True:
        raise G6ContractError("G6_INFRASTRUCTURE_RESULT_NOT_FEEDBACK")
    stage = str(result.get("stage", "verifier"))
    if result.get("passed") is True and stage == "verified":
        profile = result.get("profile") if isinstance(result.get("profile"), Mapping) else {}
        timing = profile.get("timing") if isinstance(profile.get("timing"), Mapping) else {}
        speedup = timing.get("speedup", profile.get("speedup"))
        candidate = timing.get("candidate_median_ms")
        baseline = timing.get("baseline_median_ms")
        if not all(isinstance(value, (int, float)) for value in (speedup, candidate, baseline)):
            raise G6ContractError("G6_VERIFIED_TIMING_MISSING")
        return (
            "VERIFIER_STAGE verified: Correct, authentic, normally lowered, and profiled. "
            f"speedup={float(speedup):.6f}x candidate_median_ms={float(candidate):.6f} "
            f"xla_median_ms={float(baseline):.6f} profile_marker=tpu_custom_call. "
            "The kernel is valid; continue only if you can improve measured speed without losing correctness."
        )
    descriptions = {
        "artifact_contract": "The submitted module violated the output contract.",
        "pallas_api": "The submitted module failed the authentic Pallas API inspection.",
        "tpu_compile": "The submitted kernel failed TPU compilation.",
        "full_shape_correctness": "The submitted kernel failed full-shape randomized correctness.",
        "normal_lowering": "The submitted kernel did not prove normal Pallas lowering.",
        "runtime_safety": "The submitted kernel failed TPU runtime safety.",
        "profile": "The submitted kernel failed profile-evidence capture.",
    }
    diagnostic = sanitize_diagnostic(result.get("error"))
    return f"VERIFIER_STAGE {stage}: {descriptions.get(stage, 'Verification failed.')}\nDIAGNOSTIC:\n{diagnostic}"


def kernel_score(result: Mapping[str, Any], *, correctness_bonus: float = 0.3) -> float:
    """Kevin-style score: zero unless fully verified, then bonus plus speedup."""
    if result.get("infrastructure_error") is True:
        return -1.0
    stages = result.get("stages")
    if (
        result.get("passed") is not True
        or result.get("stage") != "verified"
        or not isinstance(stages, Mapping)
        or not all(stages.get(stage) is True for stage in MANDATORY_STAGES)
    ):
        return 0.0
    profile = result.get("profile")
    if not isinstance(profile, Mapping):
        return 0.0
    speedup = profile.get("speedup")
    if speedup is None and isinstance(profile.get("timing"), Mapping):
        speedup = profile["timing"].get("speedup")
    if not isinstance(speedup, (int, float)) or not math.isfinite(speedup) or speedup <= 0:
        return 0.0
    return correctness_bonus + float(speedup)


def discounted_advantages(
    trajectory_scores: Sequence[Sequence[float]], *, gamma: float
) -> AdvantageBatch:
    """Normalize discounted future-sum returns over all trajectories and turns."""
    if not trajectory_scores or not 0 <= gamma <= 1:
        raise G6ContractError("G6_ADVANTAGE_INPUT_INVALID")
    turn_count = len(trajectory_scores[0])
    if turn_count == 0 or any(len(scores) != turn_count for scores in trajectory_scores):
        raise G6ContractError("G6_TRAJECTORY_SHAPE_INVALID")
    returns = []
    for scores in trajectory_scores:
        current = [0.0] * turn_count
        accumulator = 0.0
        for turn in range(turn_count - 1, -1, -1):
            accumulator = float(scores[turn]) + gamma * accumulator
            current[turn] = accumulator
        returns.append(tuple(current))
    flat = [value for trajectory in returns for value in trajectory]
    mean = statistics.fmean(flat)
    standard_deviation = statistics.pstdev(flat)
    if standard_deviation == 0:
        advantages = tuple(tuple(0.0 for _ in row) for row in returns)
        return AdvantageBatch(tuple(returns), advantages, mean, 0.0, False)
    advantages = tuple(
        tuple((value - mean) / standard_deviation for value in row) for row in returns
    )
    return AdvantageBatch(tuple(returns), advantages, mean, standard_deviation, True)


def load_g6_config(
    *, config_path: Path, task_manifest_path: Path, s0_manifest_path: Path, s1_manifest_path: Path
) -> dict[str, Any]:
    config = _load(config_path)
    task_manifest = _load(task_manifest_path)
    s0 = _load(s0_manifest_path)
    s1 = _load(s1_manifest_path)
    rollout = config.get("rollout", {})
    optimizer = config.get("optimizer", {})
    if (
        config.get("schema_version") != 1
        or config.get("base_model") != "thinkingmachines/Inkling-Small"
        or config.get("renderer") != "tml_v0"
        or task_manifest.get("release_sha256") != config.get("task_release_sha256")
        or rollout.get("parallel_trajectories") != 16
        or rollout.get("refinement_turns") != 4
        or rollout.get("tasks_per_step") != 8
        or rollout.get("discount_gamma") != 0.4
        or rollout.get("temperature") != 0.9
        or optimizer.get("updates_per_step") != 2
        or optimizer.get("grad_clip_norm") != 0.05
        or optimizer.get("constant_length_normalizer")
        != rollout.get("max_response_tokens")
    ):
        raise G6ContractError("G6_FROZEN_CONFIG_INVALID")
    selected = task_manifest.get("training_selection")
    if not isinstance(selected, list) or len(selected) != 32 or len(set(selected)) != 32:
        raise G6ContractError("G6_TASK_SELECTION_INVALID")
    expected_lanes = {
        "R0": ("S0", s0, "pallas_sft_run"),
        "R1": ("S1", s1, "pallas_g5_s1_run"),
    }
    lanes = config.get("lanes")
    if not isinstance(lanes, list) or {lane.get("lane_id") for lane in lanes} != set(expected_lanes):
        raise G6ContractError("G6_LANES_INVALID")
    for lane in lanes:
        parent_id, manifest, kind = expected_lanes[lane["lane_id"]]
        state = manifest.get("final_state", {}).get("path")
        sampler = manifest.get("sampler_weights", {}).get("path")
        if (
            lane.get("parent_id") != parent_id
            or lane.get("initial_state_path") != state
            or lane.get("initial_sampler_path") != sampler
            or manifest.get("kind") != kind
            or manifest.get("status") != "completed"
            or lane.get("parent_run_sha256") != manifest.get("run_sha256")
        ):
            raise G6ContractError(f"G6_LANE_LINEAGE_INVALID: {lane.get('lane_id')}")
    resolved = dict(config)
    resolved["config_sha256"] = canonical_sha256(config)
    resolved["task_ids"] = list(selected)
    return resolved
