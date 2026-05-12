"""Tests for the real-attempt iSFT pipeline (Phase 1C).

The Modal-side `Gemma4Inference.batch_click_attempts` is NOT exercised here
(it requires a Modal container). Instead we synthesize a JSONL of plausible
attempts and verify the local-side refinement pipeline correctly:
  - Re-generates the image deterministically from the seed
  - Passes through model attempts that already verify as success (no repair)
  - Routes failed attempts through the repairer
  - Skips error-marked attempts to a sidecar log
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opjax.actions import format_function_call
from opjax.isft.dataset import (
    build_real_click_isft_dataset,
    task_equals_attempt,
)
from opjax.isft.fake import FakeClickRepairer
from opjax.synthetic.click import generate_click_task


def _make_attempt(
    image_dir: Path,
    *,
    seed: int,
    tier: str,
    model_response: str,
) -> dict[str, object]:
    """Generate a task locally to extract its metadata, then build an attempt record.

    This mirrors what Modal's `batch_click_attempts` produces, byte-for-byte
    on the metadata side. The image stays in `image_dir`; the local builder
    will regenerate it on its own image_dir but the result is byte-identical
    because the seed determines the PIL output.
    """
    task = generate_click_task(image_dir, seed=seed, tier=tier)  # type: ignore[arg-type]
    return {
        "task_id": task.task_id,
        "seed": seed,
        "tier": tier,
        "prompt": task.prompt,
        "width": task.width,
        "height": task.height,
        "target_x": task.target_x,
        "target_y": task.target_y,
        "target_radius": task.target_radius,
        "model_response": model_response,
        "parsed_action": None,
        "parse_error": None,
        "verification": {
            "success": False,
            "reward": 0.0,
            "reason": "missed_target",
            "distance_px": None,
        },
        "seconds": 1.0,
    }


def _write_attempts_jsonl(path: Path, attempts: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for attempt in attempts:
            f.write(json.dumps(attempt, sort_keys=True) + "\n")


def test_real_pipeline_passes_successful_attempts_unchanged(tmp_path: Path) -> None:
    # Build an attempt whose model_response exactly clicks the target center.
    # The iSFT loop should mark it as success on round 0 and emit refined_output
    # == initial_output (no repair call).
    scratch = tmp_path / "scratch_images"
    task = generate_click_task(scratch, seed=42, tier="target")
    correct_call = format_function_call("click", {"x": task.target_x, "y": task.target_y})

    attempt = _make_attempt(scratch, seed=42, tier="target", model_response=correct_call)
    attempts_path = tmp_path / "attempts.jsonl"
    _write_attempts_jsonl(attempts_path, [attempt])

    out_dir = tmp_path / "real_isft"
    result = build_real_click_isft_dataset(
        out_dir,
        attempts_jsonl=attempts_path,
        repairer=FakeClickRepairer(),
    )

    assert result.num_records == 1
    rows = [json.loads(line) for line in Path(result.records_path).read_text().splitlines()]
    row = rows[0]
    assert row["initial_output"] == correct_call
    assert row["refined_output"] == correct_call
    assert row["attempts"][-1]["verification"]["success"] is True


def test_real_pipeline_repairs_failed_attempts(tmp_path: Path) -> None:
    # Model response is a click far from the target — verifier fails — fake
    # repairer returns the correct click via solve_click_task().
    scratch = tmp_path / "scratch_images"
    far_off = format_function_call("click", {"x": 1, "y": 1})

    attempt = _make_attempt(scratch, seed=99, tier="button", model_response=far_off)
    attempts_path = tmp_path / "attempts.jsonl"
    _write_attempts_jsonl(attempts_path, [attempt])

    out_dir = tmp_path / "real_isft"
    result = build_real_click_isft_dataset(
        out_dir,
        attempts_jsonl=attempts_path,
        repairer=FakeClickRepairer(),
    )

    rows = [json.loads(line) for line in Path(result.records_path).read_text().splitlines()]
    row = rows[0]
    assert row["initial_output"] == far_off
    assert row["initial_output"] != row["refined_output"]
    assert row["refined_output"].startswith("<start_function_call>call:click")
    # Final attempt should succeed after repair
    assert row["attempts"][-1]["verification"]["success"] is True
    # RGT format intact
    assert row["rgt_target"].startswith("[THINK]")


def test_real_pipeline_skips_error_attempts_to_sidecar(tmp_path: Path) -> None:
    # An attempt with an `error` key indicates Modal-side failure; should be
    # excluded from records.jsonl but logged in skipped_attempts.json.
    scratch = tmp_path / "scratch_images"
    good_task = generate_click_task(scratch, seed=7, tier="target")
    correct_call = format_function_call("click", {"x": good_task.target_x, "y": good_task.target_y})

    good = _make_attempt(scratch, seed=7, tier="target", model_response=correct_call)
    bad = {"task_id": "click-target-8", "seed": 8, "tier": "target", "error": "OOM at decode"}

    attempts_path = tmp_path / "attempts.jsonl"
    _write_attempts_jsonl(attempts_path, [good, bad])

    out_dir = tmp_path / "real_isft"
    result = build_real_click_isft_dataset(
        out_dir,
        attempts_jsonl=attempts_path,
        repairer=FakeClickRepairer(),
    )

    assert result.num_records == 1
    skipped_path = Path(out_dir) / "skipped_attempts.json"
    assert skipped_path.exists()
    skipped = json.loads(skipped_path.read_text())
    assert len(skipped) == 1
    assert skipped[0]["task_id"] == "click-target-8"


def test_task_regen_matches_attempt_metadata(tmp_path: Path) -> None:
    # The deterministic-seed contract: regenerating a task locally MUST produce
    # the same target_x/target_y/target_radius as the Modal-side metadata.
    # If synthetic.click ever changes its RNG protocol, this fails immediately.
    scratch = tmp_path / "scratch_images"
    attempt = _make_attempt(scratch, seed=1234, tier="distractors", model_response="")

    # Generate again in a different directory — should be byte-identical metadata.
    other = tmp_path / "other_images"
    regen = generate_click_task(other, seed=1234, tier="distractors")
    assert task_equals_attempt(regen, attempt)


def test_real_pipeline_rejects_max_rounds_zero(tmp_path: Path) -> None:
    attempts_path = tmp_path / "attempts.jsonl"
    attempts_path.write_text("")
    with pytest.raises(ValueError, match="max_rounds"):
        build_real_click_isft_dataset(
            tmp_path / "out",
            attempts_jsonl=attempts_path,
            repairer=FakeClickRepairer(),
            max_rounds=0,
        )
