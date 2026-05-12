"""Dataset builders for Phase 1 iSFT/RGT experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from opjax.actions import format_function_call
from opjax.isft.fake import FakeClickRepairer
from opjax.isft.loop import ISFTRecord, Repairer, run_isft_rgt
from opjax.synthetic.click import (
    ClickTask,
    ClickTier,
    generate_click_task,
    verify_click_output,
)


DEFAULT_TIERS: tuple[ClickTier, ...] = ("target", "distractors", "button")


@dataclass(frozen=True)
class DatasetBuildResult:
    output_dir: str
    records_path: str
    image_dir: str
    num_records: int


def build_fake_click_isft_dataset(
    output_dir: str | Path,
    *,
    count_per_tier: int,
    seed: int = 0,
    tiers: Iterable[ClickTier] = DEFAULT_TIERS,
) -> DatasetBuildResult:
    if count_per_tier < 1:
        raise ValueError("count_per_tier must be at least 1")

    image_dir, records_path = _prepare_output_paths(output_dir)
    repairer = FakeClickRepairer()
    initial_zero = format_function_call("click", {"x": 0, "y": 0})

    records: list[ISFTRecord] = []
    current_seed = seed
    for tier in tiers:
        for _ in range(count_per_tier):
            task = generate_click_task(image_dir, seed=current_seed, tier=tier)
            record = run_isft_rgt(
                task,
                initial_output=initial_zero,
                verifier=verify_click_output,
                repairer=repairer,
            )
            records.append(record)
            current_seed += 1

    return _write_jsonl(records, image_dir, records_path)


def build_real_click_isft_dataset(
    output_dir: str | Path,
    *,
    attempts_jsonl: str | Path,
    repairer: Repairer,
    max_rounds: int = 3,
) -> DatasetBuildResult:
    """Refine a JSONL of raw Gemma 4 base-model attempts into iSFT records.

    Each attempt record (one JSONL line, produced by Modal's
    `Gemma4Inference.batch_click_attempts`) carries:
      - the deterministic (seed, tier) of the task, so we regenerate the
        image LOCALLY (byte-identical to Modal's because the generator is
        deterministic).
      - the model's raw response string, which becomes the `initial_output`
        slot in the iSFT loop.
      - verification metadata for inspection (we re-verify locally to avoid
        trust issues with the JSONL).

    The repairer is invoked only on failed attempts. Successful attempts pass
    the verifier on the first iSFT round and emit a minimal RGT strategy.
    """
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")

    image_dir, records_path = _prepare_output_paths(output_dir)
    attempts = _load_attempts_jsonl(attempts_jsonl)

    records: list[ISFTRecord] = []
    skipped: list[dict[str, object]] = []
    for attempt in attempts:
        if "error" in attempt:
            skipped.append({"task_id": attempt.get("task_id"), "error": attempt["error"]})
            continue

        task = generate_click_task(
            image_dir,
            seed=int(attempt["seed"]),
            tier=attempt["tier"],  # type: ignore[arg-type]
        )
        initial_output = str(attempt.get("model_response", ""))

        record = run_isft_rgt(
            task,
            initial_output=initial_output,
            verifier=verify_click_output,
            repairer=repairer,
            max_rounds=max_rounds,
        )
        records.append(record)

    result = _write_jsonl(records, image_dir, records_path)
    if skipped:
        skip_path = Path(result.output_dir) / "skipped_attempts.json"
        skip_path.write_text(json.dumps(skipped, indent=2, sort_keys=True))
    return result


def _prepare_output_paths(output_dir: str | Path) -> tuple[Path, Path]:
    root = Path(output_dir)
    image_dir = root / "images"
    records_path = root / "records.jsonl"
    root.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    return image_dir, records_path


def _write_jsonl(
    records: list[ISFTRecord],
    image_dir: Path,
    records_path: Path,
) -> DatasetBuildResult:
    with records_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dataset_row(), sort_keys=True) + "\n")
    return DatasetBuildResult(
        output_dir=str(records_path.parent),
        records_path=str(records_path),
        image_dir=str(image_dir),
        num_records=len(records),
    )


def _load_attempts_jsonl(path: str | Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _task_summary_from_attempt(attempt: dict[str, object]) -> dict[str, object]:
    """Project an attempt record onto its task-identifying fields."""
    return {
        "task_id": attempt["task_id"],
        "seed": attempt["seed"],
        "tier": attempt["tier"],
        "width": attempt["width"],
        "height": attempt["height"],
        "target_x": attempt["target_x"],
        "target_y": attempt["target_y"],
        "target_radius": attempt["target_radius"],
    }


def task_equals_attempt(task: ClickTask, attempt: dict[str, object]) -> bool:
    """Verify a locally-regenerated task matches the Modal-side metadata.

    Used as a defensive check: if the synthetic.click generator ever changes
    (different default sizes, different RNG protocol), this catches the drift
    immediately rather than producing silently mis-matched dataset rows.
    """
    return (
        task.task_id == attempt["task_id"]
        and task.tier == attempt["tier"]
        and task.target_x == attempt["target_x"]
        and task.target_y == attempt["target_y"]
        and task.target_radius == attempt["target_radius"]
        and task.width == attempt["width"]
        and task.height == attempt["height"]
    )
