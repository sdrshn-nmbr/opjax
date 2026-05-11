"""Dataset builders for Phase 1 iSFT/RGT experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from opjax.actions import format_function_call
from opjax.isft.fake import FakeClickRepairer
from opjax.isft.loop import ISFTRecord, run_isft_rgt
from opjax.synthetic.click import ClickTier, generate_click_task, verify_click_output


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

    root = Path(output_dir)
    image_dir = root / "images"
    records_path = root / "records.jsonl"
    root.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    records: list[ISFTRecord] = []
    repairer = FakeClickRepairer()
    current_seed = seed

    for tier in tiers:
        for _ in range(count_per_tier):
            task = generate_click_task(image_dir, seed=current_seed, tier=tier)
            initial_output = format_function_call("click", {"x": 0, "y": 0})
            record = run_isft_rgt(
                task,
                initial_output=initial_output,
                verifier=verify_click_output,
                repairer=repairer,
            )
            records.append(record)
            current_seed += 1

    with records_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dataset_row(), sort_keys=True) + "\n")

    return DatasetBuildResult(
        output_dir=str(root),
        records_path=str(records_path),
        image_dir=str(image_dir),
        num_records=len(records),
    )
