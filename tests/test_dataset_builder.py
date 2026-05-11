import json
from pathlib import Path

from opjax.isft.dataset import build_fake_click_isft_dataset


def test_build_fake_click_isft_dataset(tmp_path: Path) -> None:
    result = build_fake_click_isft_dataset(tmp_path, count_per_tier=1, seed=100)

    records_path = Path(result.records_path)
    image_dir = Path(result.image_dir)

    assert result.num_records == 3
    assert records_path.exists()
    assert len(list(image_dir.glob("*.png"))) == 3

    rows = [json.loads(line) for line in records_path.read_text().splitlines()]
    assert len(rows) == 3
    assert all(row["sft_target"].startswith("<start_function_call>call:click") for row in rows)
    assert all(row["rgt_target"].startswith("[THINK]") for row in rows)
    assert all(row["attempts"][-1]["verification"]["success"] for row in rows)
