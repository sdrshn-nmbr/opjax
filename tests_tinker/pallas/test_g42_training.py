from pathlib import Path

from opjax.pallas.g42_training import prepare_g42_training

REPO_ROOT = Path(__file__).parents[2]


def test_g42_renderer_rejects_no_rows_and_truncates_none() -> None:
    preparation, rows, datums, order, _ = prepare_g42_training(
        config_path=REPO_ROOT / "config/pallas/g42-training.json",
        trace_root=REPO_ROOT / "data/pallas/runs/g42-repair-traces",
        repo_root=REPO_ROOT,
    )
    assert len(rows) == len(datums) == len(order) == 192
    assert preparation["data"]["truncated_rows"] == 0
    assert preparation["data"]["maximum_sequence_tokens"] < 8192
    assert preparation["data"]["supervised_tokens"] > 0
