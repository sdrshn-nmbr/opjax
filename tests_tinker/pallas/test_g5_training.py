from __future__ import annotations

import json
from pathlib import Path

import pytest

from opjax.pallas.g5_training import prepare_g5_dapt, validate_g5_training_run
from opjax.pallas.training import TrainingError


REPO_ROOT = Path(__file__).parents[2]
CONFIG = REPO_ROOT / "config/pallas/g5-dapt.json"
CORPUS = REPO_ROOT / "data/pallas/runs/g5-dapt-corpus"


def test_g5_packing_conserves_unique_tokens_and_balances_training_lanes() -> None:
    preparation, rows, datums, order, _, validation = prepare_g5_dapt(
        config_path=CONFIG,
        corpus_root=CORPUS,
        repo_root=REPO_ROOT,
    )

    data = preparation["data"]
    assert data["raw_rows"] == 854
    assert data["unique_tokens"] == {
        "train": {"pallas": 1_372_316, "triton": 569_245},
        "validation": {"pallas": 289_151, "triton": 34_926},
    }
    assert data["truncated_tokens"] == 0
    assert data["maximum_sequence_tokens"] == 8192
    assert len(rows) == len(datums)
    assert len(order) % preparation["training"]["batch_size"] == 0
    assert sum(rows[index]["token_count"] for index in order) == data[
        "effective_train_tokens"
    ]
    lane_tokens = data["effective_train_tokens_by_lane"]
    lane_fraction = lane_tokens["pallas"] / sum(lane_tokens.values())
    assert lane_fraction == pytest.approx(0.5, abs=0.005)
    assert sum(row["token_count"] for row in validation["rows"]) == 324_077
    assert len(validation["rows"]) == len(validation["datums"])


def test_g5_preparation_rejects_composite_dataset_hash_change(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["dataset_sha256"] = "0" * 64
    path = tmp_path / "g5-dapt.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(TrainingError, match="G5_DAPT_DATASET_HASH_MISMATCH"):
        prepare_g5_dapt(config_path=path, corpus_root=CORPUS, repo_root=REPO_ROOT)


@pytest.mark.parametrize(
    ("run_name", "kind"),
    [
        ("g5-d0-training", "pallas_g5_dapt_run"),
        ("g5-s1-training", "pallas_g5_s1_run"),
    ],
)
def test_g5_live_training_release_is_self_consistent(run_name: str, kind: str) -> None:
    result = validate_g5_training_run(
        REPO_ROOT / "data/pallas/runs" / run_name,
        expected_kind=kind,
    )
    assert result["ok"] is True
    assert result["completed_steps"] > 0
