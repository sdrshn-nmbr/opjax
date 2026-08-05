from __future__ import annotations

import json
from pathlib import Path

import pytest

from opjax.pallas.training import TrainingError, _prepare, _training_order


REPO_ROOT = Path(__file__).parents[2]
CONFIG_ROOT = REPO_ROOT / "config" / "pallas"
CORPUS_ROOT = REPO_ROOT / "data" / "pallas" / "runs" / "g41-environment-corpus"


def test_gate4_preparation_binds_verified_corpus_and_renderer() -> None:
    preparation, rows, datums, order, _ = _prepare(
        config_root=CONFIG_ROOT,
        corpus_root=CORPUS_ROOT,
        repo_root=REPO_ROOT,
    )

    assert preparation["base_model"] == "thinkingmachines/Inkling-Small"
    assert preparation["corpus_release_sha256"] == preparation["training"][
        "corpus_release_sha256"
    ]
    assert preparation["data"]["rows"] == 32
    assert preparation["data"]["sequence_tokens"] > 9104
    assert preparation["data"]["supervised_tokens"] == 5554
    assert preparation["data"]["maximum_sequence_tokens"] > 365
    assert len(rows) == len(datums) == len(order) == 32
    assert order == _training_order(rows, seed=0, num_epochs=1)


def test_gate4_preparation_rejects_dataset_hash_change(tmp_path: Path) -> None:
    config_root = tmp_path / "pallas"
    config_root.mkdir()
    for source in CONFIG_ROOT.glob("*.json"):
        (config_root / source.name).write_bytes(source.read_bytes())
    experiment_path = config_root / "experiment.json"
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    experiment["training"]["dataset_sha256"] = "0" * 64
    experiment_path.write_text(json.dumps(experiment), encoding="utf-8")

    with pytest.raises(TrainingError, match="SFT_DATASET_MISMATCH"):
        _prepare(
            config_root=config_root,
            corpus_root=CORPUS_ROOT,
            repo_root=REPO_ROOT,
        )
