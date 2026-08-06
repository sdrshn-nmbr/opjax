from __future__ import annotations

import json
from pathlib import Path

import pytest

from opjax.pallas.g5_experiment import G5ExperimentError, build_evaluation_config


REPO_ROOT = Path(__file__).parents[2]


def test_g5_evaluation_binds_dapt_lineage_and_byte_identical_sft(
    tmp_path: Path,
) -> None:
    result = build_evaluation_config(
        config_path=REPO_ROOT / "config/pallas/g5-evaluation.json",
        benchmark_root=REPO_ROOT / "data/pallas/runs/g43-benchmark-release",
        admission_root=REPO_ROOT / "data/pallas/runs/g43-admission-evidence",
        control_results_path=REPO_ROOT / "data/pallas/runs/g43-results.json",
        d0_root=REPO_ROOT / "data/pallas/runs/g5-d0-training",
        s0_root=REPO_ROOT / "data/pallas/runs/g42-training",
        s1_root=REPO_ROOT / "data/pallas/runs/g5-s1-training",
        out_path=tmp_path / "evaluation.json",
    )

    assert [model["model_id"] for model in result["models"]] == ["g5-d0", "g5-s1"]
    assert result["controls"] == {
        "base": "inkling-small-base",
        "s0": "g42-repair-sft",
    }
    assert result["lineage"]["s1_parent_run_sha256"] == result["lineage"][
        "d0_run_sha256"
    ]
    assert result["lineage"]["s1_recipe_sha256"] == result["lineage"][
        "s0_recipe_sha256"
    ]


def test_g5_evaluation_rejects_nonidentical_s1_recipe(tmp_path: Path) -> None:
    s1_root = tmp_path / "s1"
    s1_root.mkdir()
    for name in ("manifest.json", "preparation.json"):
        source = REPO_ROOT / "data/pallas/runs/g5-s1-training" / name
        (s1_root / name).write_bytes(source.read_bytes())
    preparation = json.loads((s1_root / "preparation.json").read_text(encoding="utf-8"))
    preparation["training"]["learning_rate"] = 0.001
    (s1_root / "preparation.json").write_text(json.dumps(preparation), encoding="utf-8")

    with pytest.raises(G5ExperimentError, match="G5_SFT_RECIPE_MISMATCH"):
        build_evaluation_config(
            config_path=REPO_ROOT / "config/pallas/g5-evaluation.json",
            benchmark_root=REPO_ROOT / "data/pallas/runs/g43-benchmark-release",
            admission_root=REPO_ROOT / "data/pallas/runs/g43-admission-evidence",
            control_results_path=REPO_ROOT / "data/pallas/runs/g43-results.json",
            d0_root=REPO_ROOT / "data/pallas/runs/g5-d0-training",
            s0_root=REPO_ROOT / "data/pallas/runs/g42-training",
            s1_root=s1_root,
            out_path=tmp_path / "evaluation.json",
        )
