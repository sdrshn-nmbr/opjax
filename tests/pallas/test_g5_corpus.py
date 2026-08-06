from __future__ import annotations

import json
from pathlib import Path

import pytest

from opjax.pallas.g5_corpus import (
    G5CorpusError,
    build_g5_dapt_release,
    combine_dapt_rows,
    validate_g5_dapt_release,
)


REPO_ROOT = Path(__file__).parents[2]
REPOSITORY_CORPUS = REPO_ROOT / "data/pallas/runs/g3-sft-ready-final"
HUB_CORPUS = REPO_ROOT / "data/pallas/runs/g3-hub-dapt-admission"
CONFIG = REPO_ROOT / "config/pallas/g5-dapt.json"


def _row(row_id: str, text: str, *, source: str) -> dict[str, object]:
    if source == "hub":
        return {
            "schema_version": 2,
            "row_id": row_id,
            "objective": "dapt",
            "text": text,
            "family_id": "cross_kernel_code:test",
            "split": "train",
            "sampling_weight": 1.0,
            "provenance": {
                "dataset_id": "test/dataset",
                "repository": "test/repository",
                "license": "mit",
            },
        }
    return {
        "schema_version": 1,
        "row_id": row_id,
        "objective": "dapt",
        "text": text,
        "family_id": "pallas:test",
        "provenance": {"source_id": source, "license": "Apache-2.0"},
    }


def test_g5_composite_release_uses_current_clean_sources(tmp_path: Path) -> None:
    manifest = build_g5_dapt_release(
        repo_corpus_root=REPOSITORY_CORPUS,
        hub_corpus_root=HUB_CORPUS,
        config_path=CONFIG,
        out_dir=tmp_path / "release",
    )

    assert manifest["counts"]["rows"] == 854
    assert manifest["counts"]["lanes"] == {"pallas": 179, "triton": 675}
    assert manifest["counts"]["sources"] == {
        "jax": 109,
        "maxtext": 11,
        "tokamax": 59,
    }
    assert manifest["counts"]["cross_lane_duplicates"] == 0
    assert manifest["policy"]["forbidden_source_ids"] == ["pallasbench"]
    assert validate_g5_dapt_release(tmp_path / "release")["ok"] is True

    rows = [
        json.loads(line)
        for line in (tmp_path / "release/datasets/dapt.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert not any(
        row["provenance"].get("source_id") == "pallasbench" for row in rows
    )
    repositories_by_split = {
        split: {row["repository"] for row in rows if row["split"] == split}
        for split in ("train", "validation")
    }
    assert repositories_by_split["train"].isdisjoint(
        repositories_by_split["validation"]
    )


def test_g5_composite_rejects_cross_lane_exact_duplicate() -> None:
    source = "@triton.jit\ndef kernel(x):\n    tl.store(x, tl.load(x))\n"
    with pytest.raises(G5CorpusError, match="G5_CROSS_LANE_EXACT_DUPLICATE"):
        combine_dapt_rows(
            repo_rows=[_row("repo", source, source="jax")],
            hub_rows=[_row("hub", source, source="hub")],
            validation_source_ids={"tokamax"},
            near_duplicate_threshold=0.9,
            forbidden_source_ids={"pallasbench"},
        )


def test_g5_composite_rejects_forbidden_repository_source() -> None:
    with pytest.raises(G5CorpusError, match="G5_FORBIDDEN_SOURCE"):
        combine_dapt_rows(
            repo_rows=[_row("forbidden", "def kernel():\n    pass\n", source="pallasbench")],
            hub_rows=[],
            validation_source_ids={"tokamax"},
            near_duplicate_threshold=0.9,
            forbidden_source_ids={"pallasbench"},
        )
