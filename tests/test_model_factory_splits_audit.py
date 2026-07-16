import json
from pathlib import Path

from opjax.model_factory.audit import audit_jsonl
from opjax.model_factory.splits import (
    SplitManifest,
    assert_manifest_disjoint,
    load_split_manifest,
    validate_no_train_on_sealed,
)


def test_sudarshanbench_manifest_is_disjoint_and_frozen():
    manifest = load_split_manifest(
        "docs/model-factory/02-sealed-eval/sudarshanbench/splits.json"
    )
    assert_manifest_disjoint(manifest)
    assert len(manifest.sealed) >= 1
    assert validate_no_train_on_sealed(manifest.train, manifest) == []


def test_contamination_detects_deepswe_and_sealed():
    manifest = SplitManifest(
        train=["a"],
        sealed=["b"],
        deepswe_report_split=["c"],
    )
    bad = validate_no_train_on_sealed(["b", "c", "d"], manifest)
    assert bad == ["b", "c"]


def test_audit_jsonl_counts(tmp_path: Path):
    path = tmp_path / "t.jsonl"
    records = [
        {
            "messages": [
                {"role": "user", "content": "fix it"},
                {"role": "assistant", "content": "done", "tool_calls": [{}]},
            ],
            "outcome": "success",
            "license": "MIT",
            "recovery": True,
        },
        {
            "messages": [
                {"role": "user", "content": "fix it"},
                {"role": "assistant", "content": "done", "tool_calls": [{}]},
            ],
            "outcome": "success",
            "license": "MIT",
            "recovery": True,
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    report = audit_jsonl(path)
    assert report.n_records == 2
    assert report.complete_trajectories == 2
    assert report.with_tool_use == 2
    assert report.duplicate_hashes == 1
    assert report.recovery_segments == 2
    assert report.outcome_counts["success"] == 2
