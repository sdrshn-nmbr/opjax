from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import opjax.pallas.hub_admission as admission_module
from opjax.pallas.hub_admission import (
    HubAdmissionError,
    build_hub_dapt_admission,
    validate_hub_dapt_admission,
)

ROW_RELEASE_SHA = "a" * 64
BASE_RELEASE_SHA = "b" * 64


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _candidate(candidate_id: str, content: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "dataset_id": "example/triton",
        "source_revision": "c" * 40,
        "source_path": "data.parquet",
        "source_row_id": candidate_id,
        "source_file_sha256": "d" * 64,
        "license": "mit",
        "row_license": ["MIT"],
        "role": "cross_kernel_code",
        "objective": "dapt_candidate",
        "content": content,
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "family_id": "cross_kernel_code:family",
        "provenance": {
            "repo_name": "example/repository",
            "commit_hash": "e" * 40,
        },
        "status": "curated_candidate",
        "rejection_reasons": [],
        "training_authorized": False,
    }


def _case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(
        admission_module,
        "validate_hub_row_release",
        lambda _root: {"release_sha256": ROW_RELEASE_SHA},
    )
    monkeypatch.setattr(
        admission_module,
        "validate_corpus_release",
        lambda _root: {"release_sha256": BASE_RELEASE_SHA},
    )
    row_root = tmp_path / "rows"
    base_root = tmp_path / "base"
    core = (
        "import triton\nimport triton.language as tl\n@triton.jit\n"
        "def kernel(x, out):\n"
        "    offsets = tl.arange(0, 128)\n"
        "    values = tl.load(x + offsets)\n"
        "    tl.store(out + offsets, values)\n"
    )
    helper = "@triton.jit\ndef identity(x):\n    return x\n"
    _write_jsonl(
        row_root / "row_candidates.jsonl",
        [_candidate("core", core), _candidate("helper", helper)],
    )
    (row_root / "manifest.json").write_text(
        json.dumps({"policy": {"jaxbench_scanned": True}}),
        encoding="utf-8",
    )
    _write_jsonl(
        base_root / "datasets" / "dapt.jsonl",
        [{"row_id": "base", "text": "def unrelated():\n    return 1\n"}],
    )
    config = {
        "schema_version": 2,
        "accepted_hub_row_release_sha256": ROW_RELEASE_SHA,
        "accepted_base_corpus_release_sha256": BASE_RELEASE_SHA,
        "minimum_lexical_tokens": 20,
        "near_duplicate_threshold": 0.9,
        "validation_source_fraction": 0.1,
        "required_kernel_signals": ["tl.arange", "tl.load", "tl.store"],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return {
        "repo_root": Path(__file__).parents[2],
        "row_root": row_root,
        "base_corpus_root": base_root,
        "config_path": config_path,
        "out_dir": tmp_path / "out",
    }


def test_public_admission_authorizes_structural_kernel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)

    manifest = build_hub_dapt_admission(**case)

    assert manifest["counts"]["publicly_admissible"] == 1
    assert manifest["counts"]["authorized_dapt"] == 1
    assert manifest["increase"] == {
        "public_candidate_percent": 100.0,
        "verified_train_ready_percent": 100.0,
    }
    assert manifest["counts"]["status"] == {"authorized": 1, "rejected": 1}
    assert manifest["counts"]["rejection_reasons"] == {
        "CONTENT_BELOW_TOKEN_THRESHOLD": 1,
        "KERNEL_STRUCTURE_MISSING": 1,
    }
    dapt_rows = list(
        admission_module._iter_jsonl(case["out_dir"] / "datasets" / "dapt.jsonl")
    )
    assert [row["row_id"] for row in dapt_rows] == ["core"]
    assert validate_hub_dapt_admission(case["out_dir"])["ok"] is True


def test_jaxbench_boundary_is_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    (case["row_root"] / "manifest.json").write_text(
        json.dumps({"policy": {"jaxbench_scanned": False}}),
        encoding="utf-8",
    )

    with pytest.raises(
        HubAdmissionError,
        match="HUB_ADMISSION_JAXBENCH_BOUNDARY_MISSING",
    ):
        build_hub_dapt_admission(**case)


def test_existing_dapt_duplicate_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    core = json.loads(
        (case["row_root"] / "row_candidates.jsonl").read_text().splitlines()[0]
    )["content"]
    _write_jsonl(
        case["base_corpus_root"] / "datasets" / "dapt.jsonl",
        [{"row_id": "base", "text": core}],
    )

    manifest = build_hub_dapt_admission(**case)

    assert manifest["counts"]["authorized_dapt"] == 0
    assert manifest["counts"]["rejection_reasons"]["BASE_DAPT_EXACT_DUPLICATE"] == 1


def test_validation_rejects_rehashed_invalid_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    build_hub_dapt_admission(**case)
    admission_path = case["out_dir"] / "admission.jsonl"
    rows = [json.loads(line) for line in admission_path.read_text().splitlines()]
    rows[1]["status"] = "authorized"
    rows[1]["training_authorized"] = True
    _write_jsonl(admission_path, rows)
    manifest_path = case["out_dir"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["admission.jsonl"] = hashlib.sha256(
        admission_path.read_bytes()
    ).hexdigest()
    payload = {
        key: value
        for key, value in manifest.items()
        if key not in {"created_at", "release_sha256"}
    }
    manifest["release_sha256"] = _canonical_sha256(payload)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        HubAdmissionError,
        match="HUB_ADMISSION_AUTHORIZATION_INVALID",
    ):
        validate_hub_dapt_admission(case["out_dir"])
