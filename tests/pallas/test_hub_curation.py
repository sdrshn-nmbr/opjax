from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import opjax.pallas.hub_curation as hub_curation_module
from opjax.pallas.contracts import ContractBundle
from opjax.pallas.hub_curation import (
    HubCurationError,
    curate_hub_rows,
    load_hub_curation_config,
    validate_hub_row_release,
)

REPOSITORY_ROOT = Path(__file__).parents[2]
DISCOVERY_SHA = "d" * 64
SOURCE_REVISION = "a" * 40


class FakeApi:
    def __init__(
        self,
        revision: str = SOURCE_REVISION,
        failure: Exception | None = None,
    ) -> None:
        self.revision = revision
        self.failure = failure
        self.calls: list[tuple[str, str, bool]] = []

    def dataset_info(
        self,
        repo_id: str,
        *,
        revision: str,
        files_metadata: bool,
    ) -> SimpleNamespace:
        self.calls.append((repo_id, revision, files_metadata))
        if self.failure is not None:
            raise self.failure
        return SimpleNamespace(sha=self.revision)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_repository(root: Path, files: dict[str, str]) -> str:
    root.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=opjax-test",
            "-c",
            "user.email=opjax@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _bundle(
    tmp_path: Path, forbidden: str = "def heldout(x):\n    return x + 1\n"
) -> tuple[ContractBundle, Path]:
    checkout = tmp_path / "jaxbench"
    revision = _git_repository(checkout, {"benchmarks/heldout.py": forbidden})
    bundle = ContractBundle(
        root=tmp_path,
        sources={"sources": [{"id": "jaxbench", "revision": revision}]},
        experiment={},
        splits={},
        eval_policy={},
        sha256="c" * 64,
    )
    return bundle, checkout


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _valid_row(row_id: str, code: str | None = None) -> dict[str, Any]:
    return {
        "id": row_id,
        "code": code
        or (
            "import triton\n"
            "import triton.language as tl\n"
            "@triton.jit\n"
            f"def kernel_{row_id}(x):\n"
            "    offset = tl.program_id(0)\n"
            "    return x + offset\n"
        ),
        "repository": "example/repository",
        "commit_hash": "b" * 40,
    }


def _case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    rows: Any | None = None,
    raw_source: str | None = None,
    dataset_id: str = "example/triton-kernels",
    source_overrides: dict[str, Any] | None = None,
    inventory_overrides: dict[str, Any] | None = None,
    decision_overrides: dict[str, Any] | None = None,
    forbidden: str = "def heldout(x):\n    return x + 1\n",
) -> dict[str, Any]:
    monkeypatch.setattr(
        hub_curation_module,
        "validate_hub_discovery_release",
        lambda _root: {"release_sha256": DISCOVERY_SHA},
    )
    bundle, jaxbench_root = _bundle(tmp_path, forbidden)
    source_path = tmp_path / "rows.json"
    source_path.write_text(
        raw_source
        if raw_source is not None
        else json.dumps(rows or [_valid_row("one")]),
        encoding="utf-8",
    )
    source = {
        "dataset_id": dataset_id,
        "revision": SOURCE_REVISION,
        "license": "mit",
        "role": "cross_kernel_code",
        "objective": "dapt_candidate",
        "adapter": "json_records",
        "paths": ["rows.json"],
        "row_id_field": "id",
        "content_field": "code",
        "required_fields": ["id", "code", "repository", "commit_hash"],
        "provenance_fields": ["repository", "commit_hash"],
        "required_kernel_signals": ["@triton.jit", "tl."],
        "max_rows": 20,
        "benchmark_source": False,
    }
    source.update(source_overrides or {})
    config = {
        "schema_version": 1,
        "accepted_discovery_release_sha256": DISCOVERY_SHA,
        "near_duplicate_threshold": 0.8,
        "benchmark_policy": "forbidden",
        "sources": [source],
    }
    config_path = tmp_path / "curation.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    inventory = {
        "dataset_id": dataset_id,
        "source_revision": SOURCE_REVISION,
        "license": "mit",
        "tags": [],
        "dataset_card": {},
        "files": [{"path": "rows.json"}],
        "inspection_failures": [],
        "gated": False,
    }
    inventory.update(inventory_overrides or {})
    decision = {
        "dataset_id": dataset_id,
        "source_revision": SOURCE_REVISION,
        "status": "candidate",
        "direct_training_authorized": False,
        "score": 12,
        "category": "cross_kernel_domain",
        "candidate_objectives": ["dapt_candidate"],
        "training_policy": "discovery_only",
        "risk_flags": [],
    }
    decision.update(decision_overrides or {})
    discovery_root = tmp_path / "discovery"
    discovery_root.mkdir()
    _write_jsonl(discovery_root / "candidates.jsonl", [inventory])
    _write_jsonl(discovery_root / "decisions.jsonl", [decision])

    def download(**kwargs: object) -> str:
        assert kwargs == {
            "repo_id": dataset_id,
            "filename": "rows.json",
            "repo_type": "dataset",
            "revision": SOURCE_REVISION,
        }
        return str(source_path)

    return {
        "bundle": bundle,
        "repo_root": REPOSITORY_ROOT,
        "discovery_root": discovery_root,
        "config_path": config_path,
        "jaxbench_root": jaxbench_root,
        "out_dir": tmp_path / "release",
        "api": FakeApi(),
        "downloader": download,
    }


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _rehash_release(root: Path) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative in manifest["artifacts"]:
        manifest["artifacts"][relative] = _sha256(root / relative)
    payload = {
        key: value
        for key, value in manifest.items()
        if key not in {"created_at", "release_sha256"}
    }
    manifest["release_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def test_config_rejects_ambiguous_license(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(
        tmp_path,
        monkeypatch,
        source_overrides={"license": "other"},
    )

    with pytest.raises(HubCurationError, match="HUB_ROW_LICENSE_AMBIGUOUS"):
        load_hub_curation_config(case["config_path"])


def test_config_rejects_benchmark_scope_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(
        tmp_path,
        monkeypatch,
        source_overrides={"benchmark_scope": "implementation_paths_only"},
    )

    with pytest.raises(HubCurationError, match="HUB_ROW_BENCHMARK_SCOPE_INVALID"):
        load_hub_curation_config(case["config_path"])


@pytest.mark.parametrize(
    ("inventory_overrides", "decision_overrides"),
    [
        ({"source_revision": "f" * 40}, {}),
        ({}, {"source_revision": "f" * 40}),
    ],
)
def test_source_pin_mismatch_is_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inventory_overrides: dict[str, Any],
    decision_overrides: dict[str, Any],
) -> None:
    case = _case(
        tmp_path,
        monkeypatch,
        inventory_overrides=inventory_overrides,
        decision_overrides=decision_overrides,
    )

    with pytest.raises(HubCurationError, match="HUB_ROW_SOURCE_REVISION_MISMATCH"):
        curate_hub_rows(**case)


def test_live_source_pin_mismatch_is_preserved_as_source_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    case["api"] = FakeApi(revision="f" * 40)

    manifest = curate_hub_rows(**case)

    assert manifest["counts"]["source_failures"] == 1
    result = _rows(case["out_dir"] / "source_results.jsonl")[0]
    assert result["status"] == "failed"
    assert result["failure"]["code"] == "HUB_ROW_LIVE_REVISION_MISMATCH"


def test_schema_drift_and_malformed_rows_are_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drift = _case(tmp_path / "drift", monkeypatch, raw_source=json.dumps({"rows": []}))
    drift_manifest = curate_hub_rows(**drift)
    drift_result = _rows(drift["out_dir"] / "source_results.jsonl")[0]
    assert drift_manifest["counts"]["source_failures"] == 1
    assert drift_result["failure"]["code"] == "HUB_ROW_SCHEMA_DRIFT"

    malformed = _case(
        tmp_path / "malformed",
        monkeypatch,
        rows=["not-an-object", {"id": "missing-everything"}],
    )
    malformed_manifest = curate_hub_rows(**malformed)
    malformed_rows = _rows(malformed["out_dir"] / "row_candidates.jsonl")
    assert malformed_manifest["counts"]["status"] == {"rejected": 2}
    assert all(row["status"] == "rejected" for row in malformed_rows)
    assert {"SCHEMA_INVALID", "CONTENT_MISSING", "ROW_PROVENANCE_INCOMPLETE"} <= set(
        malformed_rows[0]["rejection_reasons"]
    )


def test_discovery_objective_authorization_leakage_is_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(
        tmp_path,
        monkeypatch,
        decision_overrides={"direct_training_authorized": True},
    )

    with pytest.raises(HubCurationError, match="HUB_ROW_DISCOVERY_ROLE_LEAKAGE"):
        curate_hub_rows(**case)


def test_pallasbench_name_requires_explicit_benchmark_classification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(
        tmp_path,
        monkeypatch,
        dataset_id="example/pallasbench-training",
    )

    with pytest.raises(
        HubCurationError,
        match="HUB_ROW_BENCHMARK_CLASSIFICATION_REQUIRED",
    ):
        curate_hub_rows(**case)


def test_pallasbench_tag_is_benchmark_contamination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(
        tmp_path,
        monkeypatch,
        dataset_id="example/kernel-corpus",
        inventory_overrides={"tags": ["pallasbench"]},
    )

    curate_hub_rows(**case)

    row = _rows(case["out_dir"] / "row_candidates.jsonl")[0]
    assert row["status"] == "rejected"
    assert "BENCHMARK_CONTAMINATION" in row["rejection_reasons"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repository", "example/tritonbench"),
        ("file_path", "benchmarks/kernel.py"),
    ],
)
def test_row_provenance_benchmark_contamination_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    row = _valid_row("one")
    row["file_path"] = "src/kernel.py"
    row[field] = value
    case = _case(
        tmp_path,
        monkeypatch,
        rows=[row],
        source_overrides={
            "required_fields": ["id", "code", "repository", "file_path", "commit_hash"],
            "provenance_fields": ["repository", "file_path", "commit_hash"],
        },
    )

    curate_hub_rows(**case)

    rows = _rows(case["out_dir"] / "row_candidates.jsonl")
    assert rows[0]["status"] == "rejected"
    assert rows[0]["rejection_reasons"] == [
        f"BENCHMARK_CONTAMINATION:ROW_PROVENANCE:{field}"
    ]


def test_jaxbench_exact_and_near_contamination_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = _valid_row("exact")["code"]
    near = exact.replace("return x + offset", "return x + offset + 0")
    case = _case(
        tmp_path,
        monkeypatch,
        rows=[_valid_row("exact", exact), _valid_row("near", near)],
        forbidden=exact,
    )

    curate_hub_rows(**case)

    rows = {
        row["source_row_id"]: row
        for row in _rows(case["out_dir"] / "row_candidates.jsonl")
    }
    assert any(
        reason.startswith("JAXBENCH_EXACT_CONTAMINATION:")
        for reason in rows["exact"]["rejection_reasons"]
    )
    assert any(
        reason.startswith("JAXBENCH_NEAR_CONTAMINATION:")
        for reason in rows["near"]["rejection_reasons"]
    )


def test_exact_and_near_duplicate_rows_are_rejected_deterministically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _valid_row("a")["code"]
    near = original.replace("return x + offset", "return x + offset + 0")
    case = _case(
        tmp_path,
        monkeypatch,
        rows=[
            _valid_row("a", original),
            _valid_row("b", original),
            _valid_row("c", near),
        ],
    )

    curate_hub_rows(**case)

    rows = {
        row["source_row_id"]: row
        for row in _rows(case["out_dir"] / "row_candidates.jsonl")
    }
    assert rows["a"]["status"] == "curated_candidate"
    assert rows["b"]["rejection_reasons"] == [
        f"EXACT_DUPLICATE:{rows['a']['candidate_id']}"
    ]
    assert rows["c"]["rejection_reasons"] == [
        f"NEAR_DUPLICATE:{rows['a']['candidate_id']}"
    ]


def test_completed_resume_is_deterministic_and_rejects_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    first = curate_hub_rows(**case)
    api = FakeApi(failure=AssertionError("completed resume fetched source"))
    case["api"] = api

    resumed = curate_hub_rows(**case, resume=True)

    assert resumed["release_sha256"] == first["release_sha256"]
    assert api.calls == []
    config = json.loads(case["config_path"].read_text(encoding="utf-8"))
    config["near_duplicate_threshold"] = 0.9
    case["config_path"].write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(HubCurationError, match="HUB_ROW_RESUME_FINGERPRINT_MISMATCH"):
        curate_hub_rows(**case, resume=True)


def test_source_failure_detail_survives_release_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    case["api"] = FakeApi(
        failure=RuntimeError("REMOTE_DATASET_UNAVAILABLE: retry exhausted")
    )

    manifest = curate_hub_rows(**case)

    result = _rows(case["out_dir"] / "source_results.jsonl")[0]
    assert manifest["counts"]["source_failures"] == 1
    assert result["failure"] == {
        "code": "REMOTE_DATASET_UNAVAILABLE",
        "detail": "REMOTE_DATASET_UNAVAILABLE: retry exhausted",
        "error_type": "RuntimeError",
    }
    assert validate_hub_row_release(case["out_dir"])["ok"] is True


def test_release_validation_rejects_artifact_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path, monkeypatch)
    curate_hub_rows(**case)
    with (case["out_dir"] / "row_candidates.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write("{}\n")

    with pytest.raises(HubCurationError, match="HUB_ROW_ARTIFACT_HASH_MISMATCH"):
        validate_hub_row_release(case["out_dir"])


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("source_revision", "f" * 40, "HUB_ROW_SOURCE_BINDING_INVALID"),
        ("provenance", {}, "HUB_ROW_PROVENANCE_INVALID"),
    ],
)
def test_release_validation_rejects_rehashed_semantic_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
    error: str,
) -> None:
    case = _case(tmp_path, monkeypatch)
    curate_hub_rows(**case)
    path = case["out_dir"] / "row_candidates.jsonl"
    rows = _rows(path)
    rows[0][field] = value
    _write_jsonl(path, rows)
    _rehash_release(case["out_dir"])

    with pytest.raises(HubCurationError, match=error):
        validate_hub_row_release(case["out_dir"])
