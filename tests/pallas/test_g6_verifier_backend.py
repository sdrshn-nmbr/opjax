from __future__ import annotations

import json
from pathlib import Path

import pytest

from opjax.pallas.g6_verifier_backend import (
    G6VerifierBackendError,
    VerifierCandidate,
    _validated_results,
    build_batch,
)


def test_batch_manifest_binds_task_and_kernel_bytes(tmp_path: Path) -> None:
    task = tmp_path / "task.json"
    kernel = tmp_path / "kernel.py"
    task.write_text('{"task_id":"t"}\n', encoding="utf-8")
    kernel.write_text("def workload(x):\n    return x\n", encoding="utf-8")
    root = tmp_path / "batch"
    manifest = build_batch(
        candidates=[VerifierCandidate("u0", task, kernel)],
        batch_id="batch-0",
        out_dir=root,
    )
    assert manifest["counts"] == {"units": 1}
    assert manifest["records"][0]["unit_id"] == "u0"
    kernel.write_text("changed", encoding="utf-8")
    assert (root / "units/u0/kernel.py").read_text() != "changed"


def test_result_validation_rejects_wrong_input_release(tmp_path: Path) -> None:
    task = tmp_path / "task.json"
    kernel = tmp_path / "kernel.py"
    task.write_text("{}", encoding="utf-8")
    kernel.write_text("x", encoding="utf-8")
    root = tmp_path / "batch"
    build_batch(
        candidates=[VerifierCandidate("u0", task, kernel)],
        batch_id="batch-0",
        out_dir=root,
    )
    (root / "results.json").write_text(
        json.dumps(
            {
                "kind": "pallas_g6_verifier_batch_results",
                "input_release_sha256": "wrong",
                "counts": {"units": 1},
                "records": [],
                "release_sha256": "wrong",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(G6VerifierBackendError, match="RESULTS_INVALID"):
        _validated_results(root)
