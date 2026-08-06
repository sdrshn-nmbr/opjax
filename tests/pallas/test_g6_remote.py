from __future__ import annotations

import json
from pathlib import Path

import pytest

from opjax.pallas.g42_harness import canonical_sha256, file_sha256
from opjax.pallas.g6_remote import G6RemoteError, _static_preflight, verify_batch


def _batch(tmp_path: Path) -> Path:
    root = tmp_path / "batch"
    unit = root / "units" / "u0"
    unit.mkdir(parents=True)
    (unit / "task.json").write_text('{"task_id":"t0"}\n', encoding="utf-8")
    (unit / "kernel.py").write_text("def workload(x):\n    return x\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "kind": "pallas_g6_verifier_batch",
        "batch_id": "b0",
        "counts": {"units": 1},
        "records": [
            {
                "unit_id": "u0",
                "task_sha256": file_sha256(unit / "task.json"),
                "kernel_sha256": file_sha256(unit / "kernel.py"),
            }
        ],
    }
    manifest["release_sha256"] = canonical_sha256(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_remote_batch_rejects_changed_kernel(tmp_path: Path) -> None:
    root = _batch(tmp_path)
    (root / "units/u0/kernel.py").write_text("changed", encoding="utf-8")
    with pytest.raises(G6RemoteError, match="UNIT_HASH_MISMATCH"):
        verify_batch(batch_root=root, timeout_seconds=1)


def test_static_preflight_attributes_invalid_pallas_without_tpu_process(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).parents[2]
    task_root = (
        repo
        / "data/pallas/runs/g42-task-release/tasks/g42-activation-01-reversed-blockspec"
    )
    task = json.loads((task_root / "tests/task.json").read_text())
    result = _static_preflight(
        task=task,
        kernel_path=task_root / "environment/starter/kernel.py",
        output=tmp_path / "result",
    )
    assert result is not None
    assert result["stage"] == "pallas_api"
    assert result["infrastructure_error"] is False
    assert json.loads((tmp_path / "result/reward.json").read_text())["reward"] == 0
