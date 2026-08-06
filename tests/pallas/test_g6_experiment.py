from __future__ import annotations

import json
from pathlib import Path

from opjax.pallas.g42_harness import canonical_sha256, file_sha256
from opjax.pallas.g6_experiment import verify_release_on_pool


class FakeVerifier:
    def verify(self, *, candidates, batch_root):
        results = {}
        worker = batch_root / "worker-00"
        worker.mkdir(parents=True)
        for candidate in candidates:
            output = worker / "results" / candidate.unit_id
            output.mkdir(parents=True)
            task = json.loads(candidate.task_path.read_text(encoding="utf-8"))
            kernel_sha256 = file_sha256(candidate.kernel_path)
            result = {
                "passed": True,
                "stage": "verified",
                "kernel_sha256": kernel_sha256,
                "infrastructure_error": False,
            }
            reward = {
                "reward": 1,
                "task_id": task["task_id"],
                "kernel_sha256": kernel_sha256,
            }
            (output / "run.log").write_text(json.dumps(result), encoding="utf-8")
            (output / "reward.json").write_text(json.dumps(reward), encoding="utf-8")
            results[candidate.unit_id] = result
        (worker / "results.json").write_text(
            json.dumps({"recovery_events": []}), encoding="utf-8"
        )
        return results


def test_pool_verification_preserves_g43_release_contract(tmp_path: Path) -> None:
    root = tmp_path / "verifier"
    unit_id = "model--task--seed-0--turn-3"
    unit = root / "units" / unit_id
    unit.mkdir(parents=True)
    (unit / "task.json").write_text('{"task_id":"task"}\n', encoding="utf-8")
    (unit / "kernel.py").write_text("def workload(x):\n    return x\n", encoding="utf-8")
    record = {
        "unit_id": unit_id,
        "task_id": "task",
        "kernel_sha256": file_sha256(unit / "kernel.py"),
    }
    manifest = {
        "schema_version": 1,
        "kind": "pallas_g43_verifier_input_release",
        "counts": {"units": 1},
        "records": [record],
    }
    manifest["release_sha256"] = canonical_sha256(manifest)
    (root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    result = verify_release_on_pool(
        verifier_root=root,
        workers=["worker-0"],
        zone="zone-a",
        verifier=FakeVerifier(),
    )
    assert result["counts"] == {
        "units": 1,
        "verified": 1,
        "candidate_failures": 0,
        "infrastructure_failures": 0,
        "recovery_probes": 0,
    }
    assert (root / "results" / unit_id / "reward.json").is_file()
