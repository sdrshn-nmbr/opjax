"""Batch verifier backends for online Gate 6 trajectories."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from opjax.pallas.g42_harness import canonical_sha256, file_sha256
from opjax.pallas.g42_verifier import run_fresh_verifier


class G6VerifierBackendError(RuntimeError):
    """A verifier backend failed without candidate attribution."""


@dataclass(frozen=True)
class VerifierCandidate:
    unit_id: str
    task_path: Path
    kernel_path: Path


class VerifierBackend(Protocol):
    def verify(
        self, *, candidates: Sequence[VerifierCandidate], batch_root: Path
    ) -> dict[str, dict[str, Any]]: ...


def build_batch(
    *, candidates: Sequence[VerifierCandidate], batch_id: str, out_dir: Path
) -> dict[str, Any]:
    if out_dir.exists() or not candidates:
        raise G6VerifierBackendError(f"G6_BATCH_DESTINATION_INVALID: {out_dir}")
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+", batch_id):
        raise G6VerifierBackendError(f"G6_BATCH_ID_INVALID: {batch_id}")
    records = []
    for candidate in candidates:
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", candidate.unit_id):
            raise G6VerifierBackendError(f"G6_UNIT_ID_INVALID: {candidate.unit_id}")
        unit = out_dir / "units" / candidate.unit_id
        unit.mkdir(parents=True)
        shutil.copy2(candidate.task_path, unit / "task.json")
        shutil.copy2(candidate.kernel_path, unit / "kernel.py")
        records.append(
            {
                "unit_id": candidate.unit_id,
                "task_sha256": file_sha256(unit / "task.json"),
                "kernel_sha256": file_sha256(unit / "kernel.py"),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pallas_g6_verifier_batch",
        "batch_id": batch_id,
        "counts": {"units": len(records)},
        "records": records,
    }
    manifest["release_sha256"] = canonical_sha256(manifest)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _validated_results(batch_root: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads((batch_root / "manifest.json").read_text(encoding="utf-8"))
    results = json.loads((batch_root / "results.json").read_text(encoding="utf-8"))
    payload = dict(results)
    expected_sha = payload.pop("release_sha256", None)
    if (
        results.get("kind") != "pallas_g6_verifier_batch_results"
        or results.get("input_release_sha256") != manifest.get("release_sha256")
        or canonical_sha256(payload) != expected_sha
        or results.get("counts", {}).get("units") != len(manifest["records"])
    ):
        raise G6VerifierBackendError("G6_BATCH_RESULTS_INVALID")
    expected = {record["unit_id"]: record for record in manifest["records"]}
    observed = {record["unit_id"]: record for record in results["records"]}
    if set(expected) != set(observed):
        raise G6VerifierBackendError("G6_BATCH_RESULT_SET_MISMATCH")
    for unit_id, record in observed.items():
        if record.get("kernel_sha256") != expected[unit_id]["kernel_sha256"]:
            raise G6VerifierBackendError(f"G6_BATCH_RESULT_HASH_MISMATCH: {unit_id}")
        output = batch_root / "results" / unit_id
        if (
            file_sha256(output / "run.log") != record.get("run_log_sha256")
            or file_sha256(output / "reward.json") != record.get("reward_sha256")
        ):
            raise G6VerifierBackendError(f"G6_BATCH_EVIDENCE_HASH_MISMATCH: {unit_id}")
    return {unit_id: record["result"] for unit_id, record in observed.items()}


class LocalVerifierBackend:
    """Direct backend used by unit tests and a local TPU process."""

    def __init__(self, *, timeout_seconds: int = 180) -> None:
        self.timeout_seconds = timeout_seconds

    def verify(
        self, *, candidates: Sequence[VerifierCandidate], batch_root: Path
    ) -> dict[str, dict[str, Any]]:
        manifest = build_batch(
            candidates=candidates, batch_id=batch_root.name, out_dir=batch_root
        )
        records = []
        for record in manifest["records"]:
            unit = batch_root / "units" / record["unit_id"]
            result = run_fresh_verifier(
                task_path=unit / "task.json",
                kernel_path=unit / "kernel.py",
                output_dir=batch_root / "results" / record["unit_id"],
                timeout_seconds=self.timeout_seconds,
            )["result"]
            records.append(
                {
                    "unit_id": record["unit_id"],
                    "kernel_sha256": record["kernel_sha256"],
                    "result": result,
                    "run_log_sha256": file_sha256(
                        batch_root / "results" / record["unit_id"] / "run.log"
                    ),
                    "reward_sha256": file_sha256(
                        batch_root / "results" / record["unit_id"] / "reward.json"
                    ),
                }
            )
        result_manifest: dict[str, Any] = {
            "schema_version": 1,
            "kind": "pallas_g6_verifier_batch_results",
            "input_release_sha256": manifest["release_sha256"],
            "counts": {
                "units": len(records),
                "verified": sum(row["result"].get("passed") is True for row in records),
                "infrastructure_failures": sum(
                    row["result"].get("infrastructure_error") is True for row in records
                ),
            },
            "records": records,
        }
        result_manifest["release_sha256"] = canonical_sha256(result_manifest)
        (batch_root / "results.json").write_text(
            json.dumps(result_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return _validated_results(batch_root)


class RemoteTPUPoolVerifier:
    """Distribute immutable batches across independent single-chip TPU VMs."""

    def __init__(
        self,
        *,
        workers: Sequence[str],
        zone: str,
        remote_repo: str = "/home/sudarshan/opjax-feedback",
        timeout_seconds: int = 180,
    ) -> None:
        if (
            not workers
            or len(set(workers)) != len(workers)
            or any(not re.fullmatch(r"[a-z][a-z0-9-]{0,62}", worker) for worker in workers)
            or not re.fullmatch(r"[a-z0-9-]+", zone)
            or not re.fullmatch(r"/[a-zA-Z0-9_./-]+", remote_repo)
        ):
            raise G6VerifierBackendError("G6_TPU_WORKER_SET_INVALID")
        self.workers = tuple(workers)
        self.zone = zone
        self.remote_repo = remote_repo
        self.timeout_seconds = timeout_seconds

    def _run(self, command: list[str], *, label: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["CLOUDSDK_ACTIVE_CONFIG_NAME"] = "agent"
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(300, self.timeout_seconds * 64),
            env=environment,
        )
        if process.returncode != 0:
            raise G6VerifierBackendError(
                f"G6_REMOTE_COMMAND_FAILED: {label}: rc={process.returncode}: "
                f"{process.stderr[-4000:]}"
            )
        return process

    def _verify_worker(self, *, worker: str, root: Path) -> dict[str, dict[str, Any]]:
        archive = root.parent / f"{root.name}.tar.gz"
        output_archive = root.parent / f"{root.name}-results.tar.gz"
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(root, arcname="batch")
        remote_id = f"{root.parent.name}-{root.name}"
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", remote_id):
            raise G6VerifierBackendError(f"G6_REMOTE_ID_INVALID: {remote_id}")
        remote_root = f"/tmp/opjax-g6-{remote_id}"
        remote_archive = f"{remote_root}.tar.gz"
        remote_output = f"{remote_root}-results.tar.gz"
        self._run(
            [
                "gcloud",
                "compute",
                "tpus",
                "tpu-vm",
                "scp",
                str(archive),
                f"{worker}:{remote_archive}",
                f"--zone={self.zone}",
            ],
            label=f"upload:{worker}",
        )
        remote_command = (
            f"rm -rf {remote_root} {remote_output} && mkdir -p {remote_root} && "
            f"tar -xzf {remote_archive} -C {remote_root} && "
            f"cd {self.remote_repo} && PYTHONPATH=src .venv/bin/python -m opjax.pallas.g6_remote "
            f"--batch-root {remote_root}/batch --timeout-seconds {self.timeout_seconds} && "
            f"tar -czf {remote_output} -C {remote_root}/batch results.json results"
        )
        self._run(
            [
                "gcloud",
                "compute",
                "tpus",
                "tpu-vm",
                "ssh",
                worker,
                f"--zone={self.zone}",
                f"--command={remote_command}",
            ],
            label=f"verify:{worker}",
        )
        self._run(
            [
                "gcloud",
                "compute",
                "tpus",
                "tpu-vm",
                "scp",
                f"{worker}:{remote_output}",
                str(output_archive),
                f"--zone={self.zone}",
            ],
            label=f"download:{worker}",
        )
        with tarfile.open(output_archive, "r:gz") as handle:
            members = handle.getmembers()
            if any(member.name.startswith("/") or ".." in Path(member.name).parts for member in members):
                raise G6VerifierBackendError("G6_REMOTE_ARCHIVE_PATH_INVALID")
            handle.extractall(root, filter="data")
        return _validated_results(root)

    def verify(
        self, *, candidates: Sequence[VerifierCandidate], batch_root: Path
    ) -> dict[str, dict[str, Any]]:
        if batch_root.exists() or not candidates:
            raise G6VerifierBackendError(f"G6_BATCH_DESTINATION_INVALID: {batch_root}")
        assignments = [list(candidates[index :: len(self.workers)]) for index in range(len(self.workers))]
        assignments = [assignment for assignment in assignments if assignment]
        worker_roots = []
        for index, assignment in enumerate(assignments):
            root = batch_root / f"worker-{index:02d}"
            build_batch(
                candidates=assignment,
                batch_id=f"{batch_root.name}-w{index:02d}",
                out_dir=root,
            )
            worker_roots.append((self.workers[index], root))
        results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=len(worker_roots)) as executor:
            futures = {
                executor.submit(self._verify_worker, worker=worker, root=root): worker
                for worker, root in worker_roots
            }
            for future in as_completed(futures):
                worker_results = future.result()
                overlap = set(results) & set(worker_results)
                if overlap:
                    raise G6VerifierBackendError(f"G6_REMOTE_RESULT_DUPLICATE: {overlap}")
                results.update(worker_results)
        expected = {candidate.unit_id for candidate in candidates}
        if set(results) != expected:
            raise G6VerifierBackendError("G6_REMOTE_RESULT_SET_INCOMPLETE")
        return results
