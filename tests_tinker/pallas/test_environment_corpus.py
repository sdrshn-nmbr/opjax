from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from opjax.pallas.environment_corpus import (
    EnvironmentCorpusError,
    build_environment_corpus,
    validate_environment_corpus,
)


REPO_ROOT = Path(__file__).parents[2]
SOURCE_ROOT = REPO_ROOT / "data" / "pallas" / "runs" / "g3-sft-ready-final"
CONTRACT = REPO_ROOT / "config" / "pallas" / "g41-environment.json"


def test_environment_corpus_preserves_verified_targets_and_adds_tasks(
    tmp_path: Path,
) -> None:
    output = tmp_path / "release"
    result = build_environment_corpus(
        source_root=SOURCE_ROOT,
        contract_path=CONTRACT,
        out_dir=output,
    )

    assert result["ok"] is True
    assert result["counts"] == {"sft": 32, "tasks": 32}
    rows = [json.loads(line) for line in (output / "datasets/sft.jsonl").read_text().splitlines()]
    tasks = [json.loads(line) for line in (output / "tasks.jsonl").read_text().splitlines()]
    assert all("workload(*inputs)" in row["messages"][0]["content"] for row in rows)
    assert all(task["max_attempts"] == 3 for task in tasks)
    assert all(task["reference_solution_visible"] is False for task in tasks)


def test_environment_corpus_rejects_task_tampering(tmp_path: Path) -> None:
    output = tmp_path / "release"
    build_environment_corpus(
        source_root=SOURCE_ROOT,
        contract_path=CONTRACT,
        out_dir=output,
    )
    task_path = output / "tasks.jsonl"
    tasks = [json.loads(line) for line in task_path.read_text().splitlines()]
    tasks[0]["operation"] = "forged"
    task_path.write_text("".join(json.dumps(task) + "\n" for task in tasks))

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["tasks.jsonl"] = hashlib.sha256(task_path.read_bytes()).hexdigest()
    unhashed = {key: value for key, value in manifest.items() if key != "release_sha256"}
    manifest["release_sha256"] = hashlib.sha256(
        json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(EnvironmentCorpusError, match="ENVIRONMENT_TASK_INVALID"):
        validate_environment_corpus(output)
