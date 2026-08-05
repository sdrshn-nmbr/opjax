"""Build and validate the environment-backed Gate 4.1 SFT release."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from opjax.pallas.corpus import validate_corpus_release


class EnvironmentCorpusError(RuntimeError):
    """The environment-backed corpus is inconsistent or incomplete."""


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EnvironmentCorpusError(f"JSON_OBJECT_REQUIRED: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _corrected_prompt(prompt: str, contract: dict[str, Any]) -> str:
    return (
        f"{prompt} Return {contract['expected_interface']}. "
        "Use the current JAX API exactly: "
        f"{contract['pallas_api']['blockspec_signature']}; block_shape is the "
        "first argument and index_map is the second. Include every required "
        "import and a complete kernel body. Do not return an incomplete kernel. "
        "Return only the Python module, with no prose."
    )


def _validate_target_source(source: str, row_id: str) -> None:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise EnvironmentCorpusError(f"TARGET_SYNTAX_INVALID: {row_id}: {exc}") from exc
    functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if "workload" not in functions:
        raise EnvironmentCorpusError(f"TARGET_WORKLOAD_MISSING: {row_id}")
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and node.value.value is Ellipsis
        ):
            raise EnvironmentCorpusError(f"TARGET_PLACEHOLDER_ELLIPSIS: {row_id}")
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function = node.func
        if not (
            isinstance(function, ast.Attribute)
            and function.attr == "BlockSpec"
        ):
            continue
        if len(node.args) < 2 or isinstance(node.args[0], ast.Lambda) or not isinstance(
            node.args[1], ast.Lambda
        ):
            raise EnvironmentCorpusError(f"TARGET_BLOCKSPEC_ORDER_INVALID: {row_id}")


def build_environment_corpus(
    *, source_root: Path, contract_path: Path, out_dir: Path
) -> dict[str, Any]:
    if out_dir.exists():
        raise EnvironmentCorpusError(f"ENVIRONMENT_CORPUS_EXISTS: {out_dir}")
    source_validation = validate_corpus_release(source_root)
    contract = _load_json(contract_path)
    if contract.get("max_attempts") != 3:
        raise EnvironmentCorpusError("MAX_ATTEMPTS_INVALID")
    source_rows = _load_jsonl(source_root / "datasets" / "sft.jsonl")
    source_candidates = {
        row["candidate_id"]: row
        for row in _load_jsonl(source_root / "candidates.jsonl")
    }
    rows: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    for source_row in source_rows:
        candidate = source_candidates[source_row["row_id"]]
        metadata = candidate["metadata"]
        prompt = _corrected_prompt(source_row["messages"][0]["content"], contract)
        task_id = source_row["row_id"]
        task = {
            "schema_version": 1,
            "task_id": task_id,
            "split": "train",
            "instruction": prompt,
            "operation": metadata["operation"],
            "kernel_kind": metadata["kernel_kind"],
            "input_shapes": metadata["input_shapes"],
            "input_dtypes": metadata["input_dtypes"],
            "correctness_tolerance": metadata["correctness_tolerance"],
            "expected_interface": contract["expected_interface"],
            "max_attempts": contract["max_attempts"],
            "feedback_stages": contract["feedback_stages"],
            "reference_solution_visible": False,
            "verification": source_row["verification"],
        }
        task["task_sha256"] = _canonical_sha256(task)
        row = dict(source_row)
        row["schema_version"] = 2
        row["messages"] = [
            {"role": "user", "content": prompt},
            source_row["messages"][1],
        ]
        row["environment_task"] = {
            "task_sha256": task["task_sha256"],
            "max_attempts": contract["max_attempts"],
            "feedback_stages": contract["feedback_stages"],
        }
        rows.append(row)
        tasks.append(task)
    _write_jsonl(out_dir / "datasets" / "sft.jsonl", rows)
    _write_jsonl(out_dir / "tasks.jsonl", tasks)
    artifacts = {
        "datasets/sft.jsonl": _sha256_file(out_dir / "datasets" / "sft.jsonl"),
        "tasks.jsonl": _sha256_file(out_dir / "tasks.jsonl"),
    }
    manifest = {
        "schema_version": 1,
        "kind": "pallas_environment_corpus_release",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": _canonical_sha256(contract),
        "source_release_sha256": source_validation["release_sha256"],
        "source_release_relative_path": os.path.relpath(
            source_root.resolve(), out_dir.resolve()
        ),
        "counts": {"sft": len(rows), "tasks": len(tasks)},
        "max_attempts": contract["max_attempts"],
        "artifacts": artifacts,
    }
    manifest["release_sha256"] = _canonical_sha256(manifest)
    _write_json(out_dir / "manifest.json", manifest)
    return validate_environment_corpus(out_dir)


def validate_environment_corpus(root: Path) -> dict[str, Any]:
    manifest = _load_json(root / "manifest.json")
    if manifest.get("kind") != "pallas_environment_corpus_release":
        raise EnvironmentCorpusError("ENVIRONMENT_CORPUS_MANIFEST_INVALID")
    unhashed = {key: value for key, value in manifest.items() if key != "release_sha256"}
    if manifest.get("release_sha256") != _canonical_sha256(unhashed):
        raise EnvironmentCorpusError("ENVIRONMENT_CORPUS_HASH_INVALID")
    for relative, expected in manifest.get("artifacts", {}).items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or _sha256_file(path) != expected:
            raise EnvironmentCorpusError(f"ENVIRONMENT_CORPUS_ARTIFACT_INVALID: {relative}")
    source_root = (root / manifest["source_release_relative_path"]).resolve()
    source = validate_corpus_release(source_root)
    if source["release_sha256"] != manifest["source_release_sha256"]:
        raise EnvironmentCorpusError("ENVIRONMENT_CORPUS_SOURCE_INVALID")
    rows = _load_jsonl(root / "datasets" / "sft.jsonl")
    tasks = _load_jsonl(root / "tasks.jsonl")
    source_rows = {
        row["row_id"]: row
        for row in _load_jsonl(source_root / "datasets" / "sft.jsonl")
    }
    if len(rows) != 32 or len(tasks) != 32:
        raise EnvironmentCorpusError("ENVIRONMENT_CORPUS_COUNT_INVALID")
    row_ids = [row.get("row_id") for row in rows]
    task_ids = [task.get("task_id") for task in tasks]
    if len(set(row_ids)) != len(rows) or len(set(task_ids)) != len(tasks):
        raise EnvironmentCorpusError("ENVIRONMENT_CORPUS_ID_DUPLICATE")
    tasks_by_id = {task["task_id"]: task for task in tasks}
    for row in rows:
        prompt = row["messages"][0]["content"]
        task = tasks_by_id.get(row["row_id"])
        source_row = source_rows.get(row["row_id"])
        task_without_hash = {
            key: value for key, value in (task or {}).items() if key != "task_sha256"
        }
        if (
            task is None
            or source_row is None
            or task.get("max_attempts") != 3
            or task.get("reference_solution_visible") is not False
            or task.get("task_sha256") != _canonical_sha256(task_without_hash)
            or row.get("environment_task", {}).get("task_sha256") != task.get("task_sha256")
            or row["messages"][1] != source_row["messages"][1]
            or row.get("verification") != source_row.get("verification")
            or "workload(*inputs)" not in prompt
            or "pl.BlockSpec(block_shape, index_map)" not in prompt
            or "complete kernel body" not in prompt
        ):
            raise EnvironmentCorpusError(f"ENVIRONMENT_TASK_INVALID: {row.get('row_id')}")
        _validate_target_source(row["messages"][1]["content"], row["row_id"])
    return {
        "ok": True,
        "release_sha256": manifest["release_sha256"],
        "contract_sha256": manifest["contract_sha256"],
        "dataset_sha256": manifest["artifacts"]["datasets/sft.jsonl"],
        "counts": manifest["counts"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-environment-corpus")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--source-root", type=Path, required=True)
    build.add_argument("--contract", type=Path, required=True)
    build.add_argument("--out-dir", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = (
            build_environment_corpus(source_root=args.source_root, contract_path=args.contract, out_dir=args.out_dir)
            if args.command == "build"
            else validate_environment_corpus(args.root)
        )
    except EnvironmentCorpusError as exc:
        print(f"PALLAS_ENVIRONMENT_CORPUS_ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
