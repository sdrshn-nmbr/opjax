"""Audit Gate 4 supervision and sample a small learning-transfer ladder."""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import tinker
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from opjax.pallas.contracts import git_revision, load_contracts
from opjax.pallas.prompts import extract_code, parses, source_sha256
from opjax.pallas.sampling import _sampling_client
from opjax.pallas.scoring import inspect_pallas_source
from opjax.pallas.training import _prepare


class DiagnosticError(RuntimeError):
    """Gate 4 diagnostic evidence is invalid or incomplete."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DiagnosticError(f"DIAGNOSTIC_CONFIG_INVALID: {path}")
    return value


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _git_tracked_dirty(repo_root: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode != 0 or bool(result.stdout.strip())


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _blockspec_order(source: str) -> tuple[int, int]:
    tree = ast.parse(source)
    total = 0
    reversed_calls = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) < 2:
            continue
        function = node.func
        if not (
            isinstance(function, ast.Attribute)
            and function.attr == "BlockSpec"
        ):
            continue
        total += 1
        if isinstance(node.args[0], (ast.Lambda, ast.Name)) and isinstance(
            node.args[1], (ast.Tuple, ast.List)
        ):
            reversed_calls += 1
    return total, reversed_calls


def _source_audit(source: str) -> dict[str, Any]:
    inspection = inspect_pallas_source(source)
    blockspec_calls, reversed_calls = _blockspec_order(source)
    tree = ast.parse(source)
    imports = {
        alias.asname or alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    )
    return {
        "parses": True,
        "authentic": inspection.authentic,
        "authenticity_reasons": list(inspection.reasons),
        "blockspec_calls": blockspec_calls,
        "reversed_blockspec_calls": reversed_calls,
        "has_placeholder_ellipsis": any(
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and node.value.value is Ellipsis
            for node in ast.walk(tree)
        ),
        "imports": sorted(imports),
    }


def audit_supervision(
    *, config_root: Path, corpus_root: Path, repo_root: Path, output: Path
) -> dict[str, Any]:
    preparation, rows, datums, _, tokenizer = _prepare(
        config_root=config_root,
        corpus_root=corpus_root,
        repo_root=repo_root,
    )
    row_audits: list[dict[str, Any]] = []
    for row, datum in zip(rows, datums, strict=True):
        source = row["messages"][-1]["content"]
        weights = datum.loss_fn_inputs["weights"].tolist()
        positive = [index for index, weight in enumerate(weights) if float(weight) > 0]
        if not positive:
            raise DiagnosticError(f"SUPERVISION_EMPTY: {row['row_id']}")
        rendered_tokens = datum.model_input.to_ints()
        target_tokens = datum.loss_fn_inputs["target_tokens"].tolist()
        row_audits.append(
            {
                "row_id": row["row_id"],
                "family_category": row["family_category"],
                "sequence_tokens": datum.model_input.length + 1,
                "supervised_tokens": len(positive),
                "supervision_contiguous": positive == list(range(positive[0], positive[-1] + 1)),
                "supervision_reaches_end": positive[-1] == len(weights) - 1,
                "rendered_prefix": tokenizer.decode(rendered_tokens[: min(16, len(rendered_tokens))]),
                "target_tail": tokenizer.decode(target_tokens[-min(16, len(target_tokens)):]),
                "source": _source_audit(source),
            }
        )
    report = {
        "schema_version": 1,
        "kind": "gate4_supervision_audit",
        "created_at": _utc_now(),
        "experiment_id": preparation["experiment_id"],
        "contract_sha256": preparation["contract_sha256"],
        "dataset_sha256": preparation["dataset_sha256"],
        "renderer": preparation["training"]["renderer"],
        "train_on": preparation["training"]["train_on"],
        "maximum_length": preparation["training"]["max_length"],
        "summary": {
            "rows": len(row_audits),
            "sequence_tokens": sum(row["sequence_tokens"] for row in row_audits),
            "supervised_tokens": sum(row["supervised_tokens"] for row in row_audits),
            "rows_truncated": sum(row["sequence_tokens"] > preparation["training"]["max_length"] for row in row_audits),
            "rows_with_noncontiguous_supervision": sum(not row["supervision_contiguous"] for row in row_audits),
            "rows_without_end_supervision": sum(not row["supervision_reaches_end"] for row in row_audits),
            "blockspec_calls": sum(row["source"]["blockspec_calls"] for row in row_audits),
            "reversed_blockspec_calls": sum(row["source"]["reversed_blockspec_calls"] for row in row_audits),
            "rows_with_placeholder_ellipsis": sum(
                row["source"]["has_placeholder_ellipsis"] for row in row_audits
            ),
            "authentic_rows": sum(row["source"]["authentic"] for row in row_audits),
        },
        "rows": row_audits,
    }
    report["sha256"] = _canonical_sha256(report)
    _write_json(output, report)
    return report


def _heldout_prompt(task: dict[str, Any]) -> str:
    return (
        "Implement an authentic normal-lowering JAX Pallas kernel for the "
        f"{task['task_id']} operation. Compute {task['operation']} for full-shape "
        f"inputs {task['input_shapes']}. The callable must accept those inputs "
        f"with dtypes {task['input_dtypes']}. Return a syntactically valid, "
        "self-contained Python module defining workload(*inputs). It must match "
        "the operation semantics at the full declared shapes. Do not use "
        "interpret mode, a plain-JAX fallback, or an incomplete kernel."
    )


def _diagnostic_tasks(
    *, diagnostic: dict[str, Any], corpus_root: Path
) -> list[dict[str, Any]]:
    corpus_rows = {
        row["row_id"]: row
        for row in _load_rows(corpus_root / "datasets" / "sft.jsonl")
    }
    tasks: list[dict[str, Any]] = []
    for row_id in diagnostic["replay_rows"]:
        row = corpus_rows.get(row_id)
        if row is None:
            raise DiagnosticError(f"REPLAY_ROW_MISSING: {row_id}")
        tasks.append(
            {
                "task_id": f"replay::{row_id}",
                "tier": "training_replay",
                "prompt": row["messages"][0]["content"],
                "reference_sha256": source_sha256(row["messages"][-1]["content"]),
                "operation": _operation_from_row_id(row_id),
            }
        )
    for task in diagnostic["heldout_tasks"]:
        tasks.append({**task, "tier": "near_heldout", "prompt": _heldout_prompt(task)})
    return tasks


def _operation_from_row_id(row_id: str) -> str:
    name = row_id.split(":")[-2]
    for prefix, operation in (
        ("row-sum-", "row_sum"),
        ("matmul-", "matmul"),
        ("rmsnorm-", "rmsnorm"),
        ("add-", "add"),
    ):
        if name.startswith(prefix):
            return operation
    raise DiagnosticError(f"REPLAY_OPERATION_UNSUPPORTED: {row_id}")


async def sample_ladder(
    *,
    config_root: Path,
    corpus_root: Path,
    diagnostic_path: Path,
    repo_root: Path,
    out_dir: Path,
    arm: str,
    model_path: str | None,
) -> dict[str, Any]:
    if _git_tracked_dirty(repo_root):
        raise DiagnosticError(f"OPJAX_TRACKED_DIRTY: {repo_root}")
    bundle = load_contracts(config_root)
    diagnostic = _load_json(diagnostic_path)
    tasks = _diagnostic_tasks(diagnostic=diagnostic, corpus_root=corpus_root)
    if arm == "A" and model_path is not None:
        raise DiagnosticError("BASE_ARM_MODEL_PATH_FORBIDDEN")
    if arm != "A" and model_path is None:
        raise DiagnosticError("SFT_ARM_MODEL_PATH_REQUIRED")
    renderer_name = model_info.get_recommended_renderer_name(bundle.experiment["base_model"])
    if renderer_name != diagnostic["renderer"]:
        raise DiagnosticError("DIAGNOSTIC_RENDERER_MISMATCH")
    fingerprint = {
        "experiment_id": bundle.experiment["experiment_id"],
        "contract_sha256": bundle.sha256,
        "diagnostic_sha256": _canonical_sha256(diagnostic),
        "opjax_revision": git_revision(repo_root),
        "arm": arm,
        "base_model": bundle.experiment["base_model"],
        "model_path": model_path,
        "renderer": renderer_name,
        "sampling": diagnostic["sampling"],
        "task_ids": [task["task_id"] for task in tasks],
        "packages": {name: _package_version(name) for name in ("tinker", "tinker-cookbook")},
    }
    fingerprint["sha256"] = _canonical_sha256(fingerprint)
    if out_dir.exists():
        raise DiagnosticError(f"DIAGNOSTIC_RUN_EXISTS: {out_dir}")
    out_dir.mkdir(parents=True)
    _write_json(out_dir / "manifest.json", {"schema_version": 1, "status": "sampling", "created_at": _utc_now(), "fingerprint": fingerprint})
    tokenizer = get_tokenizer(bundle.experiment["base_model"])
    renderer = renderers.get_renderer(renderer_name, tokenizer, model_name=bundle.experiment["base_model"])
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0), follow_redirects=True)
    service = tinker.ServiceClient(http_client=http_client, max_retries=0)
    client = await _sampling_client(service=service, base_model=bundle.experiment["base_model"], model_path=model_path)
    stops = renderer.get_stop_sequences()
    semaphore = asyncio.Semaphore(4)

    async def sample_task(index: int, task: dict[str, Any]) -> dict[str, Any]:
        model_input = renderer.build_generation_prompt(
            [{"role": "user", "content": task["prompt"]}]
        )
        async with semaphore:
            result = await client.sample_async(
                prompt=model_input,
                num_samples=1,
                sampling_params=types.SamplingParams(
                    max_tokens=diagnostic["sampling"]["max_tokens"],
                    temperature=diagnostic["sampling"]["temperature"],
                    top_p=diagnostic["sampling"]["top_p"],
                    seed=diagnostic["sampling"]["seed"],
                    stop=stops or None,
                ),
            )
        sequence = result.sequences[0]
        completion = tokenizer.decode(sequence.tokens)
        code = extract_code(completion)
        candidate = code or completion
        kernel_path = Path("kernels") / f"{index:02d}.py"
        (out_dir / kernel_path).parent.mkdir(parents=True, exist_ok=True)
        (out_dir / kernel_path).write_text(candidate, encoding="utf-8")
        source_audit = (
            _source_audit(candidate) if parses(candidate) else {"parses": False}
        )
        print(
            f"G4_DIAGNOSTIC_SAMPLE arm={arm} task={task['task_id']} "
            f"parses={source_audit['parses']}",
            flush=True,
        )
        return {
            "schema_version": 1,
            "task": task,
            "kernel_path": str(kernel_path),
            "completion": completion,
            "completion_sha256": source_sha256(completion),
            "code_sha256": source_sha256(candidate),
            "n_tokens": len(sequence.tokens),
            "stop_reason": str(getattr(sequence, "stop_reason", "")),
            "source_audit": source_audit,
        }

    try:
        rows = await asyncio.gather(
            *(sample_task(index, task) for index, task in enumerate(tasks))
        )
        _write_jsonl(out_dir / "samples.jsonl", rows)
    finally:
        await http_client.aclose()
    manifest = _load_json(out_dir / "manifest.json")
    manifest.update({"status": "sampled", "completed_at": _utc_now(), "n_samples": len(rows)})
    _write_json(out_dir / "manifest.json", manifest)
    return {"n_samples": len(rows), "n_parseable": sum(row["source_audit"]["parses"] for row in rows), "out_dir": str(out_dir)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="opjax-pallas-g4-diagnostic")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit-supervision")
    audit.add_argument("--config-root", type=Path, default=Path("config/pallas"))
    audit.add_argument("--corpus-root", type=Path, required=True)
    audit.add_argument("--repo-root", type=Path, default=Path("."))
    audit.add_argument("--output", type=Path, required=True)
    sample = commands.add_parser("sample")
    sample.add_argument("--config-root", type=Path, default=Path("config/pallas"))
    sample.add_argument("--corpus-root", type=Path, required=True)
    sample.add_argument("--diagnostic", type=Path, required=True)
    sample.add_argument("--repo-root", type=Path, default=Path("."))
    sample.add_argument("--out-dir", type=Path, required=True)
    sample.add_argument("--arm", choices=["A", "B"], required=True)
    sample.add_argument("--model-path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "audit-supervision":
            result = audit_supervision(config_root=args.config_root, corpus_root=args.corpus_root, repo_root=args.repo_root, output=args.output)
        else:
            result = asyncio.run(sample_ladder(config_root=args.config_root, corpus_root=args.corpus_root, diagnostic_path=args.diagnostic, repo_root=args.repo_root, out_dir=args.out_dir, arm=args.arm, model_path=args.model_path))
    except (DiagnosticError, ValueError) as exc:
        print(f"G4_DIAGNOSTIC_ERROR {exc}", file=sys.stderr)
        return 2
    printable = result
    if args.command == "audit-supervision":
        printable = {"sha256": result["sha256"], "summary": result["summary"]}
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
