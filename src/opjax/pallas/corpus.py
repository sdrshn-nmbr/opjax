"""Deterministic discovery and promotion for the governed Pallas corpus."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Sequence

import chex
import jax
import jax.numpy as jnp

from opjax.pallas.contracts import (
    ContractBundle,
    source_by_id,
    verify_source_checkout,
)
from opjax.pallas.lowering import (
    capture_lowering_case,
    validate_calibration,
    validate_candidate_evidence,
)
from opjax.pallas.scoring import inspect_pallas_source

TEXT_SUFFIXES = {".md", ".py", ".rst"}
PALLAS_SIGNALS = (
    "jax.experimental.pallas",
    "pallas_call",
    "BlockSpec",
    "program_id",
    "pltpu",
)
SHINGLE_SIZE = 5
NEAR_DUPLICATE_THRESHOLD = 0.90


class CorpusError(RuntimeError):
    """Corpus evidence or policy is invalid."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(payload)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in materialized:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return len(materialized)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise CorpusError(f"CORPUS_ARTIFACT_MISSING: {path}") from exc
    for line_number, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CorpusError(
                f"CORPUS_JSON_INVALID: {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise CorpusError(f"CORPUS_ROW_INVALID: {path}:{line_number}")
        rows.append(value)
    return rows


def _git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _repository_revision(path: Path) -> str:
    try:
        return _git_revision(path)
    except subprocess.CalledProcessError as exc:
        raise CorpusError(
            f"REPOSITORY_REVISION_UNAVAILABLE: {path}: {exc.output.strip()}"
        ) from exc


def _normalise_python(source: str) -> str:
    try:
        return ast.unparse(ast.parse(source))
    except SyntaxError:
        return " ".join(source.split())


def _tokens(source: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\s]", source))


def _shingles(source: str) -> frozenset[str]:
    tokens = _tokens(_normalise_python(source))
    if len(tokens) < SHINGLE_SIZE:
        return frozenset({" ".join(tokens)}) if tokens else frozenset()
    return frozenset(
        " ".join(tokens[index : index + SHINGLE_SIZE])
        for index in range(len(tokens) - SHINGLE_SIZE + 1)
    )


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _family_id(category: str, source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        calls = sorted(signal for signal in PALLAS_SIGNALS if signal in source)
    else:
        calls = sorted(
            {
                node.func.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
            }
        )
    signature = {"category": category, "calls": calls}
    return f"{category}:{_canonical_sha256(signature)[:16]}"


def _safe_relative(root: Path, path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise CorpusError(f"SOURCE_PATH_ESCAPE: {path}")
    return resolved.relative_to(root.resolve()).as_posix()


def _allowlisted(source: dict[str, Any], relative: str) -> bool:
    return any(
        relative == prefix or relative.startswith(f"{prefix}/")
        for prefix in source.get("allowlisted_paths", [])
    )


def _iter_allowlisted_files(
    source: dict[str, Any],
    checkout: Path,
) -> Iterable[Path]:
    for prefix in source.get("allowlisted_paths", []):
        root = checkout / prefix
        if root.is_file():
            paths = [root]
        elif root.is_dir():
            paths = sorted(path for path in root.rglob("*") if path.is_file())
        else:
            raise CorpusError(
                f"SOURCE_ALLOWLIST_PATH_MISSING: {source['id']}:{prefix}"
            )
        for path in paths:
            if path.suffix in TEXT_SUFFIXES:
                yield path


def _inventory_git_source(
    source: dict[str, Any],
    checkout: Path,
) -> list[dict[str, Any]]:
    rows = []
    for path in _iter_allowlisted_files(source, checkout):
        relative = _safe_relative(checkout, path)
        content = path.read_text(encoding="utf-8")
        if not any(signal in content for signal in PALLAS_SIGNALS):
            continue
        rows.append(
            {
                "schema_version": 1,
                "source_id": source["id"],
                "source_kind": "git",
                "source_revision": source["revision"],
                "license": source["license"],
                "training_policy": source["training_policy"],
                "path": relative,
                "row_id": None,
                "content_sha256": _sha256_text(content),
                "bytes": len(content.encode()),
                "discovery_reasons": sorted(
                    signal for signal in PALLAS_SIGNALS if signal in content
                ),
            }
        )
    return rows


def _hf_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise CorpusError(f"HF_RESPONSE_INVALID: {url}")
    return value


def _inventory_hf_source(source: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_id = source["url"].removeprefix(
        "https://huggingface.co/datasets/"
    )
    metadata = _hf_json(f"https://huggingface.co/api/datasets/{dataset_id}")
    if metadata.get("sha") != source["revision"]:
        raise CorpusError(
            "SOURCE_REVISION_MISMATCH: "
            f"{source['id']}: expected={source['revision']} "
            f"observed={metadata.get('sha')}"
        )
    inventory: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for config in source.get("configs", []):
        for split in config.get("splits", []):
            offset = 0
            while True:
                payload = _hf_json(
                    "https://datasets-server.huggingface.co/rows",
                    {
                        "dataset": dataset_id,
                        "config": config["name"],
                        "split": split,
                        "offset": offset,
                        "length": 100,
                    },
                )
                page = payload.get("rows")
                if not isinstance(page, list):
                    raise CorpusError(
                        f"HF_ROWS_INVALID: {source['id']}:{config['name']}:{split}"
                    )
                for item in page:
                    row = item.get("row") if isinstance(item, dict) else None
                    row_index = item.get("row_idx") if isinstance(item, dict) else None
                    if not isinstance(row, dict) or not isinstance(row_index, int):
                        raise CorpusError(f"HF_ROW_INVALID: {source['id']}:{split}")
                    code = row.get("Pallas_Code")
                    if not isinstance(code, str) or not code.strip():
                        continue
                    row_id = f"{config['name']}:{split}:{row_index}"
                    inventory.append(
                        {
                            "schema_version": 1,
                            "source_id": source["id"],
                            "source_kind": "hf_dataset",
                            "source_revision": source["revision"],
                            "license": source["license"],
                            "training_policy": source["training_policy"],
                            "path": None,
                            "row_id": row_id,
                            "content_sha256": _sha256_text(code),
                            "bytes": len(code.encode()),
                            "discovery_reasons": sorted(
                                signal
                                for signal in PALLAS_SIGNALS
                                if signal in code
                            ),
                        }
                    )
                    candidates.append(
                        _candidate_row(
                            source=source,
                            path=None,
                            row_id=row_id,
                            content=code,
                            objective="discovery",
                            family_category=str(row.get("Category") or "unknown"),
                            metadata={
                                "task_name": row.get("Op_Name"),
                                "dataset_correct": row.get("Correct"),
                                "target_hardware": row.get("Target_Hardware"),
                                "pallas_backend": row.get("Pallas_Backend"),
                            },
                        )
                    )
                total = payload.get("num_rows_total")
                offset += len(page)
                if not page or not isinstance(total, int) or offset >= total:
                    break
    return inventory, candidates


def _literal_assignment(tree: ast.Module, name: str) -> Any:
    for node in tree.body:
        if (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Name) and target.id == name
                for target in (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
            )
            and node.value is not None
        ):
            return ast.literal_eval(node.value)
    raise CorpusError(f"SOURCE_ASSIGNMENT_MISSING: {name}")


def _attribute_name(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        raise CorpusError("TASK_ATTRIBUTE_INVALID")
    parts.append(node.id)
    return ".".join(reversed(parts))


def _registry_tasks(checkout: Path) -> list[dict[str, Any]]:
    registry_path = checkout / "pallasbench" / "tasks.py"
    tree = ast.parse(registry_path.read_text(encoding="utf-8"))
    registry: ast.List | None = None
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "TASK_REGISTRY"
            and isinstance(node.value, ast.List)
        ):
            registry = node.value
            break
    if registry is None:
        raise CorpusError("PALLASBENCH_REGISTRY_MISSING")
    tasks = []
    for node in registry.elts:
        if not isinstance(node, ast.Dict):
            raise CorpusError("PALLASBENCH_TASK_INVALID")
        fields = {
            ast.literal_eval(key): value
            for key, value in zip(node.keys, node.values, strict=True)
            if key is not None
        }
        task_name = ast.literal_eval(fields["name"])
        module_name, pallas_fn = _attribute_name(fields["pallas_fn"]).split(".", 1)
        _, baseline_fn = _attribute_name(fields["baseline_fn"]).split(".", 1)
        matches = sorted(
            (checkout / "pallasbench" / "kernels").rglob(f"{module_name}.py")
        )
        if len(matches) != 1:
            raise CorpusError(
                f"PALLASBENCH_KERNEL_PATH_INVALID: {task_name}: {matches}"
            )
        module_path = matches[0]
        module_tree = ast.parse(module_path.read_text(encoding="utf-8"))
        tasks.append(
            {
                "task_name": task_name,
                "level": ast.literal_eval(fields["level"]),
                "category": ast.literal_eval(fields["category"]),
                "candidate_path": _safe_relative(checkout, module_path),
                "candidate_function": pallas_fn,
                "baseline_path": "pallasbench/baselines/jax_baseline.py",
                "baseline_function": baseline_fn,
                "input_shapes": _literal_assignment(module_tree, "input_shapes"),
                "input_dtypes": (
                    ast.literal_eval(fields["input_dtypes"])
                    if "input_dtypes" in fields
                    else None
                ),
                "input_ranges": (
                    ast.literal_eval(fields["input_ranges"])
                    if "input_ranges" in fields
                    else None
                ),
            }
        )
    return tasks


def _candidate_row(
    *,
    source: dict[str, Any],
    path: str | None,
    row_id: str | None,
    content: str,
    objective: str,
    family_category: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = _normalise_python(content)
    identity = (
        f"{source['id']}:{path}" if path is not None else f"{source['id']}:{row_id}"
    )
    candidate_id = f"{identity}:{objective}"
    return {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "source_id": source["id"],
        "source_revision": source["revision"],
        "source_path": path,
        "source_row_id": row_id,
        "license": source["license"],
        "training_policy": source["training_policy"],
        "objective": objective,
        "content": content,
        "content_sha256": _sha256_text(content),
        "normalized_sha256": _sha256_text(normalized),
        "family_id": _family_id(family_category, content),
        "family_category": family_category,
        "metadata": metadata or {},
        "status": "discovered",
        "rejection_reasons": [],
    }


def _pallasbench_candidates(
    source: dict[str, Any],
    checkout: Path,
) -> list[dict[str, Any]]:
    allowlist = set(source.get("sft_task_allowlist", []))
    candidates = []
    for task in _registry_tasks(checkout):
        if task["task_name"] not in allowlist:
            continue
        path = task["candidate_path"]
        if not _allowlisted(source, path):
            raise CorpusError(f"SOURCE_PATH_NOT_ALLOWLISTED: {source['id']}:{path}")
        content = (checkout / path).read_text(encoding="utf-8")
        tree = ast.parse(content)
        candidate_function = task["candidate_function"]
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name)
                    and target.id == candidate_function
                    for target in node.targets
                )
                and isinstance(node.value, ast.Name)
            ):
                candidate_function = node.value.id
                break
        task["candidate_function"] = candidate_function
        inspection_source = (
            f"{content}\n\ndef workload(*args):\n"
            f"    return {candidate_function}(*args)\n"
        )
        inspection = inspect_pallas_source(inspection_source)
        candidate = _candidate_row(
            source=source,
            path=path,
            row_id=None,
            content=content,
            objective="sft",
            family_category=task["category"],
            metadata=task,
        )
        candidate["static_inspection"] = asdict(inspection)
        if not inspection.parses:
            candidate["rejection_reasons"].append("SYNTAX_INVALID")
        if inspection.reachable_interpret_pallas_calls:
            candidate["rejection_reasons"].append("PALLAS_INTERPRET_MODE")
        if not inspection.authentic:
            candidate["rejection_reasons"].extend(inspection.reasons)
        candidates.append(candidate)
    return candidates


def _dapt_candidates(
    source: dict[str, Any],
    checkout: Path,
) -> list[dict[str, Any]]:
    rows = []
    for path in _iter_allowlisted_files(source, checkout):
        content = path.read_text(encoding="utf-8")
        if not any(signal in content for signal in PALLAS_SIGNALS):
            continue
        relative = _safe_relative(checkout, path)
        rows.append(
            _candidate_row(
                source=source,
                path=relative,
                row_id=None,
                content=content,
                objective="dapt",
                family_category=Path(relative).parent.name or source["id"],
            )
        )
    return rows


def _forbidden_documents(
    bundle: ContractBundle,
    source_checkouts: dict[str, Path],
) -> list[tuple[str, str, frozenset[str]]]:
    documents = []
    for source in bundle.sources["sources"]:
        if source["training_policy"] != "forbidden" or source["kind"] != "git":
            continue
        checkout = source_checkouts.get(source["id"])
        if checkout is None:
            raise CorpusError(f"SOURCE_CHECKOUT_REQUIRED: {source['id']}")
        verify_source_checkout(bundle, source["id"], checkout)
        for path in sorted(candidate for candidate in checkout.rglob("*") if candidate.is_file()):
            if path.suffix not in TEXT_SUFFIXES:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            documents.append(
                (
                    f"{source['id']}:{_safe_relative(checkout, path)}",
                    _sha256_text(content),
                    _shingles(content),
                )
            )
    return documents


def _apply_policy(
    candidates: list[dict[str, Any]],
    *,
    forbidden_documents: list[tuple[str, str, frozenset[str]]],
) -> list[dict[str, Any]]:
    seen_exact: dict[str, str] = {}
    seen_normalized: dict[str, str] = {}
    seen_shingles: list[tuple[str, str, frozenset[str]]] = []
    for candidate in sorted(candidates, key=lambda row: row["candidate_id"]):
        reasons = list(candidate["rejection_reasons"])
        if candidate["training_policy"] != "allowlisted_paths_only":
            reasons.append("SOURCE_NOT_TRAINABLE")
        exact = candidate["content_sha256"]
        normalized = candidate["normalized_sha256"]
        exact_key = f"{candidate['objective']}:{exact}"
        normalized_key = f"{candidate['objective']}:{normalized}"
        if exact_key in seen_exact:
            reasons.append(f"EXACT_DUPLICATE:{seen_exact[exact_key]}")
        elif normalized_key in seen_normalized:
            reasons.append(
                f"NORMALIZED_DUPLICATE:{seen_normalized[normalized_key]}"
            )
        candidate_shingles = _shingles(candidate["content"])
        near_matches = [
            (other_id, _jaccard(candidate_shingles, other_shingles))
            for objective, other_id, other_shingles in seen_shingles
            if objective == candidate["objective"]
        ]
        if near_matches:
            other_id, similarity = max(near_matches, key=lambda item: item[1])
            if similarity >= NEAR_DUPLICATE_THRESHOLD:
                reasons.append(f"NEAR_DUPLICATE:{other_id}:{similarity:.6f}")
        for document_id, document_sha, document_shingles in forbidden_documents:
            if exact == document_sha:
                reasons.append(f"HOLDOUT_EXACT_MATCH:{document_id}")
                break
            similarity = _jaccard(candidate_shingles, document_shingles)
            if similarity >= NEAR_DUPLICATE_THRESHOLD:
                reasons.append(
                    f"HOLDOUT_NEAR_MATCH:{document_id}:{similarity:.6f}"
                )
                break
        candidate["rejection_reasons"] = sorted(set(reasons))
        candidate["status"] = (
            "rejected"
            if candidate["rejection_reasons"]
            else (
                "verification_required"
                if candidate["objective"] == "sft"
                else "eligible"
            )
        )
        seen_exact.setdefault(exact_key, candidate["candidate_id"])
        seen_normalized.setdefault(normalized_key, candidate["candidate_id"])
        seen_shingles.append(
            (candidate["objective"], candidate["candidate_id"], candidate_shingles)
        )
    return candidates


def _verification_index(
    paths: Sequence[Path],
    *,
    calibration_root: Path | None,
    expected_runtime: dict[str, str],
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for root in paths:
        for path in sorted(root.rglob("verification.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            candidate_id = value.get("candidate_id")
            if not isinstance(candidate_id, str):
                raise CorpusError(f"VERIFICATION_CANDIDATE_ID_INVALID: {path}")
            if candidate_id in records:
                raise CorpusError(f"VERIFICATION_DUPLICATE: {candidate_id}")
            expected_hash = value.get("verification_sha256")
            unhashed = {
                key: item
                for key, item in value.items()
                if key != "verification_sha256"
            }
            if expected_hash != _canonical_sha256(unhashed):
                raise CorpusError(
                    f"VERIFICATION_HASH_MISMATCH: {candidate_id}"
                )
            if value.get("verified") is False:
                value["_artifact_path"] = str(path.resolve())
                records[candidate_id] = value
                continue
            if calibration_root is None:
                raise CorpusError("VERIFICATION_CALIBRATION_REQUIRED")
            lowering = validate_candidate_evidence(
                calibration_root=calibration_root,
                candidate_root=path.parent,
                expected_kernel_sha256=value.get("kernel_sha256"),
                expected_runtime=expected_runtime,
            )
            if asdict(lowering) != value.get("lowering") or not lowering.verified:
                raise CorpusError(
                    f"VERIFICATION_LOWERING_MISMATCH: {candidate_id}"
                )
            value["_artifact_path"] = str(path.resolve())
            records[candidate_id] = value
    return records


def record_verification_failure(
    *,
    bundle: ContractBundle,
    corpus_root: Path,
    candidate_id: str,
    out_dir: Path,
    error: Exception,
) -> dict[str, Any]:
    candidates = {
        row["candidate_id"]: row
        for row in _read_jsonl(corpus_root / "candidates.jsonl")
    }
    candidate = candidates.get(candidate_id)
    if candidate is None:
        raise CorpusError(f"CANDIDATE_UNKNOWN: {candidate_id}") from error
    message = str(error)
    record = {
        "schema_version": 1,
        "kind": "pallas_corpus_verification",
        "candidate_id": candidate_id,
        "verified_at": _utc_now(),
        "contract_sha256": bundle.sha256,
        "source_revision": candidate["source_revision"],
        "kernel_sha256": candidate["content_sha256"],
        "correctness_seeds": [],
        "verified": False,
        "failure": {
            "code": message.partition(":")[0],
            "detail": message,
            "error_type": type(error).__name__,
        },
    }
    record["verification_sha256"] = _canonical_sha256(record)
    _write_json(out_dir / "verification.json", record)
    return record


def _dapt_row(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "row_id": candidate["candidate_id"],
        "objective": "dapt",
        "text": candidate["content"],
        "family_id": candidate["family_id"],
        "provenance": {
            key: candidate[key]
            for key in (
                "source_id",
                "source_revision",
                "source_path",
                "source_row_id",
                "license",
                "content_sha256",
                "family_id",
            )
        },
    }


def _sft_row(candidate: dict[str, Any], verification: dict[str, Any]) -> dict[str, Any]:
    metadata = candidate["metadata"]
    prompt = (
        "Implement an authentic normal-lowering JAX Pallas kernel for the "
        f"{metadata['task_name']} operation. The callable must accept inputs "
        f"with shapes {metadata['input_shapes']} and dtypes "
        f"{metadata.get('input_dtypes') or ['float32'] * len(metadata['input_shapes'])}. "
        "It must match the independent JAX oracle at the full declared shapes. "
        "Do not use interpret mode or a plain-JAX fallback."
    )
    return {
        "schema_version": 1,
        "row_id": candidate["candidate_id"],
        "objective": "sft",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": candidate["content"]},
        ],
        "family_id": candidate["family_id"],
        "verification": {
            "verification_sha256": verification["verification_sha256"],
            "calibration_sha256": verification["lowering"]["calibration_sha256"],
            "candidate_evidence_sha256": verification["lowering"][
                "candidate_sha256"
            ],
            "kernel_sha256": verification["kernel_sha256"],
            "correctness_seeds": verification["correctness_seeds"],
            "target": verification["runtime"],
            "evidence_relative_path": verification["evidence_relative_path"],
        },
        "provenance": {
            key: candidate[key]
            for key in (
                "source_id",
                "source_revision",
                "source_path",
                "license",
                "content_sha256",
            )
        },
    }


def build_corpus(
    *,
    bundle: ContractBundle,
    repo_root: Path,
    source_checkouts: dict[str, Path],
    out_dir: Path,
    verification_roots: Sequence[Path] = (),
    calibration_root: Path | None = None,
    include_hf: bool = True,
) -> dict[str, Any]:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise CorpusError(f"CORPUS_OUTPUT_NOT_EMPTY: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    inventory: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for source in bundle.sources["sources"]:
        if source["kind"] == "git":
            checkout = source_checkouts.get(source["id"])
            if checkout is None:
                raise CorpusError(f"SOURCE_CHECKOUT_REQUIRED: {source['id']}")
            verify_source_checkout(bundle, source["id"], checkout)
            inventory.extend(_inventory_git_source(source, checkout))
            if source["training_policy"] == "allowlisted_paths_only":
                candidates.extend(_dapt_candidates(source, checkout))
            if source["id"] == "pallasbench":
                candidates.extend(_pallasbench_candidates(source, checkout))
        elif include_hf:
            hf_inventory, hf_candidates = _inventory_hf_source(source)
            inventory.extend(hf_inventory)
            candidates.extend(hf_candidates)
    candidates = _apply_policy(
        candidates,
        forbidden_documents=_forbidden_documents(bundle, source_checkouts),
    )
    verifications = _verification_index(
        verification_roots,
        calibration_root=calibration_root,
        expected_runtime=bundle.eval_policy["runtime"],
    )
    verification_rows = []
    if any(record.get("verified") is True for record in verifications.values()):
        if calibration_root is None:
            raise CorpusError("VERIFICATION_CALIBRATION_REQUIRED")
        shutil.copytree(calibration_root, out_dir / "evidence" / "calibration")
    for candidate_id, verification in sorted(verifications.items()):
        candidate = next(
            (row for row in candidates if row["candidate_id"] == candidate_id),
            None,
        )
        if candidate is None:
            raise CorpusError(f"VERIFICATION_CANDIDATE_UNKNOWN: {candidate_id}")
        if verification.get("kernel_sha256") != candidate["content_sha256"]:
            raise CorpusError(f"VERIFICATION_KERNEL_MISMATCH: {candidate_id}")
        verification_rows.append(
            {
                key: value
                for key, value in verification.items()
                if key != "_artifact_path"
            }
        )
        if verification.get("verified") is not True:
            continue
        if candidate["status"] != "verification_required":
            raise CorpusError(
                f"VERIFICATION_CANDIDATE_INELIGIBLE: {candidate_id}: "
                f"{candidate['status']}"
            )
        candidate["status"] = "verified"
        candidate["verification_sha256"] = verification["verification_sha256"]
        evidence_relative_path = (
            Path("evidence")
            / "candidates"
            / _sha256_text(candidate_id)[:16]
        )
        shutil.copytree(
            Path(verification["_artifact_path"]).parent,
            out_dir / evidence_relative_path,
        )
        verification["evidence_relative_path"] = evidence_relative_path.as_posix()
    dapt_rows = [
        _dapt_row(candidate)
        for candidate in candidates
        if candidate["objective"] == "dapt" and candidate["status"] == "eligible"
    ]
    sft_rows = [
        _sft_row(candidate, verifications[candidate["candidate_id"]])
        for candidate in candidates
        if candidate["objective"] == "sft" and candidate["status"] == "verified"
    ]
    repair_rows: list[dict[str, Any]] = []
    corpus_rows = [*dapt_rows, *sft_rows, *repair_rows]
    _write_jsonl(out_dir / "source_inventory.jsonl", sorted(inventory, key=lambda row: (row["source_id"], str(row["path"]), str(row["row_id"]))))
    _write_jsonl(out_dir / "candidates.jsonl", sorted(candidates, key=lambda row: row["candidate_id"]))
    _write_jsonl(out_dir / "verification.jsonl", verification_rows)
    _write_jsonl(out_dir / "corpus.jsonl", corpus_rows)
    _write_jsonl(out_dir / "datasets" / "dapt.jsonl", dapt_rows)
    _write_jsonl(out_dir / "datasets" / "sft.jsonl", sft_rows)
    _write_jsonl(out_dir / "datasets" / "repair.jsonl", repair_rows)
    artifact_relatives = [
        "source_inventory.jsonl",
        "candidates.jsonl",
        "verification.jsonl",
        "corpus.jsonl",
        "datasets/dapt.jsonl",
        "datasets/sft.jsonl",
        "datasets/repair.jsonl",
    ]
    if (out_dir / "evidence").exists():
        artifact_relatives.extend(
            path.relative_to(out_dir).as_posix()
            for path in sorted((out_dir / "evidence").rglob("*"))
            if path.is_file()
        )
    artifacts = {
        relative: _sha256_file(out_dir / relative)
        for relative in artifact_relatives
    }
    status_counts = Counter(row["status"] for row in candidates)
    manifest = {
        "schema_version": 1,
        "kind": "pallas_corpus_release",
        "created_at": _utc_now(),
        "contract_sha256": bundle.sha256,
        "repository_revision": _repository_revision(repo_root),
        "source_revisions": {
            source["id"]: source["revision"] for source in bundle.sources["sources"]
        },
        "generator_command": "opjax-pallas build-corpus",
        "policy": {
            "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
            "shingle_size": SHINGLE_SIZE,
            "positive_sft_requires_tpu_verification": True,
            "reject_interpret_mode": True,
        },
        "counts": {
            "inventory": len(inventory),
            "candidates": len(candidates),
            "status": dict(sorted(status_counts.items())),
            "dapt": len(dapt_rows),
            "sft": len(sft_rows),
            "repair": len(repair_rows),
            "families": len({row["family_id"] for row in corpus_rows}),
            "holdout_contamination": sum(
                any(reason.startswith("HOLDOUT_") for reason in row["rejection_reasons"])
                for row in candidates
            ),
        },
        "artifacts": artifacts,
    }
    manifest["release_sha256"] = _canonical_sha256(manifest)
    _write_json(out_dir / "manifest.json", manifest)
    validate_corpus_release(out_dir)
    return manifest


def validate_corpus_release(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CorpusError(f"CORPUS_MANIFEST_MISSING: {manifest_path}") from exc
    if manifest.get("kind") != "pallas_corpus_release":
        raise CorpusError("CORPUS_MANIFEST_INVALID")
    expected_release_sha = manifest.get("release_sha256")
    unhashed = {key: value for key, value in manifest.items() if key != "release_sha256"}
    if expected_release_sha != _canonical_sha256(unhashed):
        raise CorpusError("CORPUS_MANIFEST_HASH_MISMATCH")
    for relative, expected in manifest.get("artifacts", {}).items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise CorpusError(f"CORPUS_ARTIFACT_MISSING: {relative}")
        if _sha256_file(path) != expected:
            raise CorpusError(f"CORPUS_ARTIFACT_HASH_MISMATCH: {relative}")
    candidates = {
        row["candidate_id"]: row for row in _read_jsonl(root / "candidates.jsonl")
    }
    sft_rows = _read_jsonl(root / "datasets" / "sft.jsonl")
    for row in sft_rows:
        candidate = candidates.get(row.get("row_id"))
        verification = row.get("verification")
        if (
            candidate is None
            or candidate.get("status") != "verified"
            or candidate.get("content_sha256") != verification.get("kernel_sha256")
            or not isinstance(verification.get("calibration_sha256"), str)
            or not isinstance(verification.get("candidate_evidence_sha256"), str)
            or len(verification["correctness_seeds"]) < 3
            or any(
                reason.startswith("HOLDOUT_")
                for reason in candidate.get("rejection_reasons", [])
            )
        ):
            raise CorpusError(f"SFT_EVIDENCE_INVALID: {row.get('row_id')}")
        evidence_root = (
            root / verification["evidence_relative_path"]
        ).resolve()
        if not evidence_root.is_relative_to(root.resolve()):
            raise CorpusError(f"SFT_EVIDENCE_PATH_INVALID: {row.get('row_id')}")
        lowering = validate_candidate_evidence(
            calibration_root=root / "evidence" / "calibration",
            candidate_root=evidence_root,
            expected_kernel_sha256=verification["kernel_sha256"],
        )
        if (
            not lowering.verified
            or lowering.calibration_sha256
            != verification["calibration_sha256"]
            or lowering.candidate_sha256
            != verification["candidate_evidence_sha256"]
        ):
            raise CorpusError(f"SFT_LOWERING_INVALID: {row.get('row_id')}")
    return {
        "ok": True,
        "release_sha256": expected_release_sha,
        "counts": manifest["counts"],
    }


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise CorpusError(f"MODULE_LOAD_FAILED: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _generate_inputs(
    shapes: Sequence[Sequence[int]],
    dtypes: Sequence[str] | None,
    ranges: Sequence[Sequence[float] | None] | None,
    seed: int,
) -> tuple[jax.Array, ...]:
    dtype_values = list(dtypes or ["float32"] * len(shapes))
    range_values = list(ranges or [None] * len(shapes))
    if len(dtype_values) != len(shapes) or len(range_values) != len(shapes):
        raise CorpusError("INPUT_SPEC_LENGTH_MISMATCH")
    key = jax.random.PRNGKey(seed)
    inputs = []
    for shape, dtype_name, bounds in zip(
        shapes,
        dtype_values,
        range_values,
        strict=True,
    ):
        key, subkey = jax.random.split(key)
        dtype = jnp.dtype(dtype_name)
        shape_tuple = tuple(shape)
        if jnp.issubdtype(dtype, jnp.integer):
            low, high = bounds or (0, max(shape_tuple[-1] if shape_tuple else 1, 2))
            value = jax.random.randint(
                subkey,
                shape_tuple,
                int(low),
                max(int(high), int(low) + 1),
                dtype=dtype,
            )
        elif jnp.issubdtype(dtype, jnp.bool_):
            value = jax.random.bernoulli(subkey, 0.5, shape_tuple)
        elif bounds:
            value = jax.random.uniform(
                subkey,
                shape_tuple,
                dtype=dtype,
                minval=bounds[0],
                maxval=bounds[1],
            )
        else:
            value = jax.random.normal(subkey, shape_tuple, dtype=dtype)
        inputs.append(value)
    return tuple(inputs)


def verify_corpus_candidate(
    *,
    bundle: ContractBundle,
    corpus_root: Path,
    candidate_id: str,
    source_checkout: Path,
    calibration_root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise CorpusError(f"VERIFICATION_OUTPUT_NOT_EMPTY: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    source = source_by_id(bundle, "pallasbench")
    verify_source_checkout(bundle, "pallasbench", source_checkout)
    candidates = {
        row["candidate_id"]: row
        for row in _read_jsonl(corpus_root / "candidates.jsonl")
    }
    candidate = candidates.get(candidate_id)
    if candidate is None:
        raise CorpusError(f"CANDIDATE_UNKNOWN: {candidate_id}")
    if (
        candidate["source_id"] != source["id"]
        or candidate["objective"] != "sft"
        or candidate["status"] != "verification_required"
        or candidate["rejection_reasons"]
    ):
        raise CorpusError(f"CANDIDATE_NOT_VERIFIABLE: {candidate_id}")
    runtime = validate_calibration(
        calibration_root,
        expected_runtime=bundle.eval_policy["runtime"],
    )["runtime"]
    try:
        chex.assert_devices_available("tpu", 1)
    except AssertionError as exc:
        raise CorpusError(f"TPU_REQUIRED: {exc}") from exc
    source_path = source_checkout / candidate["source_path"]
    if _sha256_file(source_path) != candidate["content_sha256"]:
        raise CorpusError(f"CANDIDATE_SOURCE_HASH_MISMATCH: {candidate_id}")
    sys.path.insert(0, str(source_checkout))
    try:
        candidate_module = _load_module(
            source_path,
            f"opjax_corpus_candidate_{_sha256_text(candidate_id)[:12]}",
        )
        baseline_module = _load_module(
            source_checkout / candidate["metadata"]["baseline_path"],
            f"opjax_corpus_baseline_{_sha256_text(candidate_id)[:12]}",
        )
    finally:
        sys.path.remove(str(source_checkout))
    candidate_function = getattr(
        candidate_module,
        candidate["metadata"]["candidate_function"],
        None,
    )
    baseline_function = getattr(
        baseline_module,
        candidate["metadata"]["baseline_function"],
        None,
    )
    if not callable(candidate_function) or not callable(baseline_function):
        raise CorpusError(f"CANDIDATE_CALLABLE_MISSING: {candidate_id}")
    correctness_seeds = []
    capture_inputs: tuple[jax.Array, ...] | None = None
    capture_expected: Any = None
    for seed in (0, 1, 2):
        inputs = _generate_inputs(
            candidate["metadata"]["input_shapes"],
            candidate["metadata"].get("input_dtypes"),
            candidate["metadata"].get("input_ranges"),
            seed,
        )
        expected = jax.jit(baseline_function)(*inputs)
        observed = jax.jit(candidate_function)(*inputs)
        jax.block_until_ready((expected, observed))
        try:
            chex.assert_trees_all_close(
                observed,
                expected,
                rtol=1e-3,
                atol=1e-3,
            )
        except AssertionError as exc:
            raise CorpusError(
                f"CORRECTNESS_FAILED: {candidate_id}: seed={seed}: {exc}"
            ) from exc
        correctness_seeds.append(seed)
        if seed == 0:
            capture_inputs = inputs
            capture_expected = expected
    if capture_inputs is None:
        raise CorpusError("CORRECTNESS_INPUTS_MISSING")
    evidence = capture_lowering_case(
        label="candidate",
        function=candidate_function,
        inputs=capture_inputs,
        out_dir=out_dir,
        repetitions=bundle.eval_policy["authenticity"]["profile_repetitions"],
        expected_output=capture_expected,
        rtol=1e-3,
        atol=1e-3,
    )
    candidate_summary = {
        "schema_version": 1,
        "kind": "pallas_candidate_lowering",
        "captured_at": _utc_now(),
        "capture_tool_sha256": validate_calibration(calibration_root)[
            "capture_tool_sha256"
        ],
        "workload": candidate_id,
        "kernel_sha256": candidate["content_sha256"],
        "evidence": evidence,
    }
    _write_json(out_dir / "candidate.json", candidate_summary)
    lowering = validate_candidate_evidence(
        calibration_root=calibration_root,
        candidate_root=out_dir,
        expected_kernel_sha256=candidate["content_sha256"],
        expected_runtime=bundle.eval_policy["runtime"],
    )
    if not lowering.verified:
        raise CorpusError(
            f"LOWERING_EVIDENCE_FAILED: {candidate_id}: {lowering.reasons}"
        )
    verification = {
        "schema_version": 1,
        "kind": "pallas_corpus_verification",
        "candidate_id": candidate_id,
        "verified_at": _utc_now(),
        "contract_sha256": bundle.sha256,
        "source_revision": source["revision"],
        "kernel_sha256": candidate["content_sha256"],
        "correctness_seeds": correctness_seeds,
        "correctness_tolerance": {"rtol": 1e-3, "atol": 1e-3},
        "lowering": asdict(lowering),
        "runtime": runtime,
        "verified": True,
    }
    verification["verification_sha256"] = _canonical_sha256(verification)
    _write_json(out_dir / "verification.json", verification)
    return verification
