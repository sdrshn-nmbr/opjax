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
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
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
            raise CorpusError(f"SOURCE_ALLOWLIST_PATH_MISSING: {source['id']}:{prefix}")
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


def _inventory_hf_source(
    source: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_id = source["url"].removeprefix("https://huggingface.co/datasets/")
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
                                signal for signal in PALLAS_SIGNALS if signal in code
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
        f"{source['id']}:{path}:{row_id}"
        if path is not None and row_id is not None
        else f"{source['id']}:{path}"
        if path is not None
        else f"{source['id']}:{row_id}"
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


def _binary_expression(operation: str) -> str:
    expressions = {
        "add": "x_ref[...] + y_ref[...]",
        "multiply": "x_ref[...] * y_ref[...]",
        "subtract": "x_ref[...] - y_ref[...]",
        "maximum": "jnp.maximum(x_ref[...], y_ref[...])",
    }
    try:
        return expressions[operation]
    except KeyError as exc:
        raise CorpusError(f"SFT_OPERATION_UNSUPPORTED: {operation}") from exc


def _unary_expression(operation: str) -> str:
    expressions = {
        "relu": "jnp.maximum(x_ref[...], 0.0)",
        "tanh": "jnp.tanh(x_ref[...])",
        "sigmoid": "jax.nn.sigmoid(x_ref[...])",
        "square": "jnp.square(x_ref[...])",
    }
    try:
        return expressions[operation]
    except KeyError as exc:
        raise CorpusError(f"SFT_OPERATION_UNSUPPORTED: {operation}") from exc


def _render_sft_solution(group: dict[str, Any], variant: dict[str, Any]) -> str:
    kind = group["kernel_kind"]
    operation = variant["operation"]
    shape = variant["shape"]
    header = (
        "import jax\n"
        "import jax.numpy as jnp\n"
        "from jax.experimental import pallas as pl\n\n"
    )
    if kind in {"binary", "unary", "gated"}:
        if len(shape) != 2 or any(size % 128 for size in shape):
            raise CorpusError(f"SFT_POINTWISE_SHAPE_INVALID: {variant['id']}")
        if kind == "binary":
            expression = _binary_expression(operation)
            arguments = "x_ref, y_ref, o_ref"
            call_arguments = "x, y"
            in_specs = "(spec, spec)"
        elif kind == "unary":
            expression = _unary_expression(operation)
            arguments = "x_ref, o_ref"
            call_arguments = "x"
            in_specs = "(spec,)"
        else:
            expressions = {
                "silu_gate": "jax.nn.silu(x_ref[...]) * gate_ref[...]",
                "gelu_gate": "jax.nn.gelu(x_ref[...]) * gate_ref[...]",
            }
            try:
                expression = expressions[operation]
            except KeyError as exc:
                raise CorpusError(f"SFT_OPERATION_UNSUPPORTED: {operation}") from exc
            arguments = "x_ref, gate_ref, o_ref"
            call_arguments = "x, gate"
            in_specs = "(spec, spec)"
        workload_args = (
            "x, y" if kind == "binary" else ("x" if kind == "unary" else "x, gate")
        )
        return (
            header
            + f"""SHAPE = {tuple(shape)!r}

def _kernel({arguments}):
    o_ref[...] = {expression}

def workload({workload_args}):
    spec = pl.BlockSpec((128, 128), lambda i, j: (i, j))
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(SHAPE, x.dtype),
        grid=(SHAPE[0] // 128, SHAPE[1] // 128),
        in_specs={in_specs},
        out_specs=spec,
    )({call_arguments})
"""
        )
    if kind in {"normalization", "softmax", "reduction"}:
        if len(shape) != 2 or shape[0] % 8 or shape[1] % 128:
            raise CorpusError(f"SFT_ROW_SHAPE_INVALID: {variant['id']}")
        if kind == "normalization":
            if operation == "rmsnorm":
                body = (
                    "    values = x_ref[...].astype(jnp.float32)\n"
                    "    mean_square = jnp.mean(jnp.square(values), axis=-1, keepdims=True)\n"
                    "    o_ref[...] = values * jax.lax.rsqrt(mean_square + 1e-5)"
                )
            elif operation == "layernorm":
                body = (
                    "    values = x_ref[...].astype(jnp.float32)\n"
                    "    mean = jnp.mean(values, axis=-1, keepdims=True)\n"
                    "    variance = jnp.mean(jnp.square(values - mean), axis=-1, keepdims=True)\n"
                    "    o_ref[...] = (values - mean) * jax.lax.rsqrt(variance + 1e-5)"
                )
            else:
                raise CorpusError(f"SFT_OPERATION_UNSUPPORTED: {operation}")
        elif kind == "softmax":
            body = (
                "    values = x_ref[...].astype(jnp.float32)\n"
                "    maximum = jnp.max(values, axis=-1, keepdims=True)\n"
                "    numerator = jnp.exp(values - maximum)\n"
                "    o_ref[...] = numerator / jnp.sum(numerator, axis=-1, keepdims=True)"
            )
        else:
            reducer = {"sum": "jnp.sum", "max": "jnp.max"}.get(operation)
            if reducer is None:
                raise CorpusError(f"SFT_OPERATION_UNSUPPORTED: {operation}")
            body = (
                f"    reduced = {reducer}(x_ref[...], axis=-1, keepdims=True)\n"
                "    o_ref[...] = jnp.broadcast_to(reduced, x_ref.shape)"
            )
        return (
            header
            + f"""SHAPE = {tuple(shape)!r}

def _kernel(x_ref, o_ref):
{body}

def workload(x):
    spec = pl.BlockSpec((8, SHAPE[1]), lambda i: (i, 0))
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(SHAPE, jnp.float32),
        grid=(SHAPE[0] // 8,),
        in_specs=(spec,),
        out_specs=spec,
    )(x)
"""
        )
    if kind == "transpose":
        if len(shape) != 2 or any(size % 128 for size in shape):
            raise CorpusError(f"SFT_TRANSPOSE_SHAPE_INVALID: {variant['id']}")
        return (
            header
            + f"""SHAPE = {tuple(shape)!r}

def _kernel(x_ref, o_ref):
    o_ref[...] = jnp.transpose(x_ref[...])

def workload(x):
    in_spec = pl.BlockSpec((128, 128), lambda i, j: (i, j))
    out_spec = pl.BlockSpec((128, 128), lambda i, j: (j, i))
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((SHAPE[1], SHAPE[0]), x.dtype),
        grid=(SHAPE[0] // 128, SHAPE[1] // 128),
        in_specs=(in_spec,),
        out_specs=out_spec,
    )(x)
"""
        )
    if kind == "matmul":
        if len(shape) != 3 or any(size % 128 for size in shape):
            raise CorpusError(f"SFT_MATMUL_SHAPE_INVALID: {variant['id']}")
        m, k, n = shape
        return (
            header
            + f"""M, K, N = {m}, {k}, {n}

def _kernel(x_ref, y_ref, o_ref):
    o_ref[...] = jnp.dot(
        x_ref[...], y_ref[...], preferred_element_type=jnp.float32
    )

def workload(x, y):
    x_spec = pl.BlockSpec((128, K), lambda i, j: (i, 0))
    y_spec = pl.BlockSpec((K, 128), lambda i, j: (0, j))
    out_spec = pl.BlockSpec((128, 128), lambda i, j: (i, j))
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        grid=(M // 128, N // 128),
        in_specs=(x_spec, y_spec),
        out_specs=out_spec,
    )(x, y)
"""
        )
    raise CorpusError(f"SFT_KERNEL_KIND_UNSUPPORTED: {kind}")


def _source_function_exists(source: str, function_name: str) -> bool:
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
        for node in ast.walk(ast.parse(source))
    )


def _curated_sft_candidates(
    bundle: ContractBundle,
    source: dict[str, Any],
    checkout: Path,
) -> list[dict[str, Any]]:
    candidates = []
    for group in bundle.sft_candidates["groups"]:
        if group["source_id"] != source["id"]:
            continue
        path = group["source_path"]
        if not _allowlisted(source, path):
            raise CorpusError(f"SOURCE_PATH_NOT_ALLOWLISTED: {source['id']}:{path}")
        source_path = checkout / path
        upstream_source = source_path.read_text(encoding="utf-8")
        if not _source_function_exists(upstream_source, group["source_function"]):
            raise CorpusError(
                f"SOURCE_FUNCTION_MISSING: {source['id']}:{path}:"
                f"{group['source_function']}"
            )
        upstream_sha256 = _sha256_file(source_path)
        for variant in group["variants"]:
            content = _render_sft_solution(group, variant)
            shape = variant["shape"]
            kind = group["kernel_kind"]
            input_shapes = (
                [[shape[0], shape[1]], [shape[1], shape[2]]]
                if kind == "matmul"
                else [shape, shape]
                if kind in {"binary", "gated"}
                else [shape]
            )
            input_dtypes = (
                ["bfloat16", "bfloat16"]
                if kind == "matmul"
                else ["float32"] * len(input_shapes)
            )
            metadata = {
                "task_name": variant["id"],
                "candidate_function": "workload",
                "kernel_kind": kind,
                "operation": variant["operation"],
                "input_shapes": input_shapes,
                "input_dtypes": input_dtypes,
                "input_ranges": [[-1.0, 1.0]] * len(input_shapes),
                "source_function": group["source_function"],
                "source_file_sha256": upstream_sha256,
                "derivation_policy": bundle.sft_candidates["derivation_policy"],
                "specification": (
                    f"Compute {variant['operation']} for full-shape inputs "
                    f"{input_shapes} with output semantics defined independently "
                    "by the operation name."
                ),
                "correctness_tolerance": {
                    "rtol": 0.02 if kind == "matmul" else 0.001,
                    "atol": 0.02 if kind == "matmul" else 0.001,
                },
            }
            candidate = _candidate_row(
                source=source,
                path=path,
                row_id=variant["id"],
                content=content,
                objective="sft",
                family_category=group["id"],
                metadata=metadata,
            )
            inspection = inspect_pallas_source(content)
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
        for path in sorted(
            candidate for candidate in checkout.rglob("*") if candidate.is_file()
        ):
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
            candidate["rejection_reasons"] = sorted(set(reasons))
            candidate["status"] = "rejected"
            continue
        exact = candidate["content_sha256"]
        normalized = candidate["normalized_sha256"]
        exact_key = f"{candidate['objective']}:{exact}"
        normalized_key = f"{candidate['objective']}:{normalized}"
        if exact_key in seen_exact:
            reasons.append(f"EXACT_DUPLICATE:{seen_exact[exact_key]}")
        elif normalized_key in seen_normalized:
            reasons.append(f"NORMALIZED_DUPLICATE:{seen_normalized[normalized_key]}")
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
                reasons.append(f"HOLDOUT_NEAR_MATCH:{document_id}:{similarity:.6f}")
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
                key: item for key, item in value.items() if key != "verification_sha256"
            }
            if expected_hash != _canonical_sha256(unhashed):
                raise CorpusError(f"VERIFICATION_HASH_MISMATCH: {candidate_id}")
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
            if (
                _canonical_sha256(asdict(lowering))
                != _canonical_sha256(value.get("lowering"))
                or not lowering.verified
            ):
                raise CorpusError(f"VERIFICATION_LOWERING_MISMATCH: {candidate_id}")
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
        f"{metadata['task_name']} operation. {metadata['specification']} "
        "The callable must accept inputs "
        f"with shapes {metadata['input_shapes']} and dtypes "
        f"{metadata.get('input_dtypes') or ['float32'] * len(metadata['input_shapes'])}. "
        "It must match the independent semantic oracle at the full declared shapes. "
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
        "family_category": candidate["family_category"],
        "verification": {
            "verification_sha256": verification["verification_sha256"],
            "calibration_sha256": verification["lowering"]["calibration_sha256"],
            "candidate_evidence_sha256": verification["lowering"]["candidate_sha256"],
            "kernel_sha256": verification["kernel_sha256"],
            "correctness_seeds": verification["correctness_seeds"],
            "full_declared_shapes": verification["full_declared_shapes"],
            "input_spec_sha256": verification["input_spec_sha256"],
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
        }
        | {
            "source_function": metadata["source_function"],
            "source_file_sha256": metadata["source_file_sha256"],
            "derivation_policy": metadata["derivation_policy"],
        },
    }


def _sft_readiness(
    sft_rows: Sequence[dict[str, Any]],
    *,
    policy: dict[str, Any],
    holdout_contamination: int,
) -> dict[str, Any]:
    source_counts = Counter(row["provenance"]["source_id"] for row in sft_rows)
    family_counts = Counter(row["family_category"] for row in sft_rows)
    row_count = len(sft_rows)
    maximum_source_fraction = (
        max(source_counts.values(), default=0) / row_count if row_count else 0.0
    )
    maximum_family_fraction = (
        max(family_counts.values(), default=0) / row_count if row_count else 0.0
    )
    reasons = []
    if row_count < policy["minimum_verified_rows"]:
        reasons.append("VERIFIED_ROWS_INSUFFICIENT")
    if len(family_counts) < policy["minimum_family_count"]:
        reasons.append("FAMILY_DIVERSITY_INSUFFICIENT")
    if len(source_counts) < policy["minimum_source_count"]:
        reasons.append("SOURCE_DIVERSITY_INSUFFICIENT")
    if any(
        count < policy["minimum_rows_per_family"] for count in family_counts.values()
    ):
        reasons.append("FAMILY_DEPTH_INSUFFICIENT")
    if maximum_source_fraction > policy["maximum_source_fraction"]:
        reasons.append("SOURCE_CONCENTRATION_EXCEEDED")
    if maximum_family_fraction > policy["maximum_family_fraction"]:
        reasons.append("FAMILY_CONCENTRATION_EXCEEDED")
    if holdout_contamination:
        reasons.append("HOLDOUT_CONTAMINATION_PRESENT")
    return {
        "arm_b_authorized": not reasons,
        "reasons": reasons,
        "observed": {
            "verified_rows": row_count,
            "family_count": len(family_counts),
            "source_count": len(source_counts),
            "source_counts": dict(sorted(source_counts.items())),
            "family_counts": dict(sorted(family_counts.items())),
            "maximum_source_fraction": maximum_source_fraction,
            "maximum_family_fraction": maximum_family_fraction,
            "holdout_contamination": holdout_contamination,
        },
        "required": policy,
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
                candidates.extend(_curated_sft_candidates(bundle, source, checkout))
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
            Path("evidence") / "candidates" / _sha256_text(candidate_id)[:16]
        )
        shutil.copytree(
            Path(verification["_artifact_path"]).parent,
            out_dir / evidence_relative_path,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
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
    _write_jsonl(
        out_dir / "source_inventory.jsonl",
        sorted(
            inventory,
            key=lambda row: (row["source_id"], str(row["path"]), str(row["row_id"])),
        ),
    )
    _write_jsonl(
        out_dir / "candidates.jsonl",
        sorted(candidates, key=lambda row: row["candidate_id"]),
    )
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
        relative: _sha256_file(out_dir / relative) for relative in artifact_relatives
    }
    status_counts = Counter(row["status"] for row in candidates)
    holdout_contamination = sum(
        any(reason.startswith("HOLDOUT_") for reason in row["rejection_reasons"])
        for row in candidates
    )
    readiness = _sft_readiness(
        sft_rows,
        policy=bundle.experiment["sft_readiness"],
        holdout_contamination=holdout_contamination,
    )
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
            "holdout_contamination": holdout_contamination,
        },
        "sft_readiness": readiness,
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
    unhashed = {
        key: value for key, value in manifest.items() if key != "release_sha256"
    }
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
            or verification.get("full_declared_shapes") is not True
            or not isinstance(verification.get("input_spec_sha256"), str)
            or any(
                reason.startswith("HOLDOUT_")
                for reason in candidate.get("rejection_reasons", [])
            )
        ):
            raise CorpusError(f"SFT_EVIDENCE_INVALID: {row.get('row_id')}")
        evidence_root = (root / verification["evidence_relative_path"]).resolve()
        if not evidence_root.is_relative_to(root.resolve()):
            raise CorpusError(f"SFT_EVIDENCE_PATH_INVALID: {row.get('row_id')}")
        lowering = validate_candidate_evidence(
            calibration_root=root / "evidence" / "calibration",
            candidate_root=evidence_root,
            expected_kernel_sha256=verification["kernel_sha256"],
        )
        if (
            not lowering.verified
            or lowering.calibration_sha256 != verification["calibration_sha256"]
            or lowering.candidate_sha256 != verification["candidate_evidence_sha256"]
        ):
            raise CorpusError(f"SFT_LOWERING_INVALID: {row.get('row_id')}")
    readiness = manifest.get("sft_readiness")
    if not isinstance(readiness, dict) or not isinstance(
        readiness.get("required"), dict
    ):
        raise CorpusError("CORPUS_SFT_READINESS_INVALID")
    observed_readiness = _sft_readiness(
        sft_rows,
        policy=readiness["required"],
        holdout_contamination=manifest["counts"]["holdout_contamination"],
    )
    if observed_readiness != readiness:
        raise CorpusError("CORPUS_SFT_READINESS_INVALID")
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


def _assert_tpu_available() -> None:
    try:
        chex.assert_devices_available(1, "tpu", not_less_than=True)
    except AssertionError as exc:
        raise CorpusError(f"TPU_REQUIRED: {exc}") from exc


def _semantic_oracle(operation: str, *inputs: jax.Array) -> jax.Array:
    x = inputs[0]
    if operation == "add":
        return x + inputs[1]
    if operation == "multiply":
        return x * inputs[1]
    if operation == "subtract":
        return x - inputs[1]
    if operation == "maximum":
        return jnp.maximum(x, inputs[1])
    if operation == "relu":
        return jnp.maximum(x, 0.0)
    if operation == "tanh":
        return jnp.tanh(x)
    if operation == "sigmoid":
        return jax.nn.sigmoid(x)
    if operation == "square":
        return jnp.square(x)
    if operation == "rmsnorm":
        values = x.astype(jnp.float32)
        return values * jax.lax.rsqrt(
            jnp.mean(jnp.square(values), axis=-1, keepdims=True) + 1e-5
        )
    if operation == "layernorm":
        values = x.astype(jnp.float32)
        mean = jnp.mean(values, axis=-1, keepdims=True)
        variance = jnp.mean(jnp.square(values - mean), axis=-1, keepdims=True)
        return (values - mean) * jax.lax.rsqrt(variance + 1e-5)
    if operation == "softmax":
        return jax.nn.softmax(x.astype(jnp.float32), axis=-1)
    if operation == "transpose":
        return jnp.transpose(x)
    if operation == "matmul":
        return jnp.matmul(x, inputs[1], preferred_element_type=jnp.float32)
    if operation == "silu_gate":
        return jax.nn.silu(x) * inputs[1]
    if operation == "gelu_gate":
        return jax.nn.gelu(x) * inputs[1]
    if operation == "sum":
        reduced = jnp.sum(x, axis=-1, keepdims=True)
        return jnp.broadcast_to(reduced, x.shape)
    if operation == "max":
        reduced = jnp.max(x, axis=-1, keepdims=True)
        return jnp.broadcast_to(reduced, x.shape)
    raise CorpusError(f"ORACLE_OPERATION_UNSUPPORTED: {operation}")


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
    candidates = {
        row["candidate_id"]: row
        for row in _read_jsonl(corpus_root / "candidates.jsonl")
    }
    candidate = candidates.get(candidate_id)
    if candidate is None:
        raise CorpusError(f"CANDIDATE_UNKNOWN: {candidate_id}")
    source = source_by_id(bundle, candidate["source_id"])
    verify_source_checkout(bundle, source["id"], source_checkout)
    if (
        candidate["objective"] != "sft"
        or candidate["status"] != "verification_required"
        or candidate["rejection_reasons"]
    ):
        raise CorpusError(f"CANDIDATE_NOT_VERIFIABLE: {candidate_id}")
    runtime = validate_calibration(
        calibration_root,
        expected_runtime=bundle.eval_policy["runtime"],
    )["runtime"]
    _assert_tpu_available()
    source_path = source_checkout / candidate["source_path"]
    if _sha256_file(source_path) != candidate["metadata"]["source_file_sha256"]:
        raise CorpusError(f"CANDIDATE_SOURCE_HASH_MISMATCH: {candidate_id}")
    derived_module_path = out_dir / "derived_solution.py"
    derived_module_path.write_text(candidate["content"], encoding="utf-8")
    candidate_module = _load_module(
        derived_module_path,
        f"opjax_corpus_candidate_{_sha256_text(candidate_id)[:12]}",
    )
    candidate_function = getattr(
        candidate_module,
        candidate["metadata"]["candidate_function"],
        None,
    )
    if not callable(candidate_function):
        raise CorpusError(f"CANDIDATE_CALLABLE_MISSING: {candidate_id}")
    correctness_seeds = []
    capture_inputs: tuple[jax.Array, ...] | None = None
    capture_expected: Any = None
    tolerance = candidate["metadata"]["correctness_tolerance"]
    for seed in bundle.experiment["sft_readiness"]["required_correctness_seeds"]:
        inputs = _generate_inputs(
            candidate["metadata"]["input_shapes"],
            candidate["metadata"].get("input_dtypes"),
            candidate["metadata"].get("input_ranges"),
            seed,
        )
        expected = jax.jit(
            lambda *values: _semantic_oracle(
                candidate["metadata"]["operation"], *values
            )
        )(*inputs)
        observed = jax.jit(candidate_function)(*inputs)
        jax.block_until_ready((expected, observed))
        try:
            chex.assert_trees_all_close(
                observed,
                expected,
                rtol=tolerance["rtol"],
                atol=tolerance["atol"],
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
        rtol=tolerance["rtol"],
        atol=tolerance["atol"],
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
        "correctness_tolerance": tolerance,
        "full_declared_shapes": True,
        "input_spec_sha256": _canonical_sha256(
            {
                "shapes": candidate["metadata"]["input_shapes"],
                "dtypes": candidate["metadata"]["input_dtypes"],
                "ranges": candidate["metadata"]["input_ranges"],
            }
        ),
        "lowering": asdict(lowering),
        "runtime": runtime,
        "verified": True,
    }
    verification["verification_sha256"] = _canonical_sha256(verification)
    _write_json(out_dir / "verification.json", verification)
    return verification
