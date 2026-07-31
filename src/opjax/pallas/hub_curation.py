"""Pinned row extraction and curation for Hub-discovered kernel sources."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import pyarrow.parquet as parquet
from huggingface_hub import HfApi, hf_hub_download

from opjax.pallas.contracts import ContractBundle, verify_source_checkout
from opjax.pallas.hub_discovery import validate_hub_discovery_release

HUB_CURATION_CONFIG_SCHEMA_VERSION = 1
HUB_CURATION_ARTIFACT_SCHEMA_VERSION = 1
HUB_CURATION_GENERATOR_VERSION = 1
SHINGLE_SIZE = 5
PERMISSIVE_LICENSES = frozenset(
    {
        "apache",
        "apache-2.0",
        "bsd",
        "bsd-2-clause",
        "bsd-3-clause",
        "cc0",
        "mit",
        "unlicense",
    }
)
BENCHMARK_TAGS = frozenset(
    {
        "benchmark",
        "benchmarks",
        "evaluation",
        "kernel-benchmark",
        "kernelbench",
        "llm-evaluation",
        "pallasbench",
    }
)
VALID_ROLES = frozenset({"pallas_code", "cross_kernel_code", "agent_trace"})
VALID_OBJECTIVES = frozenset({"dapt_candidate", "repair_candidate"})
VALID_ADAPTERS = frozenset(
    {"parquet_records", "json_records", "jsonl_messages", "repository_files"}
)


class HubCurationError(RuntimeError):
    """Hub row evidence violates the curation contract."""


@dataclass(frozen=True)
class HubCurationConfig:
    accepted_discovery_release_sha256: str
    near_duplicate_threshold: float
    benchmark_policy: str
    sources: tuple[dict[str, Any], ...]
    sha256: str


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


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _canonical_sha256(value: Any) -> str:
    return _sha256_text(_canonical_json(value))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    temporary.replace(path)
    return count


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    try:
        handle = path.open(encoding="utf-8")
    except FileNotFoundError as exc:
        raise HubCurationError(f"HUB_ROW_ARTIFACT_MISSING: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HubCurationError(
                    f"HUB_ROW_JSON_INVALID: {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise HubCurationError(
                    f"HUB_ROW_NOT_OBJECT: {path}:{line_number}"
                )
            yield value


def _valid_revision(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _repository_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise HubCurationError(
            f"REPOSITORY_REVISION_UNAVAILABLE: {path}: {exc.output.strip()}"
        ) from exc


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HubCurationError(f"HUB_ROW_CONFIG_MISSING: {path}") from exc
    except json.JSONDecodeError as exc:
        raise HubCurationError(f"HUB_ROW_CONFIG_INVALID: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise HubCurationError(f"HUB_ROW_CONFIG_NOT_OBJECT: {path}")
    return value


def load_hub_curation_config(path: Path) -> HubCurationConfig:
    value = _load_json_object(path)
    if value.get("schema_version") != HUB_CURATION_CONFIG_SCHEMA_VERSION:
        raise HubCurationError("HUB_ROW_CONFIG_SCHEMA_UNSUPPORTED")
    release_sha = value.get("accepted_discovery_release_sha256")
    threshold = value.get("near_duplicate_threshold")
    sources = value.get("sources")
    if not isinstance(release_sha, str) or len(release_sha) != 64:
        raise HubCurationError("HUB_ROW_CONFIG_RELEASE_INVALID")
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not 0 < float(threshold) <= 1
    ):
        raise HubCurationError("HUB_ROW_CONFIG_DEDUP_INVALID")
    if value.get("benchmark_policy") != "forbidden":
        raise HubCurationError("HUB_ROW_BENCHMARK_POLICY_INVALID")
    if not isinstance(sources, list) or not sources:
        raise HubCurationError("HUB_ROW_CONFIG_SOURCES_INVALID")
    seen: set[str] = set()
    parsed: list[dict[str, Any]] = []
    for source in sources:
        if not isinstance(source, dict):
            raise HubCurationError("HUB_ROW_CONFIG_SOURCE_INVALID")
        dataset_id = source.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id or dataset_id in seen:
            raise HubCurationError(f"HUB_ROW_CONFIG_SOURCE_INVALID: {dataset_id!r}")
        seen.add(dataset_id)
        if not _valid_revision(source.get("revision")):
            raise HubCurationError(f"HUB_ROW_SOURCE_REVISION_UNPINNED: {dataset_id}")
        if source.get("role") not in VALID_ROLES:
            raise HubCurationError(f"HUB_ROW_ROLE_INVALID: {dataset_id}")
        if source.get("objective") not in VALID_OBJECTIVES:
            raise HubCurationError(f"HUB_ROW_OBJECTIVE_INVALID: {dataset_id}")
        if source.get("adapter") not in VALID_ADAPTERS:
            raise HubCurationError(f"HUB_ROW_ADAPTER_INVALID: {dataset_id}")
        benchmark_scope = source.get("benchmark_scope")
        if benchmark_scope not in {None, "implementation_paths_only"}:
            raise HubCurationError(f"HUB_ROW_BENCHMARK_SCOPE_INVALID: {dataset_id}")
        if benchmark_scope == "implementation_paths_only" and source["adapter"] != "repository_files":
            raise HubCurationError(f"HUB_ROW_BENCHMARK_SCOPE_INVALID: {dataset_id}")
        license_id = source.get("license")
        if not isinstance(license_id, str) or license_id.casefold() not in PERMISSIVE_LICENSES:
            raise HubCurationError(f"HUB_ROW_LICENSE_AMBIGUOUS: {dataset_id}")
        max_rows = source.get("max_rows")
        if not isinstance(max_rows, int) or isinstance(max_rows, bool) or max_rows <= 0:
            raise HubCurationError(f"HUB_ROW_LIMIT_INVALID: {dataset_id}")
        parsed.append(source)
    return HubCurationConfig(
        accepted_discovery_release_sha256=release_sha,
        near_duplicate_threshold=float(threshold),
        benchmark_policy="forbidden",
        sources=tuple(parsed),
        sha256=_canonical_sha256(value),
    )


def _discovery_rows(
    discovery_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    inventory = {
        row["dataset_id"]: row
        for row in _iter_jsonl(discovery_root / "candidates.jsonl")
    }
    decisions = {
        row["dataset_id"]: row
        for row in _iter_jsonl(discovery_root / "decisions.jsonl")
        if row.get("status") == "candidate"
    }
    return inventory, decisions


def _format_score(files: Sequence[Mapping[str, Any]]) -> int:
    suffixes = {Path(str(row.get("path", ""))).suffix.casefold() for row in files}
    if suffixes & {".parquet", ".json", ".jsonl", ".csv", ".py", ".cu", ".cuh"}:
        return 2
    if suffixes & {".tar", ".gz", ".zip"}:
        return 0
    return 1


def _source_benchmark_signals(row: Mapping[str, Any]) -> list[str]:
    signals: set[str] = set()
    tags = {str(tag).casefold() for tag in row.get("tags", [])}
    card = row.get("dataset_card")
    if isinstance(card, dict):
        tags.update(str(tag).casefold() for tag in card.get("tags", []))
    signals.update(f"tag:{tag}" for tag in tags & BENCHMARK_TAGS)
    dataset_id = str(row.get("dataset_id", "")).casefold()
    for token in ("kernelbench", "pallasbench", "accel-eval", "acceleval"):
        if token in dataset_id:
            signals.add(f"dataset_id:{token}")
    if re.search(
        r"(?:^|[-_/])(eval|evaluation|benchmark|bench|test)(?:$|[-_/])",
        dataset_id,
    ):
        signals.add("dataset_id:evaluation_term")
    return sorted(signals)


def rank_hub_sources(discovery_root: Path) -> list[dict[str, Any]]:
    report = validate_hub_discovery_release(discovery_root)
    inventory, decisions = _discovery_rows(discovery_root)
    ranked: list[dict[str, Any]] = []
    for dataset_id, decision in decisions.items():
        row = inventory[dataset_id]
        license_id = str(row.get("license", "unverified")).casefold()
        benchmark_signals = _source_benchmark_signals(row)
        license_score = 3 if license_id in PERMISSIVE_LICENSES else 0
        provenance_score = 2 if _valid_revision(row.get("source_revision")) else 0
        format_score = _format_score(row.get("files", []))
        kernel_score = min(4, int(decision.get("score", 0)) // 6)
        accessibility_score = 2 if not row.get("inspection_failures") and not row.get("gated") else 0
        contamination_penalty = 20 if benchmark_signals or decision.get("training_policy") == "forbidden" else 0
        total = (
            license_score
            + provenance_score
            + format_score
            + kernel_score
            + accessibility_score
            - contamination_penalty
        )
        ranked.append(
            {
                "schema_version": HUB_CURATION_ARTIFACT_SCHEMA_VERSION,
                "dataset_id": dataset_id,
                "source_revision": row.get("source_revision"),
                "license": row.get("license"),
                "category": decision.get("category"),
                "declared_objectives": decision.get("candidate_objectives"),
                "scores": {
                    "license": license_score,
                    "provenance": provenance_score,
                    "format": format_score,
                    "kernel_density_proxy": kernel_score,
                    "accessibility": accessibility_score,
                    "contamination_penalty": contamination_penalty,
                    "total": total,
                },
                "benchmark_signals": benchmark_signals,
                "risk_flags": sorted(
                    set(decision.get("risk_flags", []))
                    | ({"BENCHMARK_CONTAMINATION"} if benchmark_signals else set())
                ),
                "training_authorized": False,
                "discovery_release_sha256": report["release_sha256"],
            }
        )
    return sorted(ranked, key=lambda row: (-row["scores"]["total"], row["dataset_id"]))


def _normalise_code(source: str) -> str:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return ast.unparse(ast.parse(source))
    except SyntaxError:
        return " ".join(source.split())


def _family_id(role: str, content: str, hint: Any) -> str:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(content)
    except SyntaxError:
        calls = sorted(
            signal
            for signal in ("@triton.jit", "__global__", "pallas_call", "tl.")
            if signal in content
        )
    else:
        calls = sorted(
            {
                node.func.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            }
        )
    return f"{role}:{_canonical_sha256({'role': role, 'hint': hint, 'calls': calls})[:16]}"


def _shingles(source: str) -> frozenset[str]:
    tokens = tuple(
        re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\s]", _normalise_code(source))
    )
    if len(tokens) < SHINGLE_SIZE:
        return frozenset({" ".join(tokens)}) if tokens else frozenset()
    return frozenset(
        " ".join(tokens[index : index + SHINGLE_SIZE])
        for index in range(len(tokens) - SHINGLE_SIZE + 1)
    )


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _read_source_records(path: Path, adapter: str) -> Iterable[dict[str, Any]]:
    if adapter == "parquet_records":
        parquet_file = parquet.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=128):
            yield from batch.to_pylist()
        return
    if adapter == "json_records":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise HubCurationError(f"HUB_ROW_SCHEMA_DRIFT: {path}: expected list")
        yield from value
        return
    if adapter == "jsonl_messages":
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    yield {"_malformed_line": line_number, "_raw": line.rstrip("\n")}
                    continue
                yield value
        return
    raise HubCurationError(f"HUB_ROW_ADAPTER_UNSUPPORTED: {adapter}")


def _source_files(
    source: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> list[dict[str, Any]]:
    available = {
        row["path"]: row
        for row in inventory.get("files", [])
        if isinstance(row, dict) and isinstance(row.get("path"), str)
    }
    if source["adapter"] == "repository_files":
        prefixes = source.get("path_prefixes", [])
        suffixes = source.get("path_suffixes", [])
        paths = [
            path
            for path in available
            if any(path.startswith(prefix) for prefix in prefixes)
            and any(path.endswith(suffix) for suffix in suffixes)
            and not any(
                Path(path).match(pattern) for pattern in source.get("path_excludes", [])
            )
        ]
    else:
        paths = source.get("paths", [])
    missing = sorted(path for path in paths if path not in available)
    if missing:
        raise HubCurationError(
            f"HUB_ROW_SOURCE_FILE_NOT_DISCOVERED: {source['dataset_id']}:{missing}"
        )
    return [available[path] for path in sorted(paths)[: source["max_rows"]]]


def _download_source_file(
    source: Mapping[str, Any],
    file: Mapping[str, Any],
    downloader: Callable[..., str],
) -> Path:
    path = Path(
        downloader(
            repo_id=source["dataset_id"],
            filename=file["path"],
            repo_type="dataset",
            revision=source["revision"],
        )
    )
    expected_lfs = file.get("lfs_sha256")
    if isinstance(expected_lfs, str) and _sha256_file(path) != expected_lfs:
        raise HubCurationError(
            f"HUB_ROW_SOURCE_FILE_HASH_MISMATCH: {source['dataset_id']}:{file['path']}"
        )
    return path


def _record_content(record: Mapping[str, Any], source: Mapping[str, Any]) -> str | None:
    if source["adapter"] == "repository_files":
        content = record.get("content")
    else:
        content = record.get(source.get("content_field"))
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _canonical_json(content)
    return None


def _row_identity(
    record: Mapping[str, Any],
    source: Mapping[str, Any],
    source_path: str,
    index: int,
) -> str:
    row_id_field = source.get("row_id_field")
    if isinstance(row_id_field, str):
        value = record.get(row_id_field)
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            return str(value)
    if source["adapter"] == "repository_files":
        return source_path
    return f"{source_path}:{index}"


def _row_reasons(
    record: Mapping[str, Any],
    source: Mapping[str, Any],
    content: str | None,
    benchmark_signals: Sequence[str],
) -> list[str]:
    reasons: set[str] = set()
    if source.get("benchmark_source") or benchmark_signals:
        reasons.add("BENCHMARK_CONTAMINATION")
    required = source.get("required_fields", [])
    if not isinstance(record, dict) or any(field not in record for field in required):
        reasons.add("SCHEMA_INVALID")
    if content is None or not content.strip():
        reasons.add("CONTENT_MISSING")
    elif not any(signal.casefold() in content.casefold() for signal in source.get("required_kernel_signals", [])):
        reasons.add("KERNEL_DENSITY_BELOW_THRESHOLD")
    provenance_fields = source.get("provenance_fields", [])
    if source["adapter"] != "repository_files" and (
        not provenance_fields
        or any(record.get(field) in {None, ""} for field in provenance_fields)
    ):
        reasons.add("ROW_PROVENANCE_INCOMPLETE")
    commit = record.get("commit_hash")
    if "commit_hash" in provenance_fields and not _valid_revision(commit):
        reasons.add("ROW_REVISION_UNPINNED")
    row_license_field = source.get("row_license_field")
    if isinstance(row_license_field, str):
        licenses = record.get(row_license_field)
        if not isinstance(licenses, list) or not licenses:
            reasons.add("ROW_LICENSE_AMBIGUOUS")
        else:
            allowed = {
                str(value).casefold() for value in source.get("allowed_row_licenses", [])
            }
            observed = {str(value).casefold() for value in licenses}
            if not observed <= allowed:
                reasons.add("ROW_LICENSE_AMBIGUOUS")
    if source["objective"] == "repair_candidate":
        reasons.add("REPAIR_EXECUTION_EVIDENCE_MISSING")
    reasons.update(str(reason) for reason in source.get("source_quarantine_reasons", []))
    return sorted(reasons)


def _forbidden_documents(
    bundle: ContractBundle,
    jaxbench_root: Path,
) -> list[tuple[str, str, frozenset[str]]]:
    verify_source_checkout(bundle, "jaxbench", jaxbench_root)
    documents = []
    for path in sorted(jaxbench_root.rglob("*.py")):
        content = path.read_text(encoding="utf-8", errors="replace")
        documents.append(
            (
                path.relative_to(jaxbench_root).as_posix(),
                _sha256_text(content),
                _shingles(content),
            )
        )
    return documents


def _apply_row_policy(
    rows: list[dict[str, Any]],
    *,
    threshold: float,
    forbidden_documents: Sequence[tuple[str, str, frozenset[str]]],
) -> list[dict[str, Any]]:
    exact_seen: dict[str, str] = {}
    normalized_seen: dict[str, str] = {}
    accepted_shingles: list[tuple[str, frozenset[str]]] = []
    for row in sorted(rows, key=lambda item: item["candidate_id"]):
        reasons = set(row["rejection_reasons"])
        content = row.get("content")
        if isinstance(content, str) and content:
            digest = row["content_sha256"]
            normalized_digest = row["normalized_sha256"]
            duplicate = exact_seen.get(digest)
            if duplicate is not None:
                reasons.add(f"EXACT_DUPLICATE:{duplicate}")
            else:
                exact_seen[digest] = row["candidate_id"]
            normalized_duplicate = normalized_seen.get(normalized_digest)
            if duplicate is None and normalized_duplicate is not None:
                reasons.add(f"NORMALIZED_DUPLICATE:{normalized_duplicate}")
            else:
                normalized_seen.setdefault(normalized_digest, row["candidate_id"])
            shingles = _shingles(content)
            for path, forbidden_sha, forbidden_shingles in forbidden_documents:
                if digest == forbidden_sha:
                    reasons.add(f"JAXBENCH_EXACT_CONTAMINATION:{path}")
                    break
                if shingles and _jaccard(shingles, forbidden_shingles) >= threshold:
                    reasons.add(f"JAXBENCH_NEAR_CONTAMINATION:{path}")
                    break
            if not reasons:
                for other_id, other_shingles in accepted_shingles:
                    if shingles and _jaccard(shingles, other_shingles) >= threshold:
                        reasons.add(f"NEAR_DUPLICATE:{other_id}")
                        break
            if not reasons:
                accepted_shingles.append((row["candidate_id"], shingles))
        row["rejection_reasons"] = sorted(reasons)
        row["status"] = "curated_candidate" if not reasons else "rejected"
        row["training_authorized"] = False
    return sorted(rows, key=lambda item: item["candidate_id"])


def _validate_source_binding(
    source: Mapping[str, Any],
    inventory: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> None:
    dataset_id = source["dataset_id"]
    if inventory.get("source_revision") != source["revision"] or decision.get("source_revision") != source["revision"]:
        raise HubCurationError(f"HUB_ROW_SOURCE_REVISION_MISMATCH: {dataset_id}")
    if str(inventory.get("license", "")).casefold() != str(source["license"]).casefold():
        raise HubCurationError(f"HUB_ROW_SOURCE_LICENSE_MISMATCH: {dataset_id}")
    if decision.get("direct_training_authorized") is not False:
        raise HubCurationError(f"HUB_ROW_DISCOVERY_ROLE_LEAKAGE: {dataset_id}")


def curate_hub_rows(
    *,
    bundle: ContractBundle,
    repo_root: Path,
    discovery_root: Path,
    config_path: Path,
    jaxbench_root: Path,
    out_dir: Path,
    resume: bool = False,
    api: Any | None = None,
    downloader: Callable[..., str] = hf_hub_download,
) -> dict[str, Any]:
    config = load_hub_curation_config(config_path)
    discovery = validate_hub_discovery_release(discovery_root)
    if discovery["release_sha256"] != config.accepted_discovery_release_sha256:
        raise HubCurationError("HUB_ROW_DISCOVERY_RELEASE_MISMATCH")
    invocation = {
        "generator_version": HUB_CURATION_GENERATOR_VERSION,
        "repository_revision": _repository_revision(repo_root),
        "contract_sha256": bundle.sha256,
        "config_sha256": config.sha256,
        "discovery_release_sha256": discovery["release_sha256"],
        "jaxbench_revision": next(
            source["revision"]
            for source in bundle.sources["sources"]
            if source["id"] == "jaxbench"
        ),
    }
    invocation_sha = _canonical_sha256(invocation)
    if out_dir.exists() and any(out_dir.iterdir()):
        if resume and (out_dir / "manifest.json").is_file():
            validated = validate_hub_row_release(out_dir)
            if validated["invocation_sha256"] != invocation_sha:
                raise HubCurationError("HUB_ROW_RESUME_FINGERPRINT_MISMATCH")
            return _load_json_object(out_dir / "manifest.json")
        raise HubCurationError(f"HUB_ROW_OUTPUT_NOT_EMPTY: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    inventory, decisions = _discovery_rows(discovery_root)
    ranked = rank_hub_sources(discovery_root)
    configured_ids = {source["dataset_id"] for source in config.sources}
    if not configured_ids <= inventory.keys() or not configured_ids <= decisions.keys():
        missing = sorted(configured_ids - (inventory.keys() & decisions.keys()))
        raise HubCurationError(f"HUB_ROW_SOURCE_NOT_DISCOVERED: {missing}")
    api = api or HfApi()
    rows: list[dict[str, Any]] = []
    source_results: list[dict[str, Any]] = []
    for source in sorted(config.sources, key=lambda item: item["dataset_id"]):
        dataset_id = source["dataset_id"]
        discovered = inventory[dataset_id]
        decision = decisions[dataset_id]
        _validate_source_binding(source, discovered, decision)
        observed_benchmark_signals = _source_benchmark_signals(discovered)
        if (
            any(signal.startswith("dataset_id:") for signal in observed_benchmark_signals)
            and not source.get("benchmark_source")
            and source.get("benchmark_scope") != "implementation_paths_only"
        ):
            raise HubCurationError(
                f"HUB_ROW_BENCHMARK_CLASSIFICATION_REQUIRED: {dataset_id}"
            )
        benchmark_signals = (
            []
            if source.get("benchmark_scope") == "implementation_paths_only"
            else observed_benchmark_signals
        )
        source_result = {
            "schema_version": HUB_CURATION_ARTIFACT_SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "source_revision": source["revision"],
            "license": source["license"],
            "role": source["role"],
            "objective": source["objective"],
            "adapter": source["adapter"],
            "benchmark_signals": observed_benchmark_signals,
            "benchmark_scope": source.get("benchmark_scope"),
            "source_family": source.get("source_family", dataset_id),
            "training_authorized": False,
            "status": "complete",
            "failure": None,
            "counts": {"gross": 0, "curated_candidate": 0, "rejected": 0},
        }
        try:
            detail = api.dataset_info(
                dataset_id,
                revision=source["revision"],
                files_metadata=True,
            )
            if getattr(detail, "sha", None) != source["revision"]:
                raise HubCurationError(f"HUB_ROW_LIVE_REVISION_MISMATCH: {dataset_id}")
            files = _source_files(source, discovered)
            remaining = source["max_rows"]
            for file in files:
                if remaining <= 0:
                    break
                local_path = _download_source_file(source, file, downloader)
                file_sha = _sha256_file(local_path)
                if source["adapter"] == "repository_files":
                    records: Iterable[dict[str, Any]] = [
                        {"content": local_path.read_text(encoding="utf-8", errors="replace")}
                    ]
                else:
                    records = _read_source_records(local_path, source["adapter"])
                for index, record in enumerate(records):
                    if remaining <= 0:
                        break
                    remaining -= 1
                    source_result["counts"]["gross"] += 1
                    if not isinstance(record, dict):
                        record = {"_invalid_record": repr(record)}
                    row_id = _row_identity(record, source, file["path"], index)
                    content = _record_content(record, source)
                    evidence = {
                        key: record.get(key) for key in source.get("provenance_fields", [])
                    }
                    family_hint = record.get("category")
                    if family_hint is None and source["adapter"] == "repository_files":
                        family_hint = Path(file["path"]).parent.as_posix()
                    if family_hint is None:
                        family_hint = record.get("entry_point")
                    candidate_id = f"hf:{dataset_id}@{source['revision']}:{row_id}"
                    rows.append(
                        {
                            "schema_version": HUB_CURATION_ARTIFACT_SCHEMA_VERSION,
                            "evidence_level": "row_candidate",
                            "candidate_id": candidate_id,
                            "dataset_id": dataset_id,
                            "source_revision": source["revision"],
                            "source_path": file["path"],
                            "source_file_sha256": file_sha,
                            "source_row_id": row_id,
                            "license": source["license"],
                            "row_license": record.get(source.get("row_license_field"))
                            if source.get("row_license_field")
                            else None,
                            "role": source["role"],
                            "objective": source["objective"],
                            "source_family": source.get("source_family", dataset_id),
                            "content": content,
                            "content_sha256": _sha256_text(content)
                            if isinstance(content, str)
                            else None,
                            "normalized_sha256": _sha256_text(_normalise_code(content))
                            if isinstance(content, str)
                            else None,
                            "family_id": _family_id(
                                source["role"],
                                content,
                                family_hint,
                            )
                            if isinstance(content, str)
                            else None,
                            "family_hint": family_hint,
                            "provenance": evidence,
                            "discovery_release_sha256": discovery["release_sha256"],
                            "rejection_reasons": _row_reasons(
                                record,
                                source,
                                content,
                                benchmark_signals,
                            ),
                            "status": "pending_policy",
                            "training_authorized": False,
                        }
                    )
        except Exception as exc:
            source_result["status"] = "failed"
            source_result["failure"] = {
                "code": str(exc).split(":", 1)[0],
                "error_type": type(exc).__name__,
                "detail": str(exc),
            }
        source_results.append(source_result)
    curated = _apply_row_policy(
        rows,
        threshold=config.near_duplicate_threshold,
        forbidden_documents=_forbidden_documents(bundle, jaxbench_root),
    )
    per_source = {row["dataset_id"]: row for row in source_results}
    for row in curated:
        per_source[row["dataset_id"]]["counts"][row["status"]] += 1
    _write_jsonl(out_dir / "source_ranking.jsonl", ranked)
    _write_jsonl(out_dir / "source_results.jsonl", source_results)
    _write_jsonl(out_dir / "row_candidates.jsonl", curated)
    artifacts = {
        relative: _sha256_file(out_dir / relative)
        for relative in (
            "source_ranking.jsonl",
            "source_results.jsonl",
            "row_candidates.jsonl",
        )
    }
    status_counts = Counter(row["status"] for row in curated)
    role_counts = Counter(
        (row["role"], row["status"]) for row in curated
    )
    reason_counts = Counter(
        reason.split(":", 1)[0]
        for row in curated
        for reason in row["rejection_reasons"]
    )
    manifest = {
        "schema_version": HUB_CURATION_ARTIFACT_SCHEMA_VERSION,
        "kind": "pallas_hub_row_candidates",
        "status": "complete",
        "created_at": _utc_now(),
        "invocation": invocation,
        "invocation_sha256": invocation_sha,
        "policy": {
            "benchmark_policy": config.benchmark_policy,
            "near_duplicate_threshold": config.near_duplicate_threshold,
            "shingle_size": SHINGLE_SIZE,
            "jaxbench_scanned": True,
            "private_holdout_material_accessed": False,
            "private_holdout_clean_claim": False,
            "direct_training_authorized": False,
            "positive_pallas_sft_authorized": False,
            "third_party_benchmark_training_authorized": False,
        },
        "counts": {
            "configured_sources": len(config.sources),
            "source_failures": sum(row["status"] == "failed" for row in source_results),
            "gross_rows": len(curated),
            "status": dict(sorted(status_counts.items())),
            "roles": {
                f"{role}:{status}": count
                for (role, status), count in sorted(role_counts.items())
            },
            "rejection_reasons": dict(sorted(reason_counts.items())),
            "configured_roles": dict(
                sorted(Counter(source["role"] for source in config.sources).items())
            ),
            "source_yields": {
                row["dataset_id"]: row["counts"] for row in source_results
            },
        },
        "artifacts": artifacts,
    }
    release_payload = {
        key: value for key, value in manifest.items() if key not in {"created_at", "release_sha256"}
    }
    manifest["release_sha256"] = _canonical_sha256(release_payload)
    _write_json(out_dir / "manifest.json", manifest)
    validate_hub_row_release(out_dir)
    return manifest


def validate_hub_row_release(root: Path) -> dict[str, Any]:
    manifest = _load_json_object(root / "manifest.json")
    if (
        manifest.get("schema_version") != HUB_CURATION_ARTIFACT_SCHEMA_VERSION
        or manifest.get("kind") != "pallas_hub_row_candidates"
        or manifest.get("status") != "complete"
    ):
        raise HubCurationError("HUB_ROW_MANIFEST_INVALID")
    release_payload = {
        key: value for key, value in manifest.items() if key not in {"created_at", "release_sha256"}
    }
    if manifest.get("release_sha256") != _canonical_sha256(release_payload):
        raise HubCurationError("HUB_ROW_MANIFEST_HASH_MISMATCH")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "source_ranking.jsonl",
        "source_results.jsonl",
        "row_candidates.jsonl",
    }:
        raise HubCurationError("HUB_ROW_ARTIFACTS_INVALID")
    for relative, expected in artifacts.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise HubCurationError(f"HUB_ROW_ARTIFACT_MISSING: {relative}")
        if _sha256_file(path) != expected:
            raise HubCurationError(f"HUB_ROW_ARTIFACT_HASH_MISMATCH: {relative}")
    previous_id: str | None = None
    counts: Counter[str] = Counter()
    for row in _iter_jsonl(root / "row_candidates.jsonl"):
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or (
            previous_id is not None and candidate_id <= previous_id
        ):
            raise HubCurationError("HUB_ROW_ORDER_INVALID")
        previous_id = candidate_id
        if row.get("evidence_level") != "row_candidate":
            raise HubCurationError(f"HUB_ROW_EVIDENCE_LEVEL_INVALID: {candidate_id}")
        if row.get("training_authorized") is not False:
            raise HubCurationError(f"HUB_ROW_TRAINING_ROLE_LEAKAGE: {candidate_id}")
        if row.get("role") not in VALID_ROLES or row.get("objective") not in VALID_OBJECTIVES:
            raise HubCurationError(f"HUB_ROW_OBJECTIVE_ROLE_INVALID: {candidate_id}")
        status = row.get("status")
        reasons = row.get("rejection_reasons")
        if status not in {"curated_candidate", "rejected"} or not isinstance(reasons, list):
            raise HubCurationError(f"HUB_ROW_STATUS_INVALID: {candidate_id}")
        if (status == "curated_candidate") == bool(reasons):
            raise HubCurationError(f"HUB_ROW_REJECTION_INVALID: {candidate_id}")
        if row.get("role") == "pallas_code" and status == "curated_candidate":
            raise HubCurationError(f"HUB_ROW_PALLAS_PROMOTION_INVALID: {candidate_id}")
        if any(reason.startswith("BENCHMARK_CONTAMINATION") for reason in reasons) and status != "rejected":
            raise HubCurationError(f"HUB_ROW_BENCHMARK_LEAKAGE: {candidate_id}")
        content = row.get("content")
        if isinstance(content, str) and row.get("content_sha256") != _sha256_text(content):
            raise HubCurationError(f"HUB_ROW_CONTENT_HASH_MISMATCH: {candidate_id}")
        counts[status] += 1
    if dict(sorted(counts.items())) != manifest.get("counts", {}).get("status"):
        raise HubCurationError("HUB_ROW_COUNT_MISMATCH")
    if manifest.get("policy", {}).get("direct_training_authorized") is not False:
        raise HubCurationError("HUB_ROW_MANIFEST_TRAINING_ROLE_LEAKAGE")
    return {
        "ok": True,
        "release_sha256": manifest["release_sha256"],
        "invocation_sha256": manifest["invocation_sha256"],
        "counts": manifest["counts"],
    }
