"""Deterministic Hugging Face dataset discovery for kernel-domain sources."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

HUB_DISCOVERY_CONFIG_SCHEMA_VERSION = 1
HUB_DISCOVERY_ARTIFACT_SCHEMA_VERSION = 3
HUB_DISCOVERY_GENERATOR_VERSION = 3
KERNEL_SIGNAL_FAMILIES = frozenset(
    {"pallas", "triton", "cuda", "cutlass", "cute", "benchmark"}
)
MAX_RATE_LIMIT_RETRIES = 20


class HubDiscoveryError(RuntimeError):
    """Hub discovery inputs or artifacts violate the evidence contract."""


@dataclass(frozen=True)
class Signal:
    id: str
    family: str
    pattern: str
    weight: int


@dataclass(frozen=True)
class HubDiscoveryConfig:
    candidate_threshold: int
    detail_threshold: int
    max_text_files_per_dataset: int
    max_text_file_bytes: int
    text_suffixes: frozenset[str]
    signals: tuple[Signal, ...]
    sha256: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(encoded)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HubDiscoveryError(f"HUB_ARTIFACT_MISSING: {path}") from exc
    except json.JSONDecodeError as exc:
        raise HubDiscoveryError(f"HUB_JSON_INVALID: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise HubDiscoveryError(f"HUB_JSON_NOT_OBJECT: {path}")
    return value


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    try:
        handle = path.open(encoding="utf-8")
    except FileNotFoundError as exc:
        raise HubDiscoveryError(f"HUB_ARTIFACT_MISSING: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HubDiscoveryError(
                    f"HUB_JSON_INVALID: {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise HubDiscoveryError(f"HUB_ROW_NOT_OBJECT: {path}:{line_number}")
            yield value


def _require_int(value: Any, *, name: str, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise HubDiscoveryError(f"HUB_CONFIG_INVALID: {name}={value!r}")
    return value


def load_hub_discovery_config(path: Path) -> HubDiscoveryConfig:
    value = _read_json(path)
    if value.get("schema_version") != HUB_DISCOVERY_CONFIG_SCHEMA_VERSION:
        raise HubDiscoveryError(f"HUB_CONFIG_SCHEMA_UNSUPPORTED: {path}")
    raw_signals = value.get("signals")
    if not isinstance(raw_signals, list) or not raw_signals:
        raise HubDiscoveryError("HUB_CONFIG_INVALID: signals")
    signals: list[Signal] = []
    ids: set[str] = set()
    for raw in raw_signals:
        if not isinstance(raw, dict):
            raise HubDiscoveryError(f"HUB_CONFIG_INVALID: signal={raw!r}")
        signal_id = raw.get("id")
        family = raw.get("family")
        pattern = raw.get("pattern")
        weight = raw.get("weight")
        if (
            not isinstance(signal_id, str)
            or not signal_id
            or signal_id in ids
            or not isinstance(family, str)
            or not family
            or not isinstance(pattern, str)
            or not pattern
            or not isinstance(weight, int)
            or isinstance(weight, bool)
            or weight <= 0
        ):
            raise HubDiscoveryError(f"HUB_CONFIG_INVALID: signal={raw!r}")
        ids.add(signal_id)
        signals.append(
            Signal(
                id=signal_id,
                family=family,
                pattern=pattern.casefold(),
                weight=weight,
            )
        )
    suffixes = value.get("text_suffixes")
    if (
        not isinstance(suffixes, list)
        or not suffixes
        or not all(isinstance(item, str) and item.startswith(".") for item in suffixes)
    ):
        raise HubDiscoveryError("HUB_CONFIG_INVALID: text_suffixes")
    return HubDiscoveryConfig(
        candidate_threshold=_require_int(
            value.get("candidate_threshold"),
            name="candidate_threshold",
            minimum=1,
        ),
        detail_threshold=_require_int(
            value.get("detail_threshold"),
            name="detail_threshold",
            minimum=1,
        ),
        max_text_files_per_dataset=_require_int(
            value.get("max_text_files_per_dataset"),
            name="max_text_files_per_dataset",
            minimum=0,
        ),
        max_text_file_bytes=_require_int(
            value.get("max_text_file_bytes"),
            name="max_text_file_bytes",
            minimum=1,
        ),
        text_suffixes=frozenset(item.casefold() for item in suffixes),
        signals=tuple(signals),
        sha256=_canonical_sha256(value),
    )


def _repository_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise HubDiscoveryError(
            f"REPOSITORY_REVISION_UNAVAILABLE: {path}: {exc.output.strip()}"
        ) from exc


def _valid_revision(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _rate_limit_delay(error: HfHubHTTPError) -> float | None:
    response = error.response
    if response is None or response.status_code != 429:
        return None
    retry_after = response.headers.get("Retry-After")
    if retry_after and retry_after.isdigit():
        return min(60.0, float(retry_after) + 1.0)
    rate_limit = response.headers.get("RateLimit", "")
    match = re.search(r"(?:^|;)t=(\d+)(?:;|$)", rate_limit)
    if match:
        return min(60.0, float(match.group(1)) + 1.0)
    return 30.0


def _call_with_rate_limit_retry(call: Callable[[], Any]) -> Any:
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return call()
        except HfHubHTTPError as exc:
            delay = _rate_limit_delay(exc)
            if delay is None:
                raise
            if attempt == MAX_RATE_LIMIT_RETRIES:
                raise HubDiscoveryError("HUB_RATE_LIMIT_RETRIES_EXHAUSTED") from exc
            time.sleep(delay)
    raise AssertionError("unreachable")


def _description(info: Any) -> str:
    value = getattr(info, "description", None)
    return value if isinstance(value, str) else ""


def _tags(info: Any) -> list[str]:
    value = getattr(info, "tags", None)
    if not isinstance(value, list):
        return []
    return sorted(item for item in value if isinstance(item, str))


def _license_from_tags(tags: Sequence[str]) -> str:
    licenses = sorted(
        {tag.removeprefix("license:") for tag in tags if tag.startswith("license:")}
    )
    if len(licenses) == 1:
        return licenses[0]
    if not licenses:
        return "unverified"
    return "multiple:" + ",".join(licenses)


def _signal_matches(
    texts: Mapping[str, str],
    signals: Sequence[Signal],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    matches: list[dict[str, Any]] = []
    family_scores: Counter[str] = Counter()
    for signal in signals:
        locations = sorted(
            location
            for location, text in texts.items()
            if signal.pattern in text.casefold()
        )
        if not locations:
            continue
        family_scores[signal.family] += signal.weight
        matches.append(
            {
                "signal_id": signal.id,
                "family": signal.family,
                "weight": signal.weight,
                "locations": locations,
            }
        )
    return sorted(matches, key=lambda row: row["signal_id"]), dict(
        sorted(family_scores.items())
    )


def _basic_inventory_row(info: Any, config: HubDiscoveryConfig) -> dict[str, Any]:
    repo_id = getattr(info, "id", None)
    if not isinstance(repo_id, str) or not repo_id:
        raise HubDiscoveryError(f"HUB_DATASET_ID_INVALID: {repo_id!r}")
    tags = _tags(info)
    description = _description(info)
    matches, family_scores = _signal_matches(
        {
            "dataset_id": repo_id,
            "description": description,
            "tags": "\n".join(tags),
        },
        config.signals,
    )
    revision = getattr(info, "sha", None)
    risk_flags = []
    if not _valid_revision(revision):
        risk_flags.append("SOURCE_REVISION_UNPINNED")
    license_id = _license_from_tags(tags)
    if license_id == "unverified" or license_id.startswith("multiple:"):
        risk_flags.append("SOURCE_LICENSE_UNVERIFIED")
    for attribute, code in (
        ("private", "SOURCE_PRIVATE"),
        ("gated", "SOURCE_GATED"),
        ("disabled", "SOURCE_DISABLED"),
    ):
        if getattr(info, attribute, False):
            risk_flags.append(code)
    return {
        "schema_version": HUB_DISCOVERY_ARTIFACT_SCHEMA_VERSION,
        "dataset_id": repo_id,
        "source_revision": revision,
        "license": license_id,
        "private": bool(getattr(info, "private", False)),
        "gated": getattr(info, "gated", False),
        "disabled": bool(getattr(info, "disabled", False)),
        "tags": tags,
        "description_sha256": _sha256_bytes(description.encode()),
        "metadata_matches": matches,
        "family_scores": family_scores,
        "score": sum(family_scores.values()),
        "risk_flags": sorted(set(risk_flags)),
        "dataset_card": {},
        "files": [],
        "content_inspections": [],
        "inspection_failures": [],
    }


def _sibling_row(sibling: Any) -> dict[str, Any] | None:
    name = getattr(sibling, "rfilename", None)
    if not isinstance(name, str) or not name:
        return None
    size = getattr(sibling, "size", None)
    lfs = getattr(sibling, "lfs", None)
    return {
        "path": name,
        "size": size if isinstance(size, int) and not isinstance(size, bool) else None,
        "blob_id": getattr(sibling, "blob_id", None),
        "lfs_sha256": (
            lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        ),
    }


def _eligible_text_files(
    files: Sequence[dict[str, Any]],
    config: HubDiscoveryConfig,
) -> list[dict[str, Any]]:
    eligible = []
    for file in files:
        suffix = Path(file["path"]).suffix.casefold()
        size = file.get("size")
        if (
            suffix in config.text_suffixes
            and isinstance(size, int)
            and size <= config.max_text_file_bytes
        ):
            eligible.append(file)
    return sorted(
        eligible,
        key=lambda row: (
            0 if Path(row["path"]).name.casefold().startswith("readme") else 1,
            row["size"],
            row["path"],
        ),
    )[: config.max_text_files_per_dataset]


def _enrich_inventory_row(
    row: dict[str, Any],
    *,
    api: Any,
    config: HubDiscoveryConfig,
    downloader: Callable[..., str],
) -> dict[str, Any]:
    revision = row["source_revision"]
    if not _valid_revision(revision):
        return row
    try:
        detail = _call_with_rate_limit_retry(
            lambda: api.dataset_info(
                row["dataset_id"],
                revision=revision,
                files_metadata=True,
            )
        )
    except Exception as exc:
        row["inspection_failures"].append(
            {
                "stage": "dataset_info",
                "error_type": type(exc).__name__,
                "detail": str(exc),
            }
        )
        return row
    observed_revision = getattr(detail, "sha", None)
    if observed_revision != revision:
        row["risk_flags"] = sorted(
            set(row["risk_flags"]) | {"SOURCE_REVISION_MISMATCH"}
        )
        row["inspection_failures"].append(
            {
                "stage": "dataset_info",
                "error_type": "RevisionMismatch",
                "detail": f"expected={revision} observed={observed_revision}",
            }
        )
        return row
    files = [
        parsed
        for sibling in (getattr(detail, "siblings", None) or [])
        if (parsed := _sibling_row(sibling)) is not None
    ]
    row["files"] = sorted(files, key=lambda item: item["path"])
    card_data = getattr(detail, "card_data", None)
    if card_data is not None and callable(getattr(card_data, "to_dict", None)):
        row["dataset_card"] = card_data.to_dict()
    card_license = row["dataset_card"].get("license")
    if isinstance(card_license, str):
        card_licenses = [card_license]
    elif isinstance(card_license, list):
        card_licenses = sorted(
            item for item in card_license if isinstance(item, str) and item
        )
    else:
        card_licenses = []
    if len(card_licenses) == 1 and row["license"] == "unverified":
        row["license"] = card_licenses[0]
        row["risk_flags"] = [
            flag for flag in row["risk_flags"] if flag != "SOURCE_LICENSE_UNVERIFIED"
        ]
    elif len(card_licenses) == 1 and row["license"] != card_licenses[0]:
        row["risk_flags"] = sorted(set(row["risk_flags"]) | {"SOURCE_LICENSE_CONFLICT"})
    content_texts: dict[str, str] = {}
    for file in _eligible_text_files(row["files"], config):
        try:
            local_path = Path(
                _call_with_rate_limit_retry(
                    lambda: downloader(
                        repo_id=row["dataset_id"],
                        filename=file["path"],
                        repo_type="dataset",
                        revision=revision,
                    )
                )
            )
            content = local_path.read_text(encoding="utf-8")
        except Exception as exc:
            row["inspection_failures"].append(
                {
                    "stage": "text_file",
                    "path": file["path"],
                    "error_type": type(exc).__name__,
                    "detail": str(exc),
                }
            )
            continue
        location = f"file:{file['path']}"
        content_texts[location] = content
        matches, family_scores = _signal_matches({location: content}, config.signals)
        row["content_inspections"].append(
            {
                "path": file["path"],
                "bytes": len(content.encode()),
                "content_sha256": _sha256_bytes(content.encode()),
                "matches": matches,
                "family_scores": family_scores,
            }
        )
    file_texts = {f"filename:{file['path']}": file["path"] for file in row["files"]}
    card_texts = (
        {"dataset_card": json.dumps(row["dataset_card"], sort_keys=True)}
        if row["dataset_card"]
        else {}
    )
    combined_matches, combined_scores = _signal_matches(
        {**card_texts, **file_texts, **content_texts},
        config.signals,
    )
    row["content_matches"] = combined_matches
    all_scores = Counter(row["family_scores"])
    all_scores.update(combined_scores)
    row["family_scores"] = dict(sorted(all_scores.items()))
    row["score"] = sum(all_scores.values())
    return row


def _classification(row: dict[str, Any], threshold: int) -> dict[str, Any]:
    scores = row["family_scores"]
    domain_families = {"pallas", "triton", "cuda", "cutlass", "cute"}
    has_domain = any(scores.get(family, 0) > 0 for family in domain_families)
    tags = {str(tag).casefold() for tag in row.get("tags", [])}
    card = row.get("dataset_card")
    if isinstance(card, dict):
        tags.update(str(tag).casefold() for tag in card.get("tags", []))
    dataset_id = str(row.get("dataset_id", "")).casefold()
    benchmark = (
        scores.get("benchmark", 0) > 0
        or bool(
            tags
            & {
                "benchmark",
                "benchmarks",
                "evaluation",
                "kernel-benchmark",
                "kernelbench",
                "pallasbench",
                "llm-evaluation",
            }
        )
        or any(
            token in dataset_id for token in ("kernelbench", "pallasbench", "jaxbench")
        )
        or re.search(
            r"(?:^|[-_/])(?:eval|evaluation|benchmark|bench|test)(?:$|[-_/])",
            dataset_id,
        )
        is not None
    )
    trace = scores.get("trace", 0) > 0
    if row["score"] < threshold or not has_domain:
        category = "irrelevant_or_ambiguous"
        candidate_objectives: list[str] = []
        training_policy = "rejected"
        risk_flags = row["risk_flags"]
    elif scores.get("pallas", 0) >= max(
        scores.get("triton", 0),
        scores.get("cuda", 0),
        scores.get("cutlass", 0),
        scores.get("cute", 0),
    ):
        category = "pallas_domain"
        candidate_objectives = ["dapt_candidate"]
        risk_flags = row["risk_flags"]
    elif trace:
        category = "kernel_agent_trace"
        candidate_objectives = ["dapt_candidate", "repair_candidate"]
        risk_flags = row["risk_flags"]
    else:
        category = "cross_kernel_domain"
        candidate_objectives = ["dapt_candidate"]
        risk_flags = row["risk_flags"]
    if category != "irrelevant_or_ambiguous":
        if benchmark and "repair_candidate" not in candidate_objectives:
            candidate_objectives.append("repair_candidate")
        training_policy = "forbidden" if benchmark else "discovery_only"
        if benchmark:
            risk_flags = sorted(set(risk_flags) | {"BENCHMARK_CONTAMINATION"})
    status = (
        "candidate"
        if training_policy in {"discovery_only", "forbidden"}
        else "rejected"
    )
    return {
        "schema_version": HUB_DISCOVERY_ARTIFACT_SCHEMA_VERSION,
        "evidence_level": "source_discovery",
        "direct_training_authorized": False,
        "dataset_id": row["dataset_id"],
        "source_revision": row["source_revision"],
        "score": row["score"],
        "family_scores": row["family_scores"],
        "category": category,
        "source_role": ("benchmark_or_evaluation" if benchmark else "domain_source"),
        "benchmark_or_evaluation": benchmark,
        "status": status,
        "training_policy": training_policy,
        "candidate_objectives": candidate_objectives,
        "risk_flags": sorted(set(risk_flags)),
        "decision_reasons": sorted(
            match["signal_id"]
            for match in row["metadata_matches"] + row.get("content_matches", [])
        ),
    }


def _detail_eligible(row: dict[str, Any], config: HubDiscoveryConfig) -> bool:
    return row["score"] >= config.detail_threshold and any(
        row["family_scores"].get(family, 0) > 0 for family in KERNEL_SIGNAL_FAMILIES
    )


def _iter_dataset_infos(
    api: Any,
    *,
    search_terms: Sequence[str],
    limit: int | None,
) -> Iterable[Any]:
    queries: Sequence[str | None] = search_terms or (None,)
    for query in queries:
        yield from api.list_datasets(
            search=query,
            limit=limit,
            expand=[
                "sha",
                "tags",
                "description",
                "gated",
                "private",
                "disabled",
            ],
        )


def _invocation_fingerprint(
    *,
    config: HubDiscoveryConfig,
    search_terms: Sequence[str],
    limit: int | None,
) -> str:
    return _canonical_sha256(
        {
            "generator_version": HUB_DISCOVERY_GENERATOR_VERSION,
            "config_sha256": config.sha256,
            "search_terms": sorted(set(search_terms)),
            "limit": limit,
        }
    )


def _open_state_database(
    path: Path,
    *,
    invocation_fingerprint: str,
) -> tuple[sqlite3.Connection, str]:
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS inventory ("
        "dataset_id TEXT PRIMARY KEY, row_json TEXT NOT NULL, "
        "detail_eligible INTEGER NOT NULL, enriched INTEGER NOT NULL)"
    )
    existing = connection.execute(
        "SELECT value FROM metadata WHERE key = 'invocation_fingerprint'"
    ).fetchone()
    if existing is not None and existing[0] != invocation_fingerprint:
        connection.close()
        raise HubDiscoveryError("HUB_RESUME_FINGERPRINT_MISMATCH")
    started = connection.execute(
        "SELECT value FROM metadata WHERE key = 'started_at'"
    ).fetchone()
    if started is None:
        started_at = _utc_now()
        connection.executemany(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            (
                ("invocation_fingerprint", invocation_fingerprint),
                ("started_at", started_at),
            ),
        )
        connection.commit()
    else:
        started_at = str(started[0])
    return connection, started_at


def _materialize_release(
    connection: sqlite3.Connection,
    *,
    out_dir: Path,
    candidate_threshold: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    filenames = (
        "source_inventory.jsonl",
        "candidates.jsonl",
        "decisions.jsonl",
    )
    temporary_paths = {
        filename: (out_dir / filename).with_suffix(".jsonl.tmp")
        for filename in filenames
    }
    category_counts: Counter[str] = Counter()
    inventory_count = 0
    candidate_count = 0
    handles = {
        filename: path.open("w", encoding="utf-8")
        for filename, path in temporary_paths.items()
    }
    try:
        for (row_json,) in connection.execute(
            "SELECT row_json FROM inventory ORDER BY dataset_id"
        ):
            row = json.loads(row_json)
            decision = _classification(row, candidate_threshold)
            handles["source_inventory.jsonl"].write(
                json.dumps(row, sort_keys=True) + "\n"
            )
            handles["decisions.jsonl"].write(
                json.dumps(decision, sort_keys=True) + "\n"
            )
            inventory_count += 1
            category_counts[decision["category"]] += 1
            if decision["status"] == "candidate":
                handles["candidates.jsonl"].write(
                    json.dumps(row, sort_keys=True) + "\n"
                )
                candidate_count += 1
    finally:
        for handle in handles.values():
            handle.close()
    for filename, temporary_path in temporary_paths.items():
        temporary_path.replace(out_dir / filename)
    counts = {
        "inventory": inventory_count,
        "candidates": candidate_count,
        "decisions": inventory_count,
        "inspection_failures": sum(
            len(json.loads(row_json)["inspection_failures"])
            for (row_json,) in connection.execute("SELECT row_json FROM inventory")
        ),
        "detail_eligible": int(
            connection.execute(
                "SELECT COUNT(*) FROM inventory WHERE detail_eligible = 1"
            ).fetchone()[0]
        ),
        "detail_enriched": int(
            connection.execute(
                "SELECT COUNT(*) FROM inventory WHERE enriched = 1"
            ).fetchone()[0]
        ),
    }
    counts["categories"] = dict(sorted(category_counts.items()))
    artifact_sha256 = {
        filename: _sha256_file(out_dir / filename) for filename in filenames
    }
    return counts, artifact_sha256


def discover_hub_datasets(
    *,
    repo_root: Path,
    config_path: Path,
    out_dir: Path,
    search_terms: Sequence[str] = (),
    limit: int | None = None,
    resume: bool = False,
    detail_workers: int = 4,
    api: Any | None = None,
    downloader: Callable[..., str] = hf_hub_download,
) -> dict[str, Any]:
    if limit is not None and (
        not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0
    ):
        raise HubDiscoveryError(f"HUB_LIMIT_INVALID: {limit!r}")
    if (
        not isinstance(detail_workers, int)
        or isinstance(detail_workers, bool)
        or detail_workers <= 0
    ):
        raise HubDiscoveryError(f"HUB_DETAIL_WORKERS_INVALID: {detail_workers!r}")
    config = load_hub_discovery_config(config_path)
    normalized_searches = tuple(sorted(set(search_terms)))
    fingerprint = _invocation_fingerprint(
        config=config,
        search_terms=normalized_searches,
        limit=limit,
    )
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        if not resume:
            raise HubDiscoveryError(f"HUB_OUTPUT_NOT_EMPTY: {out_dir}")
        completed = validate_hub_discovery_release(out_dir)
        if completed.get("invocation_fingerprint") != fingerprint:
            raise HubDiscoveryError("HUB_RESUME_FINGERPRINT_MISMATCH")
        return completed
    state_database_path = out_dir / ".hub-discovery.sqlite3"
    if out_dir.exists() and any(out_dir.iterdir()):
        if not resume or not state_database_path.exists():
            raise HubDiscoveryError(f"HUB_OUTPUT_NOT_EMPTY: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    connection, started_at = _open_state_database(
        state_database_path,
        invocation_fingerprint=fingerprint,
    )
    hub_api = api or HfApi()
    enumeration_marker = connection.execute(
        "SELECT value FROM metadata WHERE key = 'enumeration_complete'"
    ).fetchone()
    enumeration_complete = (
        enumeration_marker is not None and enumeration_marker[0] == "true"
    )
    if not enumeration_complete:
        enriched_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM inventory WHERE enriched = 1"
            ).fetchone()[0]
        )
        enumeration_complete = enriched_count > 0
        if enumeration_complete:
            connection.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ("enumeration_complete", "true"),
            )
            connection.commit()
    try:
        if not enumeration_complete:
            observed_count = int(
                connection.execute("SELECT COUNT(*) FROM inventory").fetchone()[0]
            )
            enumeration_attempt = 0
            while True:
                try:
                    infos = _iter_dataset_infos(
                        hub_api,
                        search_terms=normalized_searches,
                        limit=limit,
                    )
                    for info in infos:
                        row = _basic_inventory_row(info, config)
                        inserted = connection.execute(
                            "INSERT OR IGNORE INTO inventory "
                            "(dataset_id, row_json, detail_eligible, enriched) "
                            "VALUES (?, ?, ?, 0)",
                            (
                                row["dataset_id"],
                                json.dumps(row, sort_keys=True),
                                int(_detail_eligible(row, config)),
                            ),
                        )
                        if inserted.rowcount == 0:
                            continue
                        observed_count += 1
                        if observed_count % 500 == 0:
                            connection.commit()
                        if (
                            limit is not None
                            and not normalized_searches
                            and observed_count >= limit
                        ):
                            break
                except HfHubHTTPError as exc:
                    delay = _rate_limit_delay(exc)
                    if delay is None:
                        raise
                    enumeration_attempt += 1
                    if enumeration_attempt > MAX_RATE_LIMIT_RETRIES:
                        raise HubDiscoveryError(
                            "HUB_RATE_LIMIT_RETRIES_EXHAUSTED"
                        ) from exc
                    connection.commit()
                    time.sleep(delay)
                    continue
                break
            connection.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ("enumeration_complete", "true"),
            )
            connection.commit()
    except Exception as exc:
        connection.commit()
        if isinstance(exc, HubDiscoveryError):
            connection.close()
            raise
        connection.close()
        raise HubDiscoveryError(
            f"HUB_ENUMERATION_FAILED: {type(exc).__name__}: {exc}"
        ) from exc
    except BaseException:
        connection.commit()
        connection.close()
        raise
    try:
        pending = list(
            connection.execute(
                "SELECT dataset_id, row_json FROM inventory "
                "WHERE detail_eligible = 1 AND enriched = 0 ORDER BY dataset_id"
            )
        )

        def enrich(item: tuple[str, str]) -> tuple[str, dict[str, Any]]:
            repo_id, row_json = item
            row = json.loads(row_json)
            row = _enrich_inventory_row(
                row,
                api=hub_api,
                config=config,
                downloader=downloader,
            )
            return repo_id, row

        with ThreadPoolExecutor(max_workers=detail_workers) as executor:
            futures = [executor.submit(enrich, item) for item in pending]
            for index, future in enumerate(as_completed(futures), 1):
                repo_id, row = future.result()
                connection.execute(
                    "UPDATE inventory SET row_json = ?, enriched = 1 "
                    "WHERE dataset_id = ?",
                    (json.dumps(row, sort_keys=True), repo_id),
                )
                if index % 25 == 0:
                    connection.commit()
        connection.commit()
    except BaseException:
        connection.commit()
        connection.close()
        raise
    counts, artifact_sha256 = _materialize_release(
        connection,
        out_dir=out_dir,
        candidate_threshold=config.candidate_threshold,
    )
    manifest = {
        "schema_version": HUB_DISCOVERY_ARTIFACT_SCHEMA_VERSION,
        "kind": "pallas_hub_discovery",
        "status": "complete",
        "generator_version": HUB_DISCOVERY_GENERATOR_VERSION,
        "generator_sha256": _sha256_file(Path(__file__)),
        "opjax_revision": _repository_revision(repo_root),
        "config_sha256": config.sha256,
        "invocation_fingerprint": fingerprint,
        "search_terms": list(normalized_searches),
        "enumeration_scope": (
            "complete_registry"
            if not normalized_searches and limit is None
            else "bounded"
        ),
        "registry_snapshot_atomic": False,
        "source_revisions_pinned_individually": True,
        "limit": limit,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "counts": counts,
        "artifacts": artifact_sha256,
    }
    manifest["release_sha256"] = _canonical_sha256(
        {
            "generator_version": manifest["generator_version"],
            "generator_sha256": manifest["generator_sha256"],
            "opjax_revision": manifest["opjax_revision"],
            "config_sha256": manifest["config_sha256"],
            "invocation_fingerprint": manifest["invocation_fingerprint"],
            "registry_snapshot_atomic": manifest["registry_snapshot_atomic"],
            "source_revisions_pinned_individually": manifest[
                "source_revisions_pinned_individually"
            ],
            "counts": manifest["counts"],
            "artifacts": manifest["artifacts"],
        }
    )
    _write_json(manifest_path, manifest)
    connection.close()
    state_database_path.unlink(missing_ok=True)
    return manifest


def validate_hub_discovery_release(root: Path) -> dict[str, Any]:
    manifest = _read_json(root / "manifest.json")
    if (
        manifest.get("schema_version") != HUB_DISCOVERY_ARTIFACT_SCHEMA_VERSION
        or manifest.get("kind") != "pallas_hub_discovery"
        or manifest.get("status") != "complete"
    ):
        raise HubDiscoveryError("HUB_MANIFEST_INVALID")
    if (
        manifest.get("registry_snapshot_atomic") is not False
        or manifest.get("source_revisions_pinned_individually") is not True
    ):
        raise HubDiscoveryError("HUB_SNAPSHOT_BOUNDARY_INVALID")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "source_inventory.jsonl",
        "candidates.jsonl",
        "decisions.jsonl",
    }:
        raise HubDiscoveryError("HUB_MANIFEST_ARTIFACTS_INVALID")
    for filename, expected_sha256 in artifacts.items():
        observed_sha256 = _sha256_file(root / filename)
        if observed_sha256 != expected_sha256:
            raise HubDiscoveryError(f"HUB_ARTIFACT_HASH_MISMATCH: {filename}")
    inventory_count = 0
    candidate_count = 0
    inspection_failures = 0
    category_counts: Counter[str] = Counter()
    previous_id: str | None = None
    missing = object()
    candidates = iter(_iter_jsonl(root / "candidates.jsonl"))
    pairs = zip_longest(
        _iter_jsonl(root / "source_inventory.jsonl"),
        _iter_jsonl(root / "decisions.jsonl"),
        fillvalue=missing,
    )
    for inventory, decision in pairs:
        if inventory is missing or decision is missing:
            raise HubDiscoveryError("HUB_INVENTORY_DECISION_COUNT_MISMATCH")
        if not isinstance(inventory, dict) or not isinstance(decision, dict):
            raise HubDiscoveryError("HUB_ROW_NOT_OBJECT")
        dataset_id = inventory.get("dataset_id")
        if not isinstance(dataset_id, str) or (
            previous_id is not None and dataset_id <= previous_id
        ):
            raise HubDiscoveryError("HUB_INVENTORY_ORDER_INVALID")
        previous_id = dataset_id
        if decision.get("dataset_id") != dataset_id:
            raise HubDiscoveryError("HUB_INVENTORY_DECISION_MISMATCH")
        if decision.get("source_revision") != inventory.get("source_revision"):
            raise HubDiscoveryError("HUB_SOURCE_REVISION_MISMATCH")
        inventory_count += 1
        failures = inventory.get("inspection_failures")
        if not isinstance(failures, list):
            raise HubDiscoveryError("HUB_INSPECTION_FAILURES_INVALID")
        inspection_failures += len(failures)
        category = decision.get("category")
        if not isinstance(category, str):
            raise HubDiscoveryError("HUB_CATEGORY_INVALID")
        category_counts[category] += 1
        policy = decision.get("training_policy")
        objectives = decision.get("candidate_objectives")
        if decision.get("direct_training_authorized") is not False:
            raise HubDiscoveryError("HUB_DIRECT_TRAINING_AUTHORIZED")
        if decision.get("evidence_level") != "source_discovery":
            raise HubDiscoveryError("HUB_EVIDENCE_LEVEL_INVALID")
        if not isinstance(objectives, list) or any(
            objective not in {"dapt_candidate", "repair_candidate"}
            for objective in objectives
        ):
            raise HubDiscoveryError("HUB_OBJECTIVE_ROUTE_INVALID")
        if policy not in {"discovery_only", "forbidden", "rejected"}:
            raise HubDiscoveryError("HUB_TRAINING_POLICY_INVALID")
        benchmark = decision.get("benchmark_or_evaluation")
        source_role = decision.get("source_role")
        if not isinstance(benchmark, bool) or source_role not in {
            "benchmark_or_evaluation",
            "domain_source",
        }:
            raise HubDiscoveryError("HUB_SOURCE_ROLE_INVALID")
        if benchmark != (source_role == "benchmark_or_evaluation"):
            raise HubDiscoveryError("HUB_SOURCE_ROLE_MISMATCH")
        if policy == "forbidden" and (
            not benchmark
            or "BENCHMARK_CONTAMINATION" not in decision.get("risk_flags", [])
        ):
            raise HubDiscoveryError("HUB_BENCHMARK_POLICY_INVALID")
        if decision.get("status") == "candidate":
            if not _valid_revision(decision.get("source_revision")):
                raise HubDiscoveryError("HUB_CANDIDATE_REVISION_UNPINNED")
            candidate = next(candidates, missing)
            if candidate is missing or not isinstance(candidate, dict):
                raise HubDiscoveryError("HUB_CANDIDATE_DECISION_MISMATCH")
            if candidate.get("dataset_id") != dataset_id:
                raise HubDiscoveryError("HUB_CANDIDATE_DECISION_MISMATCH")
            if candidate.get("source_revision") != decision.get("source_revision"):
                raise HubDiscoveryError("HUB_SOURCE_REVISION_MISMATCH")
            candidate_count += 1
    if next(candidates, missing) is not missing:
        raise HubDiscoveryError("HUB_CANDIDATE_DECISION_MISMATCH")
    expected_counts = manifest.get("counts", {})
    if (
        expected_counts.get("inventory") != inventory_count
        or expected_counts.get("candidates") != candidate_count
        or expected_counts.get("decisions") != inventory_count
        or expected_counts.get("inspection_failures") != inspection_failures
        or expected_counts.get("categories") != dict(sorted(category_counts.items()))
        or expected_counts.get("detail_eligible")
        < expected_counts.get("detail_enriched", 0)
    ):
        raise HubDiscoveryError("HUB_MANIFEST_COUNT_MISMATCH")
    expected_release = _canonical_sha256(
        {
            "generator_version": manifest.get("generator_version"),
            "generator_sha256": manifest.get("generator_sha256"),
            "opjax_revision": manifest.get("opjax_revision"),
            "config_sha256": manifest.get("config_sha256"),
            "invocation_fingerprint": manifest.get("invocation_fingerprint"),
            "registry_snapshot_atomic": manifest.get("registry_snapshot_atomic"),
            "source_revisions_pinned_individually": manifest.get(
                "source_revisions_pinned_individually"
            ),
            "counts": manifest.get("counts"),
            "artifacts": manifest.get("artifacts"),
        }
    )
    if manifest.get("release_sha256") != expected_release:
        raise HubDiscoveryError("HUB_RELEASE_HASH_MISMATCH")
    return manifest
