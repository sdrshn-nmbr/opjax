"""Fail-closed DAPT admission for curated Hub kernel rows."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from opjax.pallas.corpus import validate_corpus_release
from opjax.pallas.hub_curation import validate_hub_row_release

ADMISSION_SCHEMA_VERSION = 1
ORACLE_SCHEMA_VERSION = 1
SHINGLE_SIZE = 5


class HubAdmissionError(RuntimeError):
    """A Hub DAPT admission artifact violates its contract."""


@dataclass(frozen=True)
class AdmissionConfig:
    accepted_hub_row_release_sha256: str
    accepted_base_corpus_release_sha256: str
    minimum_lexical_tokens: int
    near_duplicate_threshold: float
    validation_source_fraction: float
    required_kernel_signals: tuple[str, ...]
    sha256: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_text(_canonical_json(value))


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _repository_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise HubAdmissionError(
            f"HUB_ADMISSION_REPOSITORY_REVISION_UNAVAILABLE: {path}: {exc.output.strip()}"
        ) from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise HubAdmissionError(f"HUB_ADMISSION_JSON_INVALID: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise HubAdmissionError(f"HUB_ADMISSION_OBJECT_REQUIRED: {path}")
    return value


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise HubAdmissionError(f"HUB_ADMISSION_ARTIFACT_MISSING: {path}") from exc
    for line_number, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HubAdmissionError(
                f"HUB_ADMISSION_JSONL_INVALID: {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise HubAdmissionError(f"HUB_ADMISSION_ROW_INVALID: {path}:{line_number}")
        yield value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _tokens(source: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\s]", source))


def _normalise(source: str) -> str:
    try:
        return ast.unparse(ast.parse(source))
    except SyntaxError:
        return " ".join(source.split())


def _shingle_hashes(source: str) -> frozenset[str]:
    tokens = _tokens(_normalise(source))
    if not tokens:
        return frozenset()
    shingles = (
        (" ".join(tokens),)
        if len(tokens) < SHINGLE_SIZE
        else tuple(
            " ".join(tokens[index : index + SHINGLE_SIZE])
            for index in range(len(tokens) - SHINGLE_SIZE + 1)
        )
    )
    return frozenset(_sha256_text(shingle) for shingle in shingles)


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def load_admission_config(path: Path) -> AdmissionConfig:
    value = _read_json(path)
    if value.get("schema_version") != ADMISSION_SCHEMA_VERSION:
        raise HubAdmissionError("HUB_ADMISSION_CONFIG_SCHEMA_UNSUPPORTED")
    hub_sha = value.get("accepted_hub_row_release_sha256")
    corpus_sha = value.get("accepted_base_corpus_release_sha256")
    minimum_tokens = value.get("minimum_lexical_tokens")
    threshold = value.get("near_duplicate_threshold")
    validation_fraction = value.get("validation_source_fraction")
    signals = value.get("required_kernel_signals")
    if not _valid_sha256(hub_sha) or not _valid_sha256(corpus_sha):
        raise HubAdmissionError("HUB_ADMISSION_RELEASE_PIN_INVALID")
    if (
        not isinstance(minimum_tokens, int)
        or isinstance(minimum_tokens, bool)
        or minimum_tokens <= 0
    ):
        raise HubAdmissionError("HUB_ADMISSION_TOKEN_THRESHOLD_INVALID")
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not 0 < threshold <= 1
    ):
        raise HubAdmissionError("HUB_ADMISSION_DEDUP_THRESHOLD_INVALID")
    if (
        not isinstance(validation_fraction, (int, float))
        or isinstance(validation_fraction, bool)
        or not 0 <= validation_fraction < 1
    ):
        raise HubAdmissionError("HUB_ADMISSION_SPLIT_FRACTION_INVALID")
    if (
        not isinstance(signals, list)
        or not signals
        or any(not isinstance(signal, str) or not signal for signal in signals)
    ):
        raise HubAdmissionError("HUB_ADMISSION_KERNEL_SIGNALS_INVALID")
    return AdmissionConfig(
        accepted_hub_row_release_sha256=hub_sha,
        accepted_base_corpus_release_sha256=corpus_sha,
        minimum_lexical_tokens=minimum_tokens,
        near_duplicate_threshold=float(threshold),
        validation_source_fraction=float(validation_fraction),
        required_kernel_signals=tuple(signals),
        sha256=_canonical_sha256(value),
    )


def _load_oracle(path: Path | None) -> tuple[str | None, tuple[dict[str, Any], ...]]:
    if path is None:
        return None, ()
    value = _read_json(path)
    documents = value.get("documents")
    expected_sha = value.get("oracle_sha256")
    payload = {key: item for key, item in value.items() if key != "oracle_sha256"}
    if (
        value.get("schema_version") != ORACLE_SCHEMA_VERSION
        or not isinstance(documents, list)
        or not documents
        or expected_sha != _canonical_sha256(payload)
    ):
        raise HubAdmissionError("HUB_ADMISSION_PRIVATE_ORACLE_INVALID")
    parsed = []
    for document in documents:
        if not isinstance(document, dict):
            raise HubAdmissionError("HUB_ADMISSION_PRIVATE_ORACLE_DOCUMENT_INVALID")
        shingles = document.get("shingle_sha256")
        if (
            not isinstance(document.get("document_id"), str)
            or not _valid_sha256(document.get("content_sha256"))
            or not _valid_sha256(document.get("normalized_sha256"))
            or not isinstance(shingles, list)
            or any(not _valid_sha256(shingle) for shingle in shingles)
        ):
            raise HubAdmissionError("HUB_ADMISSION_PRIVATE_ORACLE_DOCUMENT_INVALID")
        parsed.append({**document, "shingle_sha256": frozenset(shingles)})
    return expected_sha, tuple(parsed)


def _split_for_repository(repository: str, validation_fraction: float) -> str:
    bucket = int(_sha256_text(repository)[:8], 16) / 0xFFFFFFFF
    return "validation" if bucket < validation_fraction else "train"


def _base_documents(base_corpus_root: Path) -> tuple[dict[str, Any], ...]:
    documents = []
    for row in _iter_jsonl(base_corpus_root / "datasets" / "dapt.jsonl"):
        content = row.get("text")
        if not isinstance(content, str):
            raise HubAdmissionError("HUB_ADMISSION_BASE_DAPT_CONTENT_INVALID")
        documents.append(
            {
                "row_id": row.get("row_id"),
                "content_sha256": _sha256_text(content),
                "normalized_sha256": _sha256_text(_normalise(content)),
                "shingles": _shingle_hashes(content),
            }
        )
    return tuple(documents)


def _public_rejection_reasons(
    row: Mapping[str, Any],
    config: AdmissionConfig,
    base_documents: Sequence[Mapping[str, Any]],
) -> list[str]:
    reasons: set[str] = set()
    content = row.get("content")
    if row.get("status") != "curated_candidate" or row.get("rejection_reasons"):
        reasons.add("UPSTREAM_NOT_CURATED")
    if (
        row.get("role") != "cross_kernel_code"
        or row.get("objective") != "dapt_candidate"
    ):
        reasons.add("OBJECTIVE_ROLE_INVALID")
    if row.get("training_authorized") is not False:
        reasons.add("UPSTREAM_AUTHORIZATION_LEAKAGE")
    if not isinstance(content, str):
        return sorted({*reasons, "CONTENT_MISSING"})
    try:
        ast.parse(content)
    except SyntaxError:
        reasons.add("SYNTAX_INVALID")
    if len(_tokens(content)) < config.minimum_lexical_tokens:
        reasons.add("CONTENT_BELOW_TOKEN_THRESHOLD")
    if not any(signal in content for signal in config.required_kernel_signals):
        reasons.add("KERNEL_STRUCTURE_MISSING")
    content_sha = _sha256_text(content)
    normalized_sha = _sha256_text(_normalise(content))
    shingles = _shingle_hashes(content)
    for document in base_documents:
        if content_sha == document["content_sha256"]:
            reasons.add(f"BASE_DAPT_EXACT_DUPLICATE:{document['row_id']}")
            break
        if normalized_sha == document["normalized_sha256"]:
            reasons.add(f"BASE_DAPT_NORMALIZED_DUPLICATE:{document['row_id']}")
            break
        if _jaccard(shingles, document["shingles"]) >= config.near_duplicate_threshold:
            reasons.add(f"BASE_DAPT_NEAR_DUPLICATE:{document['row_id']}")
            break
    return sorted(reasons)


def _private_rejection_reasons(
    content: str,
    documents: Sequence[Mapping[str, Any]],
    threshold: float,
) -> list[str]:
    content_sha = _sha256_text(content)
    normalized_sha = _sha256_text(_normalise(content))
    shingles = _shingle_hashes(content)
    for document in documents:
        document_id = document["document_id"]
        if content_sha == document["content_sha256"]:
            return [f"PRIVATE_HOLDOUT_EXACT_MATCH:{document_id}"]
        if normalized_sha == document["normalized_sha256"]:
            return [f"PRIVATE_HOLDOUT_NORMALIZED_MATCH:{document_id}"]
        if _jaccard(shingles, document["shingle_sha256"]) >= threshold:
            return [f"PRIVATE_HOLDOUT_NEAR_MATCH:{document_id}"]
    return []


def build_hub_dapt_admission(
    *,
    repo_root: Path,
    row_root: Path,
    base_corpus_root: Path,
    config_path: Path,
    out_dir: Path,
    private_holdout_oracle: Path | None = None,
) -> dict[str, Any]:
    config = load_admission_config(config_path)
    row_release = validate_hub_row_release(row_root)
    corpus_release = validate_corpus_release(base_corpus_root)
    if row_release["release_sha256"] != config.accepted_hub_row_release_sha256:
        raise HubAdmissionError("HUB_ADMISSION_ROW_RELEASE_MISMATCH")
    if corpus_release["release_sha256"] != config.accepted_base_corpus_release_sha256:
        raise HubAdmissionError("HUB_ADMISSION_BASE_CORPUS_RELEASE_MISMATCH")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise HubAdmissionError(f"HUB_ADMISSION_OUTPUT_NOT_EMPTY: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    oracle_sha, private_documents = _load_oracle(private_holdout_oracle)
    base_documents = _base_documents(base_corpus_root)
    evaluated_rows = []
    for row in _iter_jsonl(row_root / "row_candidates.jsonl"):
        if row.get("status") != "curated_candidate":
            continue
        repository = row.get("provenance", {}).get("repo_name")
        if not isinstance(repository, str) or not repository:
            raise HubAdmissionError(
                f"HUB_ADMISSION_REPOSITORY_MISSING: {row.get('candidate_id')}"
            )
        public_reasons = _public_rejection_reasons(row, config, base_documents)
        evaluated_rows.append((row, repository, public_reasons))
    source_counts = Counter(
        repository
        for _, repository, public_reasons in evaluated_rows
        if not public_reasons
    )
    admissions = []
    dapt_rows = []
    for row, repository, public_reasons in evaluated_rows:
        reasons = list(public_reasons)
        publicly_admissible = not reasons
        if publicly_admissible and private_documents:
            reasons.extend(
                _private_rejection_reasons(
                    row["content"],
                    private_documents,
                    config.near_duplicate_threshold,
                )
            )
        training_authorized = (
            publicly_admissible and bool(private_documents) and not reasons
        )
        status = (
            "authorized"
            if training_authorized
            else "private_holdout_required"
            if publicly_admissible and not private_documents
            else "rejected"
        )
        split = _split_for_repository(repository, config.validation_source_fraction)
        admission = {
            "schema_version": ADMISSION_SCHEMA_VERSION,
            "candidate_id": row["candidate_id"],
            "content_sha256": row["content_sha256"],
            "repository": repository,
            "family_id": row["family_id"],
            "split": split,
            "lexical_tokens": len(_tokens(row["content"])),
            "publicly_admissible": publicly_admissible,
            "private_holdout_checked": bool(private_documents),
            "training_authorized": training_authorized,
            "status": status,
            "rejection_reasons": sorted(set(reasons)),
            "sampling_weight": (
                1.0 / source_counts[repository] if publicly_admissible else 0.0
            ),
        }
        admissions.append(admission)
        if training_authorized:
            dapt_rows.append(
                {
                    "schema_version": ADMISSION_SCHEMA_VERSION,
                    "row_id": row["candidate_id"],
                    "objective": "dapt",
                    "text": row["content"],
                    "family_id": row["family_id"],
                    "split": split,
                    "sampling_weight": admission["sampling_weight"],
                    "provenance": {
                        "dataset_id": row["dataset_id"],
                        "source_revision": row["source_revision"],
                        "source_path": row["source_path"],
                        "source_row_id": row["source_row_id"],
                        "source_file_sha256": row["source_file_sha256"],
                        "repository": repository,
                        "license": row["license"],
                        "row_license": row["row_license"],
                        "content_sha256": row["content_sha256"],
                    },
                }
            )
    admissions.sort(key=lambda row: row["candidate_id"])
    dapt_rows.sort(key=lambda row: row["row_id"])
    _write_jsonl(out_dir / "admission.jsonl", admissions)
    _write_jsonl(out_dir / "datasets" / "dapt.jsonl", dapt_rows)
    artifacts = {
        relative: _sha256_file(out_dir / relative)
        for relative in ("admission.jsonl", "datasets/dapt.jsonl")
    }
    status_counts = Counter(row["status"] for row in admissions)
    reason_counts = Counter(
        reason.split(":", 1)[0]
        for row in admissions
        for reason in row["rejection_reasons"]
    )
    base_count = len(base_documents)
    authorized_count = len(dapt_rows)
    public_count = sum(row["publicly_admissible"] for row in admissions)
    manifest = {
        "schema_version": ADMISSION_SCHEMA_VERSION,
        "kind": "pallas_hub_dapt_admission",
        "status": "complete",
        "created_at": _utc_now(),
        "repository_revision": _repository_revision(repo_root),
        "config_sha256": config.sha256,
        "hub_row_release_sha256": row_release["release_sha256"],
        "base_corpus_release_sha256": corpus_release["release_sha256"],
        "private_holdout_oracle_sha256": oracle_sha,
        "training_authorized": bool(dapt_rows),
        "counts": {
            "base_dapt": base_count,
            "reviewed": len(admissions),
            "publicly_admissible": public_count,
            "authorized_dapt": authorized_count,
            "status": dict(sorted(status_counts.items())),
            "rejection_reasons": dict(sorted(reason_counts.items())),
            "combined_authorized_dapt": base_count + authorized_count,
        },
        "increase": {
            "public_candidate_percent": public_count / base_count * 100
            if base_count
            else None,
            "verified_train_ready_percent": authorized_count / base_count * 100
            if base_count
            else None,
        },
        "policy": {
            "minimum_lexical_tokens": config.minimum_lexical_tokens,
            "near_duplicate_threshold": config.near_duplicate_threshold,
            "validation_source_fraction": config.validation_source_fraction,
            "repository_balanced_sampling": True,
            "private_holdout_required": True,
        },
        "artifacts": artifacts,
    }
    payload = {
        key: value
        for key, value in manifest.items()
        if key not in {"created_at", "release_sha256"}
    }
    manifest["release_sha256"] = _canonical_sha256(payload)
    _write_json(out_dir / "manifest.json", manifest)
    validate_hub_dapt_admission(out_dir)
    return manifest


def validate_hub_dapt_admission(root: Path) -> dict[str, Any]:
    manifest = _read_json(root / "manifest.json")
    if (
        manifest.get("schema_version") != ADMISSION_SCHEMA_VERSION
        or manifest.get("kind") != "pallas_hub_dapt_admission"
        or manifest.get("status") != "complete"
    ):
        raise HubAdmissionError("HUB_ADMISSION_MANIFEST_INVALID")
    payload = {
        key: value
        for key, value in manifest.items()
        if key not in {"created_at", "release_sha256"}
    }
    if manifest.get("release_sha256") != _canonical_sha256(payload):
        raise HubAdmissionError("HUB_ADMISSION_MANIFEST_HASH_MISMATCH")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "admission.jsonl",
        "datasets/dapt.jsonl",
    }:
        raise HubAdmissionError("HUB_ADMISSION_ARTIFACTS_INVALID")
    for relative, expected in artifacts.items():
        path = (root / relative).resolve()
        if (
            not path.is_relative_to(root.resolve())
            or not path.is_file()
            or _sha256_file(path) != expected
        ):
            raise HubAdmissionError(f"HUB_ADMISSION_ARTIFACT_HASH_MISMATCH: {relative}")
    admissions = list(_iter_jsonl(root / "admission.jsonl"))
    dapt_rows = list(_iter_jsonl(root / "datasets" / "dapt.jsonl"))
    if [row.get("candidate_id") for row in admissions] != sorted(
        row.get("candidate_id") for row in admissions
    ):
        raise HubAdmissionError("HUB_ADMISSION_ORDER_INVALID")
    repositories = Counter(
        row.get("repository")
        for row in admissions
        if row.get("publicly_admissible") is True
    )
    admission_by_id: dict[str, dict[str, Any]] = {}
    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    public_count = 0
    for row in admissions:
        candidate_id = row.get("candidate_id")
        reasons = row.get("rejection_reasons")
        status = row.get("status")
        repository = row.get("repository")
        if (
            not isinstance(candidate_id, str)
            or candidate_id in admission_by_id
            or not isinstance(reasons, list)
            or status not in {"authorized", "private_holdout_required", "rejected"}
            or row.get("split") not in {"train", "validation"}
            or not isinstance(repository, str)
            or not repository
            or not isinstance(row.get("lexical_tokens"), int)
            or row["lexical_tokens"] <= 0
            or not _valid_sha256(row.get("content_sha256"))
            or row.get("sampling_weight")
            != (
                1.0 / repositories[repository]
                if row.get("publicly_admissible") is True
                else 0.0
            )
        ):
            raise HubAdmissionError(f"HUB_ADMISSION_ROW_INVALID: {candidate_id}")
        publicly_admissible = row.get("publicly_admissible") is True
        private_checked = row.get("private_holdout_checked") is True
        authorized = row.get("training_authorized") is True
        valid_state = (
            status == "authorized"
            and publicly_admissible
            and private_checked
            and authorized
            and not reasons
            or status == "private_holdout_required"
            and publicly_admissible
            and not private_checked
            and not authorized
            and not reasons
            or status == "rejected"
            and not authorized
            and bool(reasons)
        )
        if not valid_state:
            raise HubAdmissionError(
                f"HUB_ADMISSION_AUTHORIZATION_INVALID: {candidate_id}"
            )
        admission_by_id[candidate_id] = row
        status_counts[status] += 1
        reason_counts.update(reason.split(":", 1)[0] for reason in reasons)
        public_count += publicly_admissible
    authorized_ids = {
        candidate_id
        for candidate_id, row in admission_by_id.items()
        if row["training_authorized"] is True
    }
    dapt_ids = {row.get("row_id") for row in dapt_rows}
    if authorized_ids != dapt_ids or len(dapt_ids) != len(dapt_rows):
        raise HubAdmissionError("HUB_ADMISSION_AUTHORIZED_DATASET_MISMATCH")
    for row in dapt_rows:
        admission = admission_by_id[row["row_id"]]
        content = row.get("text")
        if (
            not isinstance(content, str)
            or _sha256_text(content) != admission["content_sha256"]
            or row.get("objective") != "dapt"
            or row.get("family_id") != admission["family_id"]
            or row.get("split") != admission["split"]
            or row.get("sampling_weight") != admission["sampling_weight"]
        ):
            raise HubAdmissionError(
                f"HUB_ADMISSION_DAPT_ROW_INVALID: {row.get('row_id')}"
            )
    counts = manifest.get("counts", {})
    if counts.get("status") != dict(sorted(status_counts.items())):
        raise HubAdmissionError("HUB_ADMISSION_STATUS_COUNT_MISMATCH")
    if counts.get("rejection_reasons") != dict(sorted(reason_counts.items())):
        raise HubAdmissionError("HUB_ADMISSION_REASON_COUNT_MISMATCH")
    if (
        counts.get("reviewed") != len(admissions)
        or counts.get("publicly_admissible") != public_count
        or counts.get("authorized_dapt") != len(dapt_rows)
        or counts.get("combined_authorized_dapt")
        != counts.get("base_dapt", 0) + len(dapt_rows)
    ):
        raise HubAdmissionError("HUB_ADMISSION_COUNT_MISMATCH")
    increase = manifest.get("increase", {})
    base_count = counts.get("base_dapt")
    if (
        not isinstance(base_count, int)
        or base_count <= 0
        or increase.get("public_candidate_percent") != public_count / base_count * 100
        or increase.get("verified_train_ready_percent")
        != len(dapt_rows) / base_count * 100
    ):
        raise HubAdmissionError("HUB_ADMISSION_INCREASE_INVALID")
    if manifest.get("training_authorized") is not bool(dapt_rows):
        raise HubAdmissionError("HUB_ADMISSION_MANIFEST_AUTHORIZATION_INVALID")
    return {
        "ok": True,
        "release_sha256": manifest["release_sha256"],
        "counts": counts,
        "increase": manifest["increase"],
    }
