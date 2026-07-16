"""Hard pre-upload gate: scrub canaries + rights + sealed path markers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from opjax.factory.rights import check_provider, is_public_fixture_path
from opjax.factory.scrub import find_canaries_in_file, load_canaries, sha256_file


@dataclass
class PreflightResult:
    ok: bool
    errors: list[str]
    warnings: list[str]
    dataset_sha256: str | None
    details: dict


def preflight(
    dataset: str | Path,
    *,
    provider: str = "tinker",
    manifest: str | Path | None = None,
    canary_file: str | Path | None = None,
    allow_public_fixture: bool = False,
) -> PreflightResult:
    dataset = Path(dataset)
    errors: list[str] = []
    warnings: list[str] = []
    details: dict = {}

    if not dataset.exists():
        return PreflightResult(
            ok=False,
            errors=[f"dataset missing: {dataset}"],
            warnings=[],
            dataset_sha256=None,
            details={},
        )

    digest = sha256_file(dataset)
    details["sha256"] = digest

    text = dataset.read_text(encoding="utf-8")
    for marker in ("/splits/sealed/", "deepswe-report", "sudarshanbench-sealed"):
        if marker in text:
            errors.append(f"forbidden marker present in dataset text: {marker}")

    canaries: list[str] = []
    if canary_file:
        canaries = load_canaries(canary_file)
        details["canary_count"] = len(canaries)
        hits = find_canaries_in_file(dataset, canaries)
        details["canary_hits"] = hits
        if hits:
            errors.append(f"canary leak: {len(hits)} hit(s)")

    public = allow_public_fixture or is_public_fixture_path(dataset)
    details["public_fixture"] = public

    if public and allow_public_fixture:
        warnings.append("public fixture mode — rights manifest not required")
    else:
        if manifest is None:
            errors.append("manifest required for non-public upload")
        else:
            decision = check_provider(manifest, provider)
            details["rights"] = {
                "approved": decision.approved,
                "provider_ok": decision.provider_ok,
                "slice_id": decision.slice_id,
                "reasons": decision.reasons,
            }
            if not decision.approved:
                errors.extend(decision.reasons or ["rights check failed"])

    return PreflightResult(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        dataset_sha256=digest,
        details=details,
    )
