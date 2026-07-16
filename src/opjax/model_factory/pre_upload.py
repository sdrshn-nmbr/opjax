"""Hard pre-upload gate: rights + scrub + canary + spend ledger check."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from opjax.model_factory.canary import (
    CanarySet,
    embed_canaries,
    find_canaries,
    load_canary_set,
    make_canary_set,
    save_canary_set,
)
from opjax.model_factory.scrub import scrub_text


@dataclass
class PreUploadResult:
    ok: bool
    provider: str
    reasons: list[str] = field(default_factory=list)
    scrub_hits: int = 0
    canary_path: str | None = None
    scrubbed_path: str | None = None
    manifest_path: str | None = None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "provider": self.provider,
            "reasons": self.reasons,
            "scrub_hits": self.scrub_hits,
            "canary_path": self.canary_path,
            "scrubbed_path": self.scrubbed_path,
            "manifest_path": self.manifest_path,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _provider_approved(manifest: dict, provider: str) -> bool:
    providers = manifest.get("providers", {})
    entry = providers.get(provider)
    if not entry:
        return False
    return bool(entry.get("upload_approved")) and bool(entry.get("retention_reviewed"))


def _slice_approved(manifest: dict, slice_id: str) -> bool:
    for item in manifest.get("data_slices", []):
        if item.get("id") == slice_id:
            return bool(item.get("rights_cleared")) and bool(item.get("secrets_scrubbed"))
    return False


def run_pre_upload_gate(
    *,
    source_path: Path | str,
    provider: str,
    rights_manifest_path: Path | str,
    slice_id: str,
    output_dir: Path | str,
    spend_ledger_path: Path | str | None = None,
    require_spend_headroom: bool = True,
) -> PreUploadResult:
    """Block managed-trainer upload unless governance artifacts pass.

    On success, writes a scrubbed copy with embedded canaries under ``output_dir``.
    """
    source_path = Path(source_path)
    rights_manifest_path = Path(rights_manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reasons: list[str] = []
    if not source_path.is_file():
        return PreUploadResult(
            ok=False, provider=provider, reasons=[f"missing source: {source_path}"]
        )
    if not rights_manifest_path.is_file():
        return PreUploadResult(
            ok=False,
            provider=provider,
            reasons=[f"missing rights manifest: {rights_manifest_path}"],
            manifest_path=str(rights_manifest_path),
        )

    manifest = _load_json(rights_manifest_path)
    if not _provider_approved(manifest, provider):
        reasons.append(
            f"provider '{provider}' not approved "
            "(need providers.<name>.upload_approved and retention_reviewed)"
        )
    if not _slice_approved(manifest, slice_id):
        reasons.append(
            f"data slice '{slice_id}' not cleared "
            "(need rights_cleared and secrets_scrubbed)"
        )

    if require_spend_headroom and spend_ledger_path is not None:
        ledger_path = Path(spend_ledger_path)
        if not ledger_path.is_file():
            reasons.append(f"missing spend ledger: {ledger_path}")
        else:
            ledger = _load_json(ledger_path)
            remaining = float(ledger.get("usd_remaining", 0))
            if remaining <= 0:
                reasons.append("spend ledger has no USD headroom (usd_remaining <= 0)")

    raw = source_path.read_text(encoding="utf-8", errors="replace")
    scrubbed = scrub_text(raw)
    if scrubbed.hits:
        # Hits mean secrets were present — we still emit scrubbed text, but gate fails
        # until a human re-runs after confirming redactions are acceptable.
        reasons.append(
            f"scrub found {len(scrubbed.hits)} secret-like hit(s); "
            "review redactions and re-mark secrets_scrubbed on the slice"
        )

    canaries = make_canary_set(3, seed_material=f"{provider}:{slice_id}")
    canary_path = output_dir / f"canaries-{slice_id}.json"
    save_canary_set(canary_path, canaries)
    payload = embed_canaries(scrubbed.text, canaries)
    # Ensure canaries themselves aren't stripped by a second scrub pass in CI:
    # verify they remain findable in the outgoing payload.
    missing = [c.id for c in canaries.canaries if c not in find_canaries(payload, canaries)]
    if missing:
        reasons.append(f"canary embed failed for: {missing}")

    scrubbed_path = output_dir / f"{source_path.stem}.scrubbed{source_path.suffix}"
    scrubbed_path.write_text(payload, encoding="utf-8")

    ok = not reasons
    return PreUploadResult(
        ok=ok,
        provider=provider,
        reasons=reasons,
        scrub_hits=len(scrubbed.hits),
        canary_path=str(canary_path),
        scrubbed_path=str(scrubbed_path),
        manifest_path=str(rights_manifest_path),
    )


def verify_canaries_absent_in_public_text(
    text: str, canary_set_path: Path | str
) -> list[str]:
    """Return IDs of canaries that leaked into arbitrary text (e.g. model samples)."""
    canary_set = load_canary_set(canary_set_path)
    return [c.id for c in find_canaries(text, canary_set)]
