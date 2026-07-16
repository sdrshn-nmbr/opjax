from __future__ import annotations

from pathlib import Path

import pytest

from opjax.factory.axport_ingest import load_trajectories
from opjax.factory.normalize import NormalizeConfig, normalize_trajectories
from opjax.factory.preflight import preflight
from opjax.factory.render_tinker import render_conversations_jsonl, validate_conversations_jsonl
from opjax.factory.rights import check_provider
from opjax.factory.scrub import write_canaries

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_coding_sessions.jsonl"


def test_normalize_keeps_recovery_and_filters_allowlist():
    trajs = load_trajectories(FIXTURE)
    result = normalize_trajectories(
        trajs,
        NormalizeConfig(project_allowlist={"demo-repo"}, min_turns=2),
    )
    assert result.dropped >= 1  # other-repo dropped
    outcomes = {t.trajectory_id: t.outcome for t in result.kept}
    assert outcomes.get("syn-002") == "recovery"
    assert all(t.project == "demo-repo" for t in result.kept)


def test_render_and_validate_jsonl(tmp_path: Path):
    trajs = load_trajectories(FIXTURE)
    norm = normalize_trajectories(trajs, NormalizeConfig(min_turns=2))
    out = tmp_path / "train.jsonl"
    stats = render_conversations_jsonl(norm.kept, out, scrub=True)
    assert stats["num_conversations"] == len(norm.kept)
    assert stats["scrub_substitutions"] >= 1  # dirty syn-021
    meta = validate_conversations_jsonl(out)
    assert meta["num_conversations"] == stats["num_conversations"]
    text = out.read_text(encoding="utf-8")
    assert "tinker_test_secret_value_123456" not in text
    assert "EXAMPLESECRETKEYVALUE" not in text


def test_rights_fail_closed_without_approve(tmp_path: Path):
    manifest = tmp_path / "rights.md"
    manifest.write_text(
        """
| Slice ID | `YYYYMMDD-demo` |
| Owner | | | REJECT |
| Intended provider(s) | tinker |
""",
        encoding="utf-8",
    )
    decision = check_provider(manifest, "tinker")
    assert decision.approved is False


def test_rights_approve_with_provider(tmp_path: Path):
    manifest = tmp_path / "rights.md"
    manifest.write_text(
        """
| Slice ID | `20260716-smoke1` |
| Intended provider(s) | tinker |
| Owner | agent | 2026-07-16 | APPROVE |
""",
        encoding="utf-8",
    )
    decision = check_provider(manifest, "tinker")
    assert decision.approved is True
    assert decision.provider_ok is True


def test_preflight_canary_and_public_fixture(tmp_path: Path):
    trajs = load_trajectories(FIXTURE)
    out = tmp_path / "smoke" / "conversations.jsonl"
    # put under data/factory/smoke-like path marker via allow_public_fixture
    render_conversations_jsonl(trajs[:5], out, scrub=True)

    canary_path = tmp_path / "canaries.txt"
    cans = write_canaries(canary_path, n=2)
    dirty = tmp_path / "dirty.jsonl"
    dirty.write_text(
        out.read_text(encoding="utf-8") + f'\n{{"messages":[{{"role":"user","content":"{cans[0]}"}}]}}\n',
        encoding="utf-8",
    )

    bad = preflight(
        dirty,
        provider="tinker",
        canary_file=canary_path,
        allow_public_fixture=True,
    )
    assert bad.ok is False
    assert any("canary" in e for e in bad.errors)

    good = preflight(
        out,
        provider="tinker",
        canary_file=canary_path,
        allow_public_fixture=True,
    )
    assert good.ok is True


def test_preflight_requires_manifest_for_private(tmp_path: Path):
    out = tmp_path / "private.jsonl"
    out.write_text(
        '{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"yo"}]}\n',
        encoding="utf-8",
    )
    result = preflight(out, provider="tinker", allow_public_fixture=False)
    assert result.ok is False
    assert any("manifest" in e for e in result.errors)
