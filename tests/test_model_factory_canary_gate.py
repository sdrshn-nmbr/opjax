import json
from pathlib import Path

from opjax.model_factory.canary import find_canaries, load_canary_set
from opjax.model_factory.pre_upload import run_pre_upload_gate


def test_pre_upload_gate_fails_closed_on_example_manifest(tmp_path: Path):
    source = tmp_path / "slice.jsonl"
    source.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
                "outcome": "ok",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = Path("docs/model-factory/00-governance/rights-manifest.example.json")
    ledger = Path("docs/model-factory/00-governance/spend-ledger.example.json")
    out = tmp_path / "out"

    result = run_pre_upload_gate(
        source_path=source,
        provider="tinker",
        rights_manifest_path=manifest,
        slice_id="example-public-oss",
        output_dir=out,
        spend_ledger_path=ledger,
    )
    assert result.ok is False
    assert any("not approved" in r for r in result.reasons)
    assert result.scrubbed_path is not None
    canaries = load_canary_set(result.canary_path)
    payload = Path(result.scrubbed_path).read_text(encoding="utf-8")
    assert len(find_canaries(payload, canaries)) == 3


def test_pre_upload_gate_passes_when_approved(tmp_path: Path):
    source = tmp_path / "slice.jsonl"
    source.write_text('{"messages":[{"role":"user","content":"x"},{"role":"assistant","content":"y"}]}\n')
    manifest_path = tmp_path / "rights.json"
    manifest_path.write_text(
        json.dumps(
            {
                "providers": {
                    "tinker": {"upload_approved": True, "retention_reviewed": True}
                },
                "data_slices": [
                    {
                        "id": "ok-slice",
                        "rights_cleared": True,
                        "secrets_scrubbed": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"usd_remaining": 100}), encoding="utf-8")

    result = run_pre_upload_gate(
        source_path=source,
        provider="tinker",
        rights_manifest_path=manifest_path,
        slice_id="ok-slice",
        output_dir=tmp_path / "out",
        spend_ledger_path=ledger,
    )
    assert result.ok is True
    assert result.reasons == []
    # Scrubbed JSONL must stay linewise-parseable for Tinker SFT loaders.
    payload = Path(result.scrubbed_path).read_text(encoding="utf-8")
    rows = [json.loads(ln) for ln in payload.splitlines() if ln.strip()]
    assert len(rows) >= 2  # original + canary rows
    canaries = load_canary_set(result.canary_path)
    assert len(find_canaries(payload, canaries)) == 3
