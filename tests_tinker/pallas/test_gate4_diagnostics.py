from __future__ import annotations

import json
from pathlib import Path

from opjax.pallas.gate4_diagnostics import (
    _diagnostic_tasks,
    _load_json,
    _source_audit,
    audit_sample_run,
    audit_supervision,
)


REPO_ROOT = Path(__file__).parents[2]
CONFIG_ROOT = REPO_ROOT / "config" / "pallas"
CORPUS_ROOT = REPO_ROOT / "data" / "pallas" / "runs" / "g3-sft-ready-final"
DIAGNOSTIC = CONFIG_ROOT / "gate4-diagnostic.json"


def test_supervision_audit_proves_complete_correct_targets(tmp_path: Path) -> None:
    report = audit_supervision(
        config_root=CONFIG_ROOT,
        corpus_root=CORPUS_ROOT,
        repo_root=REPO_ROOT,
        output=tmp_path / "audit.json",
    )

    assert report["summary"] == {
        "rows": 32,
        "sequence_tokens": 9104,
        "supervised_tokens": 5554,
        "rows_truncated": 0,
        "rows_with_noncontiguous_supervision": 0,
        "rows_without_end_supervision": 0,
        "blockspec_calls": 41,
        "reversed_blockspec_calls": 0,
        "unknown_blockspec_order_calls": 0,
        "rows_with_placeholder_ellipsis": 0,
        "authentic_rows": 32,
        "prompts_requiring_workload_name": 0,
        "prompts_requiring_self_contained_module": 0,
        "prompts_forbidding_incomplete_kernel": 0,
    }


def test_ladder_has_four_replays_and_four_heldout_tasks() -> None:
    tasks = _diagnostic_tasks(
        diagnostic=_load_json(DIAGNOSTIC),
        corpus_root=CORPUS_ROOT,
    )

    assert [task["tier"] for task in tasks] == ["training_replay"] * 4 + [
        "near_heldout"
    ] * 4
    assert [task["operation"] for task in tasks] == [
        "add",
        "matmul",
        "rmsnorm",
        "row_sum",
    ] * 2


def test_source_audit_distinguishes_ref_slices_from_placeholders() -> None:
    correct = _source_audit(
        "from jax.experimental import pallas as pl\n"
        "def kernel(x_ref, o_ref):\n    o_ref[...] = x_ref[...]\n"
        "def workload(x):\n"
        "    spec = pl.BlockSpec((128,), lambda i: (i,))\n"
        "    return pl.pallas_call(kernel, out_shape=x, in_specs=(spec,), "
        "out_specs=spec)(x)\n"
    )
    reversed_source = _source_audit(
        "from jax.experimental import pallas as pl\n"
        "def kernel(x_ref, o_ref):\n    ...\n"
        "def workload(x):\n"
        "    spec = pl.BlockSpec(lambda i: (i,), (128,))\n"
        "    return pl.pallas_call(kernel, out_shape=x, in_specs=(spec,), "
        "out_specs=spec)(x)\n"
    )

    assert correct["has_placeholder_ellipsis"] is False
    assert correct["reversed_blockspec_calls"] == 0
    assert reversed_source["has_placeholder_ellipsis"] is True
    assert reversed_source["reversed_blockspec_calls"] == 1


def test_sample_audit_separates_code_from_hidden_workload_contract(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        '{"status":"sampled","fingerprint":{"sha256":"abc"}}',
        encoding="utf-8",
    )
    completion = (
        "```python\n"
        "from jax.experimental import pallas as pl\n"
        "def add_kernel(x_ref, y_ref, o_ref):\n"
        "    o_ref[...] = x_ref[...] + y_ref[...]\n"
        "def add(x, y):\n"
        "    spec = pl.BlockSpec((128,), lambda i: (i,))\n"
        "    return pl.pallas_call(add_kernel, out_shape=x, "
        "in_specs=(spec, spec), out_specs=spec)(x, y)\n"
        "```"
    )
    (run_dir / "samples.jsonl").write_text(
        '{"task":{"task_id":"x","tier":"training_replay"},'
        '"n_tokens":100,"stop_reason":"stop","completion":'
        + json.dumps(completion)
        + "}\n",
        encoding="utf-8",
    )

    report = audit_sample_run(
        run_dir=run_dir,
        output=tmp_path / "audit.json",
    )

    assert report["overall"]["analysis_code_present"] == 1
    assert report["overall"]["workload_contract_met"] == 0
    assert report["overall"]["blockspec_valid"] == 1
    assert report["overall"]["complete_candidate"] == 0
