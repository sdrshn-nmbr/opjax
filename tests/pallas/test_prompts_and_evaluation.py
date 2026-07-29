from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from opjax.pallas.evaluation import (
    EvaluationError,
    SampleCandidate,
    _assert_tpu_runtime,
    _evaluate_workload,
    _load_or_create_manifest,
    _oracle_summary,
    _parse_json_output,
    _result_compiled,
    validate_sample_run,
)
from opjax.pallas.contracts import load_contracts
from opjax.pallas.prompts import extract_code, render_prompt, spec_only
from opjax.pallas.scoring import PromptContext

BASELINE = """\
import jax.numpy as jnp

CONFIG = {"size": 8}

def create_inputs():
    return (jnp.ones((CONFIG["size"],)),)

def workload(x):
    \"\"\"Square each element.\"\"\"
    return jnp.square(x)
"""
CONFIG_ROOT = Path(__file__).parents[2] / "config" / "pallas"


def test_spec_prompt_withholds_reference_body() -> None:
    specification = spec_only(BASELINE)
    prompt = render_prompt(
        workload="square",
        baseline_source=BASELINE,
        prompt_context="spec",
    )

    assert "return jnp.square(x)" not in specification
    assert "return jnp.square(x)" not in prompt
    assert "CONFIG" in prompt
    assert "create_inputs" in prompt
    assert "def workload(x)" in prompt


def test_baseline_prompt_is_explicitly_diagnostic() -> None:
    prompt = render_prompt(
        workload="square",
        baseline_source=BASELINE,
        prompt_context="baseline",
    )

    assert "diagnostic context only" in prompt
    assert "return jnp.square(x)" in prompt


def test_extractor_prefers_complete_parseable_workload() -> None:
    completion = """analysis
```python
def workload(x):
    return x
```
"""

    assert extract_code(completion) == "def workload(x):\n    return x\n"


def test_resume_requires_identical_fingerprint(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    original = {"contract": "a", "kernels": {"x": "1"}}
    _load_or_create_manifest(out_dir=out_dir, fingerprint=original, resume=False)

    resumed = _load_or_create_manifest(
        out_dir=out_dir,
        fingerprint=original,
        resume=True,
    )
    assert resumed["fingerprint"] == original

    with pytest.raises(EvaluationError, match="RESUME_FINGERPRINT_MISMATCH"):
        _load_or_create_manifest(
            out_dir=out_dir,
            fingerprint={"contract": "b"},
            resume=True,
        )


def test_existing_run_requires_explicit_resume(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    _load_or_create_manifest(out_dir=out_dir, fingerprint={"x": 1}, resume=False)

    with pytest.raises(EvaluationError, match="RUN_ALREADY_EXISTS"):
        _load_or_create_manifest(
            out_dir=out_dir,
            fingerprint={"x": 1},
            resume=False,
        )


def test_jaxbench_json_parser_accepts_log_prefix() -> None:
    payload = {"workload": "square", "status": "correct"}
    output = f"compiler log\n{json.dumps(payload)}\n"

    assert _parse_json_output(output) == payload


def test_runtime_hardware_must_match_declared_tpu_generation() -> None:
    _assert_tpu_runtime(
        {"platforms": ["tpu"], "device_kinds": ["TPU v5 lite"]},
        {"hardware": "v5e"},
    )

    with pytest.raises(EvaluationError, match="HARDWARE_TARGET_MISMATCH"):
        _assert_tpu_runtime(
            {"platforms": ["cpu"], "device_kinds": ["Apple M3"]},
            {"hardware": "v5e"},
        )

    with pytest.raises(EvaluationError, match="HARDWARE_TARGET_MISMATCH"):
        _assert_tpu_runtime(
            {"platforms": ["tpu"], "device_kinds": ["TPU v4"]},
            {"hardware": "v5e"},
        )


def test_evaluation_binds_kernel_to_completed_sample_manifest(tmp_path: Path) -> None:
    bundle = load_contracts(CONFIG_ROOT)
    sample_run = tmp_path / "sample"
    kernels = sample_run / "kernels" / "seed-0"
    kernels.mkdir(parents=True)
    source = "def workload(x):\n    return x\n"
    kernel = kernels / "1p_Flash_Attention.py"
    kernel.write_text(source, encoding="utf-8")
    code_sha256 = hashlib.sha256(source.encode()).hexdigest()
    fingerprint = {
        "sha256": "a" * 64,
        "contract_sha256": bundle.sha256,
        "jaxbench_revision": next(
            source["revision"]
            for source in bundle.sources["sources"]
            if source["id"] == "jaxbench"
        ),
        "arm": "A",
        "prompt_context": "spec",
        "model_path": None,
        "request": {
            "sample_ids": ["1p_Flash_Attention::seed=0"],
            "workloads": ["1p_Flash_Attention"],
            "seeds": [0],
        },
    }
    (sample_run / "manifest.json").write_text(
        json.dumps({"status": "sampled", "fingerprint": fingerprint}),
        encoding="utf-8",
    )
    (sample_run / "samples.jsonl").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "sample_id": "1p_Flash_Attention::seed=0",
                "workload": "1p_Flash_Attention",
                "seed": 0,
                "kernel_path": "kernels/seed-0/1p_Flash_Attention.py",
                "code_sha256": code_sha256,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    observed = validate_sample_run(
        bundle=bundle,
        sample_run=sample_run,
        model_id="thinkingmachines/Inkling",
        arm="A",
        prompt_context=PromptContext.SPEC,
    )

    assert observed.fingerprint_sha256 == "a" * 64
    assert observed.candidates[0].sample_id == "1p_Flash_Attention::seed=0"

    kernel.write_text(source + "# changed\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="SAMPLE_KERNEL_HASH_MISMATCH"):
        validate_sample_run(
            bundle=bundle,
            sample_run=sample_run,
            model_id="thinkingmachines/Inkling",
            arm="A",
            prompt_context=PromptContext.SPEC,
        )


def test_compilation_is_separate_from_correctness() -> None:
    assert _result_compiled({"status": "correct"}) is True
    assert _result_compiled({"status": "incorrect"}) is True
    assert _result_compiled({"status": "compile_error"}) is False
    assert _result_compiled({"status": "runtime_error"}) is False


def test_oracle_summary_quantifies_seed_variation(tmp_path: Path) -> None:
    bundle = load_contracts(CONFIG_ROOT)
    source = "def workload(x):\n    return x\n"
    candidates = tuple(
        SampleCandidate(
            sample_id=f"1p_Flash_Attention::seed={seed}",
            workload="1p_Flash_Attention",
            seed=seed,
            kernel=tmp_path / f"{seed}.py",
            sample={
                "status": "sampled",
                "inspection": {"authentic": seed != 1},
                "attempts": [{"attempt": 0}],
            },
        )
        for seed in (0, 1, 2)
    )
    rows = [
        {
            "sample_id": candidate.sample_id,
            "compiled": candidate.seed != 1,
            "correct": candidate.seed == 0,
            "pallas_credited": candidate.seed == 0,
            "headline_credited": False,
            "timing": {"stable": candidate.seed == 0},
            "speedup": 0.9 if candidate.seed == 0 else None,
        }
        for candidate in candidates
    ]

    summary = _oracle_summary(
        bundle=bundle,
        candidates=candidates,
        rows=rows,
    )

    assert summary["n_samples"] == 3
    assert summary["parse_rate"] == 1.0
    assert summary["compilation_rate"] == round(2 / 3, 6)
    assert summary["correctness_rate"] == round(1 / 3, 6)
    assert summary["seed_rate_ranges"]["correctness_rate"] == 1.0
    assert summary["seed_consistency"]["n_workloads_with_any_correct"] == 1
    assert summary["seed_consistency"]["n_workloads_with_all_seeds_correct"] == 0


def test_static_rejection_does_not_launch_jaxbench(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = load_contracts(CONFIG_ROOT)
    kernel = tmp_path / "candidate.py"
    kernel.write_text("this is not python", encoding="utf-8")
    baseline_dir = (
        tmp_path
        / "jaxbench"
        / "JAXBench"
        / "benchmark"
        / "1p_Flash_Attention"
    )
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "baseline.py").write_text(BASELINE, encoding="utf-8")
    candidate = SampleCandidate(
        sample_id="1p_Flash_Attention::seed=0",
        workload="1p_Flash_Attention",
        seed=0,
        kernel=kernel,
        sample={},
    )

    def fail_if_executed(**_: object) -> dict[str, object]:
        raise AssertionError("JAXBench must not execute a static rejection")

    monkeypatch.setattr(
        "opjax.pallas.evaluation._run_jaxbench_once",
        fail_if_executed,
    )

    result = _evaluate_workload(
        bundle=bundle,
        jaxbench_root=tmp_path / "jaxbench",
        candidate=candidate,
        prompt_context=PromptContext.SPEC,
        timeout_seconds=1,
    )

    assert result["execution_status"] == "STATIC_REJECTED"
    assert result["raw_runs"] == []
    assert result["compiled"] is False
    assert result["correct"] is False
