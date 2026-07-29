from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from opjax.pallas.evaluation import (
    EvaluationError,
    _assert_tpu_runtime,
    _load_or_create_manifest,
    _parse_json_output,
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
    kernels = sample_run / "kernels"
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
    }
    (sample_run / "manifest.json").write_text(
        json.dumps({"status": "sampled", "fingerprint": fingerprint}),
        encoding="utf-8",
    )
    (sample_run / "samples.jsonl").write_text(
        json.dumps(
            {
                "workload": "1p_Flash_Attention",
                "code_sha256": code_sha256,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    observed = validate_sample_run(
        bundle=bundle,
        sample_run=sample_run,
        kernels=[kernel],
        model_id="thinkingmachines/Inkling",
        arm="A",
        prompt_context=PromptContext.SPEC,
    )

    assert observed == "a" * 64

    kernel.write_text(source + "# changed\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="SAMPLE_KERNEL_HASH_MISMATCH"):
        validate_sample_run(
            bundle=bundle,
            sample_run=sample_run,
            kernels=[kernel],
            model_id="thinkingmachines/Inkling",
            arm="A",
            prompt_context=PromptContext.SPEC,
        )
