from __future__ import annotations

from opjax.pallas.scoring import (
    PromptContext,
    diagnostic_reward,
    inspect_pallas_source,
    judge,
    summarise,
    timing_evidence,
)

BASELINE = """\
import jax.numpy as jnp

def workload(x):
    return jnp.square(x)
"""

AUTHENTIC = """\
import jax
from jax.experimental import pallas as pl

def _kernel(x_ref, out_ref):
    out_ref[...] = x_ref[...] * x_ref[...]

def _launch(x):
    shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    return pl.pallas_call(_kernel, out_shape=shape)(x)

def workload(x):
    return _launch(x)
"""

DEAD_CODE = """\
import jax.numpy as jnp
from jax.experimental import pallas as pl

def _unused(x):
    return pl.pallas_call(lambda x_ref, o_ref: None, out_shape=x)(x)

def workload(x):
    return jnp.square(x)
"""

FALLBACK = """\
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def workload(x, use_pallas=True):
    if use_pallas:
        return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype))(x)
    return jnp.square(x)
"""


def _judge(
    source: str,
    *,
    compiled: bool = True,
    correct: bool = True,
    context: str = "spec",
    speedup: float | None = None,
    stable: bool | None = None,
):
    return judge(
        workload="square",
        candidate_src=source,
        baseline_src=BASELINE,
        compiled=compiled,
        correct=correct,
        prompt_context=context,
        speedup=speedup,
        timing_stable=stable,
    )


def test_reachable_helper_pallas_is_authentic() -> None:
    inspection = inspect_pallas_source(AUTHENTIC)

    assert inspection.authentic is True
    assert inspection.reachable_pallas_calls == 1
    assert inspection.reachable_functions == ("_kernel", "_launch", "workload")


def test_dead_code_pallas_is_rejected() -> None:
    inspection = inspect_pallas_source(DEAD_CODE)

    assert inspection.authentic is False
    assert inspection.reachable_pallas_calls == 0
    assert inspection.unreachable_pallas_calls == 1
    assert "PALLAS_PATH_UNREACHABLE" in inspection.reasons


def test_plain_jax_fallback_is_rejected() -> None:
    inspection = inspect_pallas_source(FALLBACK)

    assert inspection.authentic is False
    assert inspection.has_plain_jax_fallback is True
    assert "PLAIN_JAX_FALLBACK" in inspection.reasons


def test_reference_visible_context_is_never_scorable() -> None:
    verdict = _judge(AUTHENTIC, context="baseline", speedup=2, stable=True)

    assert verdict.scorable is False
    assert verdict.credited is False
    assert verdict.headline_credited is False
    assert diagnostic_reward(verdict) == 0
    assert "DIAGNOSTIC_PROMPT_CONTEXT" in verdict.no_credit_reasons


def test_correct_slow_pallas_is_not_headline_success() -> None:
    verdict = _judge(AUTHENTIC, speedup=0.9, stable=True)

    assert verdict.pallas_credited is True
    assert verdict.headline_credited is False


def test_correct_fast_stable_pallas_is_headline_success() -> None:
    verdict = _judge(AUTHENTIC, speedup=1.2, stable=True)

    assert verdict.pallas_credited is True
    assert verdict.headline_credited is True


def test_incorrect_fast_kernel_gets_no_credit() -> None:
    verdict = _judge(AUTHENTIC, correct=False, speedup=5, stable=True)

    assert verdict.credited is False
    assert verdict.pallas_credited is False
    assert verdict.headline_credited is False
    assert diagnostic_reward(verdict) == 0


def test_unstable_fast_kernel_is_not_headline_success() -> None:
    verdict = _judge(AUTHENTIC, speedup=2, stable=False)

    assert verdict.pallas_credited is True
    assert verdict.headline_credited is False
    assert "TIMING_UNSTABLE" in verdict.no_credit_reasons


def test_timing_requires_repeated_stable_measurements() -> None:
    insufficient = timing_evidence([1.0, 1.01], min_runs=3, max_coefficient_of_variation=0.1)
    stable = timing_evidence([1.0, 1.01, 0.99], min_runs=3, max_coefficient_of_variation=0.1)
    unstable = timing_evidence([1.0, 2.0, 4.0], min_runs=3, max_coefficient_of_variation=0.1)

    assert insufficient.stable is False
    assert stable.stable is True
    assert unstable.stable is False


def test_summary_separates_diagnostics_from_headline() -> None:
    verdicts = [
        _judge(BASELINE, speedup=10, stable=True),
        _judge(AUTHENTIC, speedup=0.9, stable=True),
        _judge(AUTHENTIC, speedup=1.2, stable=True),
        _judge(AUTHENTIC, context="baseline", speedup=2, stable=True),
    ]

    summary = summarise(verdicts)

    assert summary["n"] == 4
    assert summary["n_scorable"] == 3
    assert summary["n_credited"] == 3
    assert summary["n_pallas_credited"] == 2
    assert summary["n_headline_credited"] == 1
    assert summary["generalization_claim_ready"] is False
