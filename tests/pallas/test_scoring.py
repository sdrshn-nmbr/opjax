from __future__ import annotations

import pytest

from opjax.pallas.scoring import (
    PromptContext,
    TimingEvidenceError,
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

CONSTANT_DEAD = """\
import jax.numpy as jnp
from jax.experimental import pallas as pl

def workload(x):
    if False:
        return pl.pallas_call(kernel, out_shape=x)(x)
    return jnp.square(x)
"""

IDENTITY_FALLBACK = """\
import jax
from jax.experimental import pallas as pl

def workload(x, enabled=True):
    if enabled:
        shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
        return pl.pallas_call(kernel, out_shape=shape)(x)
    return x
"""

ASSIGNED_PALLAS_RESULT = """\
import jax
from jax.experimental import pallas as pl

def workload(x):
    shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    output = pl.pallas_call(kernel, out_shape=shape)(x)
    return output
"""

INTERPRETED = """\
import jax
from jax.experimental import pallas as pl

def workload(x):
    shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    return pl.pallas_call(kernel, out_shape=shape, interpret=True)(x)
"""

EXPLICIT_LOWERING = """\
import jax
from jax.experimental import pallas as pl

def workload(x):
    shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    return pl.pallas_call(kernel, out_shape=shape, interpret=False)(x)
"""

DYNAMIC_INTERPRET = """\
import jax
from jax.experimental import pallas as pl

def workload(x, interpret):
    shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    return pl.pallas_call(kernel, out_shape=shape, interpret=interpret)(x)
"""

EXPANDED_KEYWORDS = """\
import jax
from jax.experimental import pallas as pl

def workload(x):
    shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    options = {"interpret": True}
    return pl.pallas_call(kernel, out_shape=shape, **options)(x)
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
    assert inspection.reachable_lowered_pallas_calls == 1
    assert inspection.reachable_interpret_pallas_calls == 0
    assert inspection.reachable_functions == ("_kernel", "_launch", "workload")


def test_dead_code_pallas_is_rejected() -> None:
    inspection = inspect_pallas_source(DEAD_CODE)

    assert inspection.authentic is False
    assert inspection.reachable_pallas_calls == 0
    assert inspection.unreachable_pallas_calls == 1
    assert "PALLAS_PATH_UNREACHABLE" in inspection.reasons


def test_constant_false_pallas_branch_is_rejected() -> None:
    inspection = inspect_pallas_source(CONSTANT_DEAD)

    assert inspection.authentic is False
    assert inspection.reachable_pallas_calls == 0
    assert inspection.unreachable_pallas_calls == 1


def test_plain_jax_fallback_is_rejected() -> None:
    inspection = inspect_pallas_source(FALLBACK)

    assert inspection.authentic is False
    assert inspection.has_plain_jax_fallback is True
    assert "PLAIN_JAX_FALLBACK" in inspection.reasons


def test_identity_fallback_is_rejected() -> None:
    inspection = inspect_pallas_source(IDENTITY_FALLBACK)

    assert inspection.authentic is False
    assert inspection.has_plain_jax_fallback is True


def test_assigned_pallas_result_is_not_misclassified_as_fallback() -> None:
    inspection = inspect_pallas_source(ASSIGNED_PALLAS_RESULT)

    assert inspection.authentic is True
    assert inspection.has_plain_jax_fallback is False


@pytest.mark.parametrize(
    "source",
    [INTERPRETED, DYNAMIC_INTERPRET, EXPANDED_KEYWORDS],
)
def test_interpret_mode_cannot_receive_authentic_pallas_credit(
    source: str,
) -> None:
    inspection = inspect_pallas_source(source)
    verdict = _judge(source, speedup=10, stable=True)

    assert inspection.authentic is False
    assert inspection.reachable_pallas_calls == 1
    assert inspection.reachable_lowered_pallas_calls == 0
    assert inspection.reachable_interpret_pallas_calls == 1
    assert "PALLAS_INTERPRET_MODE" in inspection.reasons
    assert verdict.correct is True
    assert verdict.pallas_credited is False
    assert verdict.headline_credited is False


def test_explicit_false_interpret_mode_is_normal_lowering() -> None:
    inspection = inspect_pallas_source(EXPLICIT_LOWERING)

    assert inspection.authentic is True
    assert inspection.reachable_lowered_pallas_calls == 1
    assert inspection.reachable_interpret_pallas_calls == 0


def test_reference_visible_context_is_never_scorable() -> None:
    verdict = _judge(AUTHENTIC, context="baseline", speedup=2, stable=True)

    assert verdict.scorable is False
    assert verdict.credited is False
    assert verdict.headline_credited is False
    assert diagnostic_reward(verdict) == 0
    assert "DIAGNOSTIC_PROMPT_CONTEXT" in verdict.no_credit_reasons


def test_copied_reference_is_explicitly_rejected() -> None:
    verdict = _judge(BASELINE, context="baseline", speedup=10, stable=True)

    assert verdict.verbatim_file_copy is True
    assert verdict.copied is True
    assert verdict.credited is False
    assert "REFERENCE_COPY" in verdict.no_credit_reasons


def test_correct_slow_pallas_is_not_headline_success() -> None:
    verdict = _judge(AUTHENTIC, speedup=0.9, stable=True)

    assert verdict.pallas_credited is True
    assert verdict.headline_credited is False


def test_correct_fast_stable_pallas_is_headline_success() -> None:
    verdict = _judge(AUTHENTIC, speedup=1.2, stable=True)

    assert verdict.pallas_credited is True
    assert verdict.headline_credited is True


def test_required_lowering_evidence_fails_closed_when_missing() -> None:
    verdict = judge(
        workload="square",
        candidate_src=AUTHENTIC,
        baseline_src=BASELINE,
        compiled=True,
        correct=True,
        prompt_context="spec",
        speedup=1.2,
        timing_stable=True,
        require_lowering_evidence=True,
    )

    assert verdict.pallas_credited is False
    assert verdict.headline_credited is False
    assert "LOWERING_EVIDENCE_MISSING" in verdict.no_credit_reasons


def test_verified_lowering_evidence_unlocks_pallas_credit() -> None:
    verdict = judge(
        workload="square",
        candidate_src=AUTHENTIC,
        baseline_src=BASELINE,
        compiled=True,
        correct=True,
        prompt_context="spec",
        speedup=1.2,
        timing_stable=True,
        lowering_verified=True,
        require_lowering_evidence=True,
    )

    assert verdict.pallas_credited is True
    assert verdict.headline_credited is True
    assert verdict.lowering_verified is True


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
    empty = timing_evidence([], min_runs=3, max_coefficient_of_variation=0.1)
    insufficient = timing_evidence([1.0, 1.01], min_runs=3, max_coefficient_of_variation=0.1)
    stable = timing_evidence([1.0, 1.01, 0.99], min_runs=3, max_coefficient_of_variation=0.1)
    unstable = timing_evidence([1.0, 2.0, 4.0], min_runs=3, max_coefficient_of_variation=0.1)

    assert empty.stable is False
    assert empty.median_ms is None
    assert insufficient.stable is False
    assert stable.stable is True
    assert unstable.stable is False


@pytest.mark.parametrize(
    "invalid_sample",
    [0.0, -1.0, float("nan"), float("inf"), True, "1.0"],
)
def test_timing_rejects_non_positive_or_non_finite_samples(
    invalid_sample: object,
) -> None:
    with pytest.raises(TimingEvidenceError, match="TIMING_SAMPLES_INVALID"):
        timing_evidence(
            [1.0, invalid_sample, 1.1],
            min_runs=3,
            max_coefficient_of_variation=0.1,
        )


def test_timing_accepts_positive_integer_measurements() -> None:
    evidence = timing_evidence(
        [1, 1, 1],
        min_runs=3,
        max_coefficient_of_variation=0.1,
    )

    assert evidence.samples_ms == (1.0, 1.0, 1.0)
    assert evidence.stable is True


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
