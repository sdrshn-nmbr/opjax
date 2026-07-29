from opjax.model_factory.jaxbench_scoring import (
    COPY_SIMILARITY_THRESHOLD,
    PromptContext,
    baseline_similarity,
    is_verbatim_file_copy,
    judge,
    reward,
    summarise,
)

BASELINE = '''\
import jax.numpy as jnp

CONFIG = {"emb_dim": 8192}


def create_inputs():
    return (jnp.ones((4, 8192)),)


def workload(x, scale):
    """RMSNorm."""
    mean2 = jnp.mean(x * x, axis=-1, keepdims=True)
    return x * jnp.reciprocal(jnp.sqrt(mean2 + 1e-5)) * scale
'''

REWRITE = '''\
import jax
from jax.experimental import pallas as pl


def workload(x, scale):
    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...]

    return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype))(x)
'''


def _judge(candidate: str, context: str, **kwargs):
    defaults = dict(correct=True, uses_pallas=False, speedup=None)
    defaults.update(kwargs)
    return judge(
        workload="12p_RMSNorm",
        candidate_src=candidate,
        baseline_src=BASELINE,
        prompt_context=context,
        **defaults,
    )


def test_verbatim_copy_is_gated_when_reference_was_shown() -> None:
    verdict = _judge(BASELINE, "baseline")

    assert verdict.verbatim_file_copy is True
    assert verdict.copied is True
    assert verdict.credited is False
    assert reward(verdict) == 0.0


def test_same_copy_is_not_gated_in_spec_mode() -> None:
    """With no reference in the prompt, resembling it is convergence, not copying."""
    verdict = _judge(BASELINE, "spec")

    assert verdict.copied is False
    assert verdict.credited is True
    assert reward(verdict) > 0.0


def test_whitespace_only_edit_still_counts_as_a_copy() -> None:
    verdict = _judge(BASELINE.replace("\n\n", "\n\n\n"), "baseline")

    assert verdict.copied is True


def test_genuine_rewrite_earns_credit_despite_reference_being_shown() -> None:
    verdict = _judge(REWRITE, "baseline", uses_pallas=True)

    assert verdict.similarity is not None
    assert verdict.similarity < COPY_SIMILARITY_THRESHOLD
    assert verdict.copied is False
    assert verdict.pallas_credited is True


def test_incorrect_candidate_never_earns_credit() -> None:
    verdict = _judge(REWRITE, "spec", correct=False, uses_pallas=True, speedup=4.0)

    assert verdict.credited is False
    assert verdict.pallas_credited is False
    assert reward(verdict) == 0.0
    assert "incorrect" in verdict.no_credit_reasons


def test_speed_term_requires_pallas() -> None:
    """A fast plain-JAX rewrite must not outrank a correct Pallas kernel."""
    fast_plain = _judge(REWRITE, "spec", uses_pallas=False, speedup=15.0)
    slow_pallas = _judge(REWRITE, "spec", uses_pallas=True, speedup=0.82)

    assert reward(fast_plain) < reward(slow_pallas)


def test_reward_increases_with_speedup_and_saturates() -> None:
    at_1x = _judge(REWRITE, "spec", uses_pallas=True, speedup=1.0)
    at_1_5x = _judge(REWRITE, "spec", uses_pallas=True, speedup=1.5)
    at_2x = _judge(REWRITE, "spec", uses_pallas=True, speedup=2.0)
    at_10x = _judge(REWRITE, "spec", uses_pallas=True, speedup=10.0)

    assert reward(at_1x) < reward(at_1_5x) < reward(at_2x)
    assert reward(at_2x) == reward(at_10x) == 1.0


def test_similarity_ignores_copied_boilerplate() -> None:
    """Similarity is measured on `workload`, so copied imports cannot mask a rewrite."""
    candidate = BASELINE.replace(
        '    mean2 = jnp.mean(x * x, axis=-1, keepdims=True)\n'
        '    return x * jnp.reciprocal(jnp.sqrt(mean2 + 1e-5)) * scale\n',
        "    return pl.pallas_call(kernel, grid=(8,), out_shape=shape)(x, scale)\n",
    )

    similarity = baseline_similarity(candidate, BASELINE)

    assert similarity is not None
    assert similarity < COPY_SIMILARITY_THRESHOLD
    assert is_verbatim_file_copy(candidate, BASELINE) is False


def test_unparseable_candidate_has_no_similarity_and_no_credit() -> None:
    verdict = _judge("def workload(x:\n  <prose leaked here>", "baseline", correct=False)

    assert verdict.similarity is None
    assert verdict.credited is False


def test_summarise_flags_reference_shown_runs_as_unscorable() -> None:
    shown = summarise([_judge(BASELINE, "baseline"), _judge(REWRITE, "baseline")])
    withheld = summarise([_judge(REWRITE, "spec", uses_pallas=True)])

    assert shown["scorable"] is False
    assert shown["n_correct_raw"] == 2
    assert shown["n_copied"] == 1
    assert shown["n_credited"] == 1
    assert withheld["scorable"] is True


def test_prompt_context_gating_flag() -> None:
    assert PromptContext.BASELINE.gates_copies is True
    assert PromptContext.SPEC.gates_copies is False
