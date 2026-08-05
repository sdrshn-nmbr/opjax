from __future__ import annotations

from opjax.pallas.environment import should_continue, verifier_feedback, verify_static


VALID = """```python
import jax
from jax.experimental import pallas as pl

def kernel(x_ref, o_ref):
    o_ref[...] = x_ref[...]

def workload(x):
    spec = pl.BlockSpec((128,), lambda i: (i,))
    return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype), in_specs=(spec,), out_specs=spec)(x)
```"""


def test_static_verifier_accepts_complete_normal_lowering_module() -> None:
    verdict = verify_static(VALID)

    assert verdict.passed is True
    assert verdict.stage == "static_complete"


def test_static_verifier_returns_one_actionable_stage() -> None:
    verdict = verify_static(VALID.replace("((128,), lambda i: (i,))", "(lambda i: (i,), (128,))"))

    assert verdict.passed is False
    assert verdict.stage == "pallas_api"
    assert "block shape first" in verdict.feedback


def test_attempt_budget_stops_after_three_failures_or_success() -> None:
    assert should_continue([]) is True
    assert should_continue([{"passed": False}] * 2) is True
    assert should_continue([{"passed": False}] * 3) is False
    assert should_continue([{"passed": True}]) is False


def test_hidden_verifier_feedback_does_not_expose_solution() -> None:
    feedback = verifier_feedback({"stage": "tpu_compile", "error": "shape mismatch"})

    assert "shape mismatch" in feedback
    assert "reference" not in feedback.lower()
