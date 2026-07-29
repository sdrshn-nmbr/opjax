"""Credit rules for JAXBench candidates — the anti-mimicry gate.

Why this module exists (2026-07-28 probe, `docs/model-factory/06-env-rl/
jaxbench-base-vs-lora.md`): JAXBench prompts normally ship the reference
`baseline.py`, so "correct" can be earned by handing the reference back. Measured
on 12 priority workloads:

    reference shown   → base 4/12, Stage-5 LoRA 9/12
    reference withheld → base 3/12, Stage-5 LoRA 3/12

The 5-point spread was mimicry, not skill: the LoRA returned one baseline file
byte-for-byte (including its `benchmark()` harness) and its best speedup over 50
workloads was exactly 1.000x. An RL reward computed on reference-in-prompt tasks
is therefore farmable by echoing the prompt.

Two defences, both implemented here:

1. **Spec-mode prompts** (`PromptContext.SPEC`) — withhold the reference body, so
   there is nothing to copy. This is the scorable mode.
2. **Similarity gate** — when the reference *was* shown
   (`PromptContext.BASELINE`), a candidate whose `workload` is a near-copy of it
   earns no credit and no reward, however correct it is.

The gate is deliberately *not* applied in spec mode: with no reference in the
prompt, resembling it is convergence on the right answer (a correct RMSNorm looks
like the reference RMSNorm), not copying.
"""

from __future__ import annotations

import ast
import difflib
from dataclasses import dataclass, field
from enum import Enum

#: Similarity at or above which a reference-shown candidate is treated as a copy.
#: Chosen from the observed distribution: the Stage-5 LoRA sat at a median 0.933
#: with 27/50 above 0.9, while genuine rewrites (base's 2.36x paged attention,
#: its 15.05x algebraic fold) sat at 0.44 and below.
COPY_SIMILARITY_THRESHOLD = 0.90


class PromptContext(str, Enum):
    """What the model was shown. Determines whether the copy gate applies."""

    #: `CONFIG` + `create_inputs` + `workload` signature/docstring only.
    SPEC = "spec"
    #: Full reference implementation included — diagnostic only, gate active.
    BASELINE = "baseline"

    @property
    def gates_copies(self) -> bool:
        return self is PromptContext.BASELINE


def extract_workload_src(module_src: str) -> str | None:
    """Source of the top-level ``def workload`` in a module, or None."""
    try:
        tree = ast.parse(module_src)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "workload":
            return ast.get_source_segment(module_src, node)
    return None


def _normalise(src: str) -> str:
    return " ".join(src.split())


def baseline_similarity(candidate_src: str, baseline_src: str) -> float | None:
    """Similarity of the candidate's ``workload`` to the reference's, in [0, 1].

    Compared at function level rather than whole-file so that boilerplate the
    model copied along the way (imports, ``CONFIG``, ``create_inputs``) neither
    inflates nor masks the score. Returns None when either side has no parseable
    ``workload``.
    """
    cand = extract_workload_src(candidate_src)
    ref = extract_workload_src(baseline_src)
    if not cand or not ref:
        return None
    return difflib.SequenceMatcher(None, _normalise(ref), _normalise(cand)).ratio()


def is_verbatim_file_copy(candidate_src: str, baseline_src: str) -> bool:
    """True when the whole emitted module is the reference file, modulo whitespace."""
    return _normalise(candidate_src) == _normalise(baseline_src)


@dataclass(frozen=True)
class KernelVerdict:
    """Credit decision for one candidate on one workload."""

    workload: str
    correct: bool
    uses_pallas: bool
    prompt_context: PromptContext
    similarity: float | None = None
    verbatim_file_copy: bool = False
    speedup: float | None = None
    #: True when the candidate is a near-copy *and* the reference was shown.
    copied: bool = False
    #: Correct, not a copy — counts toward the functional score.
    credited: bool = False
    #: Correct, not a copy, and real Pallas — the headline kernel metric.
    pallas_credited: bool = False
    no_credit_reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def scorable(self) -> bool:
        """False for reference-shown runs, which cannot separate skill from mimicry."""
        return self.prompt_context is PromptContext.SPEC


def judge(
    *,
    workload: str,
    candidate_src: str,
    baseline_src: str,
    correct: bool,
    uses_pallas: bool,
    prompt_context: PromptContext | str,
    speedup: float | None = None,
    copy_threshold: float = COPY_SIMILARITY_THRESHOLD,
) -> KernelVerdict:
    """Apply the credit rules to one candidate."""
    ctx = PromptContext(prompt_context)
    similarity = baseline_similarity(candidate_src, baseline_src)
    verbatim = is_verbatim_file_copy(candidate_src, baseline_src)

    near_copy = verbatim or (similarity is not None and similarity >= copy_threshold)
    copied = bool(ctx.gates_copies and near_copy)

    reasons: list[str] = []
    if not correct:
        reasons.append("incorrect")
    if copied:
        reasons.append(
            "verbatim copy of the reference shown in the prompt"
            if verbatim
            else f"near-copy of the reference shown in the prompt "
            f"(similarity {similarity:.3f} >= {copy_threshold})"
        )
    credited = correct and not copied
    if credited and not uses_pallas:
        reasons.append("no real Pallas (`pl.pallas_call`) in the implementation")

    return KernelVerdict(
        workload=workload,
        correct=correct,
        uses_pallas=uses_pallas,
        prompt_context=ctx,
        similarity=similarity,
        verbatim_file_copy=verbatim,
        speedup=speedup,
        copied=copied,
        credited=credited,
        pallas_credited=credited and uses_pallas,
        no_credit_reasons=tuple(reasons),
    )


# Reward shaping for Stage-6b RL. Kept explicit rather than tuned so a change in
# incentives is visible in the diff.
REWARD_CORRECT = 0.3
REWARD_PALLAS = 0.3
REWARD_SPEED_MAX = 0.4
#: Speedup at which the speed term saturates (2x over the JAX baseline).
REWARD_SPEED_SATURATION = 2.0


def reward(verdict: KernelVerdict) -> float:
    """Scalar reward in [0, 1] for a candidate. Copies and failures earn zero.

    Shape: correctness is the entry ticket, real Pallas is the track we are
    climbing, and speed is the gradient *within* that track. A near-copy of a
    shown reference earns 0.0 even when correct, so the policy cannot farm reward
    by echoing the prompt.

    The speed term is deliberately gated behind Pallas. Without that gate the
    ordering inverts against the stated objective: base's 15.05x on
    ``26k_BMM_InstanceNorm`` — algebra that deletes a degenerate branch, no
    kernel involved — would outscore its one genuinely correct Pallas kernel.
    Under this shape a plain-JAX rewrite earns the correctness ticket and nothing
    more, however fast XLA makes it.
    """
    if not verdict.credited:
        return 0.0
    total = REWARD_CORRECT
    if not verdict.uses_pallas:
        return round(total, 4)
    total += REWARD_PALLAS
    if verdict.speedup and verdict.speedup > 1.0:
        span = REWARD_SPEED_SATURATION - 1.0
        frac = min((verdict.speedup - 1.0) / span, 1.0)
        total += REWARD_SPEED_MAX * frac
    return round(total, 4)


def summarise(verdicts: list[KernelVerdict]) -> dict:
    """Aggregate counters for a run, with the headline metric named explicitly."""
    n = len(verdicts)
    contexts = {v.prompt_context.value for v in verdicts}
    speedups = sorted(v.speedup for v in verdicts if v.credited and v.speedup)
    return {
        "n": n,
        "prompt_context": sorted(contexts),
        "scorable": all(v.scorable for v in verdicts),
        "n_correct_raw": sum(1 for v in verdicts if v.correct),
        "n_copied": sum(1 for v in verdicts if v.copied),
        "n_verbatim_file_copies": sum(1 for v in verdicts if v.verbatim_file_copy),
        "n_credited": sum(1 for v in verdicts if v.credited),
        "n_pallas_credited": sum(1 for v in verdicts if v.pallas_credited),
        "n_credited_faster_than_baseline": sum(1 for s in speedups if s >= 1.05),
        "best_speedup_credited": speedups[-1] if speedups else None,
        "mean_reward": (
            round(sum(reward(v) for v in verdicts) / n, 4) if n else 0.0
        ),
    }
