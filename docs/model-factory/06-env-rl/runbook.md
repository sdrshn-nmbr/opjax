# Stage 6 — Environment qualification → thin RL

**Prerequisite:** Stage 5 sealed win under kill condition.

## Qualification (before RL)

1. Measure verifier false-positive / false-negative rates on a labeled probe set.
2. Solution-channel scanners (git history, caches, package registries, web).
3. Single-repo test-reward tasks first.

## Training policy

- Prefer on-policy with version pins.
- If async: document lag bounds, importance/replay, router replay — do not claim “on-policy” without this.
- Candidate substrates: Prime Lab/verifiers (smoke first), Modal, Fireworks remote — choose by evidence.
- Pedagogy from SWE-1.7 / Composer; do not copy scale.

## Retention

Keep a replay buffer / forgetting suite when continuing from Stage 5 checkpoint.
