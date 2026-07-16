# Stage 4 — Data factory + audit

**Goal:** Hundreds–low thousands of high-quality trajectories — not character-scale dumps.

## Pipeline

1. **Ingest** via `past` / axport source adapters (structured sessions), not Markdown zip alone.
2. **Filter:** [allowlist.yaml](allowlist.yaml), `--since`, min turns, require tool use.
3. **Labels:** outcomes + keep **recovery** segments labeled (no success-only wipe).
4. **Normalize** to canon harness tool schema (train=eval=serve semantics).
5. **Rules:** system context on demonstrations; hard tigerstyle invariants also as static checks.
6. **Scrub + canary:** `opjax-model-factory scrub` / `pre-upload-gate`.
7. **Emit** backend JSONL (Tinker/`tml-renderers` when used).

## Audit metrics (binding)

Use `opjax-model-factory audit-jsonl` — report:

- complete trajectories
- assistant message count + approx assistant tokens
- duplicates
- outcome histogram
- recovery segment count
- license tags / missing license
- scrub recall (manual + canary)

**Do not** use raw character counts as dataset size.

## Mix ratio

`axport` vs `iSFT` mix is **unlocked** — choose via ablations on `dev`, never a pre-locked 40/60.
