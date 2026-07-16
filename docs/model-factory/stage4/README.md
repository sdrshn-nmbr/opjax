# Stage 4 — Data factory + audit

Pipeline (when axport ingress exists):

1. Ingest via `opjax.factory` / axport adapters (`data/axport/raw/`).
2. Filter: project allowlist, min turns, tool use.
3. Outcome labels; keep recovery segments.
4. Normalize to conversation `messages` (train=eval=serve semantics).
5. Secret scrub + canary.
6. Emit Tinker JSONL via `python -m opjax.factory render-tinker`.
7. Audit metrics → `audit-<date>.md` here (trajectory counts, assistant tokens, duplicates, outcomes, scrub recall — **not** char counts).

**Current status:** blocked on corpus ingress — see `../BLOCKED-axport-ingress.md`.
