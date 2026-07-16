# Model Factory — living experiment guide

**How to use:** Single in-repo plan for the Inkling / axport / Tinker coding FT track.
Update this file when decisions change; do not spawn parallel plans.
Every subsystem needs a **decision**, **audit status**, **reasoning**, and **primary source**.

**Status (2026-07-16):** Wave A auth done. Stages **0–2** artifacts in this tree.
Public Tinker plumbing smoke **passed** (`runs/inkling-smoke.md`).
**Not execution-ready for private training** until axport ingress + Stage-0 sign-off + scrub preflight
(`BLOCKED-axport-ingress.md`).

---

## 1. North-star verdict

| Claim | Status | Reasoning |
|-------|--------|-----------|
| Personal specialist coding model from axport + rules + open bases | **Keep** | Personalization vs chasing public SWE boards |
| Model Factory / Fusion / PorTAL ambition | **Roadmap, gated** | Hypotheses until stage gates pass |
| Ready to train on private axport | **Reject until 0–2** | Need rights, sealed evals, scrub gate |
| Durable IP today | **Governed data + sealed evals + tasksets** | Checkpoints are hypotheses until measured |
| Inkling / Prime / Fusion / OPD / PorTAL | **Competing hypotheses** | Demoted from architecture |

**Conservative spine:** curated SFT → private eval → thin RL only if SFT helps.

**Kill condition:** If Stage-5 governed LoRA fails sealed eval vs no-training controls under budget → stop weight-training claims; fix data/task or park.

---

## 2. Decision log (binding)

| ID | Binding rule |
|----|--------------|
| D1 | Continual learning fuel = train/dev only; sealed + public-report splits never train |
| D3 | Mix ratios from Stage-4 ablations only (no locked 40/60) |
| D4 | Scope = Conway + allowlisted OSS + tigerstyle first |
| D5 | DeepSWE public report split and sealed SudarshanBench never train |
| D8 | Inkling-Small contingent; tournament on currently trainable bases |
| D9 | SFT kill restored |
| D11 | DeepSWE = sealed/report; separate train task pool |
| D12 | PorTAL: reproduce Qwen3→Gemma on portallib first (late) |
| D13 | Tinker Inkling SFT path **validated** by public plumbing smoke (2026-07-16); private upload still gated by Stage-0 sign-off |

---

## 3. Four freshness problems (never collapse)

| ID | Problem | Mechanism | Metric |
|----|---------|-----------|--------|
| P1 | Task portability across bases | Re-LoRA / PorTAL (Stage 9) | Recovered lift on sealed sidekick tasks |
| P2 | Teacher-gap | OPD iff logprobs+token-compatible; else sequence distill | Teacher-gap on sealed set |
| P3 | Knowledge / API freshness | Time-forward shadow eval | Time-forward pass rate |
| P4 | Session memory | Same-checkpoint KV / text compaction | Retention + latency |

---

## 4. Architecture vs hypothesis

**Architecture (build first):** rights manifest, sealed splits, task semantics, spend gates, portable recipes (datasets/graders/evals).

**Hypotheses (fund after gates):** Inkling family, Laguna, Qwen dense, Prime Lab RL, Fusion, OPD, PorTAL, neural KV.

---

## 5. Stage sequence (summary)

| Stage | Artifact | Gate |
|-------|----------|------|
| **0** Governance | Rights + DPA/retention + scrub/canary + spend caps | No managed upload without signed-off manifest |
| **1** One claim | Hypothesis + kill + stats | Written before multi-program work |
| **2** Sealed eval | Split protocol; DeepSWE policy; empty sealed freeze | Protocol doc + frozen sealed set before train |
| **3** Base tournament | Cost-normalized compare (trainable bases) | Pick Stage-5 base from $/quality |
| **4** Data factory | Scrubbed JSONL + audit metrics | Audit complete before LoRA |
| **5** Controlled LoRA | LoRA vs prompt/RAG/static controls | Kill if sealed fails under budget |
| **6–10** | RL / Fusion / teacher / PorTAL / KV | Only after prior gates |

Detail: `stage0/`, `stage1/`, `stage2/`.

---

## 6. Immediate sequencing (this repo)

1. Bootstrap: `hf auth`, `tinker-cookbook[inkling]` — done / in progress.
2. Stage 0–2 docs — this directory.
3. `opjax.factory` package: scrub, rights, render Tinker JSONL, preflight.
4. Public Inkling smoke on fixture JSONL (no private upload).
5. Axport ingress → Stage-4 MVP → Stage-0 sign-off → Stage-5 LoRA.

---

## 7. Explicit non-goals (until gates pass)

- Training private data before Stages 0–2 + sign-off
- Locking Inkling-Small as sidekick
- Training on DeepSWE report split or sealed SudarshanBench
- OPD for arbitrary closed APIs; PorTAL agentic MoE as first spike
- Cross-model KV from shared tokenizer; Fusion before solo sealed win
- Removing kill conditions; treating axport character scale as dataset size

---

## 8. Services status (cloud VM recon 2026-07-16)

| Service | Status |
|---------|--------|
| Hugging Face | `sdrshn-nmbr`; use `hf auth` (not `huggingface-cli`) |
| Tinker | 0.23.0 + cookbook 0.5.2 + `tml-renderers`; Inkling on allowlist |
| Prime | CLI + API key present |
| Axport / R2 | **Not configured in this VM** — Stage-4 blocker until ingress |
| Fireworks / Baseten / Harbor / portallib | Deferred |

**Rule:** Stage-0 rights before any private trainer upload. Never echo secrets in terminals.

---

## 9. Change control

- Prefer semantic edits to this file over append-only drift.
- Vendor posts → artifact catalog with transfer boundary; no auto workstreams.
- “Locked” means gate-backed. Auth changes → update §8.

See also: `CHANGELOG.md`, `stage0/spend-caps.md`, run memos under `runs/`.
