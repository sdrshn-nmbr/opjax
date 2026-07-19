# Spend ledger reconcile — 2026-07-19

Source: Tinker Usage console (billing period July 2026), operator screenshot.

## Wallet snapshot

| Field | Value |
|-------|-------|
| Cost so far this period | **$286.34** |
| Tokens this period | **51.05M** |
| Checkpoint storage | 0.00 GB-months |
| Approx wallet remaining | **~$65** (operator; reload available) |
| Soft ceiling Stages 3–6 | $1,450 ([spend-caps.md](../00-governance/spend-caps.md)) |

## Attribution (best-effort from Usage chart)

| Window | Model / note | Est. USD | Stage |
|--------|--------------|----------|-------|
| Jul 13 | Non-Inkling spike (~$160 bar; not in Inkling legend) | ~160 | Pre-Stage-5 / other — **not** counted against Stage-5 LoRA claim without operator reclass |
| Jul 16–17 | `thinkingmachines/Inkling` LoRA SFT v2 (~48.4M train tokens) + sealed evals | ~126 | **Stage 5** (within $500 hard cap) |
| Period total | — | **286.34** | mixed |

Stage-5 hard-cap check: Inkling Stage-5 train+eval attributed ≈ **$126** ≪ **$500** → Stage-5 spend gate OK.

## Ledger entries to apply

Live file (gitignored): `data/model-factory/ledger/spend-ledger.json`  
Template: [`../00-governance/spend-ledger.example.json`](../00-governance/spend-ledger.example.json)

Recommended `entries[]` (append-only):

1. `stage5-inkling-lora-v2` — provider `tinker`, USD **126.00** (est.), tokens ~48.4M train, hypothesis Stage-1 sealed LoRA claim, sampler `tinker://21e391ab-7c5d-573c-9477-16c93df81a08:train:0/sampler_weights/final`
2. `period-other-jul13` — provider `tinker`, USD **160.34** (residual to match $286.34), stage `unallocated` until operator reclass

`usd_spent` after apply: **286.34**; `usd_remaining` vs soft ceiling: **1163.66**. Wallet remaining (~$65) is the Tinker console balance, not the soft ceiling.

## Stage 6 spend rule

- Cap: **$750** / 120 GPU-hrs / 21 days — only because Stage 5 kill did not trigger.
- No Inkling RL job until operator explicitly approves a token/$ line (>$25 or any reload).
- Env-qual / scanners: $0 local preferred.
