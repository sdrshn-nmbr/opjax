# Run memo — Inkling public plumbing smoke

| Field | Value |
|-------|-------|
| Date (UTC) | 2026-07-16 |
| Class | Public plumbing smoke (no private axport) |
| Label | **exploratory / plumbing** (sealed set empty) |
| Model | `thinkingmachines/Inkling` |
| Renderer | `tml_v0` (effort 0.9 default in samples) |
| Dataset | `data/factory/smoke/conversations_tiny.jsonl` (5 synthetic coding convos) |
| Dataset provenance | `tests/factory/fixtures/synthetic_coding_sessions.jsonl` → scrub → render |
| Preflight | `--allow-public-fixture` OK |
| LoRA rank | 16 |
| Learning rate | `1e-4` (manual; cookbook uncalibrated) |
| Batch size | 1 |
| max_steps | 3 |
| max_length | 4096 |
| Log path | `logs/factory/inkling_smoke` (gitignored) |
| Session / train id | `e74048cf-6a82-50af-8004-fda133b040bd:train:0` |
| Checkpoint | `tinker://e74048cf-6a82-50af-8004-fda133b040bd:train:0/weights/final` |
| Sampler | `tinker://e74048cf-6a82-50af-8004-fda133b040bd:train:0/sampler_weights/final` |
| Wall-clock | ~59 s (under 30 min cap) |
| Steps completed | 3 (train_mean_nll ≈ 0.69 → 2.78 → 2.05) |
| Result | **SUCCESS** — Tinker SFT path validated for Inkling |

## Caps

Within DRAFT smoke caps (`spend-caps.md`): ≤ $50 / 30 min / 50 steps.

## Notes

- `num_loss_tokens` logged as `1.0` per step under mean weight reduction — investigate before Stage-5 science; does not block plumbing claim.
- No private Conway/axport data left the machine.
- Next: axport ingress + Stage-0 sign-off before private LoRA.
