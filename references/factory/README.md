# Model Factory — citation pack

Pointer index for Stages 0–5. Prefer links + transfer boundaries over vendoring large binaries.
Full living guide: [`docs/model-factory/GUIDE.md`](../../docs/model-factory/GUIDE.md).

## Core (use now)

| Artifact | URL | Role / caveat |
|----------|-----|----------------|
| Inkling model card | https://thinkingmachines.ai/model-card/inkling/ | Candidate base; open **weights**, not full OSS process |
| Inkling on Hub | https://huggingface.co/thinkingmachines/Inkling | Ungated Apache-2.0 (recon 2026-07-16); ~MoE; do not pull full weights for Tinker SFT |
| Using Inkling (Tinker) | https://tinker-docs.thinkingmachines.ai/cookbook/inkling/ | `tinker-cookbook[inkling]` + `tml_renderers` |
| tml-renderers | https://tinker-docs.thinkingmachines.ai/cookbook/inkling/tml-renderers/ | Message ↔ token bridge for SFT/sample |
| First SFT tutorial | https://tinker-docs.thinkingmachines.ai/tutorials/basics/first-sft/ | LoRA primitives; JSONL conversations |
| Supervised learning cookbook | https://tinker-docs.thinkingmachines.ai/cookbook/supervised-learning/ | `FromConversationFileBuilder`, train loop |
| LoRA without regret | https://thinkingmachines.ai/blog/lora/ | LR ×10 heuristic; Inkling still needs **manual** LR |
| DeepSWE | https://deepswe.datacurve.ai/run | Public **sealed/report** only — never train the split you report |
| Tinker GA | https://thinkingmachines.ai/news/tinker-general-availability/ | Managed training API context |

## Later stages (do not auto-start workstreams)

| Artifact | URL | Gate |
|----------|-----|------|
| OPD (TML) | https://thinkingmachines.ai/blog/on-policy-distillation/ | Stage 8 — logprob + token-compatible teachers only |
| PorTAL / portallib | https://github.com/ramp-public/portallib | Stage 9 — reproduce Qwen3→Gemma first |
| Latent Briefing | https://labs.ramp.com/research/latent-briefing-kv-cache/index.md | Stage 10b — after Fusion context cost |
| STILL (Baseten) | https://www.baseten.co/research/towards-infinite-context-windows-neural-kv-cache-compaction/ | Stage 10b research |
| iSFT (Baseten) | https://www.baseten.co/research/iterative-sft/ | Data refinement method; medical evidence ≠ coding proof |
| Prime Lab / verifiers | https://www.primeintellect.ai/blog/lab | Stage 6 candidate after smoke |
| Devin Fusion | https://cognition.com/blog/devin-fusion | Stage 7 late product hypothesis |
| Composer 2.5 | https://cursor.com/blog/composer-2-5 | Harness pedagogy — not Phase-1 success metric |

## Local smoke evidence

- Run memo: [`docs/model-factory/runs/inkling-smoke.md`](../../docs/model-factory/runs/inkling-smoke.md)
- Blocked private FT: [`docs/model-factory/BLOCKED-axport-ingress.md`](../../docs/model-factory/BLOCKED-axport-ingress.md)
