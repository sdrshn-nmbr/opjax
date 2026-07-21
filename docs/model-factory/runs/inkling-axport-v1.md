# Run memo — Inkling axport v1 LoRA (Stage-5 exploratory)

| Field | Value |
|-------|-------|
| Date (UTC) | 2026-07-16 |
| Class | First private / governed LoRA |
| Label | **`v1-plumbing / exploratory`** (sealed SudarshanBench still empty — not a scientific Stage-5 win/kill) |
| Model | `thinkingmachines/Inkling` |
| Renderer | `tml_v0` (effort 0.9) |
| Slice | `20260716-axport-cursor-3ffdff36` |
| Dataset | 74 scrubbed Cursor→Tinker conversations (Conway cwd allowlist) |
| Dataset SHA256 | `3ffdff369237589cbdc1ff24f14986a9d191b9c59475d908e2b613da6f212f16` |
| Rights | `stage0/signoffs/20260716-axport-cursor-3ffdff36.md` APPROVE |
| Preflight | PASS (canaries clean) |
| LoRA rank | 16 |
| Learning rate | `1e-4` (manual) |
| Batch size | 1 |
| max_length | 8192 |
| Steps | 74 (1 epoch over slice; `max_steps=80` cap) |
| Wall-clock | ~10 min (under 4 h private cap) |
| Run ID | `067b00f3-200a-5dc9-a6df-ccd49542bb03:train:0` |
| Checkpoint | `tinker://067b00f3-200a-5dc9-a6df-ccd49542bb03:train:0/weights/final` |
| Sampler | `tinker://067b00f3-200a-5dc9-a6df-ccd49542bb03:train:0/sampler_weights/final` |
| train_mean_nll | mean ≈ 1.04 across logged steps (min ≈ 0.004, max ≈ 2.28) |
| Result | **SUCCESS** — governed upload + LoRA completed |

## Controls not yet run

Sealed tasks empty → no base / prompt-only / RAG paired comparison yet.
Do **not** treat this as hypothesis confirmation. Kill condition not evaluable until sealed freeze is non-empty.

## Hygiene

- Secrets stay in gitignored `.env` / local data paths only.
- Rotate R2 credentials if they were pasted into chat logs.
- Claude/Codex axport zips not yet mixed in.
