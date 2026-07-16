# Stage-4 audit — 20260716-axport-cursor

| Field | Value |
|-------|-------|
| Slice ID | `20260716-axport-cursor-3ffdff36` |
| Source | R2 `axport` / `exports/…/cursor.zip` + `latest/_manifest.json` |
| Train JSONL | `data/factory/tinker/train_axport_cursor.jsonl` (gitignored) |
| SHA256 | `3ffdff369237589cbdc1ff24f14986a9d191b9c59475d908e2b613da6f212f16` |
| Trajectories | **74** |
| Scrub substitutions | **18** |
| Assistant token estimate | ~541k |
| Outcomes | unknown: 74 (labels not yet mined) |
| Duplicates | 0 |
| Canary hits in train | 0 |
| Rights | `stage0/signoffs/20260716-axport-cursor-3ffdff36.md` APPROVE |
| Preflight | PASS (provider=tinker) |

## Filters applied

- source=`cursor`
- cwd allowlist: `…/Code/conway/conway`, `…/Documents/Code/conway/conway` (+ minor variants); exclude worktrees/pipeline-mvp
- entry_count ∈ [4, 80], char_count ≤ 150k
- System prompts dropped; tool_call → assistant text
- Secret scrub on render

## Notes

- Character-scale warehouse (~2.1B chars in original plan) **not** used as size metric.
- Claude/Codex zips not yet ingested (deferred).
- Sealed SudarshanBench still empty → Stage-5 labeled exploratory if run.
