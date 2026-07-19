# SudarshanBench

Private CursorBench-style suite for the Model Factory.

## Task schema (per task file)

Each task under `tasks/<id>.json`:

```json
{
  "id": "sb-0001",
  "split": "train",
  "repo": "https://github.com/org/repo",
  "commit": "<pin>",
  "prompt": "…",
  "test_command": "pytest -q",
  "rubric": ["tigerstyle-clarity", "no-drive-by"],
  "license": "MIT"
}
```

## Current state (splits.json version 2 — hardened 2026-07-19)

| Split | Count | IDs |
|-------|-------|-----|
| train | 4 | sb-0001…0004 |
| dev | 3 | sb-0005…0007 |
| sealed | **8** | sb-0008…0011 + sb-0013…0016 |
| time_forward | 1 | sb-0012 |

v1 freeze (Stage-5 claim, sealed n=4): [`splits.v1-freeze.json`](splits.v1-freeze.json).  
v1 sealed IDs remain **never-train**. New sealed IDs added via versioned manifest (protocol), not by moving train/dev.

Fixtures live under `fixtures/<id>/` (broken `solution.py` + tests). Headline metric = `pytest -q` on sealed IDs only. **Never train on sealed.**

## Stage-6 baseline

Re-run Stage-5 LoRA arm once on sealed v2 (no hparam search) before thin RL claims. See [`../../06-env-rl/sealed-harden.md`](../../06-env-rl/sealed-harden.md).
