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

## Current state (frozen 2026-07-16)

| Split | Count | IDs |
|-------|-------|-----|
| train | 4 | sb-0001…0004 |
| dev | 3 | sb-0005…0007 |
| sealed | 4 | sb-0008…0011 |
| time_forward | 1 | sb-0012 |

Fixtures live under `fixtures/<id>/` (broken `solution.py` + tests). Headline metric = `pytest -q` on sealed IDs only. **Never train on sealed.**
