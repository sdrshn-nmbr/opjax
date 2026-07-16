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

## Current state

Splits manifest is an **empty freeze scaffold**. Populate train/dev first; freeze sealed in its own commit before Stage 5.
