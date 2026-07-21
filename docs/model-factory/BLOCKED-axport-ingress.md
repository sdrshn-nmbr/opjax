# Axport corpus ingress — status

## 2026-07-16 update — **UNBLOCKED**

R2 access confirmed for bucket `axport` (ListBuckets denied; direct bucket list OK).

| Object | Used? |
|--------|-------|
| `latest/_manifest.json` | Yes — session index |
| `exports/…/cursor.zip` (~54 MB) | Yes — first train slice |
| `exports/…/claude.zip` / `codex.zip` | Not yet |
| `latest/export.zip` (~545 MB) | Not downloaded |

First governed slice: `stage4/audit-20260716-axport-cursor.md`  
Sign-off: `stage0/signoffs/20260716-axport-cursor-3ffdff36.md`

## Earlier block (historical)

Previously this VM lacked R2 secrets. Owner provided `R2_*` credentials (store in gitignored `.env` / Cloud Environment — **rotate** if exposed in chat logs).
