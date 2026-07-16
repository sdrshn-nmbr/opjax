#!/usr/bin/env python3
"""Build a scrubbed Tinker JSONL slice from downloaded axport cursor exports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opjax.factory.audit import audit_conversations_jsonl
from opjax.factory.markdown_export import index_markdown_by_uuid, parse_axport_markdown
from opjax.factory.normalize import NormalizeConfig, normalize_trajectories
from opjax.factory.render_tinker import render_conversations_jsonl
from opjax.factory.scrub import write_canaries


ALLOW_CWD_SUBSTR = (
    "/code/opjax",
    "/code/conway/conway",
    "/documents/code/conway/conway",
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/axport/meta/_manifest.json")
    p.add_argument("--md-root", default="data/axport/raw/cursor_extracted")
    p.add_argument("--out", default="data/factory/tinker/train_axport_cursor.jsonl")
    p.add_argument("--canary-file", default="data/factory/canaries.txt")
    p.add_argument("--max-trajectories", type=int, default=150)
    p.add_argument("--min-entries", type=int, default=4)
    p.add_argument("--max-entries", type=int, default=80)
    p.add_argument("--max-chars", type=int, default=150_000)
    p.add_argument("--source", default="cursor")
    args = p.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    by_uuid = index_markdown_by_uuid(Path(args.md_root))
    print(f"indexed_md={len(by_uuid)}")

    candidates = []
    for s in manifest.get("sessions") or []:
        if s.get("source") != args.source:
            continue
        cwd = (s.get("cwd") or "").lower()
        if not any(a in cwd for a in ALLOW_CWD_SUBSTR):
            continue
        # Prefer exact main repos over pipeline-mvp / worktrees for v1
        if "pipeline-mvp" in cwd or "worktree" in cwd or "conductor" in cwd:
            continue
        entries = int(s.get("entry_count") or 0)
        chars = int(s.get("char_count") or 0)
        if entries < args.min_entries or entries > args.max_entries:
            continue
        if chars > args.max_chars:
            continue
        uid = (s.get("uuid") or "").lower()
        if uid not in by_uuid:
            continue
        candidates.append((chars, entries, s, by_uuid[uid]))

    # Prefer mid-size sessions (sort by entries desc, then chars)
    candidates.sort(key=lambda t: (-t[1], t[0]))
    candidates = candidates[: args.max_trajectories]
    print(f"candidates={len(candidates)}")

    trajs = []
    for _chars, _entries, meta, path in candidates:
        text = path.read_text(encoding="utf-8", errors="replace")
        # Plant nothing here; canaries written separately for preflight
        t = parse_axport_markdown(
            text,
            trajectory_id=str(meta["uuid"]),
            project=meta.get("cwd"),
            source=f"axport-{meta.get('source')}",
            drop_system=True,
        )
        if t is None:
            continue
        t.metadata.update(
            {
                "cwd": meta.get("cwd"),
                "entry_count": meta.get("entry_count"),
                "char_count": meta.get("char_count"),
                "started_at": meta.get("started_at"),
                "md_path": str(path),
            }
        )
        trajs.append(t)

    norm = normalize_trajectories(trajs, NormalizeConfig(min_turns=2, project_allowlist=None))
    print(f"parsed={len(trajs)} kept={len(norm.kept)} dropped={norm.dropped} reasons={norm.reasons}")

    canaries = write_canaries(args.canary_file, n=3)
    # Inject canaries into a side copy for recall testing only — not into train out.
    Path(args.canary_file).write_text("\n".join(canaries) + "\n", encoding="utf-8")

    stats = render_conversations_jsonl(norm.kept, args.out, scrub=True)
    audit = audit_conversations_jsonl(args.out)
    print(json.dumps({"render": stats, "audit": audit, "canary_file": args.canary_file}, indent=2))


if __name__ == "__main__":
    main()
