"""Curate axport Markdown export → Tinker conversations JSONL."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from opjax.model_factory.scrub import scrub_text

_HEADER = re.compile(
    r"^### \[(\d+)\] role=(\w+)(?: tool=(\w+))? ts=.*$",
    re.MULTILINE,
)


@dataclass
class Stats:
    scanned: int = 0
    kept: int = 0
    skipped_short: int = 0
    skipped_no_assistant: int = 0
    scrub_hits: int = 0


def _parse_markdown(md: str) -> list[dict[str, str]]:
    matches = list(_HEADER.finditer(md))
    if not matches:
        return []
    messages: list[dict[str, str]] = []
    for i, match in enumerate(matches):
        role = match.group(2)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        content = md[start:end].strip()
        if not content:
            continue
        if role in {"user"}:
            messages.append({"role": "user", "content": content})
        elif role in {"assistant", "reasoning"}:
            # Flatten reasoning into assistant channel for first SFT pass.
            if messages and messages[-1]["role"] == "assistant":
                messages[-1]["content"] += "\n" + content
            else:
                messages.append({"role": "assistant", "content": content})
        elif role in {"tool_call", "tool_result"}:
            tool = match.group(3) or "tool"
            blob = f"[{role} {tool}]\n{content}"
            if messages and messages[-1]["role"] == "assistant":
                messages[-1]["content"] += "\n" + blob
            else:
                messages.append({"role": "assistant", "content": blob})
        # ignore other roles
    return messages


def _truncate(messages: list[dict[str, str]], max_chars: int) -> list[dict[str, str]]:
    total = sum(len(m["content"]) for m in messages)
    if total <= max_chars:
        return messages
    # Keep system-less tail-biased conversation within budget.
    out: list[dict[str, str]] = []
    budget = max_chars
    for m in reversed(messages):
        chunk = m["content"]
        if len(chunk) > budget:
            chunk = chunk[-budget:]
        out.append({"role": m["role"], "content": chunk})
        budget -= len(chunk)
        if budget <= 0:
            break
    out.reverse()
    return out


def curate_from_zip(
    zip_path: Path,
    out_path: Path,
    *,
    max_examples: int = 300,
    min_messages: int = 4,
    max_chars: int = 12000,
    path_substr: str | None = "conway",
) -> Stats:
    stats = Stats()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept_lines: list[str] = []

    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".md")]
        if path_substr:
            preferred = [n for n in names if path_substr in n.lower()]
            other = [n for n in names if n not in preferred]
            names = preferred + other

        for name in names:
            if stats.kept >= max_examples:
                break
            stats.scanned += 1
            try:
                md = zf.read(name).decode("utf-8", errors="replace")
            except Exception:
                continue
            messages = _parse_markdown(md)
            if len(messages) < min_messages:
                stats.skipped_short += 1
                continue
            if not any(m["role"] == "assistant" for m in messages):
                stats.skipped_no_assistant += 1
                continue
            if not any(m["role"] == "user" for m in messages):
                stats.skipped_short += 1
                continue
            messages = _truncate(messages, max_chars)
            # Scrub each message
            clean_msgs = []
            for m in messages:
                scrubbed = scrub_text(m["content"])
                stats.scrub_hits += len(scrubbed.hits)
                clean_msgs.append({"role": m["role"], "content": scrubbed.text})
            # Drop if still huge empty
            if sum(len(m["content"]) for m in clean_msgs) < 40:
                continue
            # System preamble: personal coding sidekick (rules lite)
            record = {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a careful coding agent. Prefer minimal diffs, "
                            "run tests, and follow tigerstyle clarity when editing code."
                        ),
                    },
                    *clean_msgs,
                ],
            }
            kept_lines.append(json.dumps(record, ensure_ascii=False))
            stats.kept += 1

    out_path.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")
    return stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--zip", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-examples", type=int, default=300)
    p.add_argument("--min-messages", type=int, default=4)
    p.add_argument("--max-chars", type=int, default=12000)
    p.add_argument("--path-substr", default="conway")
    args = p.parse_args(argv)
    stats = curate_from_zip(
        args.zip,
        args.out,
        max_examples=args.max_examples,
        min_messages=args.min_messages,
        max_chars=args.max_chars,
        path_substr=args.path_substr or None,
    )
    print(json.dumps(stats.__dict__ | {"out": str(args.out)}, indent=2))
    return 0 if stats.kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
