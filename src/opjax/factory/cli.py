"""CLI for Model Factory data path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from opjax.factory.audit import audit_conversations_jsonl
from opjax.factory.axport_ingest import load_trajectories
from opjax.factory.normalize import NormalizeConfig, normalize_trajectories
from opjax.factory.preflight import preflight
from opjax.factory.render_tinker import render_conversations_jsonl, validate_conversations_jsonl
from opjax.factory.scrub import load_canaries, scrub_text, write_canaries


def _cmd_scrub(args: argparse.Namespace) -> int:
    text = Path(args.infile).read_text(encoding="utf-8")
    result = scrub_text(text)
    out = Path(args.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.text, encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out),
                "substitutions": result.substitutions,
                "hits": [h.__dict__ for h in result.hits],
            },
            indent=2,
        )
    )
    return 0


def _cmd_canary(args: argparse.Namespace) -> int:
    cans = write_canaries(args.outfile, n=args.n)
    print(json.dumps({"path": args.outfile, "count": len(cans)}, indent=2))
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    trajs = load_trajectories(args.infile)
    cfg = NormalizeConfig(
        min_turns=args.min_turns,
        project_allowlist=set(args.allow_project) if args.allow_project else None,
    )
    norm = normalize_trajectories(trajs, cfg)
    stats = render_conversations_jsonl(norm.kept, args.outfile, scrub=not args.no_scrub)
    validate_conversations_jsonl(args.outfile)
    print(
        json.dumps(
            {
                "render": stats,
                "dropped": norm.dropped,
                "drop_reasons": norm.reasons,
            },
            indent=2,
        )
    )
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    print(json.dumps(audit_conversations_jsonl(args.dataset), indent=2, sort_keys=True))
    return 0


def _cmd_preflight(args: argparse.Namespace) -> int:
    result = preflight(
        args.dataset,
        provider=args.provider,
        manifest=args.manifest,
        canary_file=args.canary_file,
        allow_public_fixture=args.allow_public_fixture,
    )
    payload = {
        "ok": result.ok,
        "errors": result.errors,
        "warnings": result.warnings,
        "dataset_sha256": result.dataset_sha256,
        "details": result.details,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.ok else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="opjax.factory")
    sub = p.add_subparsers(dest="command", required=True)

    scrub = sub.add_parser("scrub", help="Scrub secrets from a text/JSONL file")
    scrub.add_argument("--in", dest="infile", required=True)
    scrub.add_argument("--out", dest="outfile", required=True)
    scrub.set_defaults(func=_cmd_scrub)

    canary = sub.add_parser("canary", help="Write canary strings to a file")
    canary.add_argument("--out", dest="outfile", required=True)
    canary.add_argument("-n", type=int, default=3)
    canary.set_defaults(func=_cmd_canary)

    render = sub.add_parser("render-tinker", help="Normalize + scrub + emit Tinker JSONL")
    render.add_argument("--in", dest="infile", required=True)
    render.add_argument("--out", dest="outfile", required=True)
    render.add_argument("--min-turns", type=int, default=2)
    render.add_argument("--allow-project", action="append", default=[])
    render.add_argument("--no-scrub", action="store_true")
    render.set_defaults(func=_cmd_render)

    audit = sub.add_parser("audit", help="Audit a conversations JSONL")
    audit.add_argument("--dataset", required=True)
    audit.set_defaults(func=_cmd_audit)

    pre = sub.add_parser("preflight", help="Hard pre-upload gate")
    pre.add_argument("--dataset", required=True)
    pre.add_argument("--manifest", default=None)
    pre.add_argument("--provider", default="tinker")
    pre.add_argument("--canary-file", default=None)
    pre.add_argument("--allow-public-fixture", action="store_true")
    pre.set_defaults(func=_cmd_preflight)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Quiet unused import guard for load_canaries in future commands
    _ = load_canaries
    code = args.func(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main(sys.argv[1:])
