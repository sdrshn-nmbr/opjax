"""CLI entrypoints for Model Factory governance and data tooling."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from opjax.model_factory.audit import audit_jsonl
from opjax.model_factory.pre_upload import run_pre_upload_gate
from opjax.model_factory.scrub import scrub_file
from opjax.model_factory.splits import (
    assert_manifest_disjoint,
    load_split_manifest,
    validate_no_train_on_sealed,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="opjax-model-factory")
    sub = parser.add_subparsers(dest="command", required=True)

    scrub = sub.add_parser("scrub", help="Scrub secret-like patterns from a text file.")
    scrub.add_argument("path")
    scrub.add_argument("--write", type=Path, default=None, help="Write scrubbed output.")

    gate = sub.add_parser(
        "pre-upload-gate",
        help="Hard gate before uploading a data slice to a managed trainer.",
    )
    gate.add_argument("--source", required=True, type=Path)
    gate.add_argument("--provider", required=True, choices=["tinker", "prime", "modal", "fireworks", "baseten", "hf"])
    gate.add_argument("--rights-manifest", required=True, type=Path)
    gate.add_argument("--slice-id", required=True)
    gate.add_argument("--output-dir", type=Path, default=Path("data/model-factory/audits"))
    gate.add_argument("--spend-ledger", type=Path, default=Path("docs/model-factory/00-governance/spend-ledger.example.json"))
    gate.add_argument("--allow-no-spend-check", action="store_true")

    audit = sub.add_parser("audit-jsonl", help="Audit trajectory JSONL quality metrics.")
    audit.add_argument("path", type=Path)
    audit.add_argument("--write", type=Path, default=None)

    splits = sub.add_parser("check-splits", help="Validate sealed-eval split manifest.")
    splits.add_argument("--manifest", type=Path, required=True)
    splits.add_argument(
        "--training-ids",
        type=Path,
        default=None,
        help="Optional JSON list of task IDs proposed for training.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "scrub":
        result = scrub_file(args.path)
        payload = {
            "clean": result.clean,
            "hits": [{"kind": h.kind, "start": h.start, "end": h.end} for h in result.hits],
        }
        if args.write:
            args.write.write_text(result.text, encoding="utf-8")
            payload["wrote"] = str(args.write)
        print(json.dumps(payload, indent=2))
        return 0 if result.clean else 2

    if args.command == "pre-upload-gate":
        result = run_pre_upload_gate(
            source_path=args.source,
            provider=args.provider,
            rights_manifest_path=args.rights_manifest,
            slice_id=args.slice_id,
            output_dir=args.output_dir,
            spend_ledger_path=None if args.allow_no_spend_check else args.spend_ledger,
            require_spend_headroom=not args.allow_no_spend_check,
        )
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.ok else 3

    if args.command == "audit-jsonl":
        report = audit_jsonl(args.path)
        text = json.dumps(report.to_dict(), indent=2)
        if args.write:
            args.write.parent.mkdir(parents=True, exist_ok=True)
            args.write.write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0

    if args.command == "check-splits":
        manifest = load_split_manifest(args.manifest)
        try:
            assert_manifest_disjoint(manifest)
        except ValueError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
            return 4
        contamination: list[str] = []
        if args.training_ids:
            ids = json.loads(Path(args.training_ids).read_text(encoding="utf-8"))
            contamination = validate_no_train_on_sealed(ids, manifest)
        ok = not contamination
        print(
            json.dumps(
                {
                    "ok": ok,
                    "n_train": len(manifest.train),
                    "n_dev": len(manifest.dev),
                    "n_sealed": len(manifest.sealed),
                    "n_time_forward": len(manifest.time_forward),
                    "n_deepswe_report": len(manifest.deepswe_report_split),
                    "contamination": contamination,
                },
                indent=2,
            )
        )
        return 0 if ok else 5

    parser.error(f"unknown command {args.command}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
