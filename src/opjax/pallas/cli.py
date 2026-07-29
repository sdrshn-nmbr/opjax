"""Command line interface for the governed Pallas track."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from opjax.pallas.contracts import ContractError, contract_report, load_contracts
from opjax.pallas.evaluation import (
    EvaluationError,
    assert_checkout_ready,
    evaluate_kernels,
)
from opjax.pallas.sampling import SamplingError, sample_kernels
from opjax.pallas.scoring import PromptContext, inspect_pallas_source

DEFAULT_CONFIG_ROOT = Path("config/pallas")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="opjax-pallas")
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-contracts")
    validate.add_argument("--jaxbench-root", type=Path)

    inspect = commands.add_parser("inspect-source")
    inspect.add_argument("path", type=Path)

    sample = commands.add_parser("sample")
    sample.add_argument("--repo-root", type=Path, default=Path("."))
    sample.add_argument("--jaxbench-root", type=Path, required=True)
    sample.add_argument("--out-dir", type=Path, required=True)
    sample.add_argument("--arm", choices=["A", "B", "C", "D"], default="A")
    sample.add_argument("--model-path")
    sample.add_argument(
        "--prompt-context",
        choices=["spec", "baseline"],
        default="spec",
    )
    sample.add_argument("--resume", action="store_true")
    sample.add_argument("--limit", type=int)
    sample.add_argument("--workload", action="append", dest="workloads")
    sample.add_argument("--seed", action="append", dest="seeds", type=int)
    sample.add_argument("--dry-run", action="store_true")
    sample.add_argument("--sample-timeout-seconds", type=float, default=600)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--repo-root", type=Path, default=Path("."))
    evaluate.add_argument("--jaxbench-root", type=Path, required=True)
    evaluate.add_argument(
        "--sample-run",
        type=Path,
        required=True,
        help="Completed opjax-pallas sample run",
    )
    evaluate.add_argument("--out-dir", type=Path, required=True)
    evaluate.add_argument("--model-id", required=True)
    evaluate.add_argument("--arm", choices=["A", "B", "C", "D"], required=True)
    evaluate.add_argument(
        "--prompt-context",
        choices=["spec", "baseline"],
        default="spec",
    )
    evaluate.add_argument("--resume", action="store_true")
    evaluate.add_argument("--dry-run", action="store_true")
    evaluate.add_argument("--timeout-seconds", type=float, default=900)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        bundle = load_contracts(args.config_root)
        if args.command == "validate-contracts":
            report = contract_report(bundle)
            if args.jaxbench_root:
                report["jaxbench_checkout"] = assert_checkout_ready(
                    bundle,
                    args.jaxbench_root,
                )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if args.command == "inspect-source":
            inspection = inspect_pallas_source(
                args.path.read_text(encoding="utf-8")
            )
            print(json.dumps(inspection.__dict__, indent=2, sort_keys=True))
            return 0 if inspection.authentic else 2
        if args.command == "sample":
            result = asyncio.run(
                sample_kernels(
                    bundle=bundle,
                    repo_root=args.repo_root,
                    jaxbench_root=args.jaxbench_root,
                    out_dir=args.out_dir,
                    arm=args.arm,
                    model_path=args.model_path,
                    prompt_context=PromptContext(args.prompt_context),
                    resume=args.resume,
                    limit=args.limit,
                    workloads=args.workloads,
                    seeds=args.seeds,
                    dry_run=args.dry_run,
                    sample_timeout_seconds=args.sample_timeout_seconds,
                )
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "evaluate":
            result = evaluate_kernels(
                bundle=bundle,
                repo_root=args.repo_root,
                jaxbench_root=args.jaxbench_root,
                sample_run=args.sample_run,
                out_dir=args.out_dir,
                model_id=args.model_id,
                arm=args.arm,
                prompt_context=PromptContext(args.prompt_context),
                resume=args.resume,
                dry_run=args.dry_run,
                timeout_seconds=args.timeout_seconds,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
    except (ContractError, EvaluationError, SamplingError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
