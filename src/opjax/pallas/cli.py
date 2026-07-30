"""Command line interface for the governed Pallas track."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

from opjax.pallas.contracts import ContractError, contract_report, load_contracts
from opjax.pallas.corpus import (
    CorpusError,
    build_corpus,
    validate_corpus_release,
)
from opjax.pallas.evaluation import (
    EvaluationError,
    assert_checkout_ready,
    audit_evaluation,
    audit_lowering_evidence,
    evaluate_kernels,
)
from opjax.pallas.lowering import (
    LoweringEvidenceError,
    calibrate_lowering,
    validate_candidate_evidence,
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
    evaluate.add_argument(
        "--lowering-calibration",
        type=Path,
        required=True,
        help="Completed TPU lowering-control calibration",
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

    audit = commands.add_parser("audit-evaluation")
    audit.add_argument("--repo-root", type=Path, default=Path("."))
    audit.add_argument("--jaxbench-root", type=Path, required=True)
    audit.add_argument("--sample-run", type=Path, required=True)
    audit.add_argument("--evaluation-run", type=Path, required=True)
    audit.add_argument("--model-id", required=True)
    audit.add_argument("--arm", choices=["A", "B", "C", "D"], required=True)
    audit.add_argument(
        "--prompt-context",
        choices=["spec", "baseline"],
        default="spec",
    )

    calibrate = commands.add_parser("calibrate-lowering")
    calibrate.add_argument("--out-dir", type=Path, required=True)

    verify_lowering = commands.add_parser("verify-lowering")
    verify_lowering.add_argument("--calibration-root", type=Path, required=True)
    verify_lowering.add_argument("--candidate-root", type=Path, required=True)
    verify_lowering.add_argument("--kernel", type=Path, required=True)

    audit_lowering = commands.add_parser("audit-lowering")
    audit_lowering.add_argument("--repo-root", type=Path, default=Path("."))
    audit_lowering.add_argument("--jaxbench-root", type=Path, required=True)
    audit_lowering.add_argument("--sample-run", type=Path, required=True)
    audit_lowering.add_argument("--evaluation-run", type=Path, required=True)
    audit_lowering.add_argument("--sample-id", required=True)
    audit_lowering.add_argument("--calibration-root", type=Path, required=True)
    audit_lowering.add_argument("--candidate-root", type=Path, required=True)
    audit_lowering.add_argument("--output", type=Path, required=True)
    audit_lowering.add_argument("--model-id", required=True)
    audit_lowering.add_argument(
        "--arm",
        choices=["A", "B", "C", "D"],
        required=True,
    )
    audit_lowering.add_argument(
        "--prompt-context",
        choices=["spec", "baseline"],
        default="spec",
    )
    build = commands.add_parser("build-corpus")
    build.add_argument("--repo-root", type=Path, default=Path("."))
    build.add_argument("--source-checkout", action="append", default=[])
    build.add_argument("--verification-root", type=Path, action="append", default=[])
    build.add_argument("--calibration-root", type=Path)
    build.add_argument("--out-dir", type=Path, required=True)
    build.add_argument("--skip-hf", action="store_true")

    validate_corpus = commands.add_parser("validate-corpus")
    validate_corpus.add_argument("--corpus-root", type=Path, required=True)
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
                lowering_calibration=args.lowering_calibration,
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
        if args.command == "audit-evaluation":
            result = audit_evaluation(
                bundle=bundle,
                repo_root=args.repo_root,
                jaxbench_root=args.jaxbench_root,
                sample_run=args.sample_run,
                evaluation_run=args.evaluation_run,
                model_id=args.model_id,
                arm=args.arm,
                prompt_context=PromptContext(args.prompt_context),
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "calibrate-lowering":
            result = calibrate_lowering(
                out_dir=args.out_dir,
                repetitions=bundle.eval_policy["authenticity"][
                    "profile_repetitions"
                ],
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "verify-lowering":
            result = validate_candidate_evidence(
                calibration_root=args.calibration_root,
                candidate_root=args.candidate_root,
                expected_kernel_sha256=_sha256_file(args.kernel),
                expected_runtime=bundle.eval_policy["runtime"],
            )
            print(json.dumps(asdict(result), indent=2, sort_keys=True))
            return 0 if result.verified else 2
        if args.command == "audit-lowering":
            result = audit_lowering_evidence(
                bundle=bundle,
                repo_root=args.repo_root,
                jaxbench_root=args.jaxbench_root,
                sample_run=args.sample_run,
                evaluation_run=args.evaluation_run,
                sample_id=args.sample_id,
                calibration_root=args.calibration_root,
                candidate_root=args.candidate_root,
                output=args.output,
                model_id=args.model_id,
                arm=args.arm,
                prompt_context=PromptContext(args.prompt_context),
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "build-corpus":
            result = build_corpus(
                bundle=bundle,
                repo_root=args.repo_root,
                source_checkouts=_parse_source_checkouts(args.source_checkout),
                out_dir=args.out_dir,
                verification_roots=args.verification_root,
                calibration_root=args.calibration_root,
                include_hf=not args.skip_hf,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "validate-corpus":
            result = validate_corpus_release(args.corpus_root)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
    except (
        ContractError,
        CorpusError,
        EvaluationError,
        LoweringEvidenceError,
        SamplingError,
        OSError,
        ValueError,
    ) as exc:
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_source_checkouts(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        source_id, separator, raw_path = value.partition("=")
        if not separator or not source_id or not raw_path:
            raise ValueError(
                f"SOURCE_CHECKOUT_INVALID: expected source_id=path observed={value!r}"
            )
        if source_id in parsed:
            raise ValueError(f"SOURCE_CHECKOUT_DUPLICATE: {source_id}")
        parsed[source_id] = Path(raw_path)
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
