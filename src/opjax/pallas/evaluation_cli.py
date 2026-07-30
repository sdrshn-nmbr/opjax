"""TPU evaluation CLI isolated from the Tinker sampling environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

from opjax.pallas.contracts import ContractError, load_contracts
from opjax.pallas.corpus import (
    CorpusError,
    record_verification_failure,
    verify_corpus_candidate,
)
from opjax.pallas.evaluation import (
    EvaluationError,
    audit_lowering_evidence,
    evaluate_kernels,
)
from opjax.pallas.lowering import (
    LoweringEvidenceError,
    calibrate_lowering,
    validate_candidate_evidence,
)
from opjax.pallas.scoring import PromptContext

DEFAULT_CONFIG_ROOT = Path("config/pallas")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="opjax-pallas-eval")
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)

    calibrate = commands.add_parser("calibrate-lowering")
    calibrate.add_argument("--out-dir", type=Path, required=True)

    verify = commands.add_parser("verify-lowering")
    verify.add_argument("--calibration-root", type=Path, required=True)
    verify.add_argument("--candidate-root", type=Path, required=True)
    verify.add_argument("--kernel", type=Path, required=True)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--repo-root", type=Path, default=Path("."))
    evaluate.add_argument("--jaxbench-root", type=Path, required=True)
    evaluate.add_argument("--sample-run", type=Path, required=True)
    evaluate.add_argument("--lowering-calibration", type=Path, required=True)
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

    audit = commands.add_parser("audit-lowering")
    audit.add_argument("--repo-root", type=Path, default=Path("."))
    audit.add_argument("--jaxbench-root", type=Path, required=True)
    audit.add_argument("--sample-run", type=Path, required=True)
    audit.add_argument("--evaluation-run", type=Path, required=True)
    audit.add_argument("--sample-id", required=True)
    audit.add_argument("--calibration-root", type=Path, required=True)
    audit.add_argument("--candidate-root", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--model-id", required=True)
    audit.add_argument("--arm", choices=["A", "B", "C", "D"], required=True)
    audit.add_argument(
        "--prompt-context",
        choices=["spec", "baseline"],
        default="spec",
    )
    verify_corpus = commands.add_parser("verify-corpus-candidate")
    verify_corpus.add_argument("--corpus-root", type=Path, required=True)
    verify_corpus.add_argument("--candidate-id", required=True)
    verify_corpus.add_argument("--source-checkout", type=Path, required=True)
    verify_corpus.add_argument("--calibration-root", type=Path, required=True)
    verify_corpus.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bundle = None
    try:
        bundle = load_contracts(args.config_root)
        if args.command == "calibrate-lowering":
            result = calibrate_lowering(
                out_dir=args.out_dir,
                repetitions=bundle.eval_policy["authenticity"][
                    "profile_repetitions"
                ],
            )
        elif args.command == "verify-lowering":
            verdict = validate_candidate_evidence(
                calibration_root=args.calibration_root,
                candidate_root=args.candidate_root,
                expected_kernel_sha256=_sha256_file(args.kernel),
                expected_runtime=bundle.eval_policy["runtime"],
            )
            result = asdict(verdict)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if verdict.verified else 2
        elif args.command == "evaluate":
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
        elif args.command == "audit-lowering":
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
        else:
            result = verify_corpus_candidate(
                bundle=bundle,
                corpus_root=args.corpus_root,
                candidate_id=args.candidate_id,
                source_checkout=args.source_checkout,
                calibration_root=args.calibration_root,
                out_dir=args.out_dir,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        ContractError,
        CorpusError,
        EvaluationError,
        LoweringEvidenceError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        if args.command == "verify-corpus-candidate" and bundle is not None:
            try:
                record_verification_failure(
                    bundle=bundle,
                    corpus_root=args.corpus_root,
                    candidate_id=args.candidate_id,
                    out_dir=args.out_dir,
                    error=exc,
                )
            except (CorpusError, OSError, ValueError):
                pass
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


if __name__ == "__main__":
    raise SystemExit(main())
