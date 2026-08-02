"""Capture calibrated compiler and profiler evidence for the incumbent."""

import argparse
import dataclasses
import json
from pathlib import Path

from opjax.pallas.contracts import load_contracts
from opjax.pallas.lowering import (
    calibrate_lowering,
    capture_candidate,
    validate_candidate_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-root", required=True, type=Path)
    parser.add_argument("--jaxbench-root", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    calibration_dir = args.output_dir / "calibration"
    candidate_dir = args.output_dir / "candidate"
    bundle = load_contracts(args.config_root)
    print("AR8P_LOWERING_START calibration", flush=True)
    calibrate_lowering(out_dir=calibration_dir, repetitions=3)
    print("AR8P_LOWERING_START candidate", flush=True)
    candidate = capture_candidate(
        jaxbench_root=args.jaxbench_root,
        workload="8p_GEMM",
        kernel=args.candidate,
        out_dir=candidate_dir,
        repetitions=3,
    )
    verdict = validate_candidate_evidence(
        calibration_root=calibration_dir,
        candidate_root=candidate_dir,
        expected_kernel_sha256=candidate["kernel_sha256"],
        expected_runtime=bundle.eval_policy["runtime"],
    )
    validation = dataclasses.asdict(verdict)
    (args.output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    print(f"AR8P_LOWERING_RESULT {json.dumps(validation, sort_keys=True)}", flush=True)
    return 0 if verdict.verified else 2


if __name__ == "__main__":
    raise SystemExit(main())
