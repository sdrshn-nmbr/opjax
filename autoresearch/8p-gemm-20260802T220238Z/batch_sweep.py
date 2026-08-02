"""Run isolated JAXBench candidates serially in one TPU-owning process."""

import argparse
import gc
import json
from pathlib import Path

import jax

from JAXBench.harness.evaluator import evaluate_kernel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--num-iters", type=int, default=20)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for candidate in args.candidates:
        print(f"AR8P_START candidate={candidate.name}", flush=True)
        result = evaluate_kernel(
            workload_name="8p_GEMM",
            kernel_path=str(candidate),
            tpu="v5e",
            num_warmup=args.num_warmup,
            num_iters=args.num_iters,
        )
        output = args.output_dir / f"{candidate.stem}.json"
        output.write_text(json.dumps(result, indent=2) + "\n")
        if result.get("status") == "correct":
            baseline_ms = result["baseline"]["median_ms"]
            candidate_ms = result["kernel"]["median_ms"]
            speedup = baseline_ms / candidate_ms
            print(
                "AR8P_RESULT "
                f"candidate={candidate.name} status=correct "
                f"baseline_ms={baseline_ms} candidate_ms={candidate_ms} "
                f"speedup={speedup:.9f}",
                flush=True,
            )
        else:
            error = result.get("error", "no error text")
            print(
                f"AR8P_FAILURE candidate={candidate.name} "
                f"status={result.get('status')} error={error}",
                flush=True,
            )
        jax.clear_caches()
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
