"""Run the incumbent through three frozen full-shape JAXBench evaluations."""

import argparse
import gc
import hashlib
import json
import platform
import statistics
from pathlib import Path

import chex
import jax
import jaxlib
import libtpu

from JAXBench.harness.evaluator import evaluate_kernel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for index in range(3):
        print(f"AR8P_PROOF_START run={index}", flush=True)
        result = evaluate_kernel(
            workload_name="8p_GEMM",
            kernel_path=str(args.candidate),
            tpu="v5e",
            num_warmup=3,
            num_iters=20,
        )
        (args.output_dir / f"run_{index}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        runs.append(result)
        if result.get("status") != "correct":
            print(
                f"AR8P_PROOF_FAILURE run={index} status={result.get('status')} "
                f"error={result.get('error', 'no error text')}",
                flush=True,
            )
        else:
            speedup = (
                result["baseline"]["median_ms"] / result["kernel"]["median_ms"]
            )
            print(
                f"AR8P_PROOF_RESULT run={index} "
                f"baseline_ms={result['baseline']['median_ms']} "
                f"candidate_ms={result['kernel']['median_ms']} "
                f"speedup={speedup:.9f}",
                flush=True,
            )
        jax.clear_caches()
        gc.collect()

    correct = all(result.get("status") == "correct" for result in runs)
    profiler = all(
        result.get("baseline", {}).get("timing_method") == "device_profiler"
        and result.get("kernel", {}).get("timing_method") == "device_profiler"
        for result in runs
    )
    baseline_ms = [
        result["baseline"]["median_ms"] for result in runs if result.get("baseline")
    ]
    candidate_ms = [
        result["kernel"]["median_ms"] for result in runs if result.get("kernel")
    ]
    baseline_cv = (
        statistics.pstdev(baseline_ms) / statistics.mean(baseline_ms)
        if len(baseline_ms) == 3
        else None
    )
    candidate_cv = (
        statistics.pstdev(candidate_ms) / statistics.mean(candidate_ms)
        if len(candidate_ms) == 3
        else None
    )
    stable = (
        baseline_cv is not None
        and candidate_cv is not None
        and baseline_cv <= 0.1
        and candidate_cv <= 0.1
    )
    median_baseline = statistics.median(baseline_ms) if baseline_ms else None
    median_candidate = statistics.median(candidate_ms) if candidate_ms else None
    speedup = (
        median_baseline / median_candidate
        if median_baseline is not None
        and median_candidate is not None
        and median_candidate > 0
        else None
    )
    summary = {
        "schema_version": 1,
        "workload": "8p_GEMM",
        "jaxbench_revision": "6b6c44293c43976032ba12d2f72d6bebeaf2394f",
        "candidate": str(args.candidate),
        "candidate_sha256": _sha256(args.candidate),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "libtpu": libtpu.__version__,
            "chex": chex.__version__,
            "devices": [str(device) for device in jax.devices()],
        },
        "runs": 3,
        "correct": correct,
        "device_profiler": profiler,
        "baseline_samples_ms": baseline_ms,
        "candidate_samples_ms": candidate_ms,
        "baseline_cv": baseline_cv,
        "candidate_cv": candidate_cv,
        "stable": stable,
        "baseline_median_ms": median_baseline,
        "candidate_median_ms": median_candidate,
        "speedup": speedup,
        "eligible_timing": correct and profiler and stable,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"AR8P_PROOF_SUMMARY {json.dumps(summary, sort_keys=True)}", flush=True)
    return 0 if summary["eligible_timing"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
