"""Run one JAXBench evaluation and bind it to observed TPU hardware."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import chex
import jax

from JAXBench.harness.evaluator import evaluate_kernel


def _hardware() -> dict[str, Any]:
    chex.assert_devices_available(1, "tpu", not_less_than=True)
    devices = jax.devices()
    return {
        "platforms": sorted({device.platform for device in devices}),
        "device_kinds": sorted(
            {getattr(device, "device_kind", "unknown") for device in devices}
        ),
        "device_count": len(devices),
        "process_count": jax.process_count(),
        "process_index": jax.process_index(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="opjax-pallas-jaxbench-runner")
    parser.add_argument("--workload", required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--tpu", choices=["v5e", "v6e", "auto"], required=True)
    parser.add_argument("--num-warmup", type=int, required=True)
    parser.add_argument("--num-iters", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    hardware = _hardware()
    result = evaluate_kernel(
        workload_name=args.workload,
        kernel_path=args.kernel,
        tpu=args.tpu,
        num_warmup=args.num_warmup,
        num_iters=args.num_iters,
    )
    print(json.dumps({"hardware": hardware, "result": result}, sort_keys=True))
    return 0 if result.get("status") == "correct" else 1


if __name__ == "__main__":
    sys.exit(main())
