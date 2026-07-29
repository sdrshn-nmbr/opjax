#!/usr/bin/env python3
"""Run official JAXBench evaluate over every *.py kernel in a directory.

Intended to run ON the TPU VM with JAXBench on PYTHONPATH.
Skips macOS AppleDouble junk (._*).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--kernels-dir", default="kernels")
    p.add_argument("--tpu", default="v5e")
    p.add_argument("--num-warmup", type=int, default=3)
    p.add_argument("--num-iters", type=int, default=20)
    p.add_argument("--out", default="tpu_eval_results.json")
    args = p.parse_args()

    kernels_dir = Path(args.kernels_dir)
    paths = sorted(
        f
        for f in kernels_dir.glob("*.py")
        if f.is_file() and not f.name.startswith("._")
    )
    if not paths:
        print(f"no kernels in {kernels_dir}", file=sys.stderr)
        return 1

    # Do NOT import jax in this parent process — each subprocess needs exclusive
    # libtpu access. Parent only orchestrates `python -m JAXBench evaluate`.
    print(f"evaluate {len(paths)} kernels --tpu {args.tpu} (subprocess per workload)", flush=True)

    results = []
    for i, kpath in enumerate(paths, start=1):
        workload = kpath.stem
        print(f"=== [{i}/{len(paths)}] evaluating {workload} ===", flush=True)
        cmd = [
            sys.executable,
            "-m",
            "JAXBench",
            "evaluate",
            "--workload",
            workload,
            "--kernel",
            str(kpath),
            "--tpu",
            args.tpu,
            "--num-warmup",
            str(args.num_warmup),
            "--num-iters",
            str(args.num_iters),
            "--json",
        ]
        env = os.environ.copy()
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        row = {
            "workload": workload,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-2000:],
        }
        # Prefer last JSON object on stdout
        parsed = None
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    parsed = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        if parsed is None:
            try:
                parsed = json.loads(proc.stdout.strip())
            except Exception:
                parsed = {
                    "workload": workload,
                    "status": "error",
                    "correct": None,
                    "error": (proc.stderr or proc.stdout or "no json")[-500:],
                }
        row["result"] = parsed
        print(json.dumps(parsed, default=str)[:500], flush=True)
        results.append(row)
        Path(args.out).write_text(json.dumps(results, indent=2) + "\n")

    status_counts: dict[str, int] = {}
    n_correct = 0
    for r in results:
        st = (r.get("result") or {}).get("status", "error")
        status_counts[st] = status_counts.get(st, 0) + 1
        if (r.get("result") or {}).get("correct") is True:
            n_correct += 1
    summary = {
        "n": len(results),
        "correct": n_correct,
        "status_counts": status_counts,
    }
    print(json.dumps(summary, indent=2), flush=True)
    Path(args.out.replace(".json", "_summary.json")).write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
