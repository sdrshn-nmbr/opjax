#!/usr/bin/env python3
"""Merge official TPU JAXBench records with local source inspection.

The TPU harness reports correctness and timings; whether a kernel is *actually*
Pallas — and whether it is just the reference handed back — are properties of the
emitted source, so both are read from the local eval dir and judged by
:mod:`opjax.model_factory.jaxbench_scoring`. Headline metric is `pallas_correct`
(correct AND real Pallas AND not a copy), with speedup as the secondary climb.

Usage:
  python scripts/jaxbench_merge_tpu.py \
      --arm base=data/model-factory/evals/jaxbench-fair-base:/tmp/tpu_base.jsonl \
      --arm lora=data/model-factory/evals/jaxbench-fair-lora:/tmp/tpu_lora.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opjax.model_factory.jaxbench_scoring import judge, reward, summarise  # noqa: E402

PALLAS = re.compile(r"\bpallas_call\s*\(")


def load_jsonl(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("workload"):
            out[rec["workload"]] = rec
    return out


def is_correct(rec: dict) -> bool:
    c = rec.get("correctness")
    if isinstance(c, dict) and "correct" in c:
        return bool(c["correct"])
    return bool(rec.get("correct"))


def speedup(rec: dict) -> float | None:
    """Baseline median / candidate median, however the harness spells it."""
    for key in ("speedup_vs_baseline", "speedup"):
        if isinstance(rec.get(key), (int, float)):
            return float(rec[key])
    b, k = rec.get("baseline"), rec.get("kernel") or rec.get("optimized")
    if isinstance(b, dict) and isinstance(k, dict):
        bm, km = b.get("median_ms"), k.get("median_ms")
        if isinstance(bm, (int, float)) and isinstance(km, (int, float)) and km > 0:
            return bm / km
    return None


def analyse(name: str, eval_dir: Path, jsonl: Path, bench: Path) -> dict:
    recs = load_jsonl(jsonl)
    kdir = eval_dir / "kernels"
    summary = json.loads((eval_dir / "summary.json").read_text())
    context = summary.get("prompt_context", "baseline")

    rows, verdicts = [], []
    for w, rec in recs.items():
        kpath = kdir / f"{w}.py"
        src = kpath.read_text() if kpath.exists() else ""
        bpath = bench / w / "baseline.py"
        v = judge(
            workload=w,
            candidate_src=src,
            baseline_src=bpath.read_text() if bpath.exists() else "",
            correct=is_correct(rec),
            uses_pallas=bool(PALLAS.search(src)),
            prompt_context=context,
            speedup=speedup(rec),
        )
        verdicts.append(v)
        rows.append(
            {
                "workload": w,
                "status": rec.get("status"),
                "correct": v.correct,
                "pallas": v.uses_pallas,
                "speedup": v.speedup,
                "similarity": v.similarity,
                "copied": v.copied,
                "credited": v.credited,
                "reward": reward(v),
                "error": (rec.get("error") or rec.get("reason") or "")[:120],
            }
        )
    credit = summarise(verdicts)
    sp = sorted(r["speedup"] for r in rows if r["credited"] and r["speedup"])
    return {
        "arm": name,
        "prompt_context": context,
        "n": len(rows),
        "n_correct": credit["n_correct_raw"],
        "n_copied": credit["n_copied"],
        "n_credited": credit["n_credited"],
        "n_pallas_attempt": sum(1 for r in rows if r["pallas"]),
        "n_pallas_correct": credit["n_pallas_credited"],
        "pallas_correct_workloads": [
            v.workload for v in verdicts if v.pallas_credited
        ],
        "median_speedup_of_correct": sp[len(sp) // 2] if sp else None,
        "n_speedup_ge_1_05": credit["n_credited_faster_than_baseline"],
        "best_speedup": credit["best_speedup_credited"],
        "mean_reward": credit["mean_reward"],
        "scorable": credit["scorable"],
        "by_status": dict(Counter(r["status"] for r in rows)),
        "rows": rows,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", action="append", required=True, help="name=evaldir:jsonl")
    p.add_argument("--jaxbench-root", default="/tmp/accelerator-agents")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    bench = Path(args.jaxbench_root) / "JAXBench" / "benchmark"
    arms = []
    for spec in args.arm:
        name, _, rest = spec.partition("=")
        ed, _, jl = rest.rpartition(":")
        arms.append(analyse(name, Path(ed), Path(jl), bench))

    hdr = ["| Signal | " + " | ".join(a["arm"] for a in arms) + " |",
           "|--------|" + "|".join(["---"] * len(arms)) + "|"]
    for label, key, bold in [
        ("Correct, raw (TPU official)", "n_correct", False),
        ("Copies of the shown reference, gated", "n_copied", False),
        ("Credited (correct, not a copy)", "n_credited", False),
        ("Attempts Pallas", "n_pallas_attempt", False),
        ("**pallas_correct (headline)**", "n_pallas_correct", True),
        ("Credited kernels >=1.05x baseline", "n_speedup_ge_1_05", False),
    ]:
        cells = []
        for a in arms:
            v = f"{a[key]}/{a['n']}"
            cells.append(f"**{v}**" if bold else v)
        hdr.append(f"| {label} | " + " | ".join(cells) + " |")
    med = [
        "| Median speedup of credited kernels | " + " | ".join(
            (f"{a['median_speedup_of_correct']:.3f}"
             if a["median_speedup_of_correct"] else "—") for a in arms) + " |",
        "| Mean reward | " + " | ".join(f"{a['mean_reward']:.3f}" for a in arms) + " |",
    ]
    table = "\n".join(hdr + med) + "\n\n"
    for a in arms:
        table += f"- `{a['arm']}` status: `{json.dumps(a['by_status'])}`"
        if a["pallas_correct_workloads"]:
            table += f" · pallas_correct: {a['pallas_correct_workloads']}"
        if not a["scorable"]:
            table += (
                f" · **prompt_context={a['prompt_context']}: diagnostic only** "
                "(reference was shown; copies gated)"
            )
        table += "\n"
    print(table)
    if args.out:
        Path(args.out).write_text(table)
        Path(args.out).with_suffix(".json").write_text(
            json.dumps(arms, indent=2) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
