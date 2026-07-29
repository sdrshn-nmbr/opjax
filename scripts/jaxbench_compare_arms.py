#!/usr/bin/env python3
"""Compare two JAXBench eval arms on scoring *and* on answer-shape diagnostics.

Raw `correct` conflates kernel quality with output-format compliance and with
how close a completion sits to the baseline it was handed. This prints both so a
base-vs-LoRA gap can be attributed rather than just observed.

Usage:
  python scripts/jaxbench_compare_arms.py \
      --arm lora=data/model-factory/evals/jaxbench-fair-lora \
      --arm base=data/model-factory/evals/jaxbench-fair-base \
      [--jaxbench-root /tmp/accelerator-agents] [--out docs/.../file.md]
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opjax.model_factory.jaxbench_scoring import (  # noqa: E402
    baseline_similarity,
    extract_workload_src,
    judge,
)

JUNK = re.compile(r"<\|[^|]*\|>")


def analyse(arm_dir: Path, jb_root: Path) -> dict:
    summary = json.loads((arm_dir / "summary.json").read_text())
    rows = summary["results"]
    bench = jb_root / "JAXBench" / "benchmark"
    # Read from the summary when present, so runs predating the flag are still
    # judged under the context they were actually sampled with.
    context = summary.get("prompt_context", "baseline")

    n = len(rows)
    stats: dict = {
        "n": n,
        "n_correct": sum(1 for r in rows if r.get("correct")),
        "n_pallas_correct": sum(
            1 for r in rows if r.get("correct") and r.get("uses_pallas")
        ),
        "n_attempts_pallas": sum(1 for r in rows if r.get("uses_pallas")),
        "by_status": dict(Counter(r.get("status") for r in rows)),
        "max_tokens": summary.get("max_tokens"),
        "n_truncated": sum(1 for r in rows if r.get("truncated_any")),
        "n_retried": sum(1 for r in rows if (r.get("n_attempts") or 1) > 1),
        "prompt_context": context,
    }

    parses = 0
    has_wl = 0
    n_copied = 0
    n_credited = 0
    sims: list[float] = []
    comp_lens: list[int] = []
    code_ratios: list[float] = []
    for r in rows:
        kpath = arm_dir / "kernels" / f"{r['workload']}.py"
        code = JUNK.sub("", kpath.read_text()) if kpath.exists() else ""
        try:
            ast.parse(code)
            parses += 1
        except SyntaxError:
            pass
        bpath = bench / r["workload"] / "baseline.py"
        baseline = bpath.read_text() if bpath.exists() else ""
        if extract_workload_src(code):
            has_wl += 1
            sim = baseline_similarity(code, baseline)
            if sim is not None:
                sims.append(sim)
        verdict = judge(
            workload=r["workload"],
            candidate_src=code,
            baseline_src=baseline,
            correct=bool(r.get("correct")),
            uses_pallas=bool(r.get("uses_pallas")),
            prompt_context=context,
        )
        n_copied += int(verdict.copied)
        n_credited += int(verdict.credited)
        if r.get("completion_chars"):
            comp_lens.append(r["completion_chars"])
            code_ratios.append((r.get("code_chars") or 0) / r["completion_chars"])

    def med(xs: list[float]) -> float | None:
        return round(sorted(xs)[len(xs) // 2], 3) if xs else None

    stats.update(
        {
            "n_parses": parses,
            "n_has_workload": has_wl,
            "n_copied": n_copied,
            "n_credited": n_credited,
            "median_completion_chars": med([float(x) for x in comp_lens]),
            "median_code_ratio": med(code_ratios),
            "median_sim_to_baseline": med(sims),
            "n_near_copy_of_baseline": sum(1 for s in sims if s > 0.9),
        }
    )
    return stats


ROWS = [
    ("Correct (raw / diagnostic)", "n_correct", "{}/{n}"),
    ("Copies gated", "n_copied", "{}/{n}"),
    ("Credited (correct, not a copy)", "n_credited", "{}/{n}"),
    ("**pallas_correct (headline)**", "n_pallas_correct", "**{}/{n}**"),
    ("Attempts Pallas", "n_attempts_pallas", "{}/{n}"),
    ("Syntactically valid code", "n_parses", "{}/{n}"),
    ("Defines `workload`", "n_has_workload", "{}/{n}"),
    ("Truncated on some attempt", "n_truncated", "{}/{n}"),
    ("Needed a retry", "n_retried", "{}/{n}"),
    ("Median completion chars", "median_completion_chars", "{}"),
    ("Median code/completion ratio", "median_code_ratio", "{}"),
    ("Median similarity to baseline", "median_sim_to_baseline", "{}"),
    ("Near-copies of baseline (>0.9)", "n_near_copy_of_baseline", "{}/{n}"),
]


def render(arms: dict[str, dict]) -> str:
    names = list(arms)
    out = ["| Signal | " + " | ".join(names) + " |",
           "|--------|" + "|".join(["---"] * len(names)) + "|"]
    for label, key, fmt in ROWS:
        cells = []
        for a in names:
            v = arms[a].get(key)
            cells.append("—" if v is None else fmt.format(v, n=arms[a]["n"]))
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    out.append("")
    for a in names:
        out.append(
            f"- `{a}` prompt_context=`{arms[a]['prompt_context']}` · by status: "
            f"`{json.dumps(arms[a]['by_status'])}`"
        )
    return "\n".join(out) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", action="append", required=True, help="name=path")
    p.add_argument("--jaxbench-root", default="/tmp/accelerator-agents")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    arms = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        arms[name] = analyse(Path(path), Path(args.jaxbench_root))
    table = render(arms)
    print(table)
    if args.out:
        Path(args.out).write_text(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
