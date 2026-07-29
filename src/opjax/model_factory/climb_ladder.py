"""Living stage-climb ladder: sealed pass rates + deltas across Model Factory stages.

Writes ``data/model-factory/evals/climb_ladder.json`` and a short markdown mirror
under ``docs/model-factory/06-env-rl/climb-ladder.md``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_JSON = Path("data/model-factory/evals/climb_ladder.json")
DEFAULT_MD = Path("docs/model-factory/06-env-rl/climb-ladder.md")

# Seeded historical rungs (ground truth from stage memos / prior evals).
SEED_RUNGS: list[dict] = [
    {
        "id": "stage5-before-best-control",
        "stage": 5,
        "arm": "fewshot_rag",
        "split": "sealed_v1",
        "n": 4,
        "pass_rate": 0.583,
        "notes": "Best no-training control on sealed v1 (seeds 0–2 mean)",
        "source": "docs/model-factory/05-controlled-lora/results-v2.md",
    },
    {
        "id": "stage5-lora-sealed-v1",
        "stage": 5,
        "arm": "lora",
        "split": "sealed_v1",
        "n": 4,
        "pass_rate": 1.0,
        "notes": "Stage-5 LoRA claim win; kill not triggered",
        "source": "docs/model-factory/05-controlled-lora/results-v2.md",
        "delta_vs": "stage5-before-best-control",
    },
    {
        "id": "stage6-lora-baseline-sealed-v2",
        "stage": 6,
        "arm": "lora",
        "split": "sealed_v2",
        "n": 8,
        "pass_rate": 0.875,
        "notes": "Pre-RL Stage-5 LoRA on hardened sealed v2; fail sb-0013 only",
        "source": "data/model-factory/evals/sealed-v3-baseline-lora/summary_seed0.json",
        "delta_vs": "stage5-lora-sealed-v1",
    },
]


def _delta(rungs: list[dict], rung: dict) -> float | None:
    ref_id = rung.get("delta_vs")
    if not ref_id:
        return None
    for prev in rungs:
        if prev.get("id") == ref_id and "pass_rate" in prev:
            return float(rung["pass_rate"]) - float(prev["pass_rate"])
    return None


def load_ladder(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    return {
        "updated_at": None,
        "rungs": list(SEED_RUNGS),
        "profile": {},
    }


def upsert_rung(ladder: dict, rung: dict) -> dict:
    rungs = list(ladder.get("rungs") or [])
    replaced = False
    for i, existing in enumerate(rungs):
        if existing.get("id") == rung["id"]:
            rungs[i] = {**existing, **rung}
            replaced = True
            break
    if not replaced:
        rungs.append(rung)
    ladder["rungs"] = rungs
    ladder["updated_at"] = datetime.now(timezone.utc).isoformat()
    return ladder


def render_md(ladder: dict) -> str:
    lines = [
        "# Climb ladder — sealed pass-rate deltas",
        "",
        f"**Updated:** `{ladder.get('updated_at')}`",
        "",
        "Headline metric = SudarshanBench **sealed** pytest pass rate. "
        "Never train on sealed.",
        "",
        "| Rung | Stage | Arm | Split | n | Pass rate | Δ vs prior | Notes |",
        "|------|-------|-----|-------|---|-----------|------------|-------|",
    ]
    rungs = ladder.get("rungs") or []
    for r in rungs:
        d = _delta(rungs, r)
        d_s = f"{d:+.3f}" if d is not None else "—"
        lines.append(
            f"| `{r.get('id')}` | {r.get('stage')} | `{r.get('arm')}` | "
            f"`{r.get('split')}` | {r.get('n', '—')} | "
            f"**{float(r.get('pass_rate', 0)):.3f}** | {d_s} | "
            f"{r.get('notes', '')} |"
        )
    lines.extend(
        [
            "",
            "## Profile (latest RL run)",
            "",
        ]
    )
    profile = ladder.get("profile") or {}
    if not profile:
        lines.append("_No RL profile yet._")
    else:
        lines.append("```json")
        lines.append(json.dumps(profile, indent=2))
        lines.append("```")
    lines.append("")
    return "\n".join(lines)


def write_ladder(ladder: dict, *, json_path: Path, md_path: Path) -> None:
    # Attach computed deltas for consumers
    rungs = ladder.get("rungs") or []
    for r in rungs:
        d = _delta(rungs, r)
        if d is not None:
            r["delta"] = d
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(ladder, indent=2) + "\n")
    md_path.write_text(render_md(ladder))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Update Model Factory climb ladder")
    p.add_argument("--json-path", default=str(DEFAULT_JSON))
    p.add_argument("--md-path", default=str(DEFAULT_MD))
    p.add_argument("--seed-only", action="store_true", help="Write seeded rungs only")
    p.add_argument("--rung-json", help="Path to a rung object JSON to upsert")
    p.add_argument("--profile-json", help="Path to RL profile JSON to attach")
    args = p.parse_args(argv)

    json_path = Path(args.json_path)
    md_path = Path(args.md_path)
    ladder = load_ladder(json_path)
    if args.seed_only and not json_path.is_file():
        ladder = {"updated_at": None, "rungs": list(SEED_RUNGS), "profile": {}}
    if args.rung_json:
        rung = json.loads(Path(args.rung_json).read_text())
        ladder = upsert_rung(ladder, rung)
    if args.profile_json:
        ladder["profile"] = json.loads(Path(args.profile_json).read_text())
        ladder["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_ladder(ladder, json_path=json_path, md_path=md_path)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
