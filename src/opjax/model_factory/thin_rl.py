"""Stage-6 thin on-policy GRPO loop on Tinker (spend-gated).

Warm-starts from a Stage-5 sampler path. Does **not** call Tinker unless
``--i-approve-spend`` is set (or ``dry_run=True`` for local scaffolding).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from opjax.model_factory.climb_ladder import (
    DEFAULT_JSON as CLIMB_JSON,
    DEFAULT_MD as CLIMB_MD,
    load_ladder,
    write_ladder,
)
from opjax.model_factory.reward_env import grade_solution_code

# Binding defaults from Stage-6 plan / Tinker RL hyperparams docs.
DEFAULT_SAMPLER = (
    "tinker://21e391ab-7c5d-573c-9477-16c93df81a08:train:0/sampler_weights/final"
)
# TrainingClient warm-start must use training weights, not sampler_weights.
DEFAULT_WEIGHTS = (
    "tinker://21e391ab-7c5d-573c-9477-16c93df81a08:train:0/weights/final"
)
DEFAULT_MODEL = "thinkingmachines/Inkling"


def _weights_path(sampler_or_weights: str) -> str:
    if "/sampler_weights/" in sampler_or_weights:
        return sampler_or_weights.replace("/sampler_weights/", "/weights/", 1)
    return sampler_or_weights


@dataclass
class ThinRLConfig:
    model_name: str = DEFAULT_MODEL
    sampler_path: str = DEFAULT_SAMPLER
    learning_rate: float = 1e-5
    group_size: int = 4
    max_steps: int = 15
    max_tokens: int = 512
    temperature: float = 1.0
    kl_penalty_coef: float = 0.05
    problems_per_step: int = 4
    seed: int = 0
    stage: int = 6
    # Abort if this many consecutive steps produce zero training datums.
    abort_after_idle_steps: int = 3
    fuel_splits: list[str] = field(default_factory=lambda: ["train", "dev"])


def _extract_code(text: str) -> str | None:
    import re

    fence = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)
    matches = fence.findall(text)
    if matches:
        for m in matches:
            if "def " in m:
                return m.strip() + "\n"
        return matches[-1].strip() + "\n"
    if "def " in text:
        return text[text.find("def ") :].strip() + "\n"
    return None


def plan_dict(cfg: ThinRLConfig, task_ids: list[str]) -> dict:
    """Emit a dry-run plan: expected shape, no network."""
    return {
        "mode": "thin_rl_plan",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "task_ids": task_ids,
        "substrate": "tinker",
        "warm_start": cfg.sampler_path,
        "loss_fn": "importance_sampling",
        "notes": [
            "Inkling RL on Tinker only — not Prime Hosted Training.",
            "Skip constant-reward groups.",
            "Kill if sealed does not improve vs Stage-5 LoRA.",
            "Requires --i-approve-spend to execute paid steps.",
        ],
        "est_tokens_rough": (
            cfg.max_steps
            * cfg.problems_per_step
            * cfg.group_size
            * cfg.max_tokens
            * 2  # sample + train
        ),
    }


async def _run_paid_loop(
    cfg: ThinRLConfig,
    task_ids: list[str],
    *,
    tasks_dir: Path,
    repo_root: Path,
    out_dir: Path,
) -> dict:
    """GRPO-style loop (Tinker Tutorial 104 shape). Spend-approved only."""
    import tinker
    import torch
    from tinker import TensorData, types
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    service = tinker.ServiceClient()
    weights_path = _weights_path(cfg.sampler_path)
    print(f"warm-start TrainingClient from {weights_path}", flush=True)
    training = await service.create_training_client_from_state_async(weights_path)
    renderer_name = model_info.get_recommended_renderer_name(cfg.model_name)
    tok = get_tokenizer(cfg.model_name)
    renderer = renderers.get_renderer(renderer_name, tok)
    adam = tinker.AdamParams(learning_rate=cfg.learning_rate, beta1=0.9, beta2=0.95)
    system = (
        "You are a careful coding agent. Reply with a single Python code block "
        "containing the full fixed solution.py only. No tests, no prose."
    )

    history: list[dict] = []
    t_run0 = time.perf_counter()
    idle_streak = 0
    aborted_reason: str | None = None
    for step in range(cfg.max_steps):
        t_step0 = time.perf_counter()
        print(f"thin-rl step {step + 1}/{cfg.max_steps}: save sampler …", flush=True)
        if hasattr(training, "save_weights_and_get_sampling_client_async"):
            sampling = await training.save_weights_and_get_sampling_client_async()
        else:
            sampling = await asyncio.to_thread(
                training.save_weights_and_get_sampling_client
            )
        step_tasks = [
            task_ids[(step * cfg.problems_per_step + i) % len(task_ids)]
            for i in range(min(cfg.problems_per_step, len(task_ids)))
        ]
        step_rewards: list[float] = []
        datums: list[tinker.Datum] = []
        n_degenerate = 0
        n_completion_tokens = 0
        for tid in step_tasks:
            task = json.loads((tasks_dir / f"{tid}.json").read_text())
            fixture = Path(task["fixture_dir"])
            if not fixture.is_absolute():
                fixture = repo_root / fixture
            broken = (fixture / "solution.py").read_text()
            prompt = (
                f"{task['prompt']}\n\nCurrent broken solution.py:\n```python\n"
                f"{broken}\n```\nReturn the complete fixed solution.py."
            )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
            model_input = renderer.build_generation_prompt(messages)
            params = types.SamplingParams(
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                top_p=0.95,
                seed=cfg.seed + step,
            )
            result = await asyncio.wait_for(
                sampling.sample_async(
                    prompt=model_input,
                    num_samples=cfg.group_size,
                    sampling_params=params,
                ),
                timeout=600,
            )
            rewards: list[float] = []
            tokens_g: list[list[int]] = []
            logprobs_g: list[list[float]] = []
            for seq in result.sequences:
                tokens_g.append(list(seq.tokens))
                n_completion_tokens += len(seq.tokens)
                lp = getattr(seq, "logprobs", None)
                if lp is None:
                    raise RuntimeError(
                        "sample sequences missing logprobs — cannot train with "
                        "importance_sampling"
                    )
                logprobs_g.append(list(lp))
                text = renderer.tokenizer.decode(seq.tokens)
                code = _extract_code(text) or ""
                graded = grade_solution_code(
                    task_id=tid,
                    code=code,
                    tasks_dir=tasks_dir,
                    repo_root=repo_root,
                )
                rewards.append(float(graded.reward))
            step_rewards.extend(rewards)
            mean_r = statistics.mean(rewards)
            advantages = [r - mean_r for r in rewards]
            if all(a == 0.0 for a in advantages):
                n_degenerate += 1
                continue
            # Tutorial 104 datum shape: prompt tokens get 0 advantage;
            # completion tokens get the group-relative advantage.
            ob_len = model_input.length - 1
            for tokens, logprobs, advantage in zip(
                tokens_g, logprobs_g, advantages, strict=True
            ):
                if not tokens or not logprobs:
                    continue
                train_input = model_input.append(
                    tinker.EncodedTextChunk(tokens=tokens[:-1])
                )
                target_tokens = [0] * ob_len + tokens
                padded_logprobs = [0.0] * ob_len + logprobs
                padded_advantages = [0.0] * ob_len + [advantage] * (
                    train_input.length - ob_len
                )
                datums.append(
                    tinker.Datum(
                        model_input=train_input,
                        loss_fn_inputs={
                            "target_tokens": TensorData.from_torch(
                                torch.tensor(target_tokens)
                            ),
                            "logprobs": TensorData.from_torch(
                                torch.tensor(padded_logprobs)
                            ),
                            "advantages": TensorData.from_torch(
                                torch.tensor(padded_advantages)
                            ),
                        },
                    )
                )

        trained = False
        if datums:
            fwd = await training.forward_backward_async(
                datums, loss_fn="importance_sampling"
            )
            opt = await training.optim_step_async(adam)
            await fwd.result_async()
            await opt.result_async()
            trained = True
            idle_streak = 0
        else:
            idle_streak += 1
        step_wall_s = time.perf_counter() - t_step0
        entry = {
            "step": step,
            "tasks": step_tasks,
            "mean_reward": statistics.mean(step_rewards) if step_rewards else 0.0,
            "n_rollouts": len(step_rewards),
            "n_datums": len(datums),
            "n_degenerate_groups": n_degenerate,
            "n_completion_tokens": n_completion_tokens,
            "step_wall_s": round(step_wall_s, 2),
            "cumulative_wall_s": round(time.perf_counter() - t_run0, 2),
            "trained": trained,
        }
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if idle_streak >= cfg.abort_after_idle_steps:
            aborted_reason = (
                f"no training signal for {idle_streak} consecutive steps "
                f"(fuel likely saturated; expand tasks or raise temperature)"
            )
            print(f"ABORT: {aborted_reason}", flush=True)
            break

    print("saving final sampler …", flush=True)
    save_fut = training.save_weights_for_sampler("final")
    save_res = await save_fut.result_async()
    state_fut = training.save_state("final")
    state_res = await state_fut.result_async()

    out_dir.mkdir(parents=True, exist_ok=True)
    trained_steps = sum(1 for h in history if h.get("trained"))
    profile = {
        "fuel_task_ids": task_ids,
        "fuel_splits": cfg.fuel_splits,
        "n_steps_ran": len(history),
        "n_steps_trained": trained_steps,
        "mean_reward_curve": [h.get("mean_reward") for h in history],
        "total_rollouts": sum(int(h.get("n_rollouts") or 0) for h in history),
        "total_datums": sum(int(h.get("n_datums") or 0) for h in history),
        "total_completion_tokens": sum(
            int(h.get("n_completion_tokens") or 0) for h in history
        ),
        "wall_s": round(time.perf_counter() - t_run0, 2),
        "aborted_reason": aborted_reason,
        "baseline_sealed_v2_pass_rate": 0.875,
        "kill_rule": "post_rl_sealed_pass_rate <= 0.875 under budget → kill Stage-6",
    }
    (out_dir / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    final_path = out_dir / "thin_rl_history.json"
    payload = {
        "config": asdict(cfg),
        "history": history,
        "profile": profile,
        "final_sampler": getattr(save_res, "path", None),
        "final_weights": getattr(state_res, "path", None),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    final_path.write_text(json.dumps(payload, indent=2) + "\n")
    # Attach profile to climb ladder for continuous visibility.
    ladder = load_ladder(CLIMB_JSON)
    ladder["profile"] = {
        **profile,
        "final_sampler": payload["final_sampler"],
        "final_weights": payload["final_weights"],
        "history_path": str(final_path),
    }
    ladder["updated_at"] = payload["checked_at"]
    write_ladder(ladder, json_path=CLIMB_JSON, md_path=CLIMB_MD)
    print(
        json.dumps(
            {
                "final_sampler": payload["final_sampler"],
                "final_weights": payload["final_weights"],
                "profile": profile,
                "climb_ladder": str(CLIMB_JSON),
            },
            indent=2,
        ),
        flush=True,
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage-6 thin RL (spend-gated)")
    p.add_argument("--dry-run", action="store_true", help="Emit plan JSON only")
    p.add_argument(
        "--i-approve-spend",
        action="store_true",
        help="Required to call Tinker and spend wallet balance",
    )
    p.add_argument(
        "--sampler-path",
        default=DEFAULT_SAMPLER,
        help=(
            "Stage-5 sampler or weights tinker:// path. "
            f"sampler_weights are mapped to weights (default weights: {DEFAULT_WEIGHTS})."
        ),
    )
    p.add_argument("--model-name", default=DEFAULT_MODEL)
    p.add_argument("--max-steps", type=int, default=15)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument(
        "--splits",
        default="docs/model-factory/02-sealed-eval/sudarshanbench/splits.json",
    )
    p.add_argument(
        "--tasks-dir",
        default="docs/model-factory/02-sealed-eval/sudarshanbench/tasks",
    )
    p.add_argument(
        "--split",
        default="train,dev",
        help="Comma-separated fuel splits (never sealed / time_forward)",
    )
    p.add_argument("--out-dir", default="data/model-factory/rl/thin-v1")
    p.add_argument("--repo-root", default=".")
    args = p.parse_args(argv)

    splits = json.loads(Path(args.splits).read_text())
    fuel_names = [s.strip() for s in args.split.split(",") if s.strip()]
    forbidden = {"sealed", "time_forward", "deepswe_report_split"}
    task_ids: list[str] = []
    for name in fuel_names:
        if name in forbidden:
            raise SystemExit(f"refusing to train on split={name}")
        ids = list(splits.get(name, []))
        if not ids:
            raise SystemExit(f"no task ids in split={name}")
        task_ids.extend(ids)
    # Dedupe preserve order
    seen: set[str] = set()
    task_ids = [t for t in task_ids if not (t in seen or seen.add(t))]

    cfg = ThinRLConfig(
        model_name=args.model_name,
        sampler_path=args.sampler_path,
        learning_rate=args.learning_rate,
        group_size=args.group_size,
        max_steps=args.max_steps,
        temperature=args.temperature,
        fuel_splits=fuel_names,
    )

    if args.dry_run or not args.i_approve_spend:
        plan = plan_dict(cfg, task_ids)
        plan["spend_approved"] = bool(args.i_approve_spend)
        if not args.i_approve_spend:
            plan["blocked"] = "pass --i-approve-spend after operator OK (and prefer --dry-run first)"
        print(json.dumps(plan, indent=2))
        return 0

    payload = asyncio.run(
        _run_paid_loop(
            cfg,
            task_ids,
            tasks_dir=Path(args.tasks_dir),
            repo_root=Path(args.repo_root),
            out_dir=Path(args.out_dir),
        )
    )
    print(json.dumps({"wrote": args.out_dir, "steps": len(payload["history"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
