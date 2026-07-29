"""Evaluate Tinker samplers on SudarshanBench (before/after + control arms)."""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import tinker
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

CODE_FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)

DEFAULT_SYSTEM = (
    "You are a careful coding agent. Reply with a single Python code block "
    "containing the full fixed solution.py only. No tests, no prose."
)

PROMPT_RULES_SYSTEM = (
    DEFAULT_SYSTEM
    + " Follow tigerstyle: clarity over cleverness, minimal diffs, "
    "explicit over implicit, run tests before claiming done, never edit tests "
    "unless asked, prefer small pure functions, and avoid drive-by refactors."
)

STATIC_POLICY_SYSTEM = (
    DEFAULT_SYSTEM
    + " Do not emit tool calls, shell commands, or file paths outside solution.py. "
    "Return only valid Python for solution.py."
)


def _extract_code(text: str) -> str | None:
    matches = CODE_FENCE.findall(text)
    if matches:
        for m in matches:
            if "def " in m:
                return m.strip() + "\n"
        return matches[-1].strip() + "\n"
    if "def " in text:
        idx = text.find("def ")
        return text[idx:].strip() + "\n"
    return None


def _load_split_ids(splits_path: Path, split: str) -> list[str]:
    data = json.loads(splits_path.read_text())
    return list(data.get(split, []))


def _resolve_fixture(fixture_dir: str | Path, repo_root: Path) -> Path:
    """Resolve task fixture paths for local or cloud (/workspace) checkouts."""
    path = Path(fixture_dir)
    if not path.is_absolute():
        return repo_root / path
    if path.exists():
        return path
    # Cloud agents wrote absolute /workspace/... paths; remap when missing.
    text = str(path)
    if text.startswith("/workspace/"):
        remapped = repo_root / text.removeprefix("/workspace/")
        if remapped.exists():
            return remapped
    return path


def _fewshot_block(
    tasks_dir: Path,
    fixtures_root: Path,
    train_ids: list[str],
    repo_root: Path,
    k: int = 2,
) -> str:
    """Build few-shot exemplars from train-split fixtures only (never sealed)."""
    del fixtures_root  # task JSON carries fixture_dir; keep arg for call-site stability
    chunks: list[str] = []
    for tid in train_ids[:k]:
        task_path = tasks_dir / f"{tid}.json"
        if not task_path.exists():
            continue
        task = json.loads(task_path.read_text())
        fixture = _resolve_fixture(task["fixture_dir"], repo_root)
        sol = (fixture / "solution.py").read_text()
        prompt_md = (fixture / "PROMPT.md").read_text() if (fixture / "PROMPT.md").exists() else task.get("prompt", "")
        chunks.append(
            f"### Example {tid}\n{prompt_md.strip()}\n\nFixed solution.py:\n```python\n{sol.strip()}\n```"
        )
    if not chunks:
        return ""
    return "Here are train-split exemplars (style only; solve the new task):\n\n" + "\n\n".join(chunks) + "\n\n"


def _static_policy_ok(code: str | None, completion: str) -> bool:
    if not code:
        return False
    lowered = completion.lower()
    if "tool_call" in lowered or "<tool" in lowered:
        return False
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


async def _sample(
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    system: str,
    prompt: str,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    seed: int | None = None,
) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    model_input = renderer.build_generation_prompt(messages)
    params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        seed=seed,
    )
    result = await sampling_client.sample_async(
        prompt=model_input, num_samples=1, sampling_params=params
    )
    tokens = result.sequences[0].tokens
    return renderer.tokenizer.decode(tokens)


def _run_pytest(fixture_dir: Path) -> bool:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(fixture_dir)],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def _system_for_arm(arm: str) -> str:
    if arm == "prompt_rules":
        return PROMPT_RULES_SYSTEM
    if arm == "static_policy":
        return STATIC_POLICY_SYSTEM
    return DEFAULT_SYSTEM


def _weights_path_from_sampler(sampler_path: str) -> str | None:
    """Map sampler_weights checkpoint → training weights sibling (same run)."""
    if "/sampler_weights/" not in sampler_path:
        return None
    return sampler_path.replace("/sampler_weights/", "/weights/", 1)


async def _create_sampling_client(
    service: tinker.ServiceClient,
    *,
    model_name: str,
    sampler_path: str | None,
) -> tinker.SamplingClient:
    """Create a sampling client.

    Cold ``create_sampling_client(model_path=…/sampler_weights/…)`` can hang
    indefinitely for large Inkling LoRAs. Rematerialize from ``weights/…`` and
    ``save_weights_and_get_sampling_client`` is the reliable path.
    """
    if not sampler_path:
        return await service.create_sampling_client_async(base_model=model_name)

    weights_path = _weights_path_from_sampler(sampler_path)
    if weights_path:
        print(f"rematerializing LoRA sampling client from {weights_path}", flush=True)
        training = await service.create_training_client_from_state_async(weights_path)
        if hasattr(training, "save_weights_and_get_sampling_client_async"):
            return await training.save_weights_and_get_sampling_client_async()
        return await asyncio.to_thread(training.save_weights_and_get_sampling_client)

    return await service.create_sampling_client_async(model_path=sampler_path)


async def eval_arm(
    *,
    arm: str,
    model_name: str,
    sampler_path: str | None,
    task_ids: list[str],
    tasks_dir: Path,
    fixtures_root: Path,
    out_dir: Path,
    train_ids: list[str],
    seed: int,
    stage: int,
    split: str,
    commit_sha: str | None,
    repo_root: Path,
    sample_timeout_s: float = 600.0,
) -> dict:
    t0 = time.perf_counter()
    service = tinker.ServiceClient()
    sampling = await _create_sampling_client(
        service, model_name=model_name, sampler_path=sampler_path
    )
    print(f"sampling client ready in {time.perf_counter() - t0:.1f}s", flush=True)

    renderer_name = model_info.get_recommended_renderer_name(model_name)
    tok = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tok)
    system = _system_for_arm(arm)
    fewshot = (
        _fewshot_block(tasks_dir, fixtures_root, train_ids, repo_root)
        if arm == "fewshot_rag"
        else ""
    )

    results = []
    for i, tid in enumerate(task_ids, start=1):
        task = json.loads((tasks_dir / f"{tid}.json").read_text())
        src_fix = _resolve_fixture(task["fixture_dir"], repo_root)
        broken = (src_fix / "solution.py").read_text()
        prompt = (
            f"{fewshot}"
            f"{task['prompt']}\n\n"
            f"Current broken solution.py:\n```python\n{broken}\n```\n"
            "Return the complete fixed solution.py."
        )
        print(f"[{i}/{len(task_ids)}] sampling {tid} …", flush=True)
        task_t0 = time.perf_counter()
        try:
            completion = await asyncio.wait_for(
                _sample(sampling, renderer, system, prompt, seed=seed),
                timeout=sample_timeout_s,
            )
            code = _extract_code(completion)
        except Exception as exc:
            print(
                f"[{i}/{len(task_ids)}] {tid} error {type(exc).__name__} "
                f"({time.perf_counter() - task_t0:.1f}s)",
                flush=True,
            )
            results.append(
                {"id": tid, "pass": False, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue

        if arm == "static_policy" and not _static_policy_ok(code, completion):
            results.append(
                {
                    "id": tid,
                    "pass": False,
                    "extracted": bool(code),
                    "static_policy_reject": True,
                    "completion_preview": completion[:500],
                }
            )
            continue

        work = out_dir / tid
        if work.exists():
            shutil.rmtree(work)
        shutil.copytree(src_fix, work)
        if code:
            (work / "solution.py").write_text(code)
            passed = _run_pytest(work)
        else:
            passed = False
        print(
            f"[{i}/{len(task_ids)}] {tid} pass={passed} "
            f"({time.perf_counter() - task_t0:.1f}s)",
            flush=True,
        )
        results.append(
            {
                "id": tid,
                "pass": passed,
                "extracted": bool(code),
                "completion_preview": completion[:500],
            }
        )

    n_pass = sum(1 for r in results if r.get("pass"))
    wall_s = time.perf_counter() - t0
    return {
        "stage": stage,
        "arm": arm,
        "split": split,
        "seed": seed,
        "model_name": model_name,
        "sampler_path": sampler_path,
        "commit_sha": commit_sha,
        "harness": "opjax.model_factory.eval_sudarshanbench",
        "n": len(results),
        "n_pass": n_pass,
        "pass_rate": (n_pass / len(results)) if results else 0.0,
        "wall_seconds": wall_s,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }


async def async_main(args: argparse.Namespace) -> int:
    splits = Path(args.splits)
    split_data = json.loads(splits.read_text())
    task_ids = list(split_data.get(args.split, []))
    train_ids = list(split_data.get("train", []))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    commit_sha = None
    try:
        commit_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        pass

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    summary = {"split": args.split, "seed": args.seed, "arms": {}}

    repo_root = Path(args.repo_root).resolve()
    for arm in arms:
        sampler = args.sampler_path if arm == "lora" else None
        if arm == "lora" and not sampler:
            raise SystemExit("--sampler-path required when arms includes lora")
        report = await eval_arm(
            arm=arm,
            model_name=args.model_name,
            sampler_path=sampler,
            task_ids=task_ids,
            tasks_dir=Path(args.tasks_dir),
            fixtures_root=Path(args.fixtures_root),
            out_dir=out / f"{arm}-seed{args.seed}",
            train_ids=train_ids,
            seed=args.seed,
            stage=args.stage,
            split=args.split,
            commit_sha=commit_sha,
            repo_root=repo_root,
        )
        report_path = out / f"{arm}_seed{args.seed}_report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        summary["arms"][arm] = {
            "n_pass": report["n_pass"],
            "n": report["n"],
            "pass_rate": report["pass_rate"],
            "report": str(report_path),
        }
        print(json.dumps({"arm": arm, **summary["arms"][arm]}, indent=2))

    summary_path = out / f"summary_seed{args.seed}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("wrote", summary_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="thinkingmachines/Inkling")
    p.add_argument("--sampler-path", default=None, help="Tinker sampler path for lora arm")
    p.add_argument(
        "--arms",
        default="base",
        help="Comma-separated: base,prompt_rules,fewshot_rag,static_policy,lora",
    )
    p.add_argument("--split", default="sealed")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stage", type=int, default=5)
    p.add_argument(
        "--splits",
        default="docs/model-factory/02-sealed-eval/sudarshanbench/splits.json",
    )
    p.add_argument(
        "--tasks-dir",
        default="docs/model-factory/02-sealed-eval/sudarshanbench/tasks",
    )
    p.add_argument(
        "--fixtures-root",
        default="docs/model-factory/02-sealed-eval/sudarshanbench/fixtures",
    )
    p.add_argument("--out-dir", default="data/model-factory/evals/sealed-v2-before")
    p.add_argument(
        "--repo-root",
        default=".",
        help="Repo root for resolving relative fixture_dir paths (replaces /workspace).",
    )
    args = p.parse_args(argv)
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
