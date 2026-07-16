"""Evaluate a Tinker sampler on SudarshanBench fixture tasks."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import tinker
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

CODE_FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


def _extract_code(text: str) -> str | None:
    matches = CODE_FENCE.findall(text)
    if matches:
        # Prefer a block that looks like a module definition
        for m in matches:
            if "def " in m:
                return m.strip() + "\n"
        return matches[-1].strip() + "\n"
    if "def " in text:
        # Fallback: from first def to end
        idx = text.find("def ")
        return text[idx:].strip() + "\n"
    return None


def _load_split_ids(splits_path: Path, split: str) -> list[str]:
    data = json.loads(splits_path.read_text())
    return list(data.get(split, []))


async def _sample(
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    prompt: str,
    *,
    max_tokens: int = 1024,
) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful coding agent. Reply with a single Python code block "
                "containing the full fixed solution.py only. No tests, no prose."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    model_input = renderer.build_generation_prompt(messages)
    params = types.SamplingParams(max_tokens=max_tokens, temperature=0.2, top_p=0.95)
    result = await sampling_client.sample_async(
        prompt=model_input, num_samples=1, sampling_params=params
    )
    # Decode with renderer
    tokens = result.sequences[0].tokens
    return renderer.tokenizer.decode(tokens)


def _run_pytest(fixture_dir: Path) -> bool:
    proc = subprocess.run(
        ["python", "-m", "pytest", "-q", str(fixture_dir)],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


async def eval_arm(
    *,
    model_name: str,
    sampler_path: str | None,
    task_ids: list[str],
    tasks_dir: Path,
    fixtures_root: Path,
    out_dir: Path,
) -> dict:
    service = tinker.ServiceClient()
    if sampler_path:
        sampling = service.create_sampling_client(model_path=sampler_path)
    else:
        sampling = service.create_sampling_client(base_model=model_name)

    renderer_name = model_info.get_recommended_renderer_name(model_name)
    tok = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tok)

    results = []
    for tid in task_ids:
        task = json.loads((tasks_dir / f"{tid}.json").read_text())
        src_fix = Path(task["fixture_dir"])
        if not src_fix.is_absolute():
            # repo-relative
            src_fix = Path("/workspace") / src_fix
        broken = (src_fix / "solution.py").read_text()
        prompt = (
            f"{task['prompt']}\n\n"
            f"Current broken solution.py:\n```python\n{broken}\n```\n"
            "Return the complete fixed solution.py."
        )
        try:
            completion = await _sample(sampling, renderer, prompt)
            code = _extract_code(completion)
        except Exception as exc:
            results.append(
                {"id": tid, "pass": False, "error": f"{type(exc).__name__}: {exc}"}
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
        results.append(
            {
                "id": tid,
                "pass": passed,
                "extracted": bool(code),
                "completion_preview": completion[:500],
            }
        )

    n_pass = sum(1 for r in results if r.get("pass"))
    return {
        "sampler_path": sampler_path,
        "model_name": model_name,
        "n": len(results),
        "n_pass": n_pass,
        "pass_rate": (n_pass / len(results)) if results else 0.0,
        "results": results,
    }


async def async_main(args: argparse.Namespace) -> int:
    splits = Path(args.splits)
    task_ids = _load_split_ids(splits, args.split)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    report = await eval_arm(
        model_name=args.model_name,
        sampler_path=args.sampler_path,
        task_ids=task_ids,
        tasks_dir=Path(args.tasks_dir),
        fixtures_root=Path(args.fixtures_root),
        out_dir=out / ("lora" if args.sampler_path else "base"),
    )
    report_path = out / ("lora_report.json" if args.sampler_path else "base_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("sampler_path", "n", "n_pass", "pass_rate")}, indent=2))
    print("wrote", report_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="thinkingmachines/Inkling")
    p.add_argument("--sampler-path", default=None, help="Tinker sampler path; omit for base")
    p.add_argument("--split", default="sealed")
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
    p.add_argument("--out-dir", default="data/model-factory/evals/sealed-v1")
    args = p.parse_args(argv)
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
