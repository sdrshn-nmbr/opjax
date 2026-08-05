"""Run the bounded Gate 4.1 static-feedback repair loop."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import tinker
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from opjax.pallas.contracts import load_contracts
from opjax.pallas.environment import should_continue, verify_static
from opjax.pallas.gate4_diagnostics import _git_tracked_dirty, _load_rows, _write_json, _write_jsonl
from opjax.pallas.sampling import _sampling_client


class EnvironmentSamplingError(RuntimeError):
    pass


async def run_repairs(
    *,
    config_root: Path,
    repo_root: Path,
    initial_run: Path,
    out_dir: Path,
    arm: str,
    model_path: str | None,
) -> dict[str, Any]:
    if _git_tracked_dirty(repo_root):
        raise EnvironmentSamplingError(f"OPJAX_TRACKED_DIRTY: {repo_root}")
    if out_dir.exists():
        raise EnvironmentSamplingError(f"ENVIRONMENT_RUN_EXISTS: {out_dir}")
    if (arm == "A") != (model_path is None):
        raise EnvironmentSamplingError("ARM_MODEL_PATH_INVALID")
    bundle = load_contracts(config_root)
    source_rows = [
        row for row in _load_rows(initial_run / "samples.jsonl")
        if row["task"]["tier"] == "near_heldout"
    ]
    if len(source_rows) != 4:
        raise EnvironmentSamplingError("HELDOUT_TASK_COUNT_INVALID")
    renderer_name = model_info.get_recommended_renderer_name(bundle.experiment["base_model"])
    tokenizer = get_tokenizer(bundle.experiment["base_model"])
    renderer = renderers.get_renderer(renderer_name, tokenizer, model_name=bundle.experiment["base_model"])
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0), follow_redirects=True)
    service = tinker.ServiceClient(http_client=http_client, max_retries=0)
    client = await _sampling_client(
        service=service,
        base_model=bundle.experiment["base_model"],
        model_path=model_path,
    )
    sampling = bundle.experiment["sampling"]
    rows: list[dict[str, Any]] = []
    out_dir.mkdir(parents=True)
    try:
        for task_index, source_row in enumerate(source_rows):
            task = source_row["task"]
            messages = [{"role": "user", "content": task["prompt"]}]
            completion = source_row["completion"]
            attempts: list[dict[str, Any]] = []
            while True:
                verdict = verify_static(completion)
                attempt_index = len(attempts) + 1
                kernel_path = Path("kernels") / task["task_id"] / f"attempt-{attempt_index}.py"
                candidate = verdict.code or completion
                (out_dir / kernel_path).parent.mkdir(parents=True, exist_ok=True)
                (out_dir / kernel_path).write_text(candidate, encoding="utf-8")
                attempts.append(
                    {
                        "attempt": attempt_index,
                        "passed": verdict.passed,
                        "stage": verdict.stage,
                        "feedback": verdict.feedback,
                        "evidence": verdict.evidence,
                        "kernel_path": str(kernel_path),
                        "completion": completion,
                    }
                )
                print(
                    f"G41_ENV_ATTEMPT arm={arm} task={task['task_id']} "
                    f"attempt={attempt_index} stage={verdict.stage} passed={verdict.passed}",
                    flush=True,
                )
                if not should_continue(attempts):
                    break
                messages.extend(
                    [
                        {"role": "assistant", "content": completion},
                        {"role": "user", "content": verdict.feedback},
                    ]
                )
                result = await client.sample_async(
                    prompt=renderer.build_generation_prompt(messages),
                    num_samples=1,
                    sampling_params=types.SamplingParams(
                        max_tokens=sampling["max_tokens"],
                        temperature=0.0,
                        top_p=1.0,
                        seed=task_index + attempt_index * sampling["retry_seed_stride"],
                        stop=renderer.get_stop_sequences() or None,
                    ),
                )
                completion = tokenizer.decode(result.sequences[0].tokens)
            rows.append({"task": task, "arm": arm, "attempts": attempts})
        _write_jsonl(out_dir / "results.jsonl", rows)
        manifest = {
            "schema_version": 1,
            "kind": "pallas_environment_repair_run",
            "arm": arm,
            "model_path": model_path,
            "max_attempts": 3,
            "task_count": len(rows),
            "static_pass": sum(row["attempts"][-1]["passed"] for row in rows),
        }
        _write_json(out_dir / "manifest.json", manifest)
        return manifest
    finally:
        await http_client.aclose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-environment-sample")
    parser.add_argument("--config-root", type=Path, default=Path("config/pallas"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--initial-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=["A", "B"], required=True)
    parser.add_argument("--model-path")
    args = parser.parse_args(argv)
    try:
        result = asyncio.run(run_repairs(**vars(args)))
    except (EnvironmentSamplingError, ValueError) as exc:
        print(f"G41_ENVIRONMENT_ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
