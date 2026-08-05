"""Run the bounded Gate 4.1 static-feedback repair loop."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import httpx
import tinker
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from opjax.pallas.contracts import load_contracts
from opjax.pallas.environment import should_continue, verifier_feedback, verify_static
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


async def continue_tpu_repairs(
    *,
    config_root: Path,
    repo_root: Path,
    initial_run: Path,
    verification: Path,
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
    rows = _load_rows(initial_run / "results.jsonl")
    verifier = json.loads(verification.read_text(encoding="utf-8"))
    results = {result["task_id"]: result for result in verifier["results"]}
    bundle = load_contracts(config_root)
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
    out_dir.mkdir(parents=True)
    try:
        for task_index, row in enumerate(rows):
            source_result = results[row["task"]["task_id"]]
            for attempt in row["attempts"]:
                source_path = initial_run / attempt["kernel_path"]
                target_path = out_dir / attempt["kernel_path"]
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_path, target_path)
            if source_result["passed"] or len(row["attempts"]) >= 3:
                continue
            messages = [{"role": "user", "content": row["task"]["prompt"]}]
            for attempt in row["attempts"][:-1]:
                messages.extend(
                    [
                        {"role": "assistant", "content": attempt["completion"]},
                        {"role": "user", "content": attempt["feedback"]},
                    ]
                )
            final = row["attempts"][-1]
            feedback = verifier_feedback(source_result)
            messages.extend(
                [
                    {"role": "assistant", "content": final["completion"]},
                    {"role": "user", "content": feedback},
                ]
            )
            result = await client.sample_async(
                prompt=renderer.build_generation_prompt(messages),
                num_samples=1,
                sampling_params=types.SamplingParams(
                    max_tokens=sampling["max_tokens"],
                    temperature=0.0,
                    top_p=1.0,
                    seed=task_index + 3 * sampling["retry_seed_stride"],
                    stop=renderer.get_stop_sequences() or None,
                ),
            )
            completion = tokenizer.decode(result.sequences[0].tokens)
            verdict = verify_static(completion)
            attempt_index = len(row["attempts"]) + 1
            kernel_path = Path("kernels") / row["task"]["task_id"] / f"attempt-{attempt_index}.py"
            (out_dir / kernel_path).parent.mkdir(parents=True, exist_ok=True)
            (out_dir / kernel_path).write_text(verdict.code or completion, encoding="utf-8")
            row["attempts"].append(
                {
                    "attempt": attempt_index,
                    "passed": verdict.passed,
                    "stage": verdict.stage,
                    "feedback": verdict.feedback,
                    "evidence": verdict.evidence,
                    "kernel_path": str(kernel_path),
                    "completion": completion,
                    "trigger": {"stage": source_result["stage"], "feedback": feedback},
                }
            )
            print(
                f"G41_TPU_REPAIR arm={arm} task={row['task']['task_id']} "
                f"attempt={attempt_index} static_pass={verdict.passed}",
                flush=True,
            )
        _write_jsonl(out_dir / "results.jsonl", rows)
        manifest = {
            "schema_version": 1,
            "kind": "pallas_environment_tpu_repair_run",
            "arm": arm,
            "model_path": model_path,
            "max_attempts": 3,
            "task_count": len(rows),
            "static_pass": sum(row["attempts"][-1]["passed"] for row in rows),
            "source_verification": str(verification),
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
    parser.add_argument("--verification", type=Path)
    args = parser.parse_args(argv)
    try:
        values = vars(args)
        verification = values.pop("verification")
        result = asyncio.run(
            continue_tpu_repairs(verification=verification, **values)
            if verification is not None
            else run_repairs(**values)
        )
    except (EnvironmentSamplingError, ValueError) as exc:
        print(f"G41_ENVIRONMENT_ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
