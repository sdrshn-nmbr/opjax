"""Tinker sampling into immutable, resumable Pallas run artifacts."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tinker
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from opjax.pallas.contracts import ContractBundle, verify_source_checkout
from opjax.pallas.prompts import (
    SYSTEM_PALLAS_REQUIRED,
    extract_code,
    parses,
    render_prompt,
    source_sha256,
)
from opjax.pallas.scoring import PromptContext, inspect_pallas_source


class SamplingError(RuntimeError):
    """Sampling cannot continue without corrupting comparability."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _atomic_write(
        path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _weights_path(model_path: str) -> str:
    return model_path.replace("/sampler_weights/", "/weights/", 1)


async def _sampling_client(
    *,
    service: tinker.ServiceClient,
    base_model: str,
    model_path: str | None,
) -> tinker.SamplingClient:
    if model_path is None:
        return await service.create_sampling_client_async(base_model=base_model)
    training = await service.create_training_client_from_state_async(
        _weights_path(model_path)
    )
    return await training.save_weights_and_get_sampling_client_async()


def _sampling_fingerprint(
    *,
    bundle: ContractBundle,
    jaxbench_revision: str,
    model_path: str | None,
    arm: str,
    prompt_context: PromptContext,
) -> dict[str, Any]:
    value = {
        "contract_sha256": bundle.sha256,
        "jaxbench_revision": jaxbench_revision,
        "base_model": bundle.experiment["base_model"],
        "model_path": model_path,
        "arm": arm,
        "prompt_context": prompt_context.value,
        "prompt": bundle.experiment["prompt"],
        "sampling": bundle.experiment["sampling"],
        "system_prompt_sha256": source_sha256(SYSTEM_PALLAS_REQUIRED),
    }
    value["sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return value


async def sample_kernels(
    *,
    bundle: ContractBundle,
    jaxbench_root: Path,
    out_dir: Path,
    arm: str,
    model_path: str | None,
    prompt_context: PromptContext,
    resume: bool,
    limit: int | None,
    dry_run: bool,
    sample_timeout_seconds: float,
) -> dict[str, Any]:
    if arm not in bundle.experiment["arms"]:
        raise SamplingError(f"ARM_UNKNOWN: {arm}")
    if arm != "A" and model_path is None:
        raise SamplingError(f"MODEL_PATH_REQUIRED: arm={arm}")
    if prompt_context is PromptContext.BASELINE:
        print(
            "PALLAS_SAMPLE_DIAGNOSTIC prompt_context=baseline scorable=false",
            flush=True,
        )
    revision = verify_source_checkout(bundle, "jaxbench", jaxbench_root)
    fingerprint = _sampling_fingerprint(
        bundle=bundle,
        jaxbench_revision=revision,
        model_path=model_path,
        arm=arm,
        prompt_context=prompt_context,
    )
    tasks = list(bundle.splits["public_evaluation"]["task_ids"])
    if limit is not None:
        tasks = tasks[:limit]
    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "n_requested": len(tasks),
            "fingerprint": fingerprint,
        }
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not resume:
            raise SamplingError(f"RUN_ALREADY_EXISTS: {out_dir}")
        if manifest.get("fingerprint") != fingerprint:
            raise SamplingError("RESUME_FINGERPRINT_MISMATCH")
    else:
        manifest = {
            "schema_version": 1,
            "created_at": _utc_now(),
            "status": "sampling",
            "fingerprint": fingerprint,
        }
        _write_json(manifest_path, manifest)

    samples_path = out_dir / "samples.jsonl"
    rows = _load_jsonl(samples_path)
    completed = {row["workload"] for row in rows}
    kernels_dir = out_dir / "kernels"
    kernels_dir.mkdir(parents=True, exist_ok=True)

    renderer_name = model_info.get_recommended_renderer_name(
        bundle.experiment["base_model"]
    )
    tokenizer = get_tokenizer(bundle.experiment["base_model"])
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    service = tinker.ServiceClient()
    sampling_client = await _sampling_client(
        service=service,
        base_model=bundle.experiment["base_model"],
        model_path=model_path,
    )
    sampling_config = bundle.experiment["sampling"]
    for index, workload in enumerate(tasks):
        if workload in completed:
            print(f"PALLAS_SAMPLE_RESUME workload={workload} status=skipped", flush=True)
            continue
        baseline_path = (
            jaxbench_root / "JAXBench" / "benchmark" / workload / "baseline.py"
        )
        baseline_source = baseline_path.read_text(encoding="utf-8")
        prompt = render_prompt(
            workload=workload,
            baseline_source=baseline_source,
            prompt_context=prompt_context.value,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PALLAS_REQUIRED},
            {"role": "user", "content": prompt},
        ]
        model_input = renderer.build_generation_prompt(messages)
        stops = renderer.get_stop_sequences()
        print(f"PALLAS_SAMPLE_WORKLOAD workload={workload} status=started", flush=True)
        attempts: list[dict[str, Any]] = []
        completion = ""
        code = None
        sequence = None
        for attempt in range(sampling_config["max_retries"] + 1):
            parameters = types.SamplingParams(
                max_tokens=sampling_config["max_tokens"],
                temperature=sampling_config["temperature"],
                top_p=sampling_config["top_p"],
                seed=sampling_config["seeds"][0] + index + 1000 * attempt,
                stop=stops or None,
            )
            result = await asyncio.wait_for(
                sampling_client.sample_async(
                    prompt=model_input,
                    num_samples=1,
                    sampling_params=parameters,
                ),
                timeout=sample_timeout_seconds,
            )
            sequence = result.sequences[0]
            completion = renderer.tokenizer.decode(sequence.tokens)
            code = extract_code(completion)
            attempts.append(
                {
                    "attempt": attempt,
                    "n_tokens": len(sequence.tokens),
                    "stop_reason": str(getattr(sequence, "stop_reason", "")),
                    "usable_code": bool(code and parses(code)),
                }
            )
            if code and parses(code):
                break
        assert sequence is not None
        inspection = inspect_pallas_source(code or "")
        row = {
            "schema_version": 1,
            "workload": workload,
            "sampled_at": _utc_now(),
            "prompt_context": prompt_context.value,
            "prompt_sha256": source_sha256(prompt),
            "completion_sha256": source_sha256(completion),
            "code_sha256": source_sha256(code) if code else None,
            "n_tokens": len(sequence.tokens),
            "stop_reason": str(getattr(sequence, "stop_reason", "")),
            "attempts": attempts,
            "status": (
                "sampled"
                if code and parses(code)
                else "UNPARSEABLE_CODE"
                if code
                else "CODE_MISSING"
            ),
            "inspection": {
                "authentic": inspection.authentic,
                "reasons": list(inspection.reasons),
            },
        }
        if code:
            (kernels_dir / f"{workload}.py").write_text(code, encoding="utf-8")
        rows.append(row)
        _write_jsonl(samples_path, rows)
        print(
            f"PALLAS_SAMPLE_WORKLOAD workload={workload} status={row['status']} "
            f"authentic={inspection.authentic}",
            flush=True,
        )
    manifest.update({"status": "sampled", "completed_at": _utc_now()})
    _write_json(manifest_path, manifest)
    return {
        "ok": True,
        "n_requested": len(tasks),
        "n_samples": len(rows),
        "n_authentic": sum(row["inspection"]["authentic"] for row in rows),
        "out_dir": str(out_dir),
    }
