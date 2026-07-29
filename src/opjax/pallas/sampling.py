"""Tinker sampling into immutable, resumable Pallas run artifacts."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import tinker
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from opjax.pallas.contracts import ContractBundle, git_revision, verify_source_checkout
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


@dataclass(frozen=True)
class SampleRequest:
    sample_id: str
    workload: str
    seed: int
    kernel_path: str


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


def _git_tracked_dirty(path: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode != 0 or bool(result.stdout.strip())


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sample_request(workload: str, seed: int) -> SampleRequest:
    return SampleRequest(
        sample_id=f"{workload}::seed={seed}",
        workload=workload,
        seed=seed,
        kernel_path=f"kernels/seed-{seed}/{workload}.py",
    )


def _select_values(
    *,
    requested: list[Any] | None,
    allowed: list[Any],
    error_code: str,
) -> list[Any]:
    if requested is None:
        return allowed
    if len(requested) != len(set(requested)):
        raise SamplingError(f"{error_code}_DUPLICATE: {requested}")
    unknown = sorted(set(requested) - set(allowed))
    if unknown:
        raise SamplingError(f"{error_code}_UNKNOWN: {unknown}")
    requested_set = set(requested)
    return [value for value in allowed if value in requested_set]


def _requested_samples(
    *,
    public_tasks: list[str],
    contract_seeds: list[int],
    workloads: list[str] | None,
    seeds: list[int] | None,
    limit: int | None,
) -> list[SampleRequest]:
    selected_workloads = _select_values(
        requested=workloads,
        allowed=public_tasks,
        error_code="WORKLOAD",
    )
    if limit is not None:
        if limit <= 0:
            raise SamplingError(f"LIMIT_INVALID: {limit}")
        selected_workloads = selected_workloads[:limit]
    selected_seeds = _select_values(
        requested=seeds,
        allowed=contract_seeds,
        error_code="SEED",
    )
    return [
        _sample_request(workload, seed)
        for workload in selected_workloads
        for seed in selected_seeds
    ]


def _attempt_seed(
    *,
    declared_seed: int,
    attempt: int,
    retry_seed_stride: int,
) -> int:
    return declared_seed + attempt * retry_seed_stride


def _validate_existing_rows(
    *,
    out_dir: Path,
    requests: list[SampleRequest],
    row_by_id: dict[str, dict[str, Any]],
) -> None:
    request_by_id = {request.sample_id: request for request in requests}
    for sample_id, row in row_by_id.items():
        request = request_by_id[sample_id]
        expected = {
            "workload": request.workload,
            "seed": request.seed,
            "kernel_path": request.kernel_path,
        }
        for key, value in expected.items():
            if row.get(key) != value:
                raise SamplingError(
                    "RESUME_SAMPLE_PROVENANCE_MISMATCH: "
                    f"{sample_id}: {key}: expected={value!r} "
                    f"observed={row.get(key)!r}"
                )
        kernel = out_dir / request.kernel_path
        if not kernel.is_file():
            raise SamplingError(f"RESUME_KERNEL_MISSING: {sample_id}: {kernel}")
        observed_hash = source_sha256(kernel.read_text(encoding="utf-8"))
        if row.get("code_sha256") != observed_hash:
            raise SamplingError(
                "RESUME_KERNEL_HASH_MISMATCH: "
                f"{sample_id}: expected={row.get('code_sha256')} "
                f"observed={observed_hash}"
            )


def _sampling_result(
    *,
    requests: list[SampleRequest],
    ordered_rows: list[dict[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    return {
        "ok": True,
        "n_requested": len(requests),
        "n_samples": len(ordered_rows),
        "n_parseable": sum(row["status"] == "sampled" for row in ordered_rows),
        "n_authentic": sum(
            row["inspection"]["authentic"] for row in ordered_rows
        ),
        "out_dir": str(out_dir),
    }


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
    requests: list[SampleRequest],
    opjax_revision: str,
    opjax_tracked_dirty: bool,
    jaxbench_tracked_dirty: bool,
    renderer_name: str,
) -> dict[str, Any]:
    value = {
        "contract_sha256": bundle.sha256,
        "opjax_revision": opjax_revision,
        "opjax_tracked_dirty": opjax_tracked_dirty,
        "jaxbench_revision": jaxbench_revision,
        "jaxbench_tracked_dirty": jaxbench_tracked_dirty,
        "base_model": bundle.experiment["base_model"],
        "model_path": model_path,
        "arm": arm,
        "prompt_context": prompt_context.value,
        "prompt": bundle.experiment["prompt"],
        "sampling": bundle.experiment["sampling"],
        "system_prompt_sha256": source_sha256(SYSTEM_PALLAS_REQUIRED),
        "renderer": renderer_name,
        "packages": {
            "tinker": _package_version("tinker"),
            "tinker-cookbook": _package_version("tinker-cookbook"),
        },
        "request": {
            "sample_ids": [request.sample_id for request in requests],
            "workloads": list(dict.fromkeys(request.workload for request in requests)),
            "seeds": list(dict.fromkeys(request.seed for request in requests)),
        },
    }
    value["sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return value


async def sample_kernels(
    *,
    bundle: ContractBundle,
    repo_root: Path,
    jaxbench_root: Path,
    out_dir: Path,
    arm: str,
    model_path: str | None,
    prompt_context: PromptContext,
    resume: bool,
    limit: int | None,
    workloads: list[str] | None,
    seeds: list[int] | None,
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
    public_tasks = list(bundle.splits["public_evaluation"]["task_ids"])
    sampling_config = bundle.experiment["sampling"]
    requests = _requested_samples(
        public_tasks=public_tasks,
        contract_seeds=list(sampling_config["seeds"]),
        workloads=workloads,
        seeds=seeds,
        limit=limit,
    )
    opjax_revision = git_revision(repo_root)
    opjax_tracked_dirty = _git_tracked_dirty(repo_root)
    jaxbench_revision = verify_source_checkout(bundle, "jaxbench", jaxbench_root)
    jaxbench_tracked_dirty = _git_tracked_dirty(jaxbench_root)
    if not dry_run and opjax_tracked_dirty:
        raise SamplingError(f"OPJAX_TRACKED_DIRTY: {repo_root}")
    if not dry_run and jaxbench_tracked_dirty:
        raise SamplingError(f"JAXBENCH_TRACKED_DIRTY: {jaxbench_root}")
    renderer_name = model_info.get_recommended_renderer_name(
        bundle.experiment["base_model"]
    )
    fingerprint = _sampling_fingerprint(
        bundle=bundle,
        jaxbench_revision=jaxbench_revision,
        model_path=model_path,
        arm=arm,
        prompt_context=prompt_context,
        requests=requests,
        opjax_revision=opjax_revision,
        opjax_tracked_dirty=opjax_tracked_dirty,
        jaxbench_tracked_dirty=jaxbench_tracked_dirty,
        renderer_name=renderer_name,
    )
    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "n_requested": len(requests),
            "n_workloads": len({request.workload for request in requests}),
            "n_seeds": len({request.seed for request in requests}),
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
            "schema_version": 2,
            "created_at": _utc_now(),
            "status": "sampling",
            "fingerprint": fingerprint,
            "generator": {"argv": list(sys.argv)},
        }
        _write_json(manifest_path, manifest)

    samples_path = out_dir / "samples.jsonl"
    rows = _load_jsonl(samples_path)
    row_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str):
            raise SamplingError("SAMPLE_ID_MISSING")
        if sample_id in row_by_id:
            raise SamplingError(f"SAMPLE_ID_DUPLICATE: {sample_id}")
        row_by_id[sample_id] = row
    requested_ids = [request.sample_id for request in requests]
    unexpected = sorted(set(row_by_id) - set(requested_ids))
    if unexpected:
        raise SamplingError(f"RESUME_SAMPLE_SET_MISMATCH: {unexpected}")
    _validate_existing_rows(
        out_dir=out_dir,
        requests=requests,
        row_by_id=row_by_id,
    )
    if len(row_by_id) == len(requested_ids):
        ordered_rows = [row_by_id[sample_id] for sample_id in requested_ids]
        manifest.update(
            {
                "status": "sampled",
                "completed_at": manifest.get("completed_at") or _utc_now(),
                "n_samples": len(ordered_rows),
            }
        )
        _write_json(manifest_path, manifest)
        print(
            f"PALLAS_SAMPLE_RESUME status=complete n_samples={len(ordered_rows)}",
            flush=True,
        )
        return _sampling_result(
            requests=requests,
            ordered_rows=ordered_rows,
            out_dir=out_dir,
        )

    tokenizer = get_tokenizer(bundle.experiment["base_model"])
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0),
        follow_redirects=True,
    )
    service = tinker.ServiceClient(
        http_client=http_client,
        max_retries=0,
    )
    sampling_client = await _sampling_client(
        service=service,
        base_model=bundle.experiment["base_model"],
        model_path=model_path,
    )
    semaphore = asyncio.Semaphore(sampling_config["max_concurrency"])

    async def sample_one(request: SampleRequest) -> dict[str, Any]:
        workload = request.workload
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
        print(
            "PALLAS_SAMPLE_START "
            f"sample_id={request.sample_id} workload={workload} seed={request.seed}",
            flush=True,
        )
        attempts: list[dict[str, Any]] = []
        completion = ""
        code = None
        sequence = None
        for attempt in range(sampling_config["max_retries"] + 1):
            effective_seed = _attempt_seed(
                declared_seed=request.seed,
                attempt=attempt,
                retry_seed_stride=sampling_config["retry_seed_stride"],
            )
            parameters = types.SamplingParams(
                max_tokens=sampling_config["max_tokens"],
                temperature=sampling_config["temperature"],
                top_p=sampling_config["top_p"],
                seed=effective_seed,
                stop=stops or None,
            )
            async with semaphore:
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
                    "sampling_seed": effective_seed,
                    "n_tokens": len(sequence.tokens),
                    "stop_reason": str(getattr(sequence, "stop_reason", "")),
                    "usable_code": bool(code and parses(code)),
                    "completion_sha256": source_sha256(completion),
                    "completion": completion,
                }
            )
            if code and parses(code):
                break
        assert sequence is not None
        inspection = inspect_pallas_source(code or "")
        candidate = code or completion
        row = {
            "schema_version": 2,
            "sample_id": request.sample_id,
            "workload": workload,
            "seed": request.seed,
            "kernel_path": request.kernel_path,
            "sampled_at": _utc_now(),
            "prompt_context": prompt_context.value,
            "prompt_sha256": source_sha256(prompt),
            "completion_sha256": source_sha256(completion),
            "completion": completion,
            "code_sha256": source_sha256(candidate),
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
        _atomic_write(out_dir / request.kernel_path, candidate)
        return row

    pending = [
        asyncio.create_task(sample_one(request))
        for request in requests
        if request.sample_id not in row_by_id
    ]
    for request in requests:
        if request.sample_id in row_by_id:
            print(
                f"PALLAS_SAMPLE_RESUME sample_id={request.sample_id} status=skipped",
                flush=True,
            )
    try:
        for completed_task in asyncio.as_completed(pending):
            row = await completed_task
            row_by_id[row["sample_id"]] = row
            ordered_rows = [
                row_by_id[sample_id]
                for sample_id in requested_ids
                if sample_id in row_by_id
            ]
            _write_jsonl(samples_path, ordered_rows)
            inspection = row["inspection"]
            print(
                "PALLAS_SAMPLE_DONE "
                f"sample_id={row['sample_id']} status={row['status']} "
                f"authentic={inspection['authentic']}",
                flush=True,
            )
    except BaseException:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        raise
    finally:
        await http_client.aclose()

    missing = sorted(set(requested_ids) - set(row_by_id))
    if missing:
        raise SamplingError(f"SAMPLES_INCOMPLETE: {missing}")
    ordered_rows = [row_by_id[sample_id] for sample_id in requested_ids]
    manifest.update(
        {
            "status": "sampled",
            "completed_at": _utc_now(),
            "n_samples": len(ordered_rows),
        }
    )
    _write_json(manifest_path, manifest)
    return _sampling_result(
        requests=requests,
        ordered_rows=ordered_rows,
        out_dir=out_dir,
    )
