"""Fail-closed Tinker SFT runner for the governed Pallas experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.common import compute_bpb, compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

from opjax.pallas.contracts import ContractError, git_revision, load_contracts
from opjax.pallas.corpus import CorpusError, validate_corpus_release
from opjax.pallas.environment_corpus import (
    EnvironmentCorpusError,
    validate_environment_corpus,
)


class TrainingError(RuntimeError):
    """Training cannot continue without violating the frozen contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _append_event(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _tracked_dirty(repo_root: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode != 0 or bool(result.stdout.strip())


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_value(value.model_dump())
    if hasattr(value, "__dict__"):
        return _json_value(vars(value))
    return repr(value)


def _load_training_rows(
    *, corpus_root: Path, training_config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads((corpus_root / "manifest.json").read_text(encoding="utf-8"))
    environment_release = manifest.get("kind") == "pallas_environment_corpus_release"
    validation = (
        validate_environment_corpus(corpus_root)
        if environment_release
        else validate_corpus_release(corpus_root)
    )
    if validation["release_sha256"] != training_config["corpus_release_sha256"]:
        raise TrainingError(
            "CORPUS_RELEASE_MISMATCH: "
            f"expected={training_config['corpus_release_sha256']} "
            f"observed={validation['release_sha256']}"
        )
    if manifest.get("contract_sha256") != training_config["corpus_contract_sha256"]:
        raise TrainingError(
            "CORPUS_CONTRACT_MISMATCH: "
            f"expected={training_config['corpus_contract_sha256']} "
            f"observed={manifest.get('contract_sha256')}"
        )
    if not environment_release:
        readiness = manifest.get("sft_readiness", {})
        if readiness.get("arm_b_authorized") is not True or readiness.get("reasons") != []:
            raise TrainingError(f"ARM_B_UNAUTHORIZED: {readiness.get('reasons')}")
    dataset_path = corpus_root / "datasets" / "sft.jsonl"
    observed_dataset_hash = _sha256_file(dataset_path)
    if observed_dataset_hash != training_config["dataset_sha256"]:
        raise TrainingError(
            "SFT_DATASET_MISMATCH: "
            f"expected={training_config['dataset_sha256']} "
            f"observed={observed_dataset_hash}"
        )
    rows = _read_jsonl(dataset_path)
    if len(rows) != training_config["verified_rows"]:
        raise TrainingError(
            "SFT_ROW_COUNT_MISMATCH: "
            f"expected={training_config['verified_rows']} observed={len(rows)}"
        )
    row_ids = [row.get("row_id") for row in rows]
    if any(not isinstance(row_id, str) or not row_id for row_id in row_ids):
        raise TrainingError("SFT_ROW_ID_INVALID")
    if len(row_ids) != len(set(row_ids)):
        raise TrainingError("SFT_ROW_ID_DUPLICATE")
    return rows, manifest


def _training_order(
    rows: list[dict[str, Any]], *, seed: int, num_epochs: int
) -> list[int]:
    order: list[int] = []
    for epoch in range(num_epochs):
        indices = list(range(len(rows)))
        random.Random(seed + epoch).shuffle(indices)
        order.extend(indices)
    return order


def _prepare(
    *, config_root: Path, corpus_root: Path, repo_root: Path
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[tinker.Datum],
    list[int],
    Any,
]:
    bundle = load_contracts(config_root)
    training_config = bundle.experiment["training"]
    rows, corpus_manifest = _load_training_rows(
        corpus_root=corpus_root,
        training_config=training_config,
    )
    recommended_renderer = model_info.get_recommended_renderer_name(
        bundle.experiment["base_model"]
    )
    if recommended_renderer != training_config["renderer"]:
        raise TrainingError(
            "RENDERER_MISMATCH: "
            f"expected={training_config['renderer']} observed={recommended_renderer}"
        )
    tokenizer = get_tokenizer(bundle.experiment["base_model"])
    renderer = renderers.get_renderer(
        training_config["renderer"],
        tokenizer,
        model_name=bundle.experiment["base_model"],
    )
    train_on = renderers.TrainOnWhat(training_config["train_on"])
    datums = [
        conversation_to_datum(
            row["messages"],
            renderer,
            training_config["max_length"],
            train_on,
        )
        for row in rows
    ]
    order = _training_order(
        rows,
        seed=training_config["shuffle_seed"],
        num_epochs=training_config["num_epochs"],
    )
    if len(order) % training_config["batch_size"]:
        raise TrainingError(
            "SFT_BATCH_REMAINDER: "
            f"rows={len(order)} batch_size={training_config['batch_size']}"
        )
    supervised_tokens = sum(
        sum(float(weight) > 0 for weight in datum.loss_fn_inputs["weights"].tolist())
        for datum in datums
    )
    sequence_tokens = sum(datum.model_input.length + 1 for datum in datums)
    maximum_length = max(datum.model_input.length + 1 for datum in datums)
    preparation = {
        "schema_version": 1,
        "kind": "pallas_sft_preparation",
        "experiment_id": bundle.experiment["experiment_id"],
        "contract_sha256": bundle.sha256,
        "corpus_release_sha256": corpus_manifest["release_sha256"],
        "corpus_contract_sha256": corpus_manifest["contract_sha256"],
        "dataset_sha256": training_config["dataset_sha256"],
        "base_model": bundle.experiment["base_model"],
        "training": training_config,
        "row_ids": [rows[index]["row_id"] for index in order],
        "data": {
            "rows": len(rows),
            "sequence_tokens": sequence_tokens,
            "supervised_tokens": supervised_tokens,
            "maximum_sequence_tokens": maximum_length,
        },
        "packages": {
            "tinker": _package_version("tinker"),
            "tinker-cookbook": _package_version("tinker-cookbook"),
        },
        "opjax_revision": git_revision(repo_root),
    }
    preparation["sha256"] = _canonical_sha256(preparation)
    return preparation, rows, datums, order, tokenizer


def train_sft(
    *,
    config_root: Path,
    corpus_root: Path,
    repo_root: Path,
    out_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    preparation, rows, datums, order, tokenizer = _prepare(
        config_root=config_root,
        corpus_root=corpus_root,
        repo_root=repo_root,
    )
    if dry_run:
        return {"ok": True, "dry_run": True, "preparation": preparation}
    return run_prepared_sft(
        preparation=preparation,
        rows=rows,
        datums=datums,
        order=order,
        tokenizer=tokenizer,
        repo_root=repo_root,
        out_dir=out_dir,
    )


def _evaluate_datums(
    *,
    training: Any,
    datums: list[tinker.Datum],
    rows: list[dict[str, Any]],
    batch_size: int,
    loss_fn: str,
    tokenizer: Any,
) -> dict[str, Any]:
    if not datums or len(datums) != len(rows) or batch_size <= 0:
        raise TrainingError("VALIDATION_INPUT_INVALID")
    logprobs = []
    weights = []
    targets = []
    started = time.monotonic()
    for offset in range(0, len(datums), batch_size):
        batch = datums[offset : offset + batch_size]
        forward = training.forward(batch, loss_fn=loss_fn).result()
        logprobs.extend(output["logprobs"] for output in forward.loss_fn_outputs)
        weights.extend(datum.loss_fn_inputs["weights"] for datum in batch)
        targets.extend(datum.loss_fn_inputs["target_tokens"] for datum in batch)
    by_lane: dict[str, dict[str, Any]] = {}
    for lane in sorted({str(row["lane"]) for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["lane"] == lane]
        by_lane[lane] = {
            "sequences": len(indices),
            "tokens": sum(rows[index]["token_count"] for index in indices),
            "mean_nll": compute_mean_nll(
                [logprobs[index] for index in indices],
                [weights[index] for index in indices],
            ),
            "mean_bpb": compute_bpb(
                [logprobs[index] for index in indices],
                [weights[index] for index in indices],
                [targets[index] for index in indices],
                tokenizer,
            ),
        }
    return {
        "rows": len(rows),
        "tokens": sum(row["token_count"] for row in rows),
        "mean_nll": compute_mean_nll(logprobs, weights),
        "mean_bpb": compute_bpb(logprobs, weights, targets, tokenizer),
        "by_lane": by_lane,
        "elapsed_seconds": time.monotonic() - started,
    }


def run_prepared_sft(
    *,
    preparation: dict[str, Any],
    rows: list[dict[str, Any]],
    datums: list[tinker.Datum],
    order: list[int],
    tokenizer: Any,
    repo_root: Path,
    out_dir: Path,
    initial_state_path: str | None = None,
    parent_run_sha256: str | None = None,
    run_kind: str = "pallas_sft_run",
    gate: str = "G4",
    log_prefix: str = "PALLAS_G4_STEP",
    validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute an already validated and rendered SFT preparation."""
    if _tracked_dirty(repo_root):
        raise TrainingError(f"OPJAX_TRACKED_DIRTY: {repo_root}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise TrainingError(f"TRAINING_OUTPUT_NOT_EMPTY: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "preparation.json", preparation)
    training_config = preparation["training"]
    manifest = {
        "schema_version": 1,
        "kind": run_kind,
        "status": "running",
        "started_at": _utc_now(),
        "preparation_sha256": preparation["sha256"],
        "experiment_id": preparation["experiment_id"],
        "contract_sha256": preparation["contract_sha256"],
        "corpus_release_sha256": preparation["corpus_release_sha256"],
        "dataset_sha256": preparation["dataset_sha256"],
        "base_model": preparation["base_model"],
        "initial_state_path": initial_state_path,
        "parent_run_sha256": parent_run_sha256,
        "completed_steps": 0,
        "total_steps": len(order) // training_config["batch_size"],
        "checkpoint": None,
        "sampler_weights": None,
    }
    _write_json(out_dir / "manifest.json", manifest)
    service = tinker.ServiceClient(
        user_metadata={
            "project": "opjax",
            "gate": gate,
            "experiment_id": preparation["experiment_id"],
            "preparation_sha256": preparation["sha256"],
        }
    )
    supported_models = {
        model.model_name for model in service.get_server_capabilities().supported_models
    }
    if preparation["base_model"] not in supported_models:
        raise TrainingError(f"BASE_MODEL_UNSUPPORTED: {preparation['base_model']}")
    if initial_state_path is None:
        training = service.create_lora_training_client(
            base_model=preparation["base_model"],
            rank=training_config["lora_rank"],
            seed=training_config["training_seed"],
            train_mlp=True,
            train_attn=True,
            train_unembed=True,
            user_metadata={"arm": training_config["arm"]},
        )
    else:
        training = service.create_training_client_from_state(
            initial_state_path,
            user_metadata={"arm": training_config["arm"]},
        )
    info = _json_value(training.get_info())
    if (
        info.get("is_lora") is not True
        or info.get("lora_rank") != training_config["lora_rank"]
        or info.get("model_data", {}).get("model_name") != preparation["base_model"]
    ):
        raise TrainingError(f"TRAINING_CLIENT_IDENTITY_MISMATCH: {info}")
    manifest["training_client"] = info
    if validation is not None:
        manifest["validation"] = {
            "before": _evaluate_datums(
                training=training,
                datums=validation["datums"],
                rows=validation["rows"],
                batch_size=validation["batch_size"],
                loss_fn=training_config["loss_fn"],
                tokenizer=tokenizer,
            ),
            "after": None,
        }
        _write_json(out_dir / "validation.json", manifest["validation"])
    _write_json(out_dir / "manifest.json", manifest)
    optimizer = training_config["optimizer"]
    adam = tinker.AdamParams(
        learning_rate=training_config["learning_rate"],
        beta1=optimizer["beta1"],
        beta2=optimizer["beta2"],
        eps=optimizer["eps"],
        weight_decay=optimizer["weight_decay"],
        grad_clip_norm=optimizer["grad_clip_norm"],
    )
    batch_size = training_config["batch_size"]
    events_path = out_dir / "events.jsonl"
    for step, offset in enumerate(range(0, len(order), batch_size), start=1):
        batch_indices = order[offset : offset + batch_size]
        if len(batch_indices) != batch_size:
            raise TrainingError(
                f"INCOMPLETE_BATCH: step={step} size={len(batch_indices)}"
            )
        batch = [datums[index] for index in batch_indices]
        started = time.monotonic()
        forward_future = training.forward_backward(
            batch,
            loss_fn=training_config["loss_fn"],
        )
        optimizer_future = training.optim_step(adam)
        forward = forward_future.result()
        optimizer_result = optimizer_future.result()
        train_logprobs = [output["logprobs"] for output in forward.loss_fn_outputs]
        train_weights = [datum.loss_fn_inputs["weights"] for datum in batch]
        target_tokens = [datum.loss_fn_inputs["target_tokens"] for datum in batch]
        event = {
            "schema_version": 1,
            "kind": "sft_step",
            "step": step,
            "row_ids": [rows[index]["row_id"] for index in batch_indices],
            "sequence_tokens": sum(datum.model_input.length + 1 for datum in batch),
            "supervised_tokens": sum(
                sum(
                    float(weight) > 0
                    for weight in datum.loss_fn_inputs["weights"].tolist()
                )
                for datum in batch
            ),
            "train_mean_nll": compute_mean_nll(train_logprobs, train_weights),
            "train_mean_bpb": compute_bpb(
                train_logprobs,
                train_weights,
                target_tokens,
                tokenizer,
            ),
            "optimizer_metrics": _json_value(optimizer_result.metrics),
            "elapsed_seconds": time.monotonic() - started,
            "recorded_at": _utc_now(),
        }
        _append_event(events_path, event)
        manifest["completed_steps"] = step
        if step % training_config["checkpoint_every_steps"] == 0:
            checkpoint = training.save_state(
                f"step-{step:04d}",
                ttl_seconds=604800,
            ).result()
            manifest["checkpoint"] = {
                "step": step,
                "response": _json_value(checkpoint),
            }
        _write_json(out_dir / "manifest.json", manifest)
        print(
            f"{log_prefix} "
            f"step={step}/{manifest['total_steps']} "
            f"nll={event['train_mean_nll']:.6f} "
            f"bpb={event['train_mean_bpb']:.6f}",
            flush=True,
        )
    if validation is not None:
        manifest["validation"]["after"] = _evaluate_datums(
            training=training,
            datums=validation["datums"],
            rows=validation["rows"],
            batch_size=validation["batch_size"],
            loss_fn=training_config["loss_fn"],
            tokenizer=tokenizer,
        )
        _write_json(out_dir / "validation.json", manifest["validation"])
    final_state = training.save_state("final", ttl_seconds=None).result()
    sampler_weights = training.save_weights_for_sampler(
        "final", ttl_seconds=None
    ).result()
    manifest.update(
        {
            "status": "completed",
            "completed_at": _utc_now(),
            "final_state": _json_value(final_state),
            "sampler_weights": _json_value(sampler_weights),
            "artifacts": {
                "events.jsonl": _sha256_file(events_path),
                "preparation.json": _sha256_file(out_dir / "preparation.json"),
                **(
                    {"validation.json": _sha256_file(out_dir / "validation.json")}
                    if validation is not None
                    else {}
                ),
            },
        }
    )
    manifest["run_sha256"] = _canonical_sha256(manifest)
    _write_json(out_dir / "manifest.json", manifest)
    return {"ok": True, "out_dir": str(out_dir), "manifest": manifest}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="opjax-pallas-train")
    parser.add_argument("--config-root", type=Path, default=Path("config/pallas"))
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = train_sft(
            config_root=args.config_root,
            corpus_root=args.corpus_root,
            repo_root=args.repo_root,
            out_dir=args.out_dir,
            dry_run=args.dry_run,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        ContractError,
        CorpusError,
        EnvironmentCorpusError,
        TrainingError,
        OSError,
        ValueError,
    ) as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
