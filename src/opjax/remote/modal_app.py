# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
"""Canonical Modal app for opjax remote GPU work.

Invoke with module mode so Modal treats `opjax` as a package:

  modal run -m opjax.remote.modal_app::gpu_smoke
  modal shell -m opjax.remote.modal_app::gpu_smoke --cmd=/bin/bash
"""

from __future__ import annotations

import os
import json
import subprocess

import modal

from opjax.remote.config import (
    CHECKPOINT_DIR,
    CHECKPOINT_VOLUME_NAME,
    DATA_DIR,
    DATA_VOLUME_NAME,
    EXPECTED_SECRET_KEYS,
    GPU_TYPE,
    HF_CACHE_DIR,
    HF_CACHE_VOLUME_NAME,
    MODAL_APP_NAME,
    MODAL_ENVIRONMENT,
    MODAL_SECRET_NAME,
    MODAL_VOLUME_VERSION,
    PYTHON_VERSION,
    REMOTE_ENV,
    REMOTE_IMAGE_PACKAGES,
)


app = modal.App(MODAL_APP_NAME)

hf_cache_volume = modal.Volume.from_name(
    HF_CACHE_VOLUME_NAME,
    environment_name=MODAL_ENVIRONMENT,
    create_if_missing=True,
    version=MODAL_VOLUME_VERSION,
)
data_volume = modal.Volume.from_name(
    DATA_VOLUME_NAME,
    environment_name=MODAL_ENVIRONMENT,
    create_if_missing=True,
    version=MODAL_VOLUME_VERSION,
)
checkpoint_volume = modal.Volume.from_name(
    CHECKPOINT_VOLUME_NAME,
    environment_name=MODAL_ENVIRONMENT,
    create_if_missing=True,
    version=MODAL_VOLUME_VERSION,
)
opjax_secret = modal.Secret.from_name(
    MODAL_SECRET_NAME,
    environment_name=MODAL_ENVIRONMENT,
)

volumes = {
    HF_CACHE_DIR: hf_cache_volume,
    DATA_DIR: data_volume,
    CHECKPOINT_DIR: checkpoint_volume,
}

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("git")
    .uv_pip_install(*REMOTE_IMAGE_PACKAGES)
    .env(REMOTE_ENV)
    .add_local_python_source("opjax")
)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes=volumes,
    timeout=600,
)
def gpu_smoke() -> dict[str, object]:
    """Validate Modal GPU, CUDA-visible device, JAX backend, and mounts."""
    import jax  # type: ignore[import-not-found]

    nvidia_smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    return {
        "nvidia_smi": nvidia_smi,
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "jax_device_count": jax.device_count(),
        "volume_version": MODAL_VOLUME_VERSION,
        "mounts": _mount_status(),
        "env": {key: os.environ.get(key) for key in REMOTE_ENV},
    }


@app.function(
    image=image,
    volumes=volumes,
    secrets=[opjax_secret],
    timeout=120,
)
def secret_smoke() -> dict[str, object]:
    """Validate that the single opjax Modal secret has the expected keys."""
    return {
        "secret_name": MODAL_SECRET_NAME,
        "present_keys": {key: bool(os.environ.get(key)) for key in EXPECTED_SECRET_KEYS},
        "mounts": _mount_status(),
    }


@app.function(
    image=image,
    volumes=volumes,
    timeout=120,
)
def volume_smoke() -> dict[str, object]:
    """Validate volume mount names and write/read behavior without requiring a GPU."""
    marker = os.path.join(DATA_DIR, "modal_volume_smoke.txt")
    with open(marker, "w", encoding="utf-8") as f:
        f.write("opjax modal volume smoke\n")
    data_volume.commit()
    return {
        "volume_version": MODAL_VOLUME_VERSION,
        "written": marker,
        "mounts": _mount_status(),
    }


@app.function(
    image=image,
    volumes=volumes,
    secrets=[opjax_secret],
    timeout=300,
)
def hf_gemma4_discovery() -> dict[str, object]:
    """Discover visible Hugging Face Gemma 4 repos without downloading weights."""
    from huggingface_hub import HfApi  # type: ignore[import-not-found]

    api = HfApi(token=os.environ["HF_TOKEN"])
    try:
        whoami = api.whoami()
        search_terms = ("gemma-4", "gemma4", "functiongemma")
        candidates: dict[str, dict[str, object]] = {}

        for term in search_terms:
            for model in api.list_models(author="google", search=term, limit=50, full=True):
                model_id = model.modelId
                if model_id is None:
                    continue
                lower_id = model_id.lower()
                if "gemma" not in lower_id:
                    continue
                candidates[model_id] = _model_summary(model)
    except Exception as exc:  # noqa: BLE001 - remote probes should return JSON, not transport exceptions.
        return _safe_probe_error(exc, context="hf_gemma4_discovery")

    return {
        "ok": True,
        "hf_user": whoami.get("name") or whoami.get("fullname") or "unknown",
        "candidate_count": len(candidates),
        "candidates": [candidates[key] for key in sorted(candidates)],
    }


@app.function(
    image=image,
    volumes=volumes,
    secrets=[opjax_secret],
    timeout=300,
)
def hf_model_manifest(model_id: str) -> dict[str, object]:
    """Fetch a repo manifest for a candidate model without downloading blobs."""
    from huggingface_hub import HfApi  # type: ignore[import-not-found]

    api = HfApi(token=os.environ["HF_TOKEN"])
    try:
        info = api.model_info(model_id, files_metadata=True)
    except Exception as exc:  # noqa: BLE001 - remote probes should return JSON, not transport exceptions.
        error = _safe_probe_error(exc, context="hf_model_manifest")
        error["model_id"] = model_id
        return error
    siblings = sorted(info.siblings or [], key=lambda sibling: sibling.rfilename)
    files = []
    total_bytes = 0
    weight_bytes = 0
    for sibling in siblings:
        size = sibling.size or 0
        total_bytes += size
        if _is_weight_file(sibling.rfilename):
            weight_bytes += size
        files.append(
            {
                "path": sibling.rfilename,
                "size": size,
                "size_gb": round(size / 1_000_000_000, 3),
                "is_weight": _is_weight_file(sibling.rfilename),
            }
        )

    return {
        "ok": True,
        "model_id": info.modelId,
        "private": info.private,
        "gated": info.gated,
        "disabled": info.disabled,
        "library_name": getattr(info, "library_name", None),
        "pipeline_tag": getattr(info, "pipeline_tag", None),
        "tags": sorted(info.tags or []),
        "file_count": len(files),
        "total_size_gb": round(total_bytes / 1_000_000_000, 3),
        "weight_size_gb": round(weight_bytes / 1_000_000_000, 3),
        "files": files,
    }


@app.local_entrypoint()
def cpu_smoke() -> None:
    """Run CPU-only Modal validations and print JSON results."""
    print(
        json.dumps(
            {
                "secret_smoke": secret_smoke.remote(),  # type: ignore[attr-defined]
                "volume_smoke": volume_smoke.remote(),  # type: ignore[attr-defined]
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint()
def gpu_smoke_cli() -> None:
    """Run GPU/JAX Modal validation and print JSON results."""
    print(json.dumps(gpu_smoke.remote(), indent=2, sort_keys=True))  # type: ignore[attr-defined]


@app.local_entrypoint()
def hf_gemma4_discovery_cli() -> None:
    """Print visible Gemma 4 / FunctionGemma HF candidates."""
    print(json.dumps(hf_gemma4_discovery.remote(), indent=2, sort_keys=True))  # type: ignore[attr-defined]


@app.local_entrypoint()
def hf_model_manifest_cli(model_id: str) -> None:
    """Print a candidate model manifest without downloading weights."""
    print(json.dumps(hf_model_manifest.remote(model_id), indent=2, sort_keys=True))  # type: ignore[attr-defined]


def _mount_status() -> dict[str, bool]:
    return {
        HF_CACHE_DIR: os.path.isdir(HF_CACHE_DIR),
        DATA_DIR: os.path.isdir(DATA_DIR),
        CHECKPOINT_DIR: os.path.isdir(CHECKPOINT_DIR),
    }


def _model_summary(model: object) -> dict[str, object]:
    return {
        "model_id": getattr(model, "modelId", None),
        "private": getattr(model, "private", None),
        "gated": getattr(model, "gated", None),
        "disabled": getattr(model, "disabled", None),
        "downloads": getattr(model, "downloads", None),
        "likes": getattr(model, "likes", None),
        "library_name": getattr(model, "library_name", None),
        "pipeline_tag": getattr(model, "pipeline_tag", None),
        "tags": sorted(getattr(model, "tags", None) or []),
    }


def _is_weight_file(path: str) -> bool:
    return path.endswith((".safetensors", ".bin", ".ckpt", ".msgpack", ".npz"))


def _safe_probe_error(exc: Exception, *, context: str) -> dict[str, object]:
    message = str(exc)
    if "token" in message.lower():
        message = "Hugging Face authentication failed. Check HF_TOKEN in Modal secret."
    return {
        "ok": False,
        "context": context,
        "error_type": type(exc).__name__,
        "error": message,
    }
