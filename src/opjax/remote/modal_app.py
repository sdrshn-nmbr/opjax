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
    GEMMA4_26B_A4B_IT,
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


@app.function(
    image=image,
    volumes=volumes,
    secrets=[opjax_secret],
    timeout=3600,
    cpu=4.0,
)
def gemma4_prestage_to_volume(
    source_gcs_path: str = GEMMA4_26B_A4B_IT,
    dest_subdir: str = "orbax/gemma4-26b-a4b-it",
) -> dict[str, object]:
    """Mirror a public GCS Orbax checkpoint into /mnt/hf-cache for fast reload.

    First-time GCS streaming of the 26B-A4B checkpoint took 19.79 min. After
    pre-staging, subsequent loads read from local Modal volume instead — orders
    of magnitude faster, persistent across containers. Idempotent: skips files
    that already exist with matching size.
    """
    import os
    import time
    import urllib.parse
    import urllib.request

    dest_dir = os.path.join(HF_CACHE_DIR, dest_subdir)
    os.makedirs(dest_dir, exist_ok=True)
    if not source_gcs_path.startswith("gs://"):
        return {"ok": False, "error": "source must be a gs:// path", "source": source_gcs_path}
    bucket_and_prefix = source_gcs_path[len("gs://") :]
    bucket, _, prefix = bucket_and_prefix.partition("/")
    list_url = (
        f"https://storage.googleapis.com/storage/v1/b/{bucket}/o"
        f"?prefix={prefix.rstrip('/')}/&maxResults=2000"
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(list_url, timeout=30) as resp:
            listing = json.loads(resp.read())
    except Exception as exc:  # noqa: BLE001
        return _safe_probe_error(exc, context="gemma4_prestage_to_volume.list") | {
            "list_url": list_url,
        }
    items = listing.get("items", [])

    downloaded = 0
    skipped = 0
    failures: list[dict[str, object]] = []
    total_bytes = 0
    for item in items:
        name = item["name"]
        size = int(item.get("size", 0))
        rel = name[len(prefix.rstrip("/")) + 1 :] if name.startswith(prefix.rstrip("/")) else name
        local_path = os.path.join(dest_dir, rel)
        os.makedirs(os.path.dirname(local_path) or dest_dir, exist_ok=True)
        if os.path.exists(local_path) and os.path.getsize(local_path) == size:
            skipped += 1
            continue
        media_url = (
            f"https://storage.googleapis.com/download/storage/v1/b/{bucket}/o/"
            f"{urllib.parse.quote(name, safe='')}?alt=media"
        )
        try:
            with urllib.request.urlopen(media_url, timeout=600) as resp:
                with open(local_path, "wb") as f:
                    while True:
                        chunk = resp.read(16 * 1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            downloaded += 1
            total_bytes += size
        except Exception as exc:  # noqa: BLE001
            failures.append({"name": name, "error": str(exc)})

    hf_cache_volume.commit()
    elapsed = time.time() - t0

    return {
        "ok": not failures,
        "source_gcs_path": source_gcs_path,
        "dest_dir": dest_dir,
        "elapsed_seconds": round(elapsed, 1),
        "elapsed_minutes": round(elapsed / 60.0, 2),
        "object_count": len(items),
        "downloaded": downloaded,
        "skipped": skipped,
        "bytes_downloaded": total_bytes,
        "gb_downloaded": round(total_bytes / 1e9, 3),
        "failures": failures,
    }


@app.function(
    image=image,
    volumes=volumes,
    timeout=180,
)
def gemma_import_smoke() -> dict[str, object]:
    """Force-import the gemma module surface on CPU to catch transitive-dep bugs.

    The first smoke run failed at `gm.nn.Gemma4_26B_A4B()` because the PyPI
    `dialog` package was missing `dialog.Format` (we now pin from git). This
    function exists to catch the next instance of that class of bug *before*
    paying a 20-min GPU GCS load.
    """
    import time

    t0 = time.time()
    errors: list[dict[str, object]] = []

    def _attempt(label: str, fn):
        try:
            value = fn()
            return {"ok": True, "label": label, "value": str(value)[:120]}
        except Exception as exc:  # noqa: BLE001
            errors.append({"label": label, "error_type": type(exc).__name__, "error": str(exc)[:300]})
            return {"ok": False, "label": label, "error_type": type(exc).__name__, "error": str(exc)[:300]}

    results = []
    results.append(_attempt("import dialog", lambda: __import__("dialog").Format))
    results.append(_attempt("import gemma", lambda: __import__("gemma")))
    results.append(_attempt("import gemma.gm", lambda: __import__("gemma.gm", fromlist=["gm"])))
    results.append(_attempt("gm.nn.Gemma4_26B_A4B", lambda: __import__("gemma.gm.nn", fromlist=["Gemma4_26B_A4B"]).Gemma4_26B_A4B))
    results.append(_attempt("gm.text.ChatSampler", lambda: __import__("gemma.gm.text", fromlist=["ChatSampler"]).ChatSampler))
    results.append(_attempt("gm.text.Gemma4Tokenizer", lambda: __import__("gemma.gm.text", fromlist=["Gemma4Tokenizer"]).Gemma4Tokenizer))
    results.append(_attempt("instantiate Gemma4_26B_A4B", lambda: __import__("gemma.gm.nn", fromlist=["Gemma4_26B_A4B"]).Gemma4_26B_A4B()))

    return {
        "ok": not errors,
        "elapsed_seconds": round(time.time() - t0, 2),
        "attempts": results,
        "error_count": len(errors),
        "errors": errors,
    }


@app.function(
    image=image,
    volumes=volumes,
    secrets=[opjax_secret],
    timeout=300,
)
def gemma4_metadata_probe(checkpoint_path: str = GEMMA4_26B_A4B_IT) -> dict[str, object]:
    """Read the Orbax checkpoint metadata for a Gemma 4 variant — no weight load.

    Schema discovery only: we want to know where the vision tower's
    `pos_emb` tensor lives in the param tree (the Tzafon scaling target).
    Orbax's manifest contains every leaf's shape + dtype + key path with no
    HBM cost, so a tiny CPU container reads it in seconds.

    The previous GPU-based `load_params` attempt OOM'd because GCS Orbax
    stores f32 (95.7 GB for 26B-A4B) and the loader preserves on-disk dtype.
    Metadata-only sidesteps that entirely.
    """
    import functools
    import time

    from etils import epath  # type: ignore[import-not-found]
    from orbax import checkpoint as ocp  # type: ignore[import-not-found]

    t0 = time.time()
    try:
        path = epath.Path(checkpoint_path)
        ckpt = ocp.StandardCheckpointer()
        meta = ckpt.metadata(path)
        if meta.item_metadata is None and path.joinpath("_CHECKPOINT_METADATA").exists():
            meta = ckpt.metadata(path / "default")
        if meta.item_metadata is None:
            raise ValueError(f"No item metadata found in {checkpoint_path}")
        tree = meta.item_metadata.tree
    except Exception as exc:  # noqa: BLE001 - return JSON, not transport exceptions.
        return _safe_probe_error(exc, context="gemma4_metadata_probe") | {
            "checkpoint_path": checkpoint_path,
        }
    elapsed = time.time() - t0

    def _walk(node: object, prefix: str = "") -> list[tuple[str, dict[str, object]]]:
        out: list[tuple[str, dict[str, object]]] = []
        if isinstance(node, dict):
            for k, v in node.items():
                out.extend(_walk(v, f"{prefix}.{k}" if prefix else k))
        elif hasattr(node, "shape") and hasattr(node, "dtype"):
            shape = [int(x) for x in node.shape]
            out.append(
                (
                    prefix,
                    {
                        "shape": shape,
                        "dtype": str(node.dtype),
                        "n": functools.reduce(lambda a, b: a * b, shape, 1) if shape else 1,
                    },
                )
            )
        return out

    leaves = _walk(tree)
    total_params = sum(int(info["n"]) for _, info in leaves)
    pos_emb_hits = [
        {"path": path, "shape": info["shape"], "dtype": info["dtype"], "n": info["n"]}
        for path, info in leaves
        if "pos_emb" in path
    ]
    vision_hits = sorted({
        path.split(".", 2)[0] + "." + path.split(".", 2)[1]
        for path, _info in leaves
        if path.startswith("SigLiPFromPatches_0") or path.startswith("vision_encoder") or "vision" in path[:40]
    })[:20]

    dtype_distribution: dict[str, int] = {}
    for _, info in leaves:
        dtype_distribution[info["dtype"]] = dtype_distribution.get(info["dtype"], 0) + 1

    top_level_keys = sorted(tree.keys()) if isinstance(tree, dict) else []

    return {
        "ok": True,
        "checkpoint_path": checkpoint_path,
        "metadata_seconds": round(elapsed, 2),
        "total_params": total_params,
        "total_params_billions": round(total_params / 1e9, 3),
        "weights_bf16_gb_if_cast": round(total_params * 2 / 1e9, 3),
        "leaf_count": len(leaves),
        "dtype_distribution": dtype_distribution,
        "top_level_keys": top_level_keys,
        "vision_subtree_keys": vision_hits,
        "pos_emb_matches": pos_emb_hits,
    }


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes=volumes,
    secrets=[opjax_secret],
    timeout=1800,
)
def gemma4_load_probe(checkpoint_path: str = GEMMA4_26B_A4B_IT) -> dict[str, object]:
    """Load a Gemma 4 Orbax checkpoint to GPU HBM at bf16 and report.

    Heavier than `gemma4_metadata_probe`: actually streams weights from GCS
    and casts the param tree to bf16 before restore, so 95.7 GB f32 → ~52 GB
    bf16 on H100 80GB. Used as a second-stage sanity check after metadata
    schema is known good.
    """
    import functools
    import time

    import jax  # type: ignore[import-not-found]
    import jax.numpy as jnp  # type: ignore[import-not-found]
    from etils import epath  # type: ignore[import-not-found]
    from orbax import checkpoint as ocp  # type: ignore[import-not-found]

    devices = [str(d) for d in jax.devices()]

    t0 = time.time()
    try:
        path = epath.Path(checkpoint_path)
        ckpt = ocp.StandardCheckpointer()
        meta = ckpt.metadata(path)
        if meta.item_metadata is None and path.joinpath("_CHECKPOINT_METADATA").exists():
            meta = ckpt.metadata(path / "default")
            path = path / "default"
        tree_meta = meta.item_metadata.tree

        def _to_bf16_sds(node: object) -> object:
            if hasattr(node, "shape") and hasattr(node, "dtype"):
                return jax.ShapeDtypeStruct(shape=node.shape, dtype=jnp.bfloat16)
            return node

        sds_tree = jax.tree.map(_to_bf16_sds, tree_meta, is_leaf=lambda x: hasattr(x, "shape"))
        params = ckpt.restore(path, sds_tree)
    except Exception as exc:  # noqa: BLE001 - return JSON, not transport exceptions.
        return _safe_probe_error(exc, context="gemma4_load_probe") | {
            "checkpoint_path": checkpoint_path,
            "jax_devices": devices,
        }
    load_seconds = time.time() - t0

    def _walk(node: object, prefix: str = "") -> list[tuple[str, dict[str, object]]]:
        out: list[tuple[str, dict[str, object]]] = []
        if isinstance(node, dict):
            for k, v in node.items():
                out.extend(_walk(v, f"{prefix}.{k}" if prefix else k))
        elif hasattr(node, "shape") and hasattr(node, "dtype"):
            shape = [int(x) for x in node.shape]
            out.append(
                (
                    prefix,
                    {
                        "shape": shape,
                        "dtype": str(node.dtype),
                        "n": functools.reduce(lambda a, b: a * b, shape, 1) if shape else 1,
                    },
                )
            )
        return out

    leaves = _walk(params)
    total_params = sum(int(info["n"]) for _, info in leaves)
    pos_emb_hits = [
        {"path": path, "shape": info["shape"], "dtype": info["dtype"], "n": info["n"]}
        for path, info in leaves
        if "pos_emb" in path
    ]
    top_level_keys = sorted(params.keys()) if isinstance(params, dict) else []
    has_vision_encoder = isinstance(params, dict) and (
        "vision_encoder" in params or "SigLiPFromPatches_0" in params
    )

    return {
        "ok": True,
        "checkpoint_path": checkpoint_path,
        "jax_devices": devices,
        "load_seconds": round(load_seconds, 2),
        "load_minutes": round(load_seconds / 60.0, 2),
        "total_params": total_params,
        "total_params_billions": round(total_params / 1e9, 3),
        "weights_bf16_gb": round(total_params * 2 / 1e9, 3),
        "leaf_count": len(leaves),
        "top_level_keys": top_level_keys,
        "has_vision_encoder": has_vision_encoder,
        "pos_emb_matches": pos_emb_hits,
    }


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    volumes=volumes,
    secrets=[opjax_secret],
    timeout=3600,
    scaledown_window=1800,
)
class Gemma4Inference:
    """Stateful Gemma 4 26B-A4B inference container.

    Loads weights once into HBM at bf16 (~52 GB). Exposes per-call methods for
    text/image inference, designed to support the eventual Tzafon scaling sweep
    where we mutate `vision_encoder.entry.pos_emb` in-flight between forward
    passes without reloading the model.
    """

    @modal.enter()
    def _load(self) -> None:
        import time

        import jax  # type: ignore[import-not-found]
        import jax.numpy as jnp  # type: ignore[import-not-found]
        from etils import epath  # type: ignore[import-not-found]
        from gemma import gm  # type: ignore[import-not-found]
        from orbax import checkpoint as ocp  # type: ignore[import-not-found]

        staged = os.path.join(HF_CACHE_DIR, "orbax", "gemma4-26b-a4b-it")
        if os.path.isdir(staged):
            path_str = staged
            print(f"[Gemma4Inference] using staged checkpoint: {staged}")
        else:
            path_str = GEMMA4_26B_A4B_IT
            print(f"[Gemma4Inference] staged checkpoint missing; falling back to GCS: {path_str}")

        t0 = time.time()
        path = epath.Path(path_str)
        ckpt = ocp.StandardCheckpointer()
        meta = ckpt.metadata(path)
        if meta.item_metadata is None and path.joinpath("_CHECKPOINT_METADATA").exists():
            meta = ckpt.metadata(path / "default")
            path = path / "default"
        tree_meta = meta.item_metadata.tree

        def _to_bf16_sds(node: object) -> object:
            if hasattr(node, "shape") and hasattr(node, "dtype"):
                return jax.ShapeDtypeStruct(shape=node.shape, dtype=jnp.bfloat16)
            return node

        sds_tree = jax.tree.map(_to_bf16_sds, tree_meta, is_leaf=lambda x: hasattr(x, "shape"))
        self.params = ckpt.restore(path, sds_tree)
        print(f"[Gemma4Inference] params restored in {time.time() - t0:.1f}s; building sampler")

        self.model = gm.nn.Gemma4_26B_A4B(text_only=False)
        self.sampler = gm.text.ChatSampler(
            model=self.model,
            params=self.params,
            cache_length=2048,
            max_out_length=256,
        )
        print(f"[Gemma4Inference] ready; total init {time.time() - t0:.1f}s")

    @modal.method()
    def smoke_text(self, prompt: str = "Hello, who are you?", max_new_tokens: int = 64) -> dict[str, object]:
        import time

        t0 = time.time()
        try:
            out = self.sampler.chat(prompt, max_new_tokens=max_new_tokens)
        except Exception as exc:  # noqa: BLE001
            return _safe_probe_error(exc, context="Gemma4Inference.smoke_text") | {"prompt": prompt}
        return {
            "ok": True,
            "prompt": prompt,
            "response": out,
            "seconds": round(time.time() - t0, 2),
        }

    @modal.method()
    def smoke_image(
        self,
        seed: int = 0,
        tier: str = "button",
        max_new_tokens: int = 96,
    ) -> dict[str, object]:
        """Generate one synthetic click task and ask Gemma 4 to emit a click action.

        End-to-end Phase 1A inference test: produces a click image via the
        synthetic curriculum, prompts the model in FunctionGemma action grammar,
        parses the model's response, and reports whether the click lands within
        the target radius.
        """
        import time

        from PIL import Image  # type: ignore[import-not-found]
        from opjax.actions import parse_action  # type: ignore[import-not-found]
        from opjax.synthetic.click import generate_click_task, verify_click_output  # type: ignore[import-not-found]

        t0 = time.time()
        task = generate_click_task(
            output_dir=os.path.join(DATA_DIR, "smoke_click"),
            seed=seed,
            tier=tier,  # type: ignore[arg-type]
        )
        image = Image.open(task.image_path).convert("RGB")

        prompt = (
            f"<|image|>\n\n"
            f"Look at the screenshot. {task.prompt}\n"
            f"\n"
            f"Respond with exactly one function call in this format:\n"
            f"<start_function_call>call:click{{x:<escape>X<escape>,y:<escape>Y<escape>}}<end_function_call>\n"
            f"where X and Y are integer pixel coordinates inside the {task.width}x{task.height} image."
        )

        try:
            response = self.sampler.chat(
                prompt,
                images=[image],
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            return _safe_probe_error(exc, context="Gemma4Inference.smoke_image") | {
                "task_id": task.task_id,
                "prompt": prompt,
            }

        verification = verify_click_output(task, response)
        parsed = None
        try:
            parsed_action = parse_action(response)
            parsed = {
                "type": type(parsed_action).__name__,
                "repr": repr(parsed_action),
            }
        except Exception as exc:  # noqa: BLE001
            parsed = {"parse_error": str(exc)[:200]}

        return {
            "ok": True,
            "task_id": task.task_id,
            "tier": task.tier,
            "target_xy": [task.target_x, task.target_y],
            "target_radius": task.target_radius,
            "image_size": [task.width, task.height],
            "natural_prompt": task.prompt,
            "model_response": response,
            "parsed_action": parsed,
            "verification": {
                "success": verification.success,
                "reward": verification.reward,
                "reason": verification.reason,
                "distance_px": verification.distance_px,
            },
            "seconds": round(time.time() - t0, 2),
        }

    @modal.method()
    def tzafon_sweep(
        self,
        scales: list[float] | None = None,
        n_per_tier: int = 3,
        tiers: list[str] | None = None,
        base_seed: int = 1000,
        max_new_tokens: int = 96,
    ) -> dict[str, object]:
        """Sweep `pos_emb` multiplicative scale factors and measure click accuracy.

        For each scale `k`, builds a new params tree with
        `vision_encoder.entry.pos_emb *= k`, constructs a fresh ChatSampler
        bound to those params, and runs `n_per_tier` click tasks across each
        tier. Reports per-scale per-tier accuracy. The original pos_emb tensor
        is captured at first call so subsequent scales always derive from the
        unperturbed checkpoint values, not chained scalings.
        """
        import time

        from PIL import Image  # type: ignore[import-not-found]
        from gemma import gm  # type: ignore[import-not-found]
        from opjax.actions import parse_action  # type: ignore[import-not-found]
        from opjax.synthetic.click import generate_click_task, verify_click_output  # type: ignore[import-not-found]

        if scales is None:
            scales = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
        if tiers is None:
            tiers = ["target", "distractors", "button"]

        if not hasattr(self, "_pos_emb_original"):
            self._pos_emb_original = self.params["vision_encoder"]["entry"]["pos_emb"]

        def _scaled_params(factor: float) -> dict[str, object]:
            return {
                **self.params,
                "vision_encoder": {
                    **self.params["vision_encoder"],
                    "entry": {
                        **self.params["vision_encoder"]["entry"],
                        "pos_emb": self._pos_emb_original * factor,
                    },
                },
            }

        def _prompt_for(task) -> str:
            return (
                f"<|image|>\n\n"
                f"Look at the screenshot. {task.prompt}\n"
                f"\n"
                f"Respond with exactly one function call in this format:\n"
                f"<start_function_call>call:click{{x:<escape>X<escape>,y:<escape>Y<escape>}}<end_function_call>\n"
                f"where X and Y are integer pixel coordinates inside the {task.width}x{task.height} image."
            )

        t_sweep = time.time()
        sweep_results: list[dict[str, object]] = []

        for scale in scales:
            scaled = _scaled_params(scale)
            sampler = gm.text.ChatSampler(model=self.model, params=scaled)
            t_scale = time.time()
            per_task: list[dict[str, object]] = []

            for tier_idx, tier in enumerate(tiers):
                for i in range(n_per_tier):
                    seed = base_seed + tier_idx * 1000 + i
                    task = generate_click_task(
                        output_dir=os.path.join(DATA_DIR, "tzafon_sweep", f"base{base_seed}"),
                        seed=seed,
                        tier=tier,  # type: ignore[arg-type]
                    )
                    image = Image.open(task.image_path).convert("RGB")
                    t0 = time.time()
                    try:
                        response = sampler.chat(
                            _prompt_for(task), images=[image], max_new_tokens=max_new_tokens
                        )
                    except Exception as exc:  # noqa: BLE001
                        per_task.append({
                            "task_id": task.task_id,
                            "tier": tier,
                            "seed": seed,
                            "error": str(exc)[:200],
                            "seconds": round(time.time() - t0, 2),
                        })
                        continue

                    verification = verify_click_output(task, response)
                    parsed_repr = None
                    try:
                        parsed_repr = repr(parse_action(response))
                    except Exception:  # noqa: BLE001
                        pass

                    per_task.append({
                        "task_id": task.task_id,
                        "tier": tier,
                        "seed": seed,
                        "model_response": response[:300],
                        "parsed_action": parsed_repr,
                        "success": verification.success,
                        "reward": verification.reward,
                        "distance_px": verification.distance_px,
                        "reason": verification.reason,
                        "seconds": round(time.time() - t0, 2),
                    })

            n_total = len(per_task)
            n_success = sum(1 for r in per_task if r.get("success"))
            per_tier_stats = {
                tier: {
                    "success": sum(
                        1 for r in per_task if r["tier"] == tier and r.get("success")
                    ),
                    "total": sum(1 for r in per_task if r["tier"] == tier),
                }
                for tier in tiers
            }
            sweep_results.append({
                "scale_factor": scale,
                "n_total": n_total,
                "n_success": n_success,
                "accuracy": round(n_success / n_total, 3) if n_total else 0.0,
                "per_tier": per_tier_stats,
                "scale_seconds": round(time.time() - t_scale, 2),
                "per_task": per_task,
            })

        return {
            "ok": True,
            "scales_swept": scales,
            "n_per_tier": n_per_tier,
            "tiers": tiers,
            "base_seed": base_seed,
            "total_sweep_seconds": round(time.time() - t_sweep, 2),
            "summary": [
                {
                    "scale_factor": r["scale_factor"],
                    "accuracy": r["accuracy"],
                    "n_success": r["n_success"],
                    "n_total": r["n_total"],
                    "scale_seconds": r["scale_seconds"],
                }
                for r in sweep_results
            ],
            "results": sweep_results,
        }


@app.local_entrypoint()
def gemma4_smoke_text_cli(prompt: str = "Hello, who are you?") -> None:
    """Run a text-only smoke inference through the loaded Gemma 4 26B-A4B model."""
    inst = Gemma4Inference()
    print(json.dumps(inst.smoke_text.remote(prompt), indent=2, sort_keys=True))  # type: ignore[attr-defined]


@app.local_entrypoint()
def gemma4_smoke_image_cli(seed: int = 0, tier: str = "button") -> None:
    """Run a click-task image inference; reports click accuracy vs ground truth."""
    inst = Gemma4Inference()
    print(json.dumps(inst.smoke_image.remote(seed, tier), indent=2, sort_keys=True))  # type: ignore[attr-defined]


@app.local_entrypoint()
def gemma4_tzafon_sweep_cli(n_per_tier: int = 3, base_seed: int = 1000) -> None:
    """Run the Tzafon pos_emb scaling sweep; prints the per-scale accuracy table."""
    inst = Gemma4Inference()
    result = inst.tzafon_sweep.remote(  # type: ignore[attr-defined]
        scales=[1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
        n_per_tier=n_per_tier,
        base_seed=base_seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


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


@app.local_entrypoint()
def gemma_import_smoke_cli() -> None:
    """Force-import the gemma module surface on CPU; quickly catches dep bugs."""
    print(json.dumps(gemma_import_smoke.remote(), indent=2, sort_keys=True))  # type: ignore[attr-defined]


@app.local_entrypoint()
def gemma4_prestage_to_volume_cli(
    source_gcs_path: str = GEMMA4_26B_A4B_IT,
    dest_subdir: str = "orbax/gemma4-26b-a4b-it",
) -> None:
    """Pre-stage a Gemma 4 GCS Orbax checkpoint to /mnt/hf-cache for fast reload."""
    print(json.dumps(gemma4_prestage_to_volume.remote(source_gcs_path, dest_subdir), indent=2, sort_keys=True))  # type: ignore[attr-defined]


@app.local_entrypoint()
def gemma4_metadata_probe_cli(checkpoint_path: str = GEMMA4_26B_A4B_IT) -> None:
    """Read Orbax metadata for a Gemma 4 variant on CPU; report pos_emb location."""
    print(json.dumps(gemma4_metadata_probe.remote(checkpoint_path), indent=2, sort_keys=True))  # type: ignore[attr-defined]


@app.local_entrypoint()
def gemma4_load_probe_cli(checkpoint_path: str = GEMMA4_26B_A4B_IT) -> None:
    """Load a Gemma 4 Orbax checkpoint on H100 at bf16 and report pos_emb location."""
    print(json.dumps(gemma4_load_probe.remote(checkpoint_path), indent=2, sort_keys=True))  # type: ignore[attr-defined]


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
    import traceback

    raw_message = str(exc)
    hint = None
    if context == "hf_gemma4_discovery" and "401" in raw_message and "token" in raw_message.lower():
        hint = "Hugging Face authentication failed. Check HF_TOKEN in Modal secret."
    return {
        "ok": False,
        "context": context,
        "error_type": type(exc).__name__,
        "error": raw_message[:1500],
        "traceback": traceback.format_exc()[-2500:],
        "hint": hint,
    }
