# Modal GPU Notes

## Role

Modal replaces Lambda as the cloud GPU layer for Phase 1 and Phase 2 experiments. TRC remains the TPU scaling target.

The canonical app lives at `opjax.remote.modal_app`, not in `scripts/`, because Modal's project-structure docs recommend module mode for package projects.

## First Gate

The first Modal gate is not Gemma. It is lower-level:

- Can we launch the exact image we will use for probes?
- Does `jax[cuda12]` import inside that image?
- Does JAX see the H100/H200 device?
- Can we enter the same environment with `modal shell` for interactive pairing?

## Commands

```bash
uv sync
uv run modal profile current
uv run modal run -e main -m opjax.remote.modal_app::gpu_smoke
uv run modal shell -e main -m opjax.remote.modal_app::gpu_smoke --cmd=/bin/bash
```

## Learning Hook

This isolates infrastructure failure from model failure. If Gemma loading fails later, we want to already know CUDA, JAX, and Modal image construction are sound.

## Configuration

- Workspace/profile: `conway`
- Modal environment: `main`
- App name: `opjax`
- GPU request: `H100`, allowing Modal's normal H200 auto-upgrade behavior
- Secret: `opjax-secrets`
- Default volumes: `opjax-hf-cache-v2`, `opjax-data-v2`, `opjax-checkpoints-v2`
- Fallback volumes: set `OPJAX_MODAL_VOLUME_VERSION=1` to use `*-v1` names
