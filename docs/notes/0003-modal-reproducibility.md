# Modal Reproducibility

## Why This Shape

Modal gives us the fast pairing loop we want: `modal shell` drops us into the same image, volumes, mounts, and GPU request as the function we run. That avoids the usual remote-debug split where the notebook environment and training environment drift.

## One-Time Setup

Confirm profile and environment:

```bash
uv run modal profile current
uv run modal environment list
```

Expected profile is `conway`; expected environment is `main`.

Create the single project secret from local environment variables:

```bash
uv run modal secret create -e main opjax-secrets \
  HF_TOKEN="$HF_TOKEN" \
  ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  WANDB_API_KEY="$WANDB_API_KEY"
```

If the variables live in `.env`, use Modal's dotenv loader instead:

```bash
uv run modal secret create -e main opjax-secrets --from-dotenv .env
```

Create v2 volumes:

```bash
uv run modal volume create -e main --version 2 opjax-hf-cache-v2
uv run modal volume create -e main --version 2 opjax-data-v2
uv run modal volume create -e main --version 2 opjax-checkpoints-v2
```

Fallback to v1 if v2 blocks us:

```bash
uv run modal volume create -e main --version 1 opjax-hf-cache-v1
uv run modal volume create -e main --version 1 opjax-data-v1
uv run modal volume create -e main --version 1 opjax-checkpoints-v1
OPJAX_MODAL_VOLUME_VERSION=1 uv run modal run -e main -m opjax.remote.modal_app::gpu_smoke
```

## Validation Commands

GPU/JAX smoke:

```bash
uv run modal run -e main -m opjax.remote.modal_app::gpu_smoke
```

Interactive shell in the same image/function spec:

```bash
uv run modal shell -e main -m opjax.remote.modal_app::gpu_smoke --cmd=/bin/bash
```

Secret visibility smoke:

```bash
uv run modal run -e main -m opjax.remote.modal_app::secret_smoke
```

Volume write/read smoke:

```bash
uv run modal run -e main -m opjax.remote.modal_app::volume_smoke
```

Combined CPU-only smoke with printed JSON output:

```bash
uv run modal run -e main -m opjax.remote.modal_app::cpu_smoke
```

GPU smoke with printed JSON output:

```bash
uv run modal run -e main -m opjax.remote.modal_app::gpu_smoke_cli
```

## Reproducibility Rules

- Keep Modal image package versions exact in `opjax.remote.config.REMOTE_IMAGE_PACKAGES`.
- Keep local package dependencies pinned in `pyproject.toml`; `uv.lock` records the full resolution.
- Use Modal Secrets for credentials; do not commit tokens or write them into docs.
- Use v2 volumes by default, with `OPJAX_MODAL_VOLUME_VERSION=1` as the explicit fallback switch.
- Use module mode (`-m opjax.remote.modal_app`) so Modal packages the project as a Python package.
- Prefer `uv run modal` over bare `modal` for app commands so the CLI can import the local `opjax` package from the project environment.
