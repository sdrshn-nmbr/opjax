"""Shared Modal configuration for reproducible remote runs."""

from __future__ import annotations

import os


MODAL_APP_NAME = "opjax"
MODAL_PROFILE = "conway"
MODAL_ENVIRONMENT = "main"
MODAL_SECRET_NAME = "opjax-secrets"

PYTHON_VERSION = "3.12"
GPU_TYPE = "H100"

# Modal Volumes v2 are the default. Set OPJAX_MODAL_VOLUME_VERSION=1 to use the
# v1 fallback names if v2 blocks us during an experiment.
MODAL_VOLUME_VERSION = int(os.environ.get("OPJAX_MODAL_VOLUME_VERSION", "2"))
if MODAL_VOLUME_VERSION not in {1, 2}:
    raise ValueError("OPJAX_MODAL_VOLUME_VERSION must be 1 or 2")

VOLUME_SUFFIX = f"v{MODAL_VOLUME_VERSION}"
HF_CACHE_VOLUME_NAME = f"opjax-hf-cache-{VOLUME_SUFFIX}"
DATA_VOLUME_NAME = f"opjax-data-{VOLUME_SUFFIX}"
CHECKPOINT_VOLUME_NAME = f"opjax-checkpoints-{VOLUME_SUFFIX}"

HF_CACHE_DIR = "/mnt/hf-cache"
DATA_DIR = "/mnt/data"
CHECKPOINT_DIR = "/mnt/checkpoints"

EXPECTED_SECRET_KEYS = ("HF_TOKEN", "ANTHROPIC_API_KEY", "WANDB_API_KEY")

REMOTE_IMAGE_PACKAGES = (
    "anthropic==0.97.0",
    "chex==0.1.91",
    "hf-transfer==0.1.9",
    "huggingface-hub==0.36.2",
    "jax[cuda12]==0.10.0",
    "jaxtyping==0.3.9",
    "orbax-checkpoint==0.11.39",
    "pillow==12.2.0",
    "wandb==0.26.1",
)

REMOTE_ENV = {
    "HF_HOME": HF_CACHE_DIR,
    "HF_HUB_CACHE": f"{HF_CACHE_DIR}/hub",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "OPJAX_DATA_DIR": DATA_DIR,
    "OPJAX_CHECKPOINT_DIR": CHECKPOINT_DIR,
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
}
