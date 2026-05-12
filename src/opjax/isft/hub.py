"""Hugging Face Hub utilities for the Phase 1C iSFT dataset.

Upload behavior is INTENTIONALLY explicit-only: there is no auto-push
side-channel. The CLI command `opjax push-isft-to-hub` is the single
entry point, and it requires an HF write token via env var (default
HF_TOKEN). This guards against accidental publication of in-progress
datasets.

The dataset itself lives at `sudarshan/opjax-click-isft` by default but
is configurable via the CLI flag; the repo will be created on first push
if it doesn't exist yet.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, create_repo


def push_dataset_to_hub(
    *,
    local_dir: str | Path,
    repo_id: str,
    token: str,
    private: bool = False,
    commit_message: str = "phase 1c: opjax click iSFT/RGT dataset",
) -> dict[str, object]:
    """Upload an iSFT dataset directory to HF Hub as a dataset repo.

    `local_dir` should be the output_dir from `build_real_click_isft_dataset`,
    which contains `records.jsonl`, `images/`, and (if present)
    `skipped_attempts.json`. Everything in that directory is uploaded
    verbatim — no transformations.
    """
    local_path = Path(local_dir)
    if not local_path.exists():
        raise FileNotFoundError(f"local_dir does not exist: {local_path}")
    records_path = local_path / "records.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(
            f"records.jsonl missing in {local_path}; expected a built iSFT dataset"
        )

    num_records = sum(1 for line in records_path.read_text().splitlines() if line.strip())
    num_images = len(list((local_path / "images").glob("*.png"))) if (local_path / "images").exists() else 0

    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        private=private,
        exist_ok=True,
    )
    api = HfApi(token=token)
    commit_info = api.upload_folder(
        folder_path=str(local_path),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
    )

    return {
        "ok": True,
        "repo_id": repo_id,
        "repo_url": f"https://huggingface.co/datasets/{repo_id}",
        "commit_url": str(commit_info.commit_url) if hasattr(commit_info, "commit_url") else None,
        "num_records": num_records,
        "num_images": num_images,
        "local_dir": str(local_path),
    }
