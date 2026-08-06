#!/usr/bin/env python3
"""Export a Tinker sampler checkpoint to a private Hugging Face model repo."""

import argparse
import fcntl
import hashlib
import json
import shutil
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

import tinker
from huggingface_hub import HfApi


CHUNK_BYTES = 16 << 20
REPORT_BYTES = 1 << 30
REQUIRED_ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "checkpoint_complete",
)
REQUIRED_REPO_FILES = {
    "README.md",
    "adapter_config.json",
    "adapter_model.safetensors",
    "checkpoint_complete",
    "checkpoint_provenance.json",
    "opjax_training_manifest.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--sampler-path", required=True)
    parser.add_argument("--state-path", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--base-model", default="thinkingmachines/Inkling-Small")
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=Path("/tmp/opjax-hf-checkpoints"),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_url(sampler_path: str) -> str:
    rest = tinker.ServiceClient().create_rest_client()
    attempts = 5
    for attempt in range(1, attempts + 1):
        try:
            response = rest.get_checkpoint_archive_url_from_tinker_path(sampler_path).result()
            return response.url
        except tinker.TinkerError:
            if attempt == attempts:
                raise
            delay_seconds = min(30, 2**attempt)
            print(
                f"[opjax-checkpoint] archive-retry attempt={attempt}/{attempts} "
                f"delay_seconds={delay_seconds}",
                flush=True,
            )
            time.sleep(delay_seconds)
    raise RuntimeError("checkpoint archive retry loop exhausted")


def expected_archive_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def download_archive(url: str, destination: Path, expected_size: int, label: str) -> None:
    existing = destination.stat().st_size if destination.exists() else 0
    if existing > expected_size:
        raise RuntimeError(
            f"partial archive is larger than source for {label}: {existing} > {expected_size}"
        )
    if existing == expected_size:
        print(f"[opjax-checkpoint] download-reused label={label} bytes={existing}", flush=True)
        return

    headers = {"Range": f"bytes={existing}-"} if existing else {}
    request = urllib.request.Request(url, headers=headers)
    mode = "ab" if existing else "wb"
    with urllib.request.urlopen(request, timeout=120) as response, destination.open(mode) as output:
        if existing and response.status != 206:
            raise RuntimeError(f"resume rejected for {label}: HTTP {response.status}")
        copied = existing
        next_report = ((copied // REPORT_BYTES) + 1) * REPORT_BYTES
        while True:
            chunk = response.read(CHUNK_BYTES)
            if not chunk:
                break
            output.write(chunk)
            copied += len(chunk)
            if copied >= next_report:
                print(
                    f"[opjax-checkpoint] download-progress label={label} "
                    f"bytes={copied}/{expected_size}",
                    flush=True,
                )
                next_report += REPORT_BYTES

    actual_size = destination.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"download size mismatch for {label}: {actual_size} != {expected_size}"
        )


def extract_adapter(archive: Path, staging: Path, label: str) -> None:
    with tarfile.open(archive, "r:") as bundle:
        bundle.extractall(staging, filter="data")
    missing = [name for name in REQUIRED_ADAPTER_FILES if not (staging / name).is_file()]
    if missing:
        raise RuntimeError(f"missing adapter artifacts for {label}: {missing}")


def write_metadata(args: argparse.Namespace, archive: Path, staging: Path) -> None:
    exported_files = {}
    for filename in REQUIRED_ADAPTER_FILES:
        path = staging / filename
        exported_files[filename] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    training_manifest = json.loads(args.manifest.read_text())
    (staging / "opjax_training_manifest.json").write_text(
        json.dumps(training_manifest, indent=2, sort_keys=True) + "\n"
    )
    provenance = {
        "schema_version": 1,
        "repo_id": args.repo_id,
        "checkpoint_label": args.label,
        "description": args.description,
        "base_model": args.base_model,
        "sampler_checkpoint": args.sampler_path,
        "resumable_training_state": args.state_path,
        "training_state_exportable_from_tinker": False,
        "training_state_note": (
            "Tinker archive export supports sampler_weights checkpoints only; "
            "the permanent Tinker URI is retained for resume."
        ),
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": sha256_file(args.manifest),
        "tinker_archive_sha256": sha256_file(archive),
        "exported_files": exported_files,
        "evaluation_note": args.evaluation,
    }
    (staging / "checkpoint_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    repo_name = args.repo_id.split("/", 1)[1]
    readme = f"""---
base_model: {args.base_model}
library_name: peft
pipeline_tag: text-generation
tags:
- jax
- pallas
- tinker
- lora
- peft
---

# {repo_name}

Private research checkpoint from the `opjax` Pallas-kernel specialization ladder.

- Intervention: {args.description}
- Base model: `{args.base_model}`
- Portable artifact: PEFT LoRA adapter exported from Tinker sampler weights
- Evaluation: {args.evaluation}

`checkpoint_provenance.json` records byte hashes, the source manifest, the sampler
checkpoint, and the resumable Tinker training-state URI. Tinker does not permit
archive export of optimizer-bearing `/weights/` checkpoints, so this repository
preserves the portable inference adapter but not an independent optimizer-state backup.
"""
    (staging / "README.md").write_text(readme)


def upload_and_verify(args: argparse.Namespace, staging: Path) -> str:
    lock_path = args.staging_root / ".hf-upload.lock"
    with lock_path.open("w") as lock:
        print(f"[opjax-checkpoint] upload-lock-wait label={args.label}", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        print(f"[opjax-checkpoint] upload-lock-acquired label={args.label}", flush=True)
        subprocess.run(
            [
                "hf",
                "upload-large-folder",
                args.repo_id,
                str(staging),
                "--repo-type",
                "model",
                "--private",
                "--num-workers",
                "8",
                "--no-bars",
            ],
            check=True,
        )
    api = HfApi()
    remote_files = set(api.list_repo_files(args.repo_id, repo_type="model"))
    if not REQUIRED_REPO_FILES.issubset(remote_files):
        missing = sorted(REQUIRED_REPO_FILES - remote_files)
        raise RuntimeError(f"remote verification failed for {args.label}: missing {missing}")
    info = api.model_info(args.repo_id, files_metadata=True)
    if not info.private:
        raise RuntimeError(f"remote repository is not private for {args.label}")
    remote = {sibling.rfilename: sibling for sibling in info.siblings}
    for filename in REQUIRED_REPO_FILES:
        local_size = (staging / filename).stat().st_size
        remote_size = remote[filename].size
        if remote_size != local_size:
            raise RuntimeError(
                f"remote size mismatch for {args.label}/{filename}: "
                f"{remote_size} != {local_size}"
            )
    provenance = json.loads((staging / "checkpoint_provenance.json").read_text())
    expected_adapter_sha256 = provenance["exported_files"]["adapter_model.safetensors"][
        "sha256"
    ]
    adapter_lfs = remote["adapter_model.safetensors"].lfs
    if adapter_lfs is None or adapter_lfs.sha256 != expected_adapter_sha256:
        actual_sha256 = None if adapter_lfs is None else adapter_lfs.sha256
        raise RuntimeError(
            f"remote hash mismatch for {args.label}/adapter_model.safetensors: "
            f"{actual_sha256} != {expected_adapter_sha256}"
        )
    return info.sha


def main() -> None:
    args = parse_args()
    staging = args.staging_root / args.label
    archive = args.staging_root / f"{args.label}.tar"
    args.staging_root.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)

    url = checkpoint_url(args.sampler_path)
    expected_size = expected_archive_size(url)
    print(
        f"[opjax-checkpoint] download-start label={args.label} bytes={expected_size}",
        flush=True,
    )
    download_archive(url, archive, expected_size, args.label)
    print(f"[opjax-checkpoint] extract label={args.label}", flush=True)
    extract_adapter(archive, staging, args.label)
    print(f"[opjax-checkpoint] metadata label={args.label}", flush=True)
    write_metadata(args, archive, staging)
    print(
        f"[opjax-checkpoint] upload-start label={args.label} repo={args.repo_id}",
        flush=True,
    )
    revision = upload_and_verify(args, staging)
    print(
        f"[opjax-checkpoint] verified label={args.label} repo={args.repo_id} sha={revision}",
        flush=True,
    )
    archive.unlink()
    shutil.rmtree(staging)
    print(f"[opjax-checkpoint] cleanup-complete label={args.label}", flush=True)


if __name__ == "__main__":
    main()
