from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from opjax.pallas.hub_discovery import (
    HubDiscoveryError,
    discover_hub_datasets,
    validate_hub_discovery_release,
)

CONFIG_PATH = Path(__file__).parents[2] / "config" / "pallas" / "hub-discovery.json"
REVISION = "a" * 40


def _dataset(
    dataset_id: str,
    *,
    description: str,
    tags: list[str] | None = None,
    revision: str | None = REVISION,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=dataset_id,
        description=description,
        tags=tags or [],
        sha=revision,
        private=False,
        gated=False,
        disabled=False,
    )


class FakeApi:
    def __init__(
        self,
        datasets: list[SimpleNamespace],
        files: dict[str, list[SimpleNamespace]] | None = None,
    ) -> None:
        self.datasets = datasets
        self.files = files or {}
        self.list_calls: list[dict[str, object]] = []
        self.info_calls: list[tuple[str, str, bool]] = []

    def list_datasets(self, **kwargs: object) -> list[SimpleNamespace]:
        self.list_calls.append(kwargs)
        search = kwargs.get("search")
        rows = self.datasets
        if isinstance(search, str):
            rows = [row for row in rows if search.casefold() in row.id.casefold()]
        limit = kwargs.get("limit")
        return rows[:limit] if isinstance(limit, int) else rows

    def dataset_info(
        self,
        repo_id: str,
        *,
        revision: str,
        files_metadata: bool,
    ) -> SimpleNamespace:
        self.info_calls.append((repo_id, revision, files_metadata))
        return SimpleNamespace(
            sha=revision,
            siblings=self.files.get(repo_id, []),
        )


def _downloader(files: dict[tuple[str, str], Path]):
    def download(**kwargs: object) -> str:
        return str(files[(str(kwargs["repo_id"]), str(kwargs["filename"]))])

    return download


def _load_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_discovery_rejects_noisy_bare_keywords_and_separates_roles(
    tmp_path: Path,
) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("Uses @triton.jit and tl.program_id for CUDA kernels.")
    api = FakeApi(
        [
            _dataset(
                "CyberHarem/pallas_arknights",
                description="Images of Pallas from Arknights",
                tags=["license:mit", "task_categories:text-to-image"],
            ),
            _dataset(
                "example/triton-kernels",
                description="GPU compiler examples",
                tags=["license:apache-2.0"],
            ),
            _dataset(
                "example/kernelbench-traces",
                description="Agent trajectories solving KernelBench CUDA tasks",
                tags=["license:mit"],
            ),
        ],
        files={
            "example/triton-kernels": [
                SimpleNamespace(
                    rfilename="README.md",
                    size=readme.stat().st_size,
                    blob_id="blob",
                    lfs=None,
                )
            ]
        },
    )
    out = tmp_path / "release"

    manifest = discover_hub_datasets(
        repo_root=Path(__file__).parents[2],
        config_path=CONFIG_PATH,
        out_dir=out,
        api=api,
        downloader=_downloader(
            {("example/triton-kernels", "README.md"): readme}
        ),
    )

    assert manifest["counts"]["inventory"] == 3
    decisions = {
        row["dataset_id"]: row for row in _load_rows(out / "decisions.jsonl")
    }
    assert decisions["CyberHarem/pallas_arknights"]["status"] == "rejected"
    assert decisions["example/triton-kernels"]["category"] == "cross_kernel_domain"
    assert decisions["example/triton-kernels"]["candidate_objectives"] == [
        "dapt_candidate"
    ]
    benchmark = decisions["example/kernelbench-traces"]
    assert benchmark["training_policy"] == "forbidden"
    assert benchmark["candidate_objectives"] == []
    assert "BENCHMARK_CONTAMINATION" in benchmark["risk_flags"]


def test_discovery_is_deterministic_and_pins_detail_requests(tmp_path: Path) -> None:
    datasets = [
        _dataset(
            "example/pallas-code",
            description="jax.experimental.pallas pallas_call examples",
            tags=["license:apache-2.0"],
        )
    ]
    api_one = FakeApi(datasets)
    api_two = FakeApi(datasets)
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_manifest = discover_hub_datasets(
        repo_root=Path(__file__).parents[2],
        config_path=CONFIG_PATH,
        out_dir=first,
        api=api_one,
    )
    second_manifest = discover_hub_datasets(
        repo_root=Path(__file__).parents[2],
        config_path=CONFIG_PATH,
        out_dir=second,
        api=api_two,
    )

    assert first_manifest["release_sha256"] == second_manifest["release_sha256"]
    for filename in ("source_inventory.jsonl", "candidates.jsonl", "decisions.jsonl"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    assert api_one.info_calls == [("example/pallas-code", REVISION, True)]


def test_discovery_preserves_existing_output_and_accepts_completed_resume(
    tmp_path: Path,
) -> None:
    out = tmp_path / "release"
    out.mkdir()
    api = FakeApi(
        [_dataset("example/cuda", description="CUDA kernels", tags=["license:mit"])]
    )
    (out / "unrelated").write_text("preserve")
    with pytest.raises(HubDiscoveryError, match="HUB_OUTPUT_NOT_EMPTY"):
        discover_hub_datasets(
            repo_root=Path(__file__).parents[2],
            config_path=CONFIG_PATH,
            out_dir=out,
            api=api,
        )

    resumable = tmp_path / "resumable"
    discover_hub_datasets(
        repo_root=Path(__file__).parents[2],
        config_path=CONFIG_PATH,
        out_dir=resumable,
        search_terms=["cuda"],
        api=api,
    )
    resumed = discover_hub_datasets(
        repo_root=Path(__file__).parents[2],
        config_path=CONFIG_PATH,
        out_dir=resumable,
        search_terms=["cuda"],
        resume=True,
        api=FakeApi([]),
    )
    assert resumed["release_sha256"] == validate_hub_discovery_release(
        resumable
    )["release_sha256"]
    with pytest.raises(HubDiscoveryError, match="HUB_RESUME_FINGERPRINT_MISMATCH"):
        discover_hub_datasets(
            repo_root=Path(__file__).parents[2],
            config_path=CONFIG_PATH,
            out_dir=resumable,
            search_terms=["pallas"],
            resume=True,
            api=FakeApi([]),
        )


def test_interrupted_detail_enrichment_resumes_from_checkpoint(
    tmp_path: Path,
) -> None:
    datasets = [
        _dataset("a/cuda", description="CUDA kernels", tags=["license:mit"]),
        _dataset("b/cuda", description="CUDA kernels", tags=["license:mit"]),
    ]

    class InterruptingApi(FakeApi):
        def dataset_info(
            self,
            repo_id: str,
            *,
            revision: str,
            files_metadata: bool,
        ) -> SimpleNamespace:
            if repo_id == "b/cuda":
                raise KeyboardInterrupt
            return super().dataset_info(
                repo_id,
                revision=revision,
                files_metadata=files_metadata,
            )

    out = tmp_path / "release"
    with pytest.raises(KeyboardInterrupt):
        discover_hub_datasets(
            repo_root=Path(__file__).parents[2],
            config_path=CONFIG_PATH,
            out_dir=out,
            detail_workers=1,
            api=InterruptingApi(datasets),
        )

    resumed_api = FakeApi(datasets)
    manifest = discover_hub_datasets(
        repo_root=Path(__file__).parents[2],
        config_path=CONFIG_PATH,
        out_dir=out,
        resume=True,
        detail_workers=1,
        api=resumed_api,
    )

    assert manifest["status"] == "complete"
    assert resumed_api.list_calls == []
    assert resumed_api.info_calls == [("b/cuda", REVISION, True)]


def test_unpinned_source_never_enters_trainable_role(tmp_path: Path) -> None:
    api = FakeApi(
        [_dataset("example/cuda", description="CUDA kernels", revision=None)]
    )
    out = tmp_path / "release"
    discover_hub_datasets(
        repo_root=Path(__file__).parents[2],
        config_path=CONFIG_PATH,
        out_dir=out,
        api=api,
    )

    decision = _load_rows(out / "decisions.jsonl")[0]
    assert decision["training_policy"] == "discovery_only"
    assert "SOURCE_REVISION_UNPINNED" in decision["risk_flags"]


def test_release_validation_detects_tampering(tmp_path: Path) -> None:
    out = tmp_path / "release"
    discover_hub_datasets(
        repo_root=Path(__file__).parents[2],
        config_path=CONFIG_PATH,
        out_dir=out,
        api=FakeApi(
            [_dataset("example/cuda", description="CUDA kernels", tags=["license:mit"])]
        ),
    )
    with (out / "decisions.jsonl").open("a") as handle:
        handle.write("{}\n")

    with pytest.raises(HubDiscoveryError, match="HUB_ARTIFACT_HASH_MISMATCH"):
        validate_hub_discovery_release(out)
