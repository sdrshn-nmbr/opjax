from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub.errors import HfHubHTTPError
from requests import Response

import opjax.pallas.hub_discovery as hub_discovery_module
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


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _rate_limit_error(reset_seconds: int = 1) -> HfHubHTTPError:
    response = Response()
    response.status_code = 429
    response.headers["RateLimit"] = f'"api";r=0;t={reset_seconds}'
    return HfHubHTTPError("rate limited", response=response)


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
            _dataset(
                "example/generic-trajectories",
                description="Agent trajectory data with no kernel content",
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

    assert manifest["counts"]["inventory"] == 4
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
    assert decisions["example/generic-trajectories"]["status"] == "rejected"
    assert all(
        call[0] != "example/generic-trajectories" for call in api.info_calls
    )


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


def test_enumeration_retries_hub_rate_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datasets = [
        _dataset("example/cuda", description="CUDA kernels", tags=["license:mit"])
    ]

    class RateLimitedApi(FakeApi):
        def list_datasets(self, **kwargs: object) -> list[SimpleNamespace]:
            if not self.list_calls:
                self.list_calls.append(kwargs)
                raise _rate_limit_error()
            return super().list_datasets(**kwargs)

    sleeps: list[float] = []
    monkeypatch.setattr(hub_discovery_module.time, "sleep", sleeps.append)
    api = RateLimitedApi(datasets)

    manifest = discover_hub_datasets(
        repo_root=Path(__file__).parents[2],
        config_path=CONFIG_PATH,
        out_dir=tmp_path / "release",
        api=api,
    )

    assert manifest["counts"]["inventory"] == 1
    assert len(api.list_calls) == 2
    assert sleeps == [2.0]


def test_release_validation_rejects_role_leakage_after_rehash(
    tmp_path: Path,
) -> None:
    out = tmp_path / "release"
    discover_hub_datasets(
        repo_root=Path(__file__).parents[2],
        config_path=CONFIG_PATH,
        out_dir=out,
        api=FakeApi(
            [_dataset("example/cuda", description="CUDA kernels", tags=["license:mit"])]
        ),
    )
    decisions = _load_rows(out / "decisions.jsonl")
    decisions[0]["direct_training_authorized"] = True
    (out / "decisions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in decisions)
    )
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["artifacts"]["decisions.jsonl"] = hashlib.sha256(
        (out / "decisions.jsonl").read_bytes()
    ).hexdigest()
    manifest["release_sha256"] = _canonical_sha256(
        {
            "generator_version": manifest["generator_version"],
            "generator_sha256": manifest["generator_sha256"],
            "opjax_revision": manifest["opjax_revision"],
            "config_sha256": manifest["config_sha256"],
            "invocation_fingerprint": manifest["invocation_fingerprint"],
            "registry_snapshot_atomic": manifest["registry_snapshot_atomic"],
            "source_revisions_pinned_individually": manifest[
                "source_revisions_pinned_individually"
            ],
            "counts": manifest["counts"],
            "artifacts": manifest["artifacts"],
        }
    )
    (out / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(HubDiscoveryError, match="HUB_DIRECT_TRAINING_AUTHORIZED"):
        validate_hub_discovery_release(out)
