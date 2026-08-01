from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import opjax.pallas.corpus as corpus_module
from opjax.pallas.contracts import load_contracts
from opjax.pallas.corpus import (
    CorpusError,
    build_corpus,
    record_verification_failure,
    validate_corpus_release,
)

CONFIG_ROOT = Path(__file__).parents[2] / "config" / "pallas"


def _git_repository(root: Path, files: dict[str, str]) -> str:
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=opjax-test",
            "-c",
            "user.email=opjax@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _fixture_contracts(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    jax = tmp_path / "jax"
    jaxbench = tmp_path / "jaxbench"
    pallasbench = tmp_path / "pallasbench"
    tokamax = tmp_path / "tokamax"
    maxtext = tmp_path / "maxtext"
    revisions = {
        "jax": _git_repository(
            jax,
            {
                "docs/pallas/example.py": (
                    "from jax.experimental import pallas as pl\n"
                    "def example(x):\n"
                    "    return pl.pallas_call\n"
                ),
                "jax/_src/pallas/core.py": "PALLAS = 'BlockSpec'\n",
                "tests/pallas/test_core.py": "def test_program_id():\n    pass\n",
            },
        ),
        "jaxbench": _git_repository(
            jaxbench,
            {"benchmark/example/baseline.py": "def workload(x):\n    return x + 1\n"},
        ),
        "pallasbench": _git_repository(
            pallasbench,
            {
                "pallasbench/tasks.py": (
                    "TASK_REGISTRY: list[dict] = [\n"
                    "  {'name': 'L1/relu', 'level': 1, 'category': 'activation', "
                    "'pallas_fn': relu.pallas_kernel, "
                    "'baseline_fn': jax_baseline.jax_relu, "
                    "'input_shapes': relu.input_shapes},\n"
                    "]\n"
                ),
                "pallasbench/kernels/level1/relu.py": (
                    "import jax\n"
                    "import jax.numpy as jnp\n"
                    "from jax.experimental import pallas as pl\n"
                    "def _kernel(x_ref, o_ref):\n"
                    "    o_ref[...] = jnp.maximum(x_ref[...], 0)\n"
                    "def pallas_relu(x):\n"
                    "    spec = pl.BlockSpec((128, 128), lambda i, j: (i, j))\n"
                    "    return pl.pallas_call(_kernel, "
                    "out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype), "
                    "grid=(1, 1), in_specs=[spec], out_specs=spec)(x)\n"
                    "pallas_kernel = pallas_relu\n"
                    "input_shapes = [(128, 128)]\n"
                ),
                "pallasbench/baselines/jax_baseline.py": (
                    "import jax.numpy as jnp\n"
                    "def jax_relu(x):\n"
                    "    return jnp.maximum(x, 0)\n"
                ),
                "pallasbench/provenance.py": "",
            },
        ),
        "tokamax": _git_repository(
            tokamax,
            {
                "tokamax/_src/ops/example.py": "from jax.experimental import pallas as pl\n",
                "tokamax/_src/pallas/example.py": "from jax.experimental import pallas as pl\n",
                "tokamax/_src/mosaic_tpu.py": "from jax.experimental import pallas as pl\n",
                "tokamax/_src/mosaic_gpu.py": "from jax.experimental import pallas as pl\n",
            },
        ),
        "maxtext": _git_repository(
            maxtext,
            {
                "src/maxtext/kernels/example.py": "from jax.experimental import pallas as pl\n",
            },
        ),
    }
    config = tmp_path / "config"
    config.mkdir()
    for source in CONFIG_ROOT.glob("*.json"):
        (config / source.name).write_text(
            source.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    sources_path = config / "sources.json"
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    for source in sources["sources"]:
        if source["id"] in revisions:
            source["revision"] = revisions[source["id"]]
        if source["id"] == "pallasbench":
            source["sft_task_allowlist"] = ["L1/relu"]
    sources_path.write_text(json.dumps(sources), encoding="utf-8")
    return config, {
        "jax": jax,
        "jaxbench": jaxbench,
        "pallasbench": pallasbench,
        "tokamax": tokamax,
        "maxtext": maxtext,
    }


def test_build_corpus_keeps_sft_pending_until_tpu_verification(
    tmp_path: Path,
) -> None:
    config, checkouts = _fixture_contracts(tmp_path)
    out_dir = tmp_path / "release"

    manifest = build_corpus(
        bundle=load_contracts(config),
        repo_root=Path(__file__).parents[2],
        source_checkouts=checkouts,
        out_dir=out_dir,
        include_hf=False,
    )

    assert manifest["counts"]["sft"] == 0
    assert manifest["counts"]["status"]["verification_required"] == 1
    assert manifest["counts"]["holdout_contamination"] == 0
    candidates = [
        json.loads(line)
        for line in (out_dir / "candidates.jsonl").read_text().splitlines()
    ]
    sft = next(row for row in candidates if row["objective"] == "sft")
    assert sft["static_inspection"]["authentic"] is True
    assert sft["rejection_reasons"] == []
    assert "def jax_relu(x):" in sft["metadata"]["oracle_source"]
    assert validate_corpus_release(out_dir)["ok"] is True


def test_corpus_release_rejects_artifact_tampering(tmp_path: Path) -> None:
    config, checkouts = _fixture_contracts(tmp_path)
    out_dir = tmp_path / "release"
    build_corpus(
        bundle=load_contracts(config),
        repo_root=Path(__file__).parents[2],
        source_checkouts=checkouts,
        out_dir=out_dir,
        include_hf=False,
    )
    with (out_dir / "candidates.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(CorpusError, match="CORPUS_ARTIFACT_HASH_MISMATCH"):
        validate_corpus_release(out_dir)


def test_build_corpus_rejects_nonempty_output(tmp_path: Path) -> None:
    config, checkouts = _fixture_contracts(tmp_path)
    out_dir = tmp_path / "release"
    out_dir.mkdir()
    (out_dir / "existing").write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(CorpusError, match="CORPUS_OUTPUT_NOT_EMPTY"):
        build_corpus(
            bundle=load_contracts(config),
            repo_root=Path(__file__).parents[2],
            source_checkouts=checkouts,
            out_dir=out_dir,
            include_hf=False,
        )


def test_failed_verification_is_preserved_without_sft_promotion(
    tmp_path: Path,
) -> None:
    config, checkouts = _fixture_contracts(tmp_path)
    bundle = load_contracts(config)
    discovery = tmp_path / "discovery"
    build_corpus(
        bundle=bundle,
        repo_root=Path(__file__).parents[2],
        source_checkouts=checkouts,
        out_dir=discovery,
        include_hf=False,
    )
    candidate = next(
        json.loads(line)
        for line in (discovery / "candidates.jsonl").read_text().splitlines()
        if json.loads(line)["objective"] == "sft"
    )
    verification_root = tmp_path / "verification"
    record_verification_failure(
        bundle=bundle,
        corpus_root=discovery,
        candidate_id=candidate["candidate_id"],
        out_dir=verification_root,
        error=CorpusError("TPU_COMPILE_FAILED: unsupported layout"),
    )

    release = tmp_path / "release"
    manifest = build_corpus(
        bundle=bundle,
        repo_root=Path(__file__).parents[2],
        source_checkouts=checkouts,
        out_dir=release,
        verification_roots=[verification_root],
        include_hf=False,
    )

    assert manifest["counts"]["sft"] == 0
    verification = json.loads(
        (release / "verification.jsonl").read_text(encoding="utf-8")
    )
    assert verification["verified"] is False
    assert verification["failure"]["code"] == "TPU_COMPILE_FAILED"


def test_tpu_preflight_uses_chex_0190_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def assert_devices(*args: object, **kwargs: object) -> None:
        observed.append((args, kwargs))

    monkeypatch.setattr(
        corpus_module.chex,
        "assert_devices_available",
        assert_devices,
    )

    corpus_module._assert_tpu_available()

    assert observed == [((1, "tpu"), {"not_less_than": True})]
