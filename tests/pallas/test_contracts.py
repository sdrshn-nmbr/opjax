from __future__ import annotations

import json
from pathlib import Path

import pytest

from opjax.pallas.contracts import ContractError, contract_report, load_contracts

CONFIG_ROOT = Path(__file__).parents[2] / "config" / "pallas"


def _copy_contracts(tmp_path: Path) -> Path:
    for source in CONFIG_ROOT.glob("*.json"):
        (tmp_path / source.name).write_text(
            source.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    return tmp_path


def test_repository_contract_is_valid_and_not_generalization_ready() -> None:
    bundle = load_contracts(CONFIG_ROOT)
    report = contract_report(bundle)

    assert report["ok"] is True
    assert report["target"]["hardware"] == "v5e"
    assert report["split_counts"]["public"] == 50
    assert report["split_counts"]["private"] == 0
    assert report["generalization_claim_ready"] is False
    assert len(report["contract_sha256"]) == 64


def test_experiment_is_locked_to_inkling_small(tmp_path: Path) -> None:
    root = _copy_contracts(tmp_path)
    path = root / "experiment.json"
    experiment = json.loads(path.read_text(encoding="utf-8"))
    experiment["base_model"] = "thinkingmachines/Inkling"
    for arm in experiment["arms"].values():
        arm["start"] = "thinkingmachines/Inkling"
    path.write_text(json.dumps(experiment), encoding="utf-8")

    with pytest.raises(ContractError, match="BASE_MODEL_INVALID"):
        load_contracts(root)


def test_experiment_has_frozen_gate4_training_config(tmp_path: Path) -> None:
    root = _copy_contracts(tmp_path)
    path = root / "experiment.json"
    experiment = json.loads(path.read_text(encoding="utf-8"))
    experiment["training"]["batch_size"] = 32
    path.write_text(json.dumps(experiment), encoding="utf-8")

    with pytest.raises(ContractError, match="TRAINING_CONFIG_INVALID"):
        load_contracts(root)


def test_public_benchmark_source_is_forbidden_from_training() -> None:
    bundle = load_contracts(CONFIG_ROOT)
    jaxbench = next(
        source for source in bundle.sources["sources"] if source["id"] == "jaxbench"
    )

    assert jaxbench["training_policy"] == "forbidden"
    assert "jaxbench" in bundle.splits["train"]["forbidden_source_ids"]
    assert "pallasbench" in bundle.splits["train"]["forbidden_source_ids"]
    assert "pallasbench_unified" in bundle.splits["train"]["forbidden_source_ids"]


def test_trainable_sources_require_a_verified_license(tmp_path: Path) -> None:
    root = _copy_contracts(tmp_path)
    path = root / "sources.json"
    sources = json.loads(path.read_text(encoding="utf-8"))
    jax = next(source for source in sources["sources"] if source["id"] == "jax")
    jax["license"] = "unverified"
    path.write_text(json.dumps(sources), encoding="utf-8")

    with pytest.raises(ContractError, match="SOURCE_LICENSE_UNVERIFIED"):
        load_contracts(root)


def test_pallasbench_cannot_be_made_trainable(tmp_path: Path) -> None:
    root = _copy_contracts(tmp_path)
    path = root / "sources.json"
    sources = json.loads(path.read_text(encoding="utf-8"))
    pallasbench = next(
        source for source in sources["sources"] if source["id"] == "pallasbench"
    )
    pallasbench["training_policy"] = "allowlisted_paths_only"
    pallasbench["allowlisted_paths"] = ["pallasbench/kernels"]
    path.write_text(json.dumps(sources), encoding="utf-8")

    with pytest.raises(ContractError, match="PALLASBENCH_TRAINING_FORBIDDEN"):
        load_contracts(root)


def test_sft_readiness_minimums_cannot_be_weakened(tmp_path: Path) -> None:
    root = _copy_contracts(tmp_path)
    path = root / "experiment.json"
    experiment = json.loads(path.read_text(encoding="utf-8"))
    experiment["sft_readiness"]["minimum_verified_rows"] = 2
    path.write_text(json.dumps(experiment), encoding="utf-8")

    with pytest.raises(ContractError, match="SFT_READINESS_MINIMUM_INVALID"):
        load_contracts(root)


def test_split_overlap_fails_closed(tmp_path: Path) -> None:
    root = _copy_contracts(tmp_path)
    path = root / "splits.json"
    splits = json.loads(path.read_text(encoding="utf-8"))
    splits["train"]["task_ids"] = [splits["public_evaluation"]["task_ids"][0]]
    path.write_text(json.dumps(splits), encoding="utf-8")

    with pytest.raises(ContractError, match="TASK_SPLIT_OVERLAP"):
        load_contracts(root)


def test_empty_private_split_cannot_enable_generalization_claim(tmp_path: Path) -> None:
    root = _copy_contracts(tmp_path)
    path = root / "splits.json"
    splits = json.loads(path.read_text(encoding="utf-8"))
    splits["private_evaluation"]["generalization_claim_ready"] = True
    path.write_text(json.dumps(splits), encoding="utf-8")

    with pytest.raises(ContractError, match="PRIVATE_GENERALIZATION_NOT_READY"):
        load_contracts(root)


def test_jaxbench_cannot_be_made_trainable(tmp_path: Path) -> None:
    root = _copy_contracts(tmp_path)
    path = root / "sources.json"
    sources = json.loads(path.read_text(encoding="utf-8"))
    jaxbench = next(
        source for source in sources["sources"] if source["id"] == "jaxbench"
    )
    jaxbench["training_policy"] = "allowlisted_paths_only"
    path.write_text(json.dumps(sources), encoding="utf-8")

    with pytest.raises(ContractError, match="JAXBENCH_TRAINING_FORBIDDEN"):
        load_contracts(root)


def test_sampling_seeds_must_be_unique_and_retry_disjoint(tmp_path: Path) -> None:
    root = _copy_contracts(tmp_path)
    path = root / "experiment.json"
    experiment = json.loads(path.read_text(encoding="utf-8"))
    experiment["sampling"]["seeds"] = [0, 0]
    path.write_text(json.dumps(experiment), encoding="utf-8")

    with pytest.raises(ContractError, match="SAMPLING_SEEDS_INVALID"):
        load_contracts(root)

    experiment["sampling"]["seeds"] = [0, 2]
    experiment["sampling"]["retry_seed_stride"] = 2
    path.write_text(json.dumps(experiment), encoding="utf-8")

    with pytest.raises(ContractError, match="RETRY_SEED_STRIDE_INVALID"):
        load_contracts(root)


def test_evaluation_runtime_and_lowering_probe_are_binding(tmp_path: Path) -> None:
    root = _copy_contracts(tmp_path)
    path = root / "eval-policy.json"
    policy = json.loads(path.read_text(encoding="utf-8"))
    policy["authenticity"]["require_empirical_tpu_lowering"] = False
    path.write_text(json.dumps(policy), encoding="utf-8")

    with pytest.raises(ContractError, match="LOWERING_EVIDENCE_POLICY_INVALID"):
        load_contracts(root)

    policy["authenticity"]["require_empirical_tpu_lowering"] = True
    policy["runtime"]["jax"] = "0.10.0"
    path.write_text(json.dumps(policy), encoding="utf-8")

    with pytest.raises(ContractError, match="EVALUATION_RUNTIME_INVALID"):
        load_contracts(root)
