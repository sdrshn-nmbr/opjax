from __future__ import annotations

from pathlib import Path

import pytest

from opjax.pallas.contracts import load_contracts
from opjax.pallas.sampling import (
    SamplingError,
    _attempt_seed,
    _requested_samples,
    _sampling_fingerprint,
)
from opjax.pallas.scoring import PromptContext

CONFIG_ROOT = Path(__file__).parents[2] / "config" / "pallas"


def test_requested_samples_are_workload_seed_pairs_in_contract_order() -> None:
    requests = _requested_samples(
        public_tasks=["b", "a"],
        contract_seeds=[0, 1, 2],
        workloads=["a", "b"],
        seeds=[2, 0],
        limit=None,
    )

    assert [request.sample_id for request in requests] == [
        "b::seed=0",
        "b::seed=2",
        "a::seed=0",
        "a::seed=2",
    ]
    assert requests[0].kernel_path == "kernels/seed-0/b.py"


def test_requested_samples_reject_unknown_and_duplicate_selectors() -> None:
    with pytest.raises(SamplingError, match="WORKLOAD_UNKNOWN"):
        _requested_samples(
            public_tasks=["a"],
            contract_seeds=[0, 1],
            workloads=["missing"],
            seeds=None,
            limit=None,
        )

    with pytest.raises(SamplingError, match="SEED_DUPLICATE"):
        _requested_samples(
            public_tasks=["a"],
            contract_seeds=[0, 1],
            workloads=None,
            seeds=[0, 0],
            limit=None,
        )


def test_retry_seed_derivation_preserves_declared_seed_identity() -> None:
    assert _attempt_seed(
        declared_seed=2,
        attempt=0,
        retry_seed_stride=1_000_000,
    ) == 2
    assert _attempt_seed(
        declared_seed=2,
        attempt=2,
        retry_seed_stride=1_000_000,
    ) == 2_000_002


def test_sampling_fingerprint_binds_exact_requested_pairs() -> None:
    bundle = load_contracts(CONFIG_ROOT)
    first = _requested_samples(
        public_tasks=["a", "b"],
        contract_seeds=[0, 1],
        workloads=["a"],
        seeds=[0],
        limit=None,
    )
    second = _requested_samples(
        public_tasks=["a", "b"],
        contract_seeds=[0, 1],
        workloads=["a"],
        seeds=[1],
        limit=None,
    )
    kwargs = {
        "bundle": bundle,
        "jaxbench_revision": "a" * 40,
        "model_path": None,
        "arm": "A",
        "prompt_context": PromptContext.SPEC,
        "opjax_revision": "b" * 40,
        "opjax_tracked_dirty": False,
        "jaxbench_tracked_dirty": False,
        "renderer_name": "test-renderer",
    }

    first_fingerprint = _sampling_fingerprint(requests=first, **kwargs)
    second_fingerprint = _sampling_fingerprint(requests=second, **kwargs)

    assert first_fingerprint["request"]["sample_ids"] == ["a::seed=0"]
    assert first_fingerprint["sha256"] != second_fingerprint["sha256"]
