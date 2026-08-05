from pathlib import Path

from opjax.pallas.g42_admission import validate_admission_release
from opjax.pallas.g42_curriculum import validate_benchmark_release
from opjax.pallas.g42_harness import validate_task_release
from opjax.pallas.g42_traces import validate_trace_release

REPO_ROOT = Path(__file__).parents[2]


def test_committed_g42_releases_are_hash_valid() -> None:
    task = validate_task_release(REPO_ROOT / "data/pallas/runs/g42-task-release")
    benchmark = validate_benchmark_release(REPO_ROOT / "data/pallas/runs/g42-benchmark-release")
    admission = validate_admission_release(REPO_ROOT / "data/pallas/runs/g42-admission-evidence")
    traces = validate_trace_release(REPO_ROOT / "data/pallas/runs/g42-repair-traces")
    assert task["training_count"] == 32
    assert benchmark["task_count"] == 4
    assert admission["verified_solutions"] == 36
    assert admission["deterministic_failed_starters"] == 36
    assert traces["trajectories"] == 32
    assert traces["prefix_sft_rows"] == 192
