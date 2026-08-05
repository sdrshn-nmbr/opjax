import json
from pathlib import Path

from opjax.pallas.environment import verify_static
from opjax.pallas.g42_harness import load_task_package
from opjax.pallas.g43_corpus import (
    FAMILIES,
    build_benchmark_release,
    build_learning_curve_release,
    validate_benchmark_release,
    validate_learning_curve_release,
    validate_trace_subset,
)

REPO_ROOT = Path(__file__).parents[2]
CONFIG = REPO_ROOT / "config/pallas/g43-learning-curve.json"
TASKS = REPO_ROOT / "data/pallas/runs/g42-task-release"
TRACES = REPO_ROOT / "data/pallas/runs/g42-repair-traces"


def test_g43_benchmark_is_balanced_static_valid_and_disjoint(tmp_path: Path) -> None:
    root = tmp_path / "benchmark"
    build_benchmark_release(
        config_path=CONFIG,
        training_task_root=TASKS,
        out_dir=root,
    )
    validation = validate_benchmark_release(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    assert validation["task_count"] == 16
    assert validation["families"] == {family: 2 for family in FAMILIES}
    assert not set(manifest["task_signatures"]) & set(
        manifest["training_task_signatures"]
    )
    for relative in manifest["tasks"]:
        package = load_task_package(root / relative)
        source = (package.root / "solution/kernel.py").read_text(encoding="utf-8")
        verdict = verify_static(f"```python\n{source}\n```")
        assert verdict.passed, (package.task_id, verdict)


def test_g43_learning_curve_is_nested_balanced_and_complete(tmp_path: Path) -> None:
    root = tmp_path / "learning-curve"
    build_learning_curve_release(
        config_path=CONFIG,
        training_task_root=TASKS,
        trace_root=TRACES,
        out_dir=root,
    )
    validation = validate_learning_curve_release(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    assert validation["trajectory_counts"] == [8, 16, 32]
    assert validation["training_configs"] == 9
    previous: set[str] = set()
    for count in (8, 16, 32):
        subset = validate_trace_subset(root / f"n{count}")
        current = set(
            next(row for row in manifest["subsets"] if row["trajectory_count"] == count)[
                "task_ids"
            ]
        )
        assert previous <= current
        assert subset["row_count"] == count * 6
        assert subset["family_counts"] == {family: count // 8 for family in FAMILIES}
        previous = current
