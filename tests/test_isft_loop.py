from pathlib import Path
from typing import cast

from opjax.actions import format_function_call
from opjax.isft import FakeClickRepairer, run_isft_rgt
from opjax.synthetic.click import generate_click_task, verify_click_output


def test_isft_rgt_repairs_failed_click_output(tmp_path: Path) -> None:
    task = generate_click_task(tmp_path, seed=23, tier="button")
    wrong = format_function_call("click", {"x": 0, "y": 0})

    record = run_isft_rgt(
        task,
        initial_output=wrong,
        verifier=verify_click_output,
        repairer=FakeClickRepairer(),
        max_rounds=2,
    )

    assert record.initial_output == wrong
    assert record.refined_output != wrong
    assert len(record.attempts) == 2
    assert record.attempts[-1].verification["success"] is True
    assert record.sft_target == record.refined_output
    assert record.rgt_target.startswith("[THINK] Click the known target center")
    assert "[/THINK] <start_function_call>call:click" in record.rgt_target


def test_isft_record_dataset_row_contains_training_targets(tmp_path: Path) -> None:
    task = generate_click_task(tmp_path, seed=29, tier="target")
    wrong = format_function_call("click", {"x": 0, "y": 0})

    record = run_isft_rgt(
        task,
        initial_output=wrong,
        verifier=verify_click_output,
        repairer=FakeClickRepairer(),
    )

    row = record.to_dataset_row()
    task_payload = cast(dict[str, object], row["task"])

    assert row["sft_target"] == record.refined_output
    assert row["rgt_target"] == record.rgt_target
    assert task_payload["task_id"] == task.task_id
