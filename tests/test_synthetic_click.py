from pathlib import Path

from opjax.actions import format_function_call
from opjax.synthetic.click import ClickTier, generate_click_task, solve_click_task, verify_click_output


def test_generate_and_solve_click_task(tmp_path: Path) -> None:
    task = generate_click_task(tmp_path, seed=7, tier="target")

    assert Path(task.image_path).exists()
    result = verify_click_output(task, solve_click_task(task))

    assert result.success
    assert result.reward == 1.0
    assert result.reason == "success"


def test_click_task_curriculum_tiers(tmp_path: Path) -> None:
    tiers: tuple[ClickTier, ...] = ("target", "distractors", "button")
    for tier in tiers:
        task = generate_click_task(tmp_path, seed=11, tier=tier)
        result = verify_click_output(task, solve_click_task(task))

        assert task.tier == tier
        assert result.success


def test_verifier_rejects_missed_target(tmp_path: Path) -> None:
    task = generate_click_task(tmp_path, seed=17, tier="distractors")
    wrong = format_function_call("click", {"x": 0, "y": 0})

    result = verify_click_output(task, wrong)

    assert not result.success
    assert result.reason == "missed_target"
    assert result.distance_px is not None
