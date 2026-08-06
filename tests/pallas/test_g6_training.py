from __future__ import annotations

from tinker import ModelInput

from opjax.pallas.g6_rollout import TurnSample
from opjax.pallas.g6_training import sample_to_datum


def test_rl_datum_masks_prompt_and_uses_constant_length_normalization() -> None:
    sample = TurnSample(
        task_id="task",
        trajectory=0,
        turn=1,
        prompt=ModelInput.from_ints([10, 11, 12]),
        prompt_messages=[],
        response_tokens=[20, 21, 22],
        behavior_logprobs=[-1.0, -2.0, -3.0],
        response_text="response",
        stop_reason="stop",
        action=None,
        action_result={},
        snapshot={},
        advantage=2.0,
    )
    datum = sample_to_datum(sample, length_normalizer=4.0)
    assert datum.model_input.to_ints() == [10, 11, 12, 20, 21]
    assert datum.loss_fn_inputs["target_tokens"].tolist() == [0, 0, 20, 21, 22]
    assert datum.loss_fn_inputs["logprobs"].tolist() == [0.0, 0.0, -1.0, -2.0, -3.0]
    assert datum.loss_fn_inputs["advantages"].tolist() == [0.0, 0.0, 0.5, 0.5, 0.5]
