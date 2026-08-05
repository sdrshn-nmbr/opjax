from types import SimpleNamespace

import pytest
from minisweagent.exceptions import FormatError

from opjax.pallas.g42_agent import TinkerMiniSWEModel


class _Future:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


class _Client:
    def __init__(self, text: str):
        self.text = text
        self.params = None

    def sample(self, **kwargs):
        self.params = kwargs["sampling_params"]
        sequence = SimpleNamespace(tokens=[1, 2], stop_reason="stop")
        return _Future(SimpleNamespace(sequences=[sequence]))


class _Renderer:
    def build_generation_prompt(self, messages):
        return messages

    def get_stop_sequences(self):
        return ["stop"]


class _Tokenizer:
    def __init__(self, text: str):
        self.text = text

    def decode(self, tokens):
        return self.text


def _model(text: str) -> TinkerMiniSWEModel:
    return TinkerMiniSWEModel(
        client=_Client(text),
        renderer=_Renderer(),
        tokenizer=_Tokenizer(text),
        checkpoint=None,
        seed=2,
        max_tokens=8192,
        temperature=0.2,
        top_p=0.95,
    )


def test_tinker_adapter_emits_one_mini_swe_action_and_records_identity() -> None:
    model = _model("```mswea_bash_command\npython dev_check.py kernel.py\n```")
    message = model.query([{"role": "user", "content": "repair"}])
    assert message["extra"]["actions"] == [{"command": "python dev_check.py kernel.py"}]
    assert message["extra"]["checkpoint"] is None
    assert message["extra"]["seed"] == 2
    assert message["extra"]["completion_tokens"] == 2


def test_tinker_adapter_counts_malformed_response_as_a_call() -> None:
    model = _model("not an action")
    with pytest.raises(FormatError):
        model.query([{"role": "user", "content": "repair"}])
    assert model.calls == 1
    assert model.samples[0]["call"] == 1
