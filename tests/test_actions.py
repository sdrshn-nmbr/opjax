import pytest

from opjax.actions import ActionParseError, ClickAction, action_to_cua_snake, format_function_call, parse_action


def test_parse_functiongemma_click_action() -> None:
    text = format_function_call("click", {"x": 123, "y": 456})

    action = parse_action(text)

    assert action == ClickAction(x=123, y=456)
    assert action_to_cua_snake(action) == "click(123, 456)"


def test_parse_action_ignores_surrounding_text() -> None:
    text = f"I will click now. {format_function_call('click', {'x': 10, 'y': 20})}"

    assert parse_action(text) == ClickAction(x=10, y=20)


def test_parse_action_rejects_missing_coordinate() -> None:
    text = format_function_call("click", {"x": 10})

    with pytest.raises(ActionParseError, match="Missing required argument 'y'"):
        parse_action(text)


def test_parse_action_rejects_non_integer_pixel() -> None:
    text = format_function_call("click", {"x": 10.5, "y": 20})
    with pytest.raises(ActionParseError, match="integer pixel"):
        parse_action(text)


def test_parse_action_accepts_bare_base_model_output() -> None:
    text = "<|channel>thought\n<channel|><start_function_call>call:click{x:251,y:102}<end_function_call><turn|>"

    assert parse_action(text) == ClickAction(x=251, y=102)


def test_parse_action_accepts_bare_with_whitespace() -> None:
    text = "<start_function_call>call:click{ x : 42 , y : 8 }<end_function_call>"

    assert parse_action(text) == ClickAction(x=42, y=8)
