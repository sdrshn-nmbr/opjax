"""FunctionGemma-style action parsing for GUI control."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


START_CALL = "<start_function_call>"
END_CALL = "<end_function_call>"
ESCAPE = "<escape>"


class ActionParseError(ValueError):
    """Raised when model text cannot be converted into a supported action."""


@dataclass(frozen=True)
class FunctionCall:
    name: str
    arguments: dict[str, str]

    def require(self, key: str) -> str:
        if key not in self.arguments:
            raise ActionParseError(f"Missing required argument '{key}' for call:{self.name}")
        return self.arguments[key]


@dataclass(frozen=True)
class ClickAction:
    x: int
    y: int


@dataclass(frozen=True)
class TypeAction:
    text: str


@dataclass(frozen=True)
class HotkeyAction:
    keys: tuple[str, ...]


@dataclass(frozen=True)
class ScrollAction:
    direction: str = "down"
    amount: int = 100


@dataclass(frozen=True)
class WaitAction:
    seconds: float = 1.0


@dataclass(frozen=True)
class DoneAction:
    pass


Action = ClickAction | TypeAction | HotkeyAction | ScrollAction | WaitAction | DoneAction


_CALL_RE = re.compile(
    rf"{re.escape(START_CALL)}\s*call:(?P<name>[a-zA-Z_][\w]*)\s*\{{(?P<body>.*?)\}}\s*{re.escape(END_CALL)}",
    re.DOTALL,
)
_ARG_RE = re.compile(rf"(?P<key>[a-zA-Z_][\w]*)\s*:\s*{re.escape(ESCAPE)}(?P<value>.*?){re.escape(ESCAPE)}", re.DOTALL)
_BARE_ARG_RE = re.compile(r"(?P<key>[a-zA-Z_][\w]*)\s*:\s*(?P<value>[^,{}]+?)(?=\s*,\s*[a-zA-Z_]|\s*$)", re.DOTALL)


def format_function_call(name: str, arguments: dict[str, Any]) -> str:
    body = ",".join(f"{key}:{ESCAPE}{value}{ESCAPE}" for key, value in arguments.items())
    return f"{START_CALL}call:{name}{{{body}}}{END_CALL}"


def parse_function_call(text: str) -> FunctionCall:
    if not isinstance(text, str):
        raise ActionParseError("Model output must be a string")

    match = _CALL_RE.search(text.strip())
    if match is None:
        raise ActionParseError("No FunctionGemma function call found")

    body = match.group("body")
    arguments = {arg.group("key"): arg.group("value") for arg in _ARG_RE.finditer(body)}
    if not arguments and body.strip():
        arguments = {arg.group("key"): arg.group("value").strip() for arg in _BARE_ARG_RE.finditer(body)}
    if body.strip() and not arguments:
        raise ActionParseError(f"Malformed function call arguments: {body!r}")

    return FunctionCall(name=match.group("name"), arguments=arguments)


def parse_action(text: str) -> Action:
    call = parse_function_call(text)
    name = call.name.lower()

    if name == "click":
        return ClickAction(x=_parse_int(call.require("x"), "x"), y=_parse_int(call.require("y"), "y"))
    if name == "type":
        return TypeAction(text=call.require("text"))
    if name == "hotkey":
        raw_keys = call.require("keys")
        keys = tuple(key.strip() for key in re.split(r"[+,]", raw_keys) if key.strip())
        if not keys:
            raise ActionParseError("hotkey requires at least one key")
        return HotkeyAction(keys=keys)
    if name == "scroll":
        direction = call.arguments.get("direction", "down").lower()
        amount = _parse_int(call.arguments.get("amount", "100"), "amount")
        return ScrollAction(direction=direction, amount=amount)
    if name == "wait":
        return WaitAction(seconds=_parse_float(call.arguments.get("seconds", "1.0"), "seconds"))
    if name == "done":
        return DoneAction()

    raise ActionParseError(f"Unsupported action call:{call.name}")


def action_to_cua_snake(action: Action) -> str:
    if isinstance(action, ClickAction):
        return f"click({action.x}, {action.y})"
    if isinstance(action, TypeAction):
        escaped = action.text.replace("\\", "\\\\").replace('"', '\\"')
        return f'type("{escaped}")'
    if isinstance(action, HotkeyAction):
        return f"hotkey({'+'.join(action.keys)})"
    if isinstance(action, ScrollAction):
        return f"scroll({action.direction}, {action.amount})"
    if isinstance(action, WaitAction):
        return f"wait({action.seconds:g})"
    if isinstance(action, DoneAction):
        return "done()"
    raise TypeError(f"Unsupported action type: {type(action)!r}")


def action_to_dict(action: Action) -> dict[str, Any]:
    if isinstance(action, ClickAction):
        return {"type": "ClickAction", "x": action.x, "y": action.y}
    if isinstance(action, TypeAction):
        return {"type": "TypeAction", "text": action.text}
    if isinstance(action, HotkeyAction):
        return {"type": "HotkeyAction", "keys": list(action.keys)}
    if isinstance(action, ScrollAction):
        return {"type": "ScrollAction", "direction": action.direction, "amount": action.amount}
    if isinstance(action, WaitAction):
        return {"type": "WaitAction", "seconds": action.seconds}
    if isinstance(action, DoneAction):
        return {"type": "DoneAction"}
    raise TypeError(f"Unsupported action type: {type(action)!r}")


def _parse_int(value: str, key: str) -> int:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ActionParseError(f"Argument '{key}' must be numeric, got {value!r}") from exc
    if parsed != int(parsed):
        raise ActionParseError(f"Argument '{key}' must be an integer pixel coordinate, got {value!r}")
    return int(parsed)


def _parse_float(value: str, key: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ActionParseError(f"Argument '{key}' must be numeric, got {value!r}") from exc
