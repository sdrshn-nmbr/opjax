from __future__ import annotations

from opjax.factory.markdown_export import parse_axport_markdown


SAMPLE = """=== cursor session abc ===
### [0000] role=system
You are a huge system prompt.

### [0001] role=user
Add a unit test for add().

### [0002] role=assistant
I'll add a test.

### [0003] role=tool_call
run_terminal_cmd: pytest

### [0004] role=user
Looks good.
"""


def test_parse_drops_system_keeps_user_assistant():
    t = parse_axport_markdown(SAMPLE, trajectory_id="abc", project="/tmp/demo")
    assert t is not None
    assert t.trajectory_id == "abc"
    roles = [m.role for m in t.messages]
    assert "system" not in roles
    assert roles[0] == "user"
    assert "assistant" in roles
