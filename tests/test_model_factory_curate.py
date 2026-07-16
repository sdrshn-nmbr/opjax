import json
import zipfile
from pathlib import Path

from opjax.model_factory.curate_axport import (
    curate_from_zip,
    flatten_to_singleturn,
)


def _md(turns: list[tuple[str, str]]) -> str:
    parts = []
    for i, (role, content) in enumerate(turns):
        parts.append(f"### [{i}] role={role} ts=2026-01-01T00:00:00Z\n\n{content}\n")
    return "\n".join(parts)


def test_flatten_to_singleturn_one_per_assistant():
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    flats = flatten_to_singleturn(messages)
    assert len(flats) == 2
    assert [m["role"] for m in flats[0]] == ["system", "user", "assistant"]
    assert flats[0][1]["content"] == "u1"
    assert flats[1][1]["content"] == "u2"


def test_curate_full_zip_unlimited_and_tool_filter(tmp_path: Path):
    zpath = tmp_path / "export.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr(
            "agent/a.md",
            _md(
                [
                    ("user", "fix this"),
                    ("assistant", "looking"),
                    ("tool_call", "edit_file\npath=x"),
                    ("tool_result", "ok"),
                    ("user", "done?"),
                    ("assistant", "yes"),
                ]
            ),
        )
        zf.writestr(
            "agent/b.md",
            _md(
                [
                    ("user", "hello"),
                    ("assistant", "hi"),
                    ("user", "more"),
                    ("assistant", "ok"),
                ]
            ),
        )
        # short — skipped
        zf.writestr("agent/c.md", _md([("user", "x"), ("assistant", "y")]))

    out = tmp_path / "multi.jsonl"
    single = tmp_path / "single.jsonl"
    stats = curate_from_zip(
        zpath,
        out,
        max_examples=0,
        require_tool_use=True,
        singleturn_out=single,
    )
    assert stats.kept == 1
    assert stats.skipped_no_tool == 1
    assert stats.singleturn_examples >= 1
    records = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(records) == 1
    singles = [json.loads(l) for l in single.read_text().splitlines() if l.strip()]
    assert all(len(r["messages"]) == 3 for r in singles)
