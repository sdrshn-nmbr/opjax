import json
import zipfile
from pathlib import Path

from opjax.model_factory.canary import embed_canaries, make_canary_set
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


def test_flatten_keeps_orphan_assistant_with_synthetic_user():
    messages = [{"role": "assistant", "content": "solo work"}]
    flats = flatten_to_singleturn(messages, ensure_user=True)
    assert len(flats) == 1
    assert flats[0][1]["role"] == "user"
    assert flats[0][2]["content"] == "solo work"


def test_curate_max_data_keeps_non_tool_and_emits_many_singleturns(tmp_path: Path):
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
                    ("assistant", "hi there with enough content"),
                    ("user", "more"),
                    ("assistant", "ok more content here"),
                ]
            ),
        )
        # assistant-leading — rescued via synthetic user
        zf.writestr(
            "agent/c.md",
            _md([("assistant", "starting work with enough content for keep")]),
        )

    out = tmp_path / "multi.jsonl"
    single = tmp_path / "single.jsonl"
    stats = curate_from_zip(
        zpath,
        out,
        max_examples=0,
        min_messages=2,
        require_tool_use=False,
        ensure_user=True,
        singleturn_out=single,
    )
    assert stats.kept == 3
    assert stats.singleturn_examples >= 4
    singles = [json.loads(l) for l in single.read_text().splitlines() if l.strip()]
    assert all(len(r["messages"]) == 3 for r in singles)


def test_embed_canaries_jsonl_remains_linewise_parseable():
    rows = [
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "a"},
                ]
            }
        )
        for _ in range(3)
    ]
    text = "\n".join(rows) + "\n"
    canaries = make_canary_set(3, seed_material="test")
    out = embed_canaries(text, canaries)
    parsed = [json.loads(ln) for ln in out.splitlines() if ln.strip()]
    assert len(parsed) == 6  # 3 originals + 3 canary rows
    blob = out
    for c in canaries.canaries:
        assert c.token in blob
