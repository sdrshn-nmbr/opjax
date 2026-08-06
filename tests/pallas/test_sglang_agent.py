from __future__ import annotations

import json

import pytest
from minisweagent.exceptions import FormatError

from opjax.pallas.g42_harness import G42HarnessError, canonical_sha256, file_sha256
from opjax.pallas.sglang_agent import SGLangMiniSWEModel, parse_sglang_action
from opjax.remote.laguna_baseline import summarize_baseline


def _model(generate):
    return SGLangMiniSWEModel(
        generate=generate,
        model_id="poolside/Laguna-XS-2.1",
        model_revision="model-revision",
        runtime_revision="runtime-revision",
        precision="bfloat16",
        seed=7,
        max_tokens=1024,
        temperature=0.2,
        top_p=0.95,
    )


def test_sglang_model_records_identity_sampling_and_action() -> None:
    calls = []

    def generate(messages, sampling):
        calls.append((messages, sampling))
        return {
            "text": "```mswea_bash_command\npython dev_check.py\n```",
            "prompt_tokens": 12,
            "completion_tokens": 9,
            "stop_reason": "stop",
            "latency_seconds": 0.25,
        }

    model = _model(generate)
    result = model.query([{"role": "system", "content": "system"}, {"role": "user", "content": "task"}])

    assert result["extra"]["actions"] == [{"command": "python dev_check.py"}]
    assert calls[0][1] == {
        "max_new_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.95,
        "sampling_seed": 7,
    }
    assert model.serialize()["info"]["model"]["provider"] == "sglang"
    assert model.samples[0]["model_revision"] == "model-revision"


def test_sglang_model_counts_format_failure_as_call() -> None:
    model = _model(lambda messages, sampling: {"text": "not an action"})

    with pytest.raises(FormatError):
        model.query([{"role": "user", "content": "task"}])

    assert model.calls == 1
    assert model.samples[0]["content"] == "not an action"


def test_poolside_native_action_is_normalized() -> None:
    content = (
        "Inspecting.</think><tool_call>mswea_bash_command"
        "<arg_key>command</arg_key><arg_value>sed -n '1,80p' instruction.md &amp;&amp; "
        "python dev_check.py</arg_value></tool_call>"
    )

    assert parse_sglang_action(content) == {
        "command": "sed -n '1,80p' instruction.md && python dev_check.py"
    }


def test_mixed_action_protocol_is_rejected() -> None:
    content = (
        "```mswea_bash_command\nls\n```"
        "<tool_call>mswea_bash_command<arg_key>command</arg_key>"
        "<arg_value>pwd</arg_value></tool_call>"
    )

    with pytest.raises(G42HarnessError, match="ACTION_PROTOCOL_MIXED"):
        parse_sglang_action(content)


def test_laguna_baseline_summary_validates_authoritative_evidence(tmp_path) -> None:
    unit_id = "laguna--task--seed-0--turn-3"
    result_dir = tmp_path / "results" / unit_id
    result_dir.mkdir(parents=True)
    reward_path = result_dir / "reward.json"
    reward_path.write_text(
        json.dumps({"reward": 0, "failure_stage": "artifact_contract"}) + "\n"
    )
    manifest = {
        "schema_version": 1,
        "records": [
            {
                "unit_id": unit_id,
                "task_id": "task",
                "family": "elementwise_binary",
                "patch_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            }
        ],
    }
    manifest["release_sha256"] = canonical_sha256(manifest)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest) + "\n")
    verification = {
        "input_release_sha256": manifest["release_sha256"],
        "counts": {"infrastructure_failures": 0},
        "records": [
            {
                "unit_id": unit_id,
                "reward": 0,
                "artifacts": {"reward.json": file_sha256(reward_path)},
            }
        ],
    }
    verification["release_sha256"] = canonical_sha256(verification)
    (tmp_path / "verification.json").write_text(json.dumps(verification) + "\n")

    output_path = tmp_path / "summary.json"
    result = summarize_baseline(verifier_root=tmp_path, out_path=output_path)

    assert result["counts"] == {
        "tasks": 1,
        "profile_verified": 0,
        "candidate_failures": 1,
        "infrastructure_failures": 0,
        "nonempty_patches": 0,
    }
    assert result["failure_stages"] == {"artifact_contract": 1}
    assert json.loads(output_path.read_text())["result_sha256"] == result["result_sha256"]
