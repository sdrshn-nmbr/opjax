from __future__ import annotations

from opjax.factory.scrub import find_canaries, make_canary, scrub_text


def test_scrub_removes_anthropic_and_env_secrets():
    raw = (
        "debug sk-ant-api03-EXAMPLESECRETKEYVALUE000001 end\n"
        "TINKER_API_KEY=tinker_test_secret_value_123456\n"
        "Authorization: Bearer SUPERSECRETTOKENVALUE12345\n"
    )
    result = scrub_text(raw)
    assert "EXAMPLESECRETKEYVALUE" not in result.text
    assert "tinker_test_secret_value_123456" not in result.text
    assert "SUPERSECRETTOKENVALUE12345" not in result.text
    assert "REDACTED_" in result.text
    assert result.substitutions >= 2


def test_scrub_pem_block():
    pem = (
        "-----BEGIN PRIVATE KEY-----\n"
        "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7\n"
        "-----END PRIVATE KEY-----\n"
    )
    result = scrub_text(pem)
    assert "BEGIN PRIVATE KEY" not in result.text
    assert result.substitutions == 1


def test_canary_detection():
    c = make_canary()
    assert find_canaries(f"hello {c} world", [c]) == [c]
    assert find_canaries("clean", [c]) == []
