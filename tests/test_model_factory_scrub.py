from opjax.model_factory.scrub import scrub_text


def test_scrubs_hf_and_github_tokens():
    raw = (
        "Leaked hf_abcdefghijklmnopqrstuvwxyz12 and "
        "ghp_abcdefghijklmnopqrstuvwxyz0123456789 in the log"
    )
    result = scrub_text(raw)
    assert not result.clean
    assert "[REDACTED:hf_token]" in result.text
    assert "[REDACTED:github_pat]" in result.text
    assert "hf_abcdefghijklmnopqrstuvwxyz12" not in result.text
    assert "ghp_abcdefghijklmnopqrstuvwxyz0123456789" not in result.text


def test_scrubs_dotenv_assignment():
    raw = "HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz12\nkeep this line\n"
    result = scrub_text(raw)
    assert "[REDACTED:dotenv_assignment]" in result.text or "[REDACTED:hf_token]" in result.text
    assert "keep this line" in result.text


def test_clean_text_passes():
    result = scrub_text("Implement the feature and run pytest -q")
    assert result.clean
    assert result.text == "Implement the feature and run pytest -q"
