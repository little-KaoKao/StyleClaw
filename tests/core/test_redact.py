from __future__ import annotations

import pytest

from styleclaw.core.redact import redact, redact_exc


class TestRedact:
    def test_bearer_token_masked(self):
        out = redact("Authorization: Bearer abc123def456ghi789jkl0")
        assert "abc123def456ghi789jkl0" not in out
        # The prefix label is preserved so the line still reads naturally;
        # the actual secret value is what must disappear.
        assert "Authorization" in out
        assert "***" in out

    def test_plain_bearer_masked(self):
        out = redact("got Bearer abc123def456ghi789jkl0 back")
        assert "abc123def456ghi789jkl0" not in out
        assert "Bearer" in out
        assert "***" in out

    def test_authorization_without_bearer_masked(self):
        # `Authorization: <raw token>` without the `Bearer` prefix used to
        # slip through — the upgraded regex covers it now.
        out = redact("Authorization: abc123def456ghi789jkl0")
        assert "abc123def456ghi789jkl0" not in out
        assert "Authorization" in out
        assert "***" in out

    def test_api_key_masked(self):
        out = redact("api_key=mySecretKey123456789")
        assert "mySecretKey123456789" not in out
        assert "api_key" in out

    def test_x_api_key_header_masked(self):
        out = redact("x-api-key: xyzSecretValue1234567890")
        assert "xyzSecretValue1234567890" not in out
        assert "x-api-key" in out

    def test_sk_prefix_masked(self):
        out = redact("Using sk-abcdef0123456789xyz to call")
        assert "sk-abcdef0123456789xyz" not in out

    def test_known_env_secret_masked(self, monkeypatch):
        monkeypatch.setenv("RUNNINGHUB_API_KEY", "supersecret123456")
        out = redact("Got error using supersecret123456 for upload")
        assert "supersecret123456" not in out
        assert "***" in out

    def test_safe_url_path_segment_not_masked(self):
        # URL-like segments under 25 chars stay readable.
        out = redact("https://example.com/task/abc")
        assert "https" in out

    def test_empty_input_returns_empty(self):
        assert redact("") == ""

    def test_redact_exc_includes_type(self):
        exc = ValueError("Authorization: Bearer leakedToken1234567890")
        msg = redact_exc(exc)
        assert "ValueError" in msg
        assert "leakedToken1234567890" not in msg
