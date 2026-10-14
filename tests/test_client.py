"""
Unit tests for GeminiClient error handling, rate-limiting backoff,
timeout management, and environment variable resolution.
"""

import os
from unittest.mock import MagicMock, patch
import pytest

from src.client import (
    GeminiClient,
    MissingAPIKeyError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMAPIError,
    get_gemini_api_key,
)
from src.schemas import CustomerSupportTriage


class TestEnvironmentAndConfig:
    def test_missing_api_key_raises_error(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("Gemini_API_key", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        with pytest.raises(MissingAPIKeyError):
            get_gemini_api_key()

    def test_detects_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test_mock_key_12345")
        key = get_gemini_api_key()
        assert key == "test_mock_key_12345"


class TestRetryAndErrorHandling:
    @patch("google.genai.Client")
    def test_client_init_with_explicit_key(self, mock_genai):
        client = GeminiClient(api_key="explicit_key_test")
        assert client.api_key == "explicit_key_test"
        mock_genai.assert_called_once_with(api_key="explicit_key_test")

    @patch("google.genai.Client")
    def test_rate_limit_retry_success(self, mock_genai):
        """
        Simulates an API call that fails once with a 429 RateLimitError,
        then succeeds on retry.
        """
        from google.genai.errors import ClientError

        client = GeminiClient(api_key="mock_key", max_retries=3)

        mock_response = MagicMock()
        mock_response.text = "Success after retry"

        # First call fails with 429, second call succeeds
        rate_limit_error = ClientError(429, {"error": {"message": "RESOURCE_EXHAUSTED: Rate limit exceeded"}})
        mock_call = MagicMock(side_effect=[rate_limit_error, mock_response])

        result = client._execute_with_retry(mock_call)
        assert result.text == "Success after retry"
        assert mock_call.call_count == 2

    @patch("google.genai.Client")
    def test_rate_limit_exceeds_max_retries(self, mock_genai):
        """
        Simulates persistent rate-limiting that exceeds max_retries
        and cleanly raises LLMRateLimitError rather than crashing unhandled.
        """
        from google.genai.errors import ClientError

        client = GeminiClient(api_key="mock_key", max_retries=2)
        rate_limit_error = ClientError(429, {"error": {"message": "RESOURCE_EXHAUSTED: Quota depleted"}})
        mock_call = MagicMock(side_effect=rate_limit_error)

        with pytest.raises(LLMRateLimitError):
            client._execute_with_retry(mock_call)

        assert mock_call.call_count == 2

    @patch("google.genai.Client")
    def test_timeout_retry_handling(self, mock_genai):
        """
        Simulates a timeout that retries and recovers.
        """
        client = GeminiClient(api_key="mock_key", max_retries=3)
        mock_response = MagicMock()
        mock_response.text = "Recovered from timeout"

        timeout_error = TimeoutError("Connection to API timed out")
        mock_call = MagicMock(side_effect=[timeout_error, mock_response])

        result = client._execute_with_retry(mock_call)
        assert result.text == "Recovered from timeout"
        assert mock_call.call_count == 2

    @patch("google.genai.Client")
    def test_generate_structured_success(self, mock_genai):
        """
        Verifies that generate_structured parses a valid JSON payload into the Pydantic schema.
        """
        client = GeminiClient(api_key="mock_key")

        valid_json = """
        {
            "ticket_id": "TCK-555",
            "sentiment": "negative",
            "urgency": "high",
            "category": "billing",
            "entities": ["Invoice #12", "Card ending 4401"],
            "action_items": ["Process refund", "Email receipt"],
            "summary": "Customer was double charged on Invoice 12."
        }
        """
        mock_response = MagicMock()
        mock_response.text = valid_json
        client.client.models.generate_content = MagicMock(return_value=mock_response)

        result = client.generate_structured(
            prompt="Analyze ticket",
            response_schema=CustomerSupportTriage,
        )

        assert isinstance(result, CustomerSupportTriage)
        assert result.ticket_id == "TCK-555"
        assert result.urgency == "high"
        assert result.category == "billing"
