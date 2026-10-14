"""
Track K: AI Engineering - K2 Prompt Engineering & AI APIs
Robust AI API client with exponential backoff, rate-limit catching,
timeout management, and structured schema enforcement.
"""

import os
import json
import logging
import time
from typing import Optional, Type, TypeVar, Any
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

# Load environment variables from .env file
# Looks in current working directory and parent paths
load_dotenv()
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AIClient")

T = TypeVar("T", bound=BaseModel)


class MissingAPIKeyError(Exception):
    """Raised when the required API key cannot be located in the environment."""
    pass


class LLMRateLimitError(Exception):
    """Raised when the LLM provider returns a 429 / Quota Exceeded error."""
    pass


class LLMTimeoutError(Exception):
    """Raised when an API call exceeds the configured network timeout."""
    pass


class LLMAPIError(Exception):
    """Generic wrapper for upstream API errors."""
    pass


def get_gemini_api_key() -> str:
    """
    Retrieves the Gemini API key from environment variables.
    Checks multiple common key names to ensure compatibility.
    """
    for key_name in ["GEMINI_API_KEY", "Gemini_API_key", "GOOGLE_API_KEY"]:
        key = os.getenv(key_name)
        if key and key.strip():
            return key.strip()
    raise MissingAPIKeyError(
        "Gemini API key not found. Please set 'GEMINI_API_KEY' in your environment or in a .env file."
    )


class GeminiClient:
    """
    Production-grade client wrapper for the Google GenAI SDK (Gemini models).
    Implements structured outputs, system prompts, temperature tuning,
    and resilient error handling for rate limits and timeouts.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        default_temperature: Optional[float] = None,
        request_timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        self.api_key = api_key or get_gemini_api_key()
        self.default_model = default_model or os.getenv("DEFAULT_MODEL", "gemini-3.8-flash")
        
        env_temp = os.getenv("DEFAULT_TEMPERATURE")
        self.default_temperature = default_temperature if default_temperature is not None else (float(env_temp) if env_temp else 0.2)
        
        env_timeout = os.getenv("REQUEST_TIMEOUT_SECONDS")
        self.request_timeout = request_timeout if request_timeout is not None else (float(env_timeout) if env_timeout else 30.0)
        
        env_retries = os.getenv("MAX_RETRIES")
        self.max_retries = max_retries if max_retries is not None else (int(env_retries) if env_retries else 3)

        # Lazy initialize SDK client
        from google import genai
        self.client = genai.Client(api_key=self.api_key)

    def _execute_with_retry(self, api_func, *args, **kwargs):
        """
        Executes an API call with exponential backoff for rate limits and transient errors.
        """
        from google.genai import errors

        retryer = retry(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((LLMRateLimitError, LLMTimeoutError)),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )

        @retryer
        def _call():
            try:
                return api_func(*args, **kwargs)
            except errors.ClientError as e:
                # HTTP 429: Too Many Requests / Resource Exhausted
                if getattr(e, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(e).upper():
                    logger.warning("Rate limit hit (429 / Resource Exhausted). Backing off and retrying...")
                    raise LLMRateLimitError(f"Rate limit exceeded: {e}") from e
                logger.error(f"ClientError from Gemini API: {e}")
                raise LLMAPIError(f"API ClientError: {e}") from e
            except errors.ServerError as e:
                logger.warning(f"Server error from Gemini API: {e}. Retrying...")
                raise LLMRateLimitError(f"Server error (temporary): {e}") from e
            except TimeoutError as e:
                logger.warning(f"Request timed out. Retrying: {e}")
                raise LLMTimeoutError(f"Request timed out after {self.request_timeout}s") from e
            except Exception as e:
                err_str = str(e).upper()
                if "RESOURCE_EXHAUSTED" in err_str or "RATE_LIMIT" in err_str or "429" in err_str:
                    logger.warning("Rate limit detected in generic exception. Retrying...")
                    raise LLMRateLimitError(f"Rate limit exceeded: {e}") from e
                if "TIMEOUT" in err_str:
                    raise LLMTimeoutError(f"Timeout detected: {e}") from e
                logger.error(f"Unexpected error calling Gemini API: {e}")
                raise LLMAPIError(f"Unexpected API failure: {e}") from e

        return _call()

    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        Standard text generation for testing zero-shot, few-shot, and CoT prompts.
        """
        from google.genai import types

        target_model = model or self.default_model
        temp = self.default_temperature if temperature is None else temperature

        config = types.GenerateContentConfig(
            temperature=temp,
            system_instruction=system_instruction,
        )

        def _call_api():
            return self.client.models.generate_content(
                model=target_model,
                contents=prompt,
                config=config,
            )

        response = self._execute_with_retry(_call_api)
        return response.text or ""

    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
    ) -> T:
        """
        Reliably returns parsed, schema-validated Pydantic objects using native JSON schema mode.
        Guarantees structured parseability without manual cleanup.
        """
        from google.genai import types

        target_model = model or self.default_model
        temp = self.default_temperature if temperature is None else temperature

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=temp,
            system_instruction=system_instruction,
        )

        def _call_api():
            return self.client.models.generate_content(
                model=target_model,
                contents=prompt,
                config=config,
            )

        response = self._execute_with_retry(_call_api)
        raw_text = response.text or "{}"

        # Parse and validate with Pydantic
        try:
            parsed_json = json.loads(raw_text)
            validated_obj = response_schema.model_validate(parsed_json)
            return validated_obj
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to parse structured output: {e}\nRaw output: {raw_text}")
            raise ValueError(f"Model response did not conform to schema {response_schema.__name__}: {e}") from e
