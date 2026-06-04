"""
LLM Client Utility Module
===========================

Handles communication with the DeepInfra OpenAI-compatible API endpoint
for the Meta-Llama-3.1-8B-Instruct-Turbo model.

Uses the raw `requests` library (not LangChain's ChatOpenAI) for full
control over the HTTP request lifecycle, timeout handling, and latency
measurement.

Environment Variables Required:
    DEEPINFRA_API_KEY  — Your DeepInfra API key (never hardcoded).
    DEEPINFRA_MODEL    — Model identifier (defaults to meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo).
"""

import logging
import os
import time
from typing import Optional

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# ---------------------------------------------------------------------------
# Load environment variables from .env file
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEEPINFRA_BASE_URL: str = "https://api.deepinfra.com/v1/openai/chat/completions"
DEFAULT_MODEL: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
REQUEST_TIMEOUT_SECONDS: int = 60


def query_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> tuple[str, float]:
    """
    Send a chat completion request to the DeepInfra API and return the response.

    The function uses the OpenAI-compatible chat completions format, sending
    a system message (to define assistant behaviour) and a user message
    (containing the RAG context and question).

    Args:
        system_prompt: Instructions that define the assistant's role and constraints.
        user_prompt:   The user-facing prompt containing context and question.
        temperature:   Sampling temperature (0.1 for near-deterministic output).
        max_tokens:    Maximum number of tokens in the generated response.

    Returns:
        A tuple of (response_text, latency_seconds):
            - response_text (str):    The LLM's generated answer.
            - latency_seconds (float): Round-trip time in seconds.

    Raises:
        EnvironmentError: If DEEPINFRA_API_KEY is not set.
        requests.exceptions.Timeout: If the API does not respond in time.
        requests.exceptions.HTTPError: If the API returns a non-2xx status.
        RuntimeError: For any other unexpected failure.
    """
    # ------------------------------------------------------------------
    # 1. Validate API credentials
    # ------------------------------------------------------------------
    api_key: Optional[str] = os.getenv("DEEPINFRA_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        raise EnvironmentError(
            "DEEPINFRA_API_KEY is not set or still contains the placeholder value. "
            "Please set it in your .env file."
        )

    model: str = os.getenv("DEEPINFRA_MODEL", DEFAULT_MODEL)

    logger.info("Querying LLM model: %s", model)

    # ------------------------------------------------------------------
    # 2. Build the request payload (OpenAI-compatible format)
    # ------------------------------------------------------------------
    headers: dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # ------------------------------------------------------------------
    # 3. Send request and measure latency
    # ------------------------------------------------------------------
    try:
        start_time: float = time.time()

        response = requests.post(
            DEEPINFRA_BASE_URL,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        latency: float = round(time.time() - start_time, 2)

        # Raise an HTTPError for 4xx / 5xx responses
        response.raise_for_status()

        # ------------------------------------------------------------------
        # 4. Parse the response
        # ------------------------------------------------------------------
        response_json: dict = response.json()

        # Extract the assistant's message from the choices array
        choices = response_json.get("choices", [])
        if not choices:
            raise RuntimeError(
                "DeepInfra API returned an empty 'choices' array. "
                f"Full response: {response_json}"
            )

        answer_text: str = choices[0]["message"]["content"].strip()

        logger.info("LLM responded in %.2f seconds (%d chars).", latency, len(answer_text))

        return answer_text, latency

    except requests.exceptions.Timeout:
        logger.error("DeepInfra API request timed out after %ds.", REQUEST_TIMEOUT_SECONDS)
        raise

    except requests.exceptions.HTTPError as http_err:
        logger.error(
            "DeepInfra API HTTP error %s: %s",
            response.status_code,
            response.text[:500],
        )
        raise RuntimeError(
            f"DeepInfra API returned HTTP {response.status_code}: {response.text[:300]}"
        ) from http_err

    except requests.exceptions.ConnectionError as conn_err:
        logger.error("Could not connect to DeepInfra API: %s", conn_err)
        raise RuntimeError(
            "Failed to connect to DeepInfra API. Check your internet connection."
        ) from conn_err

    except Exception as exc:
        logger.error("Unexpected error during LLM query: %s", exc)
        raise RuntimeError(f"LLM query failed unexpectedly: {exc}") from exc
