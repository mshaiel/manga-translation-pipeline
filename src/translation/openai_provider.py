"""OpenAI translation provider fallback (e.g., gpt-4o-mini).

Features:
    - Structured output using JSON mode or Pydantic parsing.
    - Multimodal vision support via base64 or image URLs.
    - Exponential backoff retry logic.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from src.translation.base_provider import TranslationProvider
from src.translation.prompt_builder import build_translation_prompt
from src.translation.schemas import TranslationRequest, TranslationResponse

logger = logging.getLogger(__name__)


def encode_image_to_base64(image_path: str | Path) -> str:
    """Encode an image file to a base64 string for OpenAI vision API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class OpenAIProvider(TranslationProvider):
    """Fallback translation provider powered by OpenAI (default: gpt-4o-mini)."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_retries: int = 3,
        initial_retry_delay: float = 1.0,
        backoff_factor: float = 2.0,
    ) -> None:
        """Initialize OpenAI translation provider.

        Args:
            api_key: OpenAI API key. If omitted, read from OPENAI_API_KEY env var.
            model_name: Target OpenAI model (default 'gpt-4o-mini').
            temperature: Sampling temperature.
            max_retries: Maximum number of retry attempts.
            initial_retry_delay: Initial delay in seconds before first retry.
            backoff_factor: Multiplier for exponential backoff.
        """
        self._model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.backoff_factor = backoff_factor
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

        self._client: Any = None

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_configured(self) -> None:
        """Verify API key and instantiate OpenAI client."""
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")

        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError as err:
                raise ImportError("openai package is not installed. Install via `pip install openai`.") from err

    def translate(self, request: TranslationRequest) -> TranslationResponse:
        """Translate dialogue items using OpenAI chat completions."""
        self._ensure_configured()
        client = self._client

        request_summary = (request.page_index + 1) % 5 == 0
        prompt_text = build_translation_prompt(
            text_boxes=request.text_boxes,
            context=request.context,
            vision_mode=request.vision_mode,
            request_scene_summary=request_summary,
        )

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a professional Japanese-to-English manga translator. "
                    "Always output valid JSON conforming to the requested schema."
                ),
            }
        ]

        user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]

        if request.vision_mode and request.page_image_path:
            try:
                b64 = encode_image_to_base64(request.page_image_path)
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    }
                )
            except Exception as e:
                logger.warning("Could not encode vision image for OpenAI: %s", e)

        messages.append({"role": "user", "content": user_content})

        delay = self.initial_retry_delay
        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )

                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Received empty response from OpenAI.")

                data = json.loads(content)
                return TranslationResponse.model_validate(data)

            except Exception as err:
                last_exception = err
                logger.warning("OpenAI API call failed on attempt %d: %s", attempt + 1, err)

                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= self.backoff_factor

        raise RuntimeError(
            f"OpenAI translation failed after {self.max_retries + 1} attempts. Last error: {last_exception}"
        ) from last_exception
