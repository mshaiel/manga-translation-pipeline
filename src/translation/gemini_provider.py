"""Google Gemini translation provider using Gemini 2.5 Flash and structured outputs.

Features:
    - Native JSON schema enforcement via `response_mime_type="application/json"` and `response_schema`.
    - Multimodal panel/page attachment for vision verification mode.
    - Exponential backoff retry logic (1s, 2s, 4s, 8s) with recovery from schema parsing errors.
    - Flexible API key resolution (Colab userdata, environment variables, or explicit argument).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from PIL import Image

from src.translation.base_provider import TranslationProvider
from src.translation.prompt_builder import build_translation_prompt
from src.translation.schemas import TranslationRequest, TranslationResponse

logger = logging.getLogger(__name__)


def resolve_gemini_api_key(explicit_key: str | None = None) -> str | None:
    """Retrieve Gemini API key from explicit arg, env var, or Google Colab secrets."""
    if explicit_key:
        return explicit_key

    # Check local environment / .env
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key

    # Check Google Colab userdata secrets
    try:
        from google.colab import userdata  # type: ignore[import-not-found]
        colab_key = userdata.get("GEMINI_API_KEY")
        if colab_key:
            return str(colab_key)
    except (ImportError, Exception):
        pass

    return None


class GeminiProvider(TranslationProvider):
    """Translation provider powered by Google Gemini (default: gemini-2.5-flash)."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.2,
        max_retries: int = 3,
        initial_retry_delay: float = 1.0,
        backoff_factor: float = 2.0,
    ) -> None:
        """Initialize Gemini translation provider.

        Args:
            api_key: Gemini API key. If omitted, resolved from environment/Colab secrets.
            model_name: Target Gemini model identifier (e.g., 'gemini-2.5-flash').
            temperature: Sampling temperature for translation creativity (default 0.2).
            max_retries: Maximum number of retry attempts upon failure.
            initial_retry_delay: Initial delay in seconds before first retry.
            backoff_factor: Multiplier for exponential backoff.
        """
        self._model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.backoff_factor = backoff_factor
        self.api_key = resolve_gemini_api_key(api_key)

        self._genai_client: Any = None

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_configured(self) -> None:
        """Verify API key and configure google.generativeai SDK."""
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Please set the GEMINI_API_KEY environment variable "
                "or store it in Google Colab's secrets tab."
            )

        if self._genai_client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._genai_client = genai
            except ImportError as err:
                raise ImportError(
                    "google-generativeai is not installed. Install via `pip install google-generativeai`."
                ) from err

    def translate(self, request: TranslationRequest) -> TranslationResponse:
        """Translate a page's dialogue using Gemini with structured output enforcement.

        Args:
            request: TranslationRequest with text boxes and rolling context.

        Returns:
            Validated TranslationResponse instance.
        """
        self._ensure_configured()
        genai = self._genai_client

        # Build structured translation prompt
        request_summary = (request.page_index + 1) % 5 == 0
        prompt_text = build_translation_prompt(
            text_boxes=request.text_boxes,
            context=request.context,
            vision_mode=request.vision_mode,
            request_scene_summary=request_summary,
        )

        # Prepare multimodal inputs if vision mode is enabled
        content_parts: list[Any] = [prompt_text]
        if request.vision_mode and request.page_image_path:
            try:
                img = Image.open(request.page_image_path)
                content_parts.append(img)
                logger.info("Attached panel crop to multimodal Gemini prompt.")
            except Exception as e:
                logger.warning("Could not load vision crop from %s: %s", request.page_image_path, e)

        # Configure GenerativeModel with enforced Pydantic response_schema
        model = genai.GenerativeModel(
            model_name=self._model_name,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=TranslationResponse,
                temperature=self.temperature,
            ),
        )

        # Execution loop with exponential backoff
        last_exception: Exception | None = None
        delay = self.initial_retry_delay

        for attempt in range(self.max_retries + 1):
            try:
                logger.debug("Dispatching Gemini translation request (attempt %d/%d)...", attempt + 1, self.max_retries + 1)
                response = model.generate_content(content_parts)

                if not response or not response.text:
                    raise ValueError("Received empty response from Gemini API.")

                # Parse and validate response against Pydantic schema
                data = json.loads(response.text)
                parsed_response = TranslationResponse.model_validate(data)

                # Ensure all requested text box IDs are present in the response
                requested_ids = {tb.id for tb in request.text_boxes}
                received_ids = {item.id for item in parsed_response.translations}
                missing_ids = requested_ids - received_ids

                if missing_ids:
                    logger.warning("Gemini response omitted text IDs: %s. Reconstructing missing items.", missing_ids)
                    for mid in missing_ids:
                        matching_tb = next((t for t in request.text_boxes if t.id == mid), None)
                        parsed_response.translations.append(
                            TranslationResponse(
                                id=mid,
                                english=matching_tb.japanese if matching_tb else "[Translation unavailable]",
                                translator_note="Omitted in initial LLM response",
                            )  # type: ignore[arg-type]
                        )

                return parsed_response

            except Exception as err:
                last_exception = err
                logger.warning("Gemini API call failed on attempt %d: %s", attempt + 1, err)

                if attempt < self.max_retries:
                    logger.info("Retrying in %.2f seconds (exponential backoff)...", delay)
                    time.sleep(delay)
                    delay *= self.backoff_factor

        raise RuntimeError(
            f"Gemini translation failed after {self.max_retries + 1} attempts. Last error: {last_exception}"
        ) from last_exception
