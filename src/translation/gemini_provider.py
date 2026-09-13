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
from src.translation.schemas import TranslationItem, TranslationRequest, TranslationResponse

logger = logging.getLogger(__name__)

# Protobuf/OpenAPI 3.0 schema strictly compatible with Google Gemini API
# (Omits unsupported Pydantic v2 metadata fields like 'default', '$defs', 'title')
GEMINI_TRANSLATION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "translations": {
            "type": "ARRAY",
            "description": "List of translations mapping 1:1 by id to the requested text boxes",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {
                        "type": "INTEGER",
                        "description": "Maps 1:1 back to Magi's text box index",
                    },
                    "english": {
                        "type": "STRING",
                        "description": "Natural, idiomatic English translation",
                    },
                    "translator_note": {
                        "type": "STRING",
                        "nullable": True,
                        "description": "Optional translator note or ambiguity flag",
                    },
                },
                "required": ["id", "english"],
            },
        },
        "scene_summary_update": {
            "type": "STRING",
            "nullable": True,
            "description": "Updated 2-3 sentence summary of current events if requested",
        },
        "confidence": {
            "type": "NUMBER",
            "description": "Self-reported translation confidence in [0.0, 1.0]",
        },
    },
    "required": ["translations"],
}


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

        # Build candidate models list for automatic failover when high demand (503/429) occurs
        candidates = [model_name]
        for fallback in ("gemini-2.5-flash", "gemini-3.6-flash", "gemini-2.5-pro"):
            if fallback not in candidates:
                candidates.append(fallback)
        self.candidate_models = candidates

        self._genai_client: Any = None
        self._use_new_sdk: bool = False

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_configured(self) -> None:
        """Verify API key and configure Gemini SDK (supporting both google.genai and google.generativeai)."""
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Please set the GEMINI_API_KEY environment variable "
                "or store it in Google Colab's secrets tab."
            )

        if self._genai_client is None:
            # 1. Prefer new official google.genai SDK if present
            try:
                from google import genai
                self._genai_client = genai.Client(api_key=self.api_key)
                self._use_new_sdk = True
                logger.debug("Configured official google.genai client.")
                return
            except (ImportError, AttributeError):
                pass

            # 2. Fallback to legacy google.generativeai SDK
            try:
                import google.generativeai as legacy_genai
                legacy_genai.configure(api_key=self.api_key)
                self._genai_client = legacy_genai
                self._use_new_sdk = False
                logger.debug("Configured google.generativeai legacy client.")
            except ImportError as err:
                raise ImportError(
                    "Neither google-genai nor google-generativeai is installed. "
                    "Install via `pip install google-genai` or `pip install google-generativeai`."
                ) from err

    def _parse_and_validate_response(
        self,
        raw_output_text: str,
        request: TranslationRequest,
    ) -> TranslationResponse:
        """Sanitize raw LLM response text, parse JSON, and validate with Pydantic."""
        raw_text = raw_output_text.strip()
        # Clean markdown code fences if wrapped by the model
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw_text = "\n".join(lines).strip()

        data = json.loads(raw_text)
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
                    TranslationItem(
                        id=mid,
                        english=matching_tb.japanese if matching_tb else "[Translation unavailable]",
                        translator_note="Omitted in initial LLM response",
                    )
                )

        return parsed_response

    def translate(self, request: TranslationRequest) -> TranslationResponse:
        """Translate a page's dialogue using Gemini with structured output enforcement.

        Args:
            request: TranslationRequest with text boxes and rolling context.

        Returns:
            Validated TranslationResponse instance.
        """
        self._ensure_configured()

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

        # Execution loop with exponential backoff and model failover
        last_exception: Exception | None = None
        delay = self.initial_retry_delay
        candidate_models = list(self.candidate_models)

        for attempt in range(self.max_retries + 1):
            current_model = candidate_models[min(attempt, len(candidate_models) - 1)]
            try:
                logger.info(
                    "Dispatching Gemini translation request (attempt %d/%d) using model '%s'...",
                    attempt + 1,
                    self.max_retries + 1,
                    current_model,
                )

                if self._use_new_sdk:
                    # New google.genai SDK
                    client = self._genai_client
                    config_dict: dict[str, Any] = {
                        "response_mime_type": "application/json",
                        "response_schema": GEMINI_TRANSLATION_RESPONSE_SCHEMA,
                        "temperature": self.temperature,
                    }
                    response = client.models.generate_content(
                        model=current_model,
                        contents=content_parts,
                        config=config_dict,
                    )
                    raw_text = response.text if response else ""
                else:
                    # Legacy google.generativeai SDK
                    genai = self._genai_client
                    gen_config_kwargs: dict[str, Any] = {
                        "response_mime_type": "application/json",
                        "temperature": self.temperature,
                    }
                    try:
                        gen_config = genai.GenerationConfig(
                            response_schema=GEMINI_TRANSLATION_RESPONSE_SCHEMA,
                            **gen_config_kwargs,
                        )
                    except Exception as schema_err:
                        logger.warning(
                            "Could not set response_schema on GenerationConfig (%s). Falling back to JSON mime-type.",
                            schema_err,
                        )
                        gen_config = genai.GenerationConfig(**gen_config_kwargs)

                    model = genai.GenerativeModel(
                        model_name=current_model,
                        generation_config=gen_config,
                    )
                    response = model.generate_content(content_parts)
                    raw_text = response.text if response else ""

                if not raw_text:
                    raise ValueError("Received empty response from Gemini API.")

                return self._parse_and_validate_response(raw_text, request)

            except Exception as err:
                last_exception = err
                logger.warning("Gemini API call failed on attempt %d (%s): %s", attempt + 1, current_model, err)

                if attempt < self.max_retries:
                    next_model = candidate_models[min(attempt + 1, len(candidate_models) - 1)]
                    if next_model != current_model:
                        logger.warning(
                            "Model '%s' failed. Automatically failing over to '%s' on next attempt.",
                            current_model,
                            next_model,
                        )
                    logger.info("Retrying in %.2f seconds (exponential backoff)...", delay)
                    time.sleep(delay)
                    delay *= self.backoff_factor

        raise RuntimeError(
            f"Gemini translation failed after {self.max_retries + 1} attempts. Last error: {last_exception}"
        ) from last_exception
