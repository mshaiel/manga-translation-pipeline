"""Deterministic mock translation provider for unit testing and offline development."""

from __future__ import annotations

import logging

from src.translation.base_provider import TranslationProvider
from src.translation.schemas import (
    TranslationItem,
    TranslationRequest,
    TranslationResponse,
)

logger = logging.getLogger(__name__)


class MockTranslationProvider(TranslationProvider):
    """Mock translation provider returning synthetic, deterministic translations."""

    def __init__(
        self,
        model_name: str = "mock-translator",
        simulate_failures: int = 0,
        fail_permanently: bool = False,
    ) -> None:
        """Initialize mock provider.

        Args:
            model_name: Identifier for mock engine.
            simulate_failures: Number of consecutive transient failures before succeeding.
            fail_permanently: If True, always raises an error (for testing fallback).
        """
        self._model_name = model_name
        self.simulate_failures = simulate_failures
        self.fail_permanently = fail_permanently
        self.failure_counter = 0

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model_name

    def translate(self, request: TranslationRequest) -> TranslationResponse:
        """Generate deterministic translations mapping 1:1 by id to requested text boxes."""
        if self.fail_permanently:
            raise RuntimeError("Simulated permanent API failure in MockTranslationProvider")

        if self.failure_counter < self.simulate_failures:
            self.failure_counter += 1
            raise ConnectionError(f"Simulated transient error #{self.failure_counter}")

        translations: list[TranslationItem] = []

        for item in request.text_boxes:
            if item.is_sfx:
                sfx_map = {
                    "ドン": "BOOM!",
                    "ドーン": "THUD!",
                    "ゴゴゴ": "RUMBLE...",
                    "バーン": "BAM!",
                }
                cleaned_jp = item.japanese.strip("！？!?。、… ")
                en_sfx = sfx_map.get(cleaned_jp, "SOUND EFFECT")
                translations.append(
                    TranslationItem(
                        id=item.id,
                        english=en_sfx,
                        translator_note="SFX translation",
                    )
                )
            else:
                prefix = f"[{item.speaker}] " if item.speaker else ""
                en_text = f"{prefix}English translation for: {item.japanese}"
                translations.append(
                    TranslationItem(
                        id=item.id,
                        english=en_text,
                        translator_note=None,
                    )
                )

        scene_update: str | None = None
        if (request.page_index + 1) % 5 == 0:
            scene_update = f"Scene progression update after page {request.page_index + 1}."

        return TranslationResponse(
            translations=translations,
            scene_summary_update=scene_update,
            confidence=0.96,
        )
