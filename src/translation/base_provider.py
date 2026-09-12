"""Abstract base class for translation LLM providers.

Enables swappable LLM engines (Gemini, OpenAI, Claude, Mock) without modifying
core pipeline orchestration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.translation.schemas import TranslationRequest, TranslationResponse


class TranslationProvider(ABC):
    """Abstract interface for manga translation providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier (e.g., 'gemini', 'openai')."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Underlying model identifier (e.g., 'gemini-2.5-flash', 'gpt-4o-mini')."""
        pass

    @abstractmethod
    def translate(self, request: TranslationRequest) -> TranslationResponse:
        """Translate a page's dialogue and SFX items using structured reasoning.

        Args:
            request: TranslationRequest containing ordered dialogue items,
                     rolling JSON context, and optional vision crop.

        Returns:
            TranslationResponse matching the Pydantic schema with 1:1 ID mapping.

        Raises:
            RuntimeError: If all retries and validation attempts are exhausted.
        """
        pass
