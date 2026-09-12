"""Translation and schema definitions for the manga translation pipeline."""

from src.translation.base_provider import TranslationProvider
from src.translation.gemini_provider import GeminiProvider, resolve_gemini_api_key
from src.translation.mock_provider import MockTranslationProvider
from src.translation.openai_provider import OpenAIProvider
from src.translation.prompt_builder import (
    build_dialogue_payload,
    build_translation_prompt,
)
from src.translation.schemas import (
    BoundingBox,
    ChapterInfo,
    CharacterSeen,
    DialogueEntry,
    PageDetection,
    PageQuality,
    PageResult,
    PipelineMetadata,
    QualityFlag,
    QualityReport,
    RollingContext,
    TextBox,
    TranslationItem,
    TranslationRequest,
    TranslationRequestItem,
    TranslationResponse,
)

__all__ = [
    "BoundingBox",
    "TextBox",
    "PageDetection",
    "TranslationRequestItem",
    "TranslationItem",
    "TranslationRequest",
    "TranslationResponse",
    "PageResult",
    "QualityFlag",
    "PageQuality",
    "PipelineMetadata",
    "QualityReport",
    "ChapterInfo",
    "CharacterSeen",
    "DialogueEntry",
    "RollingContext",
    "TranslationProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "MockTranslationProvider",
    "resolve_gemini_api_key",
    "build_translation_prompt",
    "build_dialogue_payload",
]
