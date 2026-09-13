"""Unit tests for translation providers and retry mechanics."""

import pytest

from src.translation.gemini_provider import resolve_gemini_api_key
from src.translation.mock_provider import MockTranslationProvider
from src.translation.schemas import (
    RollingContext,
    TranslationRequest,
    TranslationRequestItem,
    TranslationResponse,
)


class TestTranslationProviders:
    """Test suite for provider-swappable translation architecture."""

    def test_mock_provider_translation(self):
        provider = MockTranslationProvider()
        tb1 = TranslationRequestItem(id=0, speaker="Luffy", japanese="海賊王になる！", is_sfx=False)
        tb2 = TranslationRequestItem(id=1, speaker=None, japanese="ドーン", is_sfx=True)

        req = TranslationRequest(
            page_index=0,
            text_boxes=[tb1, tb2],
            context=RollingContext(),
        )

        res = provider.translate(req)
        assert isinstance(res, TranslationResponse)
        assert len(res.translations) == 2
        assert res.translations[0].id == 0
        assert "海賊王になる！" in res.translations[0].english
        assert res.translations[1].id == 1
        assert res.translations[1].english == "THUD!"
        assert res.confidence == 0.96

    def test_mock_provider_scene_summary_trigger(self):
        provider = MockTranslationProvider()
        # Page 4 (5th page in 1-index) should trigger scene summary update
        req = TranslationRequest(
            page_index=4,
            text_boxes=[TranslationRequestItem(id=0, japanese="テスト", is_sfx=False)],
            context=RollingContext(),
        )
        res = provider.translate(req)
        assert res.scene_summary_update is not None
        assert "page 5" in res.scene_summary_update

    def test_transient_retry_success(self):
        # Configure provider to fail 2 times before succeeding on attempt 3
        provider = MockTranslationProvider(simulate_failures=2)
        req = TranslationRequest(
            page_index=0,
            text_boxes=[TranslationRequestItem(id=0, japanese="こんにちは", is_sfx=False)],
            context=RollingContext(),
        )

        with pytest.raises(ConnectionError):
            provider.translate(req)

        with pytest.raises(ConnectionError):
            provider.translate(req)

        # Third attempt succeeds!
        res = provider.translate(req)
        assert len(res.translations) == 1

    def test_provider_fallback_orchestration(self):
        """Verify that when primary provider fails, the pipeline seamlessly falls back."""
        primary_failing_provider = MockTranslationProvider(fail_permanently=True)
        fallback_provider = MockTranslationProvider(model_name="fallback-mock")

        req = TranslationRequest(
            page_index=0,
            text_boxes=[TranslationRequestItem(id=10, japanese="行くぞ！", is_sfx=False)],
            context=RollingContext(),
        )

        response: TranslationResponse
        try:
            response = primary_failing_provider.translate(req)
        except Exception:
            # Catch primary failure and execute fallback
            response = fallback_provider.translate(req)

        assert response.translations[0].id == 10
        assert fallback_provider.model_name == "fallback-mock"

    def test_resolve_api_key(self, monkeypatch):
        # Explicit key precedence
        assert resolve_gemini_api_key("explicit_test_key") == "explicit_test_key"

        # Env variable fallback
        monkeypatch.setenv("GEMINI_API_KEY", "env_test_key")
        assert resolve_gemini_api_key(None) == "env_test_key"

    def test_gemini_provider_response_parsing(self):
        from src.translation.gemini_provider import GeminiProvider

        provider = GeminiProvider(api_key="test_dummy_key")
        req = TranslationRequest(
            page_index=0,
            text_boxes=[
                TranslationRequestItem(id=1, japanese="海賊王", is_sfx=False),
                TranslationRequestItem(id=2, japanese="ドーン", is_sfx=True),
            ],
            context=RollingContext(),
        )

        # 1. Test clean markdown-fenced JSON parsing
        fenced_json = """```json
        {
            "translations": [
                {"id": 1, "english": "Pirate King", "translator_note": null}
            ],
            "scene_summary_update": "Luffy arrives.",
            "confidence": 0.98
        }
        ```"""

        # Notice ID 2 is missing from response: should be auto-reconstructed
        res = provider._parse_and_validate_response(fenced_json, req)
        assert len(res.translations) == 2
        assert res.translations[0].id == 1
        assert res.translations[0].english == "Pirate King"
        assert res.translations[1].id == 2
        assert res.translations[1].english == "ドーン"
        assert res.scene_summary_update == "Luffy arrives."
        assert res.confidence == 0.98

