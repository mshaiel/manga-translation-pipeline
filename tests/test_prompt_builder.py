"""Unit tests for prompt builder in src/translation/prompt_builder.py."""

from src.translation.prompt_builder import build_dialogue_payload, build_translation_prompt
from src.translation.schemas import ChapterInfo, RollingContext, TextBox


class TestPromptBuilder:
    """Test suite for LLM translation prompt formatting and instructions."""

    def test_build_dialogue_payload(self):
        tb1 = TextBox(id=0, bbox=[0.1, 0.1, 0.4, 0.4], ocr_text="こんにちは", speaker_name="Naruto", is_sfx=False)
        tb2 = TextBox(id=1, bbox=[0.5, 0.1, 0.8, 0.4], ocr_text="ドーン！", is_sfx=True)

        payload = build_dialogue_payload([tb1, tb2])
        assert len(payload) == 2
        assert payload[0]["id"] == 0
        assert payload[0]["speaker"] == "Naruto"
        assert payload[0]["japanese"] == "こんにちは"
        assert payload[0]["is_sfx"] is False

        assert payload[1]["id"] == 1
        assert payload[1]["speaker"] is None
        assert payload[1]["japanese"] == "ドーン！"
        assert payload[1]["is_sfx"] is True

    def test_build_translation_prompt_standard(self):
        tb = TextBox(id=0, bbox=[0.1, 0.1, 0.4, 0.4], ocr_text="行くぞ！", speaker_name="Luffy")
        context = RollingContext(chapter_info=ChapterInfo(title="One Piece", user_context="Sea voyage"))

        prompt = build_translation_prompt([tb], context, vision_mode=False, request_scene_summary=False)

        assert "You are an expert Japanese-to-English manga translator." in prompt
        assert "CONTEXT:" in prompt
        assert "One Piece" in prompt
        assert "Sea voyage" in prompt
        assert "THIS PAGE'S DIALOGUE (in reading order):" in prompt
        assert "行くぞ！" in prompt
        assert "Luffy" in prompt
        assert "MULTIMODAL VERIFICATION" not in prompt

    def test_build_translation_prompt_vision_mode(self):
        tb = TextBox(id=0, bbox=[0.1, 0.1, 0.4, 0.4], ocr_text="???", speaker_name=None)
        context = RollingContext()

        prompt = build_translation_prompt([tb], context, vision_mode=True, request_scene_summary=True)

        assert "MULTIMODAL VERIFICATION" in prompt
        assert "SCENE SUMMARY" in prompt

    def test_build_dialogue_payload_with_translation_request_items(self):
        from src.translation.schemas import TranslationRequestItem

        item1 = TranslationRequestItem(id=0, speaker="Zoro", japanese="何だ？", is_sfx=False)
        item2 = TranslationRequestItem(id=1, speaker=None, japanese="ザッ", is_sfx=True)

        payload = build_dialogue_payload([item1, item2])
        assert len(payload) == 2
        assert payload[0]["id"] == 0
        assert payload[0]["speaker"] == "Zoro"
        assert payload[0]["japanese"] == "何だ？"
        assert payload[1]["is_sfx"] is True

        # Test prompt building directly with TranslationRequestItems
        context = RollingContext()
        prompt = build_translation_prompt([item1, item2], context)
        assert "何だ？" in prompt
        assert "Zoro" in prompt

    def test_translation_request_item_duck_typing(self):
        from src.translation.schemas import TranslationRequestItem

        item = TranslationRequestItem(id=5, speaker="Sanji", japanese="クソ野郎", is_sfx=False)
        # Verify duck typing properties match TextBox interface
        assert item.speaker_name == "Sanji"
        assert item.speaker_cluster_id is None
        assert item.ocr_text == "クソ野郎"

