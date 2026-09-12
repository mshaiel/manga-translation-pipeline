"""Unit tests for SFX heuristic classifier."""

from src.ocr.sfx_classifier import is_predominantly_katakana, is_sfx_candidate


class TestSfxClassifier:
    """Test suite for SFX detection and classification logic."""

    def test_predominantly_katakana(self):
        assert is_predominantly_katakana("ドーン！") is True
        assert is_predominantly_katakana("ゴゴゴゴ…") is True
        assert is_predominantly_katakana("バーン") is True
        assert is_predominantly_katakana("お前は誰だ？") is False
        assert is_predominantly_katakana("ありがとう") is False

    def test_magi_non_essential_signal(self):
        # Even if text has kanji, if Magi says is_essential=False, it's SFX
        assert is_sfx_candidate("爆発", is_essential=False) is True
        assert is_sfx_candidate("ドン", is_essential=False) is True

    def test_dialogue_essential(self):
        # Standard dialogue is NOT SFX
        assert is_sfx_candidate("おれは海賊王になる！", is_essential=True) is False
        assert is_sfx_candidate("ここが目的地か…。", is_essential=True) is False

    def test_short_katakana_heuristic(self):
        # Even if is_essential is mistakenly True, short Katakana tokens trigger SFX
        assert is_sfx_candidate("ドン！", is_essential=True) is True
        assert is_sfx_candidate("ゴゴ", is_essential=True) is True
        assert is_sfx_candidate("バン", is_essential=True) is True

    def test_low_confidence_katakana(self):
        # Low confidence Katakana triggers SFX candidate
        assert is_sfx_candidate("パチパチパチ", is_essential=True, ocr_confidence=0.3) is True
