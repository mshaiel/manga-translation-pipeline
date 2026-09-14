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
        # Non-essential text without Japanese sentence grammar is routed to SFX
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

    def test_dialogue_overrides_magi_non_essential(self):
        # Crucial bug fix: Japanese dialogue speech bubbles misclassified by Magi as non-essential
        # must NOT be routed to SFX!
        assert is_sfx_candidate("おれは海賊王になる！", is_essential=False) is False
        assert is_sfx_candidate("お前は誰だ？", is_essential=False) is False
        assert is_sfx_candidate("ありがとう、助かったよ。", is_essential=False) is False
        assert is_sfx_candidate("待て！そこから先へは行かせない！", is_essential=False) is False

    def test_bubble_tail_and_speaker_override(self):
        # Text with a speech bubble tail is dialogue
        assert is_sfx_candidate("ドン", is_essential=False, has_tail=True) is False
        # Text with attributed speaker is dialogue
        assert is_sfx_candidate("バーン", is_essential=False, speaker_cluster_id=1) is False

    def test_is_punctuation_only(self):
        from src.ocr.sfx_classifier import is_punctuation_only

        assert is_punctuation_only("？") is True
        assert is_punctuation_only("！") is True
        assert is_punctuation_only("！？") is True
        assert is_punctuation_only("…") is True
        assert is_punctuation_only("?!") is True
        assert is_punctuation_only(" ... ") is True
        assert is_punctuation_only("") is True
        assert is_punctuation_only("  ") is True
        # Contains letters/words -> False
        assert is_punctuation_only("何？") is False
        assert is_punctuation_only("What?") is False
        assert is_punctuation_only("ドン！") is False

    def test_is_silence_bubble(self):
        from src.ocr.sfx_classifier import is_silence_bubble

        # Ellipses, dots, leaders, dashes, and empty strings
        assert is_silence_bubble("") is True
        assert is_silence_bubble("   ") is True
        assert is_silence_bubble("……") is True
        assert is_silence_bubble("...") is True
        assert is_silence_bubble("‥") is True
        assert is_silence_bubble("⋮⋮") is True
        assert is_silence_bubble("︙") is True
        assert is_silence_bubble("―――") is True
        assert is_silence_bubble("---") is True
        assert is_silence_bubble("…っ") is True
        assert is_silence_bubble(" … ") is True
        assert is_silence_bubble("っ") is True
        assert is_silence_bubble("・") is True
        assert is_silence_bubble("こ") is True

        # Standalone punctuation without dots is not a silence bubble
        assert is_silence_bubble("？") is False
        assert is_silence_bubble("！") is False
        assert is_silence_bubble("！？") is False

        # Actual dialogue is not a silence bubble
        assert is_silence_bubble("もちろんモチちゃんとも") is False
        assert is_silence_bubble("待て！") is False

