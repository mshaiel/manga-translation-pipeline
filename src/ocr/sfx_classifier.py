"""SFX (Sound Effects) classification and routing heuristics.

Combines Magi's visual `is_essential_text` signal with linguistic heuristics
(e.g., Katakana ratio, short token patterns, and OCR confidence) to route dialogue
bubbles to inpainting and sound effects to Viz Media style offset subtitles.
"""

from __future__ import annotations

import re

# Regex matching Japanese Katakana characters (standard, phonetic extensions, half-width)
KATAKANA_REGEX = re.compile(r"[\u30A0-\u30FF\u31F0-\u31FF\uFF65-\uFF9F]")
# Full-width and standard punctuation often found in manga sound effects
PUNCTUATION_REGEX = re.compile(r"[!?.~〜…ーッっ・、。！？\s\uff01\uff1f]")


def is_predominantly_katakana(text: str, threshold: float = 0.60) -> bool:
    """Check if the non-punctuation characters in a text are predominantly Katakana.

    Japanese manga sound effects are overwhelmingly written in Katakana.

    Args:
        text: Japanese text string.
        threshold: Minimum fraction of Katakana characters (0.0 to 1.0).

    Returns:
        True if the proportion of Katakana exceeds threshold, False otherwise.
    """
    clean_text = PUNCTUATION_REGEX.sub("", text)
    if not clean_text:
        return False

    katakana_count = len(KATAKANA_REGEX.findall(clean_text))
    return (katakana_count / len(clean_text)) >= threshold


def is_sfx_candidate(
    ocr_text: str,
    is_essential: bool = True,
    ocr_confidence: float = 1.0,
    confidence_threshold: float = 0.40,
    short_token_threshold: int = 2,
) -> bool:
    """Determine whether a detected text box represents a sound effect (SFX).

    Decision logic:
    1. Primary signal: Magi's `is_essential_text` flag. If False, it is dialogue-irrelevant SFX.
    2. Fallback heuristic: If OCR confidence is low, or text is very short (<= 2 chars) and
       predominantly Katakana, flag as SFX candidate.

    Args:
        ocr_text: Recognized Japanese text string.
        is_essential: Boolean flag from Magi (`is_essential_text`).
        ocr_confidence: Confidence score of OCR recognition in [0.0, 1.0].
        confidence_threshold: Confidence below which text is flagged as candidate.
        short_token_threshold: Character length threshold for short tokens.

    Returns:
        True if text box should be rendered as an SFX subtitle, False for regular dialogue.
    """
    # 1. Primary signal from Magi detector
    if not is_essential:
        return True

    text = ocr_text.strip()
    if not text:
        return False

    # 2. Heuristic checks for edge cases where Magi missed SFX tagging
    clean_text = PUNCTUATION_REGEX.sub("", text)
    char_len = len(clean_text)

    # Heuristic A: Very short Katakana token (e.g., ドン, バーン, ドキ, ゴゴ)
    if 0 < char_len <= short_token_threshold and is_predominantly_katakana(text):
        return True

    # Heuristic B: Low OCR confidence combined with high Katakana proportion
    if ocr_confidence < confidence_threshold and is_predominantly_katakana(text):
        return True

    return False
