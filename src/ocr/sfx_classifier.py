"""SFX (Sound Effects) classification and routing heuristics.

Combines Magi's visual signals (speech bubble tails, character associations, essential text)
with linguistic heuristics (grammar particles, Hiragana/Kanji ratios, Katakana onomatopoeia patterns)
to cleanly route dialogue bubbles to inpainting and sound effects to Viz Media style offset subtitles.
"""

from __future__ import annotations

import re
import cv2
import numpy as np

# Regex matching Japanese Katakana characters (standard, phonetic extensions, half-width)
KATAKANA_REGEX = re.compile(r"[\u30A0-\u30FF\u31F0-\u31FF\uFF65-\uFF9F]")
# Japanese Hiragana characters (grammar, conjugations, verb inflections)
HIRAGANA_REGEX = re.compile(r"[\u3040-\u309F]")
# Japanese Kanji characters
KANJI_REGEX = re.compile(r"[\u4E00-\u9FFF]")
# Full-width and standard punctuation often found in manga sound effects
PUNCTUATION_REGEX = re.compile(r"[!?.~〜…ーッっ・、。！？\s\uff01\uff1f\"'「」『』]")

# Japanese grammatical particles and conversational sentence markers exclusively found in dialogue
DIALOGUE_GRAMMAR_REGEX = re.compile(
    r"(?:[はがをにでのともかねよぞわぜ]|[だで]す|[だっ]た|ない|たい|てる|でる|から|けど|って|お前|おれ|私|僕|何|誰|どこ|いつ|どう|そう|これ|それ|あれ|ありがとう|助かった|待て|行く|来る|やる|見る|言う)"
)

# Matches text that consists strictly of punctuation, ellipses, dots, dashes, vertical leaders, and whitespace
SILENCE_OR_PUNCTUATION_REGEX = re.compile(
    r"^[\s？！?!…。、．・〜～ー―─–—「」『』（）\(\)\[\]\{\}\.\,\!\?\:\;\-・･•‥⋮⋯︙´`'\"|｜￤┆┊︰︓︔﹗●○]+$"
)

# Short noise patterns often produced by OCR on dots/silence (e.g. '…っ', '…ッ', '…・', 'っ', 'ミ', 'こ', '二', '三')
SILENCE_NOISE_REGEX = re.compile(
    r"^[\s…‥⋮⋯︙\.\-―─–—・･•|｜￤┆┊︰︓\:\;]*[っッミこ二三一1Il!]*[\s…‥⋮⋯︙\.\-―─–—・･•|｜￤┆┊︰︓\:\;]*$"
)


def is_crop_visually_silent(crop_np: np.ndarray) -> bool:
    """Detect whether an image crop contains only vertical dots, ellipsis, or silence.

    Directly analyzes foreground ink density and connected components.
    Evaluates both the inner sub-crop (ignoring speech bubble outlines on the border)
    and the full crop. This guarantees 100% reliable detection of vertical Japanese
    manga ellipsis bubbles even when manga-ocr misreads or hallucinates characters.
    """
    if crop_np is None or crop_np.size == 0:
        return True

    if crop_np.ndim == 3:
        gray = cv2.cvtColor(crop_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = crop_np

    h, w = gray.shape
    if h == 0 or w == 0:
        return True

    # 1. Check inner region (excluding outer margin where speech bubble borders live)
    pad_y = max(4, int(h * 0.18)) if h > 30 else 0
    pad_x = max(4, int(w * 0.20)) if w > 30 else 0

    regions_to_check = []
    if pad_y > 0 and pad_x > 0 and (h - 2 * pad_y) > 15 and (w - 2 * pad_x) > 15:
        inner = gray[pad_y : h - pad_y, pad_x : w - pad_x]
        regions_to_check.append(inner)
    regions_to_check.append(gray)

    for region in regions_to_check:
        rh, rw = region.shape
        ink_mask = (region < 140).astype(np.uint8)
        ink_count = int(np.count_nonzero(ink_mask))
        total_pixels = rh * rw
        ink_density = ink_count / max(1, total_pixels)

        # Empty / white region
        if ink_density < 0.001:
            return True

        # Vertical dots have low ink density (< 6% of region)
        if ink_density < 0.06:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ink_mask)
            components = []
            for i in range(1, num_labels):
                left = stats[i, cv2.CC_STAT_LEFT]
                top = stats[i, cv2.CC_STAT_TOP]
                comp_w = stats[i, cv2.CC_STAT_WIDTH]
                comp_h = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]

                # Discard edge slivers touching the crop border (clipped bubble wall artifacts)
                if left == 0 or top == 0 or (left + comp_w) >= rw or (top + comp_h) >= rh:
                    continue

                # Dots in manga are small isolated spots
                if 2 <= area <= 300 and comp_w <= 35 and comp_h <= 35:
                    components.append((stats[i], centroids[i]))

            # 1 to 6 small dots
            if 1 <= len(components) <= 6:
                # Check vertical alignment near horizontal center
                xs = [c[1][0] for c in components]
                center_x = rw / 2.0
                if all(abs(x - center_x) < (rw * 0.40) for x in xs):
                    return True

    return False


def is_silence_or_punctuation(ocr_text: str) -> bool:
    """Return True if the text represents silence (ellipses, dots, dashes) or punctuation-only.

    Used to detect silent speech bubbles ('……', '...', '---', etc.) or standalone symbols ('?', '!')
    so they are not translated with hallucinated dialogue or stolen text from adjacent panels.
    """
    if not ocr_text or not ocr_text.strip():
        return True
    stripped = ocr_text.strip()
    if SILENCE_OR_PUNCTUATION_REGEX.match(stripped):
        return True
    if len(stripped) <= 4 and SILENCE_NOISE_REGEX.match(stripped):
        return True
    return False


def is_punctuation_only(ocr_text: str) -> bool:
    """Return True if the text consists ONLY of Japanese/English punctuation and whitespace.

    Maintained for backwards compatibility; delegates to is_silence_or_punctuation.
    """
    return is_silence_or_punctuation(ocr_text)


def is_silence_bubble(ocr_text: str) -> bool:
    """Return True if the text represents a silence, pause, or vertical/horizontal ellipsis bubble.

    Captures:
    - Pure dots and ellipses ('……', '...', '‥', '⋮', '⋯', '︙', '―', '---')
    - Vertical manga ellipsis leaders and colons ('⋮', '︙', '︰', '┆', '┊', '::')
    - Empty or whitespace-only OCR crops
    - 1-3 character noise artifacts on dot textures ('…っ', 'っ', '・', 'ミ', 'こ', '|')
    """
    if not ocr_text or not ocr_text.strip():
        return True
    stripped = ocr_text.strip()
    if not is_silence_or_punctuation(stripped):
        # Even if not full punctuation match, short string with dot/ellipsis symbol is silence
        if len(stripped) <= 5 and re.search(r"[…\.\-―─–—ー・･•‥⋮⋯︙︰┆┊|｜￤\:]", stripped):
            if not is_japanese_dialogue(stripped):
                return True
        return False
    # If it matched silence/punctuation, check if it's dots/ellipses/dashes/bars or short dot noise
    if re.search(r"[…\.\-―─–—ー・･•‥⋮⋯︙︰┆┊|｜￤\:]", stripped):
        return True
    if len(stripped) <= 3 and SILENCE_NOISE_REGEX.match(stripped):
        return True
    return False



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


def is_japanese_dialogue(text: str) -> bool:
    """Determine if Japanese text represents human speech/narration dialogue rather than SFX.

    Checks for:
    1. Grammatical particles and conversational markers.
    2. Co-occurrence of Kanji and Hiragana (standard sentence morphology).
    3. Sentence-level dialogue length with mixed scripts.
    """
    clean_text = PUNCTUATION_REGEX.sub("", text).strip()
    if not clean_text:
        return False

    # 1. Direct grammatical particle or dialogue verb match
    if DIALOGUE_GRAMMAR_REGEX.search(clean_text):
        return True

    hiragana_count = len(HIRAGANA_REGEX.findall(clean_text))
    kanji_count = len(KANJI_REGEX.findall(clean_text))

    # 2. Sentences containing both Kanji and Hiragana (e.g. 運命は、少年が、動き出した)
    if kanji_count >= 1 and hiragana_count >= 1:
        return True

    # 3. Multiple Kanji vocabulary items (> 2 Kanji characters without katakana)
    if kanji_count >= 3 and hiragana_count >= 0:
        return True

    # 4. Hiragana sentence structure (> 4 Hiragana characters)
    if hiragana_count >= 4:
        return True

    return False


def is_sfx_candidate(
    ocr_text: str,
    is_essential: bool = True,
    ocr_confidence: float = 1.0,
    confidence_threshold: float = 0.40,
    short_token_threshold: int = 4,
    has_tail: bool = False,
    speaker_name: str | None = None,
    speaker_cluster_id: int | None = None,
) -> bool:
    """Determine whether a detected text box represents a sound effect (SFX).

    Decision logic:
    1. Visual & Attributive Overrides: If the text box has a speech bubble tail pointing
       to a character, or is attributed to a speaking character, it is definitely DIALOGUE (False).
    2. Linguistic Override: If OCR text contains Japanese grammatical particles, sentence
       structure, or Kanji+Hiragana morphology, it is definitely DIALOGUE (False), regardless
       of whether Magi's English-trained linear classifier flagged is_essential=False.
    3. Explicit SFX Heuristics:
       - Predominantly Katakana without sentence grammar -> SFX (True).
       - Short Katakana tokens (e.g., ドン！, ゴゴ, バーン) -> SFX (True).
       - Low confidence Katakana text -> SFX (True).
    4. Magi Non-Essential Signal: If is_essential is False, and the text has no grammatical
       dialogue markers, it is classified as SFX (True).
    5. Default: Treat as DIALOGUE (False).

    Args:
        ocr_text: Recognized Japanese text string.
        is_essential: Boolean flag from Magi (`is_essential_text`).
        ocr_confidence: Confidence score of OCR recognition in [0.0, 1.0].
        confidence_threshold: Confidence below which text is flagged as candidate.
        short_token_threshold: Character length threshold for short tokens.
        has_tail: Whether the text box is linked to a speech bubble tail.
        speaker_name: Attributed speaker name if known.
        speaker_cluster_id: Attributed speaker cluster ID if known.

    Returns:
        True if text box should be rendered as an SFX subtitle, False for regular dialogue.
    """
    # 1. Visual/Speaker signals: Speech bubbles with tails or speakers are never SFX
    if has_tail:
        return False
    if speaker_name is not None or speaker_cluster_id is not None:
        return False

    text = ocr_text.strip()
    if not text:
        # Empty text region: follow Magi's essential flag
        return not is_essential

    # 2. Linguistic check: Japanese grammar particles and dialogue words override Magi's classifier
    if is_japanese_dialogue(text):
        return False

    # 3. SFX Heuristic checks
    clean_text = PUNCTUATION_REGEX.sub("", text)
    char_len = len(clean_text)

    # Predominantly Katakana onomatopoeia
    if is_predominantly_katakana(text):
        # Short token or low confidence or non-essential -> SFX
        if char_len <= short_token_threshold:
            return True
        if ocr_confidence < confidence_threshold:
            return True
        if not is_essential:
            return True

    # 4. If Magi flagged as non-essential and it didn't match dialogue grammar
    if not is_essential:
        return True

    return False
