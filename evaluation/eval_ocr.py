"""Evaluation script for Japanese Manga OCR Character Error Rate (CER).

Measures character-level transcription accuracy of manga-ocr against ground-truth
Japanese text annotations (e.g. Manga109 benchmark).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def compute_levenshtein_distance(seq1: str, seq2: str) -> int:
    """Compute the minimum edit distance (insertions, deletions, substitutions) between two sequences."""
    n1, n2 = len(seq1), len(seq2)
    dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]

    for i in range(n1 + 1):
        dp[i][0] = i
    for j in range(n2 + 1):
        dp[0][j] = j

    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # deletion
                dp[i][j - 1] + 1,        # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )

    return dp[n1][n2]


def calculate_cer(predictions: list[str], references: list[str]) -> tuple[float, dict[str, float]]:
    """Calculate Character Error Rate (CER) across a set of predicted and ground-truth Japanese strings.

    CER = (Substitutions + Insertions + Deletions) / Total Reference Characters

    Args:
        predictions: List of transcribed Japanese strings from manga-ocr.
        references: List of corresponding ground-truth Japanese strings.

    Returns:
        Tuple of (overall_cer, metrics_dict).
    """
    if len(predictions) != len(references):
        raise ValueError(f"Mismatch: {len(predictions)} predictions vs {len(references)} references.")

    total_distance = 0
    total_ref_chars = 0
    exact_matches = 0

    for pred, ref in zip(predictions, references, strict=False):
        pred_clean = pred.replace(" ", "").strip()
        ref_clean = ref.replace(" ", "").strip()

        dist = compute_levenshtein_distance(pred_clean, ref_clean)
        total_distance += dist
        total_ref_chars += len(ref_clean)

        if pred_clean == ref_clean:
            exact_matches += 1

    overall_cer = total_distance / max(1, total_ref_chars)
    accuracy = (1.0 - overall_cer) * 100.0
    exact_match_ratio = (exact_matches / max(1, len(references))) * 100.0

    metrics = {
        "cer": round(overall_cer, 4),
        "character_accuracy_percent": round(max(0.0, accuracy), 2),
        "exact_match_percent": round(exact_match_ratio, 2),
        "total_evaluated_samples": len(references),
        "total_reference_characters": total_ref_chars,
    }

    return overall_cer, metrics


def main() -> None:
    """CLI runner for CER evaluation on a JSON benchmark dataset."""
    parser = argparse.ArgumentParser(description="Evaluate Manga OCR CER accuracy")
    parser.add_argument("--data", type=str, required=True, help="Path to JSON file with [{'pred': ..., 'ref': ...}]")
    args = parser.parse_args()

    data_path = Path(args.data)
    with open(data_path, encoding="utf-8") as f:
        samples = json.load(f)

    preds = [s["pred"] for s in samples]
    refs = [s["ref"] for s in samples]

    cer, stats = calculate_cer(preds, refs)
    print("=" * 50)
    print("Manga OCR Evaluation Results:")
    print(f"Character Error Rate (CER): {stats['cer']:.4f}")
    print(f"Character Accuracy:         {stats['character_accuracy_percent']}%")
    print(f"Exact Match Rate:           {stats['exact_match_percent']}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
