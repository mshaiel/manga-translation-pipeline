"""Translation quality evaluation and ablation benchmarking.

Evaluates BLEU score against professional English references, comparing:
    1. Full Pipeline (With 6-page rolling context memory & visual speaker diarization).
    2. Baseline (Raw isolated LLM translation without context).
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import re
from typing import Any

logger = logging.getLogger(__name__)


def tokenize_words(text: str) -> list[str]:
    """Tokenize English text into lowercase words."""
    return re.findall(r"\b\w+\b", text.lower())


def compute_sentence_bleu(hypothesis: str, reference: str, max_n: int = 4) -> float:
    """Compute sentence-level BLEU score with smoothing and brevity penalty.

    Args:
        hypothesis: Machine-translated candidate string.
        reference: Reference human translation.
        max_n: Maximum n-gram order (default 4).

    Returns:
        BLEU score between 0.0 and 100.0.
    """
    hyp_tokens = tokenize_words(hypothesis)
    ref_tokens = tokenize_words(reference)

    if not hyp_tokens or not ref_tokens:
        return 0.0

    # Brevity penalty
    c = len(hyp_tokens)
    r = len(ref_tokens)
    bp = math.exp(min(0.0, 1.0 - (r / max(1, c))))

    precisions: list[float] = []

    for n in range(1, max_n + 1):
        hyp_ngrams = collections.Counter(tuple(hyp_tokens[i : i + n]) for i in range(len(hyp_tokens) - n + 1))
        ref_ngrams = collections.Counter(tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1))

        if not hyp_ngrams:
            continue

        clipped_count = sum(min(count, ref_ngrams[ng]) for ng, count in hyp_ngrams.items())
        total_count = sum(hyp_ngrams.values())
        precisions.append((clipped_count + 1e-4) / total_count)

    if not precisions:
        return 0.0

    log_sum = sum(math.log(p) for p in precisions) / len(precisions)
    bleu = bp * math.exp(log_sum) * 100.0
    return round(bleu, 2)


def evaluate_corpus_bleu(hypotheses: list[str], references: list[str]) -> float:
    """Calculate mean BLEU score across a corpus of sentences."""
    if not hypotheses or not references:
        return 0.0

    scores = [
        compute_sentence_bleu(h, r)
        for h, r in zip(hypotheses, references, strict=False)
    ]
    return round(sum(scores) / len(scores), 2)


def run_translation_ablation_benchmark(
    samples: list[dict[str, str]],
) -> dict[str, Any]:
    """Run ablation comparison between full context pipeline vs raw baseline.

    Args:
        samples: List of dicts with keys:
                 'japanese', 'reference', 'ours_with_context', 'raw_baseline'

    Returns:
        Dictionary containing comparative BLEU scores and lift metrics.
    """
    refs = [s["reference"] for s in samples]
    ours = [s["ours_with_context"] for s in samples]
    baseline = [s["raw_baseline"] for s in samples]

    ours_bleu = evaluate_corpus_bleu(ours, refs)
    base_bleu = evaluate_corpus_bleu(baseline, refs)
    lift = ours_bleu - base_bleu

    return {
        "num_evaluated_dialogue_lines": len(samples),
        "ours_with_context_bleu": ours_bleu,
        "raw_baseline_bleu": base_bleu,
        "context_lift_bleu": round(lift, 2),
        "percentage_improvement": round((lift / max(1e-4, base_bleu)) * 100.0, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Translation BLEU and Context Ablation")
    parser.add_argument("--data", type=str, required=True, help="Path to evaluation JSON")
    args = parser.parse_args()

    with open(args.data, encoding="utf-8") as f:
        samples = json.load(f)

    results = run_translation_ablation_benchmark(samples)
    print("=" * 60)
    print("TRANSLATION QUALITY ABLATION BENCHMARK (BLEU Score)")
    print("=" * 60)
    print(f"Total Lines Evaluated:            {results['num_evaluated_dialogue_lines']}")
    print(f"Ours (Rolling Context + Diarization): {results['ours_with_context_bleu']} BLEU")
    print(f"Baseline (Raw Isolated LLM):      {results['raw_baseline_bleu']} BLEU")
    print(f"Net Improvement from Context:     +{results['context_lift_bleu']} BLEU ({results['percentage_improvement']}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
