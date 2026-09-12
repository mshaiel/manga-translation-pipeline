"""Unit tests for evaluation metrics (CER, Detection, BLEU)."""

import pytest

from evaluation.eval_detection import (
    compute_bbox_iou,
    evaluate_detection_category,
    evaluate_speaker_attribution,
)
from evaluation.eval_ocr import calculate_cer, compute_levenshtein_distance
from evaluation.eval_translation import (
    compute_sentence_bleu,
    run_translation_ablation_benchmark,
)


class TestEvaluationMetrics:
    """Test suite for research and benchmarking evaluation scripts."""

    def test_levenshtein_and_cer(self):
        # 1 substitution: '海賊王' vs '海賊犬' (1 edit in 3 chars)
        dist = compute_levenshtein_distance("海賊王", "海賊犬")
        assert dist == 1

        cer, stats = calculate_cer(["海賊王"], ["海賊王"])
        assert cer == 0.0
        assert stats["character_accuracy_percent"] == 100.0

        cer_sub, stats_sub = calculate_cer(["海賊犬"], ["海賊王"])
        assert cer_sub == pytest.approx(1 / 3)

    def test_bbox_iou_and_detection_metrics(self):
        b1 = [0.0, 0.0, 10.0, 10.0]
        b2 = [0.0, 0.0, 10.0, 10.0]
        assert compute_bbox_iou(b1, b2) == pytest.approx(1.0)

        b3 = [5.0, 5.0, 15.0, 15.0]
        # Inter: 5x5=25, Union: 100+100-25=175 -> IoU = 25/175 = 0.1428
        assert compute_bbox_iou(b1, b3) == pytest.approx(25 / 175)

        metrics = evaluate_detection_category([[0.0, 0.0, 10.0, 10.0]], [[0.0, 0.0, 10.0, 10.0]], iou_threshold=0.5)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_speaker_attribution_accuracy(self):
        preds = [(0, 1), (1, 2), (2, 3)]
        gts = [(0, 1), (1, 2), (2, 4)]
        res = evaluate_speaker_attribution(preds, gts)
        assert res["correct_attributions"] == 2
        assert res["total_dialogue_boxes"] == 3
        assert res["speaker_accuracy_percent"] == pytest.approx(66.67, rel=1e-2)

    def test_bleu_and_translation_ablation(self):
        hyp = "I will become the Pirate King!"
        ref = "I will become the Pirate King!"
        score = compute_sentence_bleu(hyp, ref)
        assert score > 90.0

        samples = [
            {
                "japanese": "おれは海賊王になる！",
                "reference": "I'm gonna be King of the Pirates!",
                "ours_with_context": "I'm gonna be King of the Pirates!",
                "raw_baseline": "He becomes the king of pirates.",
            }
        ]
        res = run_translation_ablation_benchmark(samples)
        assert res["ours_with_context_bleu"] > res["raw_baseline_bleu"]
        assert res["context_lift_bleu"] > 0
