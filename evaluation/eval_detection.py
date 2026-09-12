"""Evaluation metrics for Manga Detection and Speaker Diarization.

Evaluates:
    1. Bounding box mAP@50 for panels, text boxes, and characters.
    2. Speaker attribution accuracy (% of dialogue boxes linked to the correct speaker character).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence

logger = logging.getLogger(__name__)


def compute_bbox_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def evaluate_detection_category(
    predicted_boxes: list[list[float]],
    ground_truth_boxes: list[list[float]],
    iou_threshold: float = 0.50,
) -> dict[str, float]:
    """Compute Precision, Recall, and F1 score for a category of bounding boxes at specified IoU threshold."""
    if not ground_truth_boxes:
        return {"precision": 1.0 if not predicted_boxes else 0.0, "recall": 1.0, "f1": 1.0}

    matched_gt: set[int] = set()
    true_positives = 0

    for pred in predicted_boxes:
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truth_boxes):
            if gt_idx in matched_gt:
                continue
            iou = compute_bbox_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx != -1:
            matched_gt.add(best_gt_idx)
            true_positives += 1

    precision = true_positives / max(1, len(predicted_boxes))
    recall = true_positives / max(1, len(ground_truth_boxes))
    f1 = (2 * precision * recall) / max(1e-6, precision + recall)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": true_positives,
        "ground_truth_count": len(ground_truth_boxes),
        "prediction_count": len(predicted_boxes),
    }


def evaluate_speaker_attribution(
    predicted_associations: list[tuple[int, int]],
    ground_truth_associations: list[tuple[int, int]],
) -> dict[str, float]:
    """Calculate speaker diarization accuracy (% of speech bubbles correctly matched to character).

    Args:
        predicted_associations: List of (text_id, char_id) predictions.
        ground_truth_associations: List of (text_id, char_id) ground truths.

    Returns:
        Dictionary reporting accuracy and total count.
    """
    gt_map = dict(ground_truth_associations)
    pred_map = dict(predicted_associations)

    if not gt_map:
        return {"speaker_accuracy_percent": 100.0, "total_dialogue_boxes": 0}

    correct = 0
    total = len(gt_map)

    for text_id, target_char in gt_map.items():
        if pred_map.get(text_id) == target_char:
            correct += 1

    acc = (correct / total) * 100.0
    return {
        "speaker_accuracy_percent": round(acc, 2),
        "correct_attributions": correct,
        "total_dialogue_boxes": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Magi Detection and Speaker Diarization")
    parser.add_argument("--data", type=str, required=True, help="Path to evaluation JSON")
    args = parser.parse_args()

    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 50)
    print("Magi Detection & Diarization Benchmark:")
    print("Evaluated on:", args.data)
    print("Samples loaded:", len(data) if isinstance(data, list) else 1)
    print("=" * 50)


if __name__ == "__main__":
    main()
