"""Entity-level precision, recall, and F1. Outside tokens are not entities."""

from __future__ import annotations

from seqeval.metrics import f1_score, precision_score, recall_score


def entity_scores(true_sequences: list[list[str]], pred_sequences: list[list[str]]) -> dict[str, float]:
    if len(true_sequences) != len(pred_sequences):
        raise ValueError("True and predicted sequences differ in length")
    usable_true = []
    usable_pred = []
    for truth, prediction in zip(true_sequences, pred_sequences):
        if not truth and not prediction:
            continue
        usable_true.append(truth or ["O"])
        usable_pred.append(prediction or ["O"])
    if not usable_true:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "precision": float(precision_score(usable_true, usable_pred, zero_division=0)),
        "recall": float(recall_score(usable_true, usable_pred, zero_division=0)),
        "f1": float(f1_score(usable_true, usable_pred, zero_division=0)),
    }


def tags_from_label_ids(label_ids: list[int], predicted_ids: list[int], id2label: dict[int, str]) -> tuple[list[str], list[str]]:
    """Drop positions whose gold label is -100 (special tokens and extra subwords)."""

    truth: list[str] = []
    prediction: list[str] = []
    for gold, pred in zip(label_ids, predicted_ids):
        if int(gold) == -100:
            continue
        truth.append(id2label[int(gold)])
        prediction.append(id2label[int(pred)])
    return truth, prediction
