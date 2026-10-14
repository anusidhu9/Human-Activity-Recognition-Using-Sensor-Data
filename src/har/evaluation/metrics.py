"""Classification metrics shared by all models (one implementation = fair comparison)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from har.config import ACTIVITY_LABELS

LABEL_IDS = list(ACTIVITY_LABELS)  # 1..6


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy, macro/weighted precision-recall-F1, per-class scores, confusion matrix.

    Labels are official activity ids 1-6; the confusion matrix rows are true
    classes and columns predicted classes, both in id order.
    """
    out: dict = {"accuracy": accuracy_score(y_true, y_pred)}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABEL_IDS,
                                                     average=avg, zero_division=0)
        out[avg] = {"precision": p, "recall": r, "f1": f}
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=LABEL_IDS,
                                                       average=None, zero_division=0)
    out["per_class"] = {
        aid: {"precision": p[i], "recall": r[i], "f1": f[i], "support": int(support[i])}
        for i, aid in enumerate(LABEL_IDS)
    }
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=LABEL_IDS)
    return out


def metrics_to_rows(m: dict) -> list[dict]:
    """Flatten ``compute_metrics`` output into rows for the long-format ``metrics`` table."""
    rows = [{"metric_name": "accuracy", "value": m["accuracy"]}]
    for avg in ("macro", "weighted"):
        rows += [{"metric_name": k, "average": avg, "value": v} for k, v in m[avg].items()]
    for aid, scores in m["per_class"].items():
        rows += [{"metric_name": k, "activity_id": aid, "value": v}
                 for k, v in scores.items() if k != "support"]
    return rows


def summary_line(m: dict) -> str:
    return (f"acc {m['accuracy']:.4f} | macro P {m['macro']['precision']:.4f} "
            f"R {m['macro']['recall']:.4f} F1 {m['macro']['f1']:.4f}")
