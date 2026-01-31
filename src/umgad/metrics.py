from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

def evaluate(y_true: np.ndarray, scores: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    out = {}
    try:
        out["auc"] = float(roc_auc_score(y_true, scores))
    except Exception:
        out["auc"] = None
    out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["num_anom_true"] = int(y_true.sum())
    out["num_anom_pred"] = int(y_pred.sum())
    return out
