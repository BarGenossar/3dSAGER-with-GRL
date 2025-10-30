from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import numpy as np


def compute_metrics(preds, labels, threshold=0.5):
    preds = np.asarray(preds)

    # Case 1: already class indices (0/1)
    if preds.ndim == 1 and set(np.unique(preds)).issubset({0,1}):
        preds_bin = preds
        auc = None  # can't compute AUC without probabilities
    else:
        # Case 2: probabilities (floats)
        preds_bin = (preds >= threshold).astype(int)

    return {
        "accuracy": round(accuracy_score(labels, preds_bin), 3),
        "precision": round(precision_score(labels, preds_bin, zero_division=0), 3),
        "recall": round(recall_score(labels, preds_bin, zero_division=0), 3),
        "f1": round(f1_score(labels, preds_bin, zero_division=0), 3),
    }