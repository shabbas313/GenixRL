"""
Metrics-computation and decision curve analysis.
"""
import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    precision_score, recall_score, f1_score, matthews_corrcoef
)

def compute_metrics(y_true, y_pred, threshold):
    """
    Compute evaluation metrics.
    """
    auc_score = roc_auc_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc_score = auc(recall, precision)
    pred_labels = [1 if p >= threshold else 0 for p in y_pred]
    precision_score_val = precision_score(y_true, pred_labels, zero_division=0)
    recall_score_val = recall_score(y_true, pred_labels, zero_division=0)
    f1_score_val = f1_score(y_true, pred_labels, zero_division=0)
    mcc_score_val = matthews_corrcoef(y_true, pred_labels)
    return {
        "auc": auc_score,
        "pr_auc": pr_auc_score,
        "precision": precision_score_val,
        "recall": recall_score_val,
        "f1": f1_score_val,
        "mcc": mcc_score_val,
        "Threshold": threshold,
    }

def evaluate_sub_models(preds_valid, y_valid, model_names, thresholds):
    """
    Evaluate sub-models on validation set.
    """
    sub_model_metrics = {}
    for name in model_names:
        metrics = compute_metrics(y_valid, preds_valid[name], thresholds[name])
        sub_model_metrics[name] = metrics
        print(f"\nSub-Model {name} Metrics on RL Validation Set:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    return sub_model_metrics

def decision_curve_analysis(y_true, y_pred, thresholds=None):
    """
    Perform decision curve analysis.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only binary labels (0 or 1)")
    if not np.all((y_pred >= 0) & (y_pred <= 1)):
        raise ValueError("y_pred must contain probabilities between 0 and 1")
    
    if thresholds is None:
        thresholds = np.arange(0.01, 0.96, 0.01)
    thresholds = np.array(thresholds)
    if not np.all((thresholds >= 0) & (thresholds <= 1)):
        raise ValueError("Thresholds must be between 0 and 1")

    n = len(y_true)
    if n == 0:
        raise ValueError("Empty input arrays")
    prevalence = np.mean(y_true)

    net_benefits = []
    net_benefits_all = []
    net_benefits_none = []

    for p_t in thresholds:
        y_pred_labels = (y_pred >= p_t).astype(int)
        tp = np.sum((y_pred_labels == 1) & (y_true == 1))
        fp = np.sum((y_pred_labels == 1) & (y_true == 0))
        w = p_t / (1 - p_t) if p_t < 1 else float("inf")
        net_benefit = (tp - w * fp) / n if n > 0 and not np.isinf(w) else 0.0
        net_benefits.append(net_benefit)
        
        nb_all_pt = prevalence - w * (1 - prevalence) if not np.isinf(w) else 0.0
        net_benefits_all.append(max(nb_all_pt, 0.0))
        net_benefits_none.append(0.0)
        
    return thresholds, net_benefits, net_benefits_all, net_benefits_none