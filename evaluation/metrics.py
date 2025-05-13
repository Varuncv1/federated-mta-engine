#!/usr/bin/env python3
"""
metrics.py: Evaluation metrics for attribution modeling and federated learning.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score


def compute_classification_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute common classification metrics: AUC, log loss, accuracy, precision, recall.

    Args:
        y_true: True binary labels (0 or 1).
        y_pred_proba: Predicted probabilities for the positive class.
        threshold: Threshold to convert probabilities to class labels.

    Returns:
        Dictionary of metrics.
    """
    # Ensure numpy arrays
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    # Binarize predictions
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        'auc': roc_auc_score(y_true, y_pred_proba),
        'log_loss': log_loss(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred),
    }
    # Optional: precision, recall
    try:
        from sklearn.metrics import precision_score, recall_score
        metrics['precision'] = precision_score(y_true, y_pred)
        metrics['recall'] = recall_score(y_true, y_pred)
    except ImportError:
        pass

    return metrics


def aggregate_attribution_metrics(credit_df: pd.DataFrame, y_true: pd.Series) -> dict:
    """
    Evaluate attribution credit against actual conversions.

    Args:
        credit_df: DataFrame with columns ['user_id', 'credit'], total credit per user across channels.
        y_true: Series mapping user_id to true conversion label.

    Returns:
        Dictionary with correlation and MAE of attribution credit vs conversion.
    """
    # Merge user-level credit with truth
    merged = credit_df.groupby('user_id')['credit'].sum().rename('pred_credit').reset_index()
    truth = y_true.reset_index().rename(columns={y_true.name: 'actual'})
    df = pd.merge(merged, truth, on='user_id', how='inner')

    # Compute metrics
    corr = df['pred_credit'].corr(df['actual'])
    mae = np.mean(np.abs(df['pred_credit'] - df['actual']))

    return {'credit_conversion_correlation': corr, 'credit_mae': mae}


def summary_report(metrics: dict) -> pd.DataFrame:
    """
    Summarize metrics into a pandas DataFrame for reporting.

    Args:
        metrics: Dictionary of metric names to values.

    Returns:
        DataFrame with columns ['metric', 'value'].
    """
    report = pd.DataFrame([{'metric': k, 'value': v} for k, v in metrics.items()])
    return report


if __name__ == "__main__":
    # Example usage
    import os
    # Load true labels and model scores
    # Placeholder: implement loading logic as needed
    print("Run compute_classification_metrics or aggregate_attribution_metrics from your evaluation scripts.")
