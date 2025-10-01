"""fraud_evaluator.py

Load a CSV of transactions with ground truth and predictions and compute
classification metrics (accuracy, precision, recall, F1, confusion matrix).
If a probability column is found, compute ROC AUC (simple trapezoidal AUC).

Usage examples:

# Basic (defaults: truth=isFraud, infer pred from pred_label or pred_prob)
python3 fraud_evaluator.py --input transactions_with_preds.csv

# Specify custom columns
python3 fraud_evaluator.py --input out.csv --truth-col actual_is_fraud --prob-col score --threshold 0.6

# Save metrics to JSON
python3 fraud_evaluator.py --input out.csv --output-json metrics.json

This script intentionally avoids sklearn to keep dependencies minimal.
"""

from __future__ import annotations
import argparse
import json
import math
from typing import Optional, Tuple
import pandas as pd
import numpy as np

DEFAULT_TRUTH_COL = 'isFraud'
POSSIBLE_PROB_COLS = ['pred_prob', 'probability', 'score', 'fraud_prob', 'prob']
POSSIBLE_PRED_COLS = ['pred_label', 'prediction', 'predicted', 'pred']


def load_dataframe(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def infer_columns(df: pd.DataFrame, truth_col: Optional[str], prob_col: Optional[str], pred_col: Optional[str]) -> Tuple[str, Optional[str], Optional[str]]:
    # Determine truth column
    if truth_col and truth_col in df.columns:
        truth = truth_col
    elif DEFAULT_TRUTH_COL in df.columns:
        truth = DEFAULT_TRUTH_COL
    else:
        raise ValueError(f"Truth column not found: provided {truth_col}, tried default '{DEFAULT_TRUTH_COL}'. Available columns: {list(df.columns)}")

    # Determine probability column
    prob = None
    if prob_col and prob_col in df.columns:
        prob = prob_col
    else:
        for c in POSSIBLE_PROB_COLS:
            if c in df.columns:
                prob = c
                break

    # Determine predicted label column
    pred = None
    if pred_col and pred_col in df.columns:
        pred = pred_col
    else:
        for c in POSSIBLE_PRED_COLS:
            if c in df.columns:
                pred = c
                break

    return truth, prob, pred


def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    # y_true and y_pred are 0/1 arrays
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, fp, tn, fn


def compute_metrics(tp: int, fp: int, tn: int, fn: int) -> dict:
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else float('nan')
    precision = tp / (tp + fp) if (tp + fp) else float('nan')
    recall = tp / (tp + fn) if (tp + fn) else float('nan')
    if math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0:
        f1 = float('nan')
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return {
        'total': int(total),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    # Sort by decreasing score
    desc_idx = np.argsort(-y_score)
    y_true_sorted = y_true[desc_idx]
    y_score_sorted = y_score[desc_idx]

    # TPR and FPR at each threshold (unique scores)
    thresholds = np.unique(y_score_sorted)
    tprs = []
    fprs = []
    P = float((y_true == 1).sum())
    N = float((y_true == 0).sum())
    if P == 0 or N == 0:
        return float('nan'), np.array([]), np.array([])

    for t in thresholds:
        preds = (y_score >= t).astype(int)
        tp = int(((y_true == 1) & (preds == 1)).sum())
        fp = int(((y_true == 0) & (preds == 1)).sum())
        tprs.append(tp / P)
        fprs.append(fp / N)

    # Add (0,0) and (1,1) endpoints for proper trapezoid integration
    fprs_arr = np.concatenate(([0.0], np.array(fprs), [1.0]))
    tprs_arr = np.concatenate(([0.0], np.array(tprs), [1.0]))

    # Sort by FPR increasing for trapezoid rule
    order = np.argsort(fprs_arr)
    fprs_arr = fprs_arr[order]
    tprs_arr = tprs_arr[order]

    auc = 0.0
    for i in range(1, len(fprs_arr)):
        x_diff = fprs_arr[i] - fprs_arr[i - 1]
        y_avg = (tprs_arr[i] + tprs_arr[i - 1]) / 2.0
        auc += x_diff * y_avg

    return float(auc), fprs_arr, tprs_arr


def evaluate(df: pd.DataFrame, truth_col: str, prob_col: Optional[str], pred_col: Optional[str], threshold: float = 0.5) -> dict:
    # Normalize truth to 0/1
    y_true_raw = df[truth_col]
    # Accept booleans, 0/1, or 'True'/'False' strings
    y_true = pd.Series(y_true_raw).apply(lambda x: 1 if (str(x).lower() in ['1', 'true', 't', 'yes']) or x == 1 or x is True else 0).to_numpy(dtype=int)

    # Determine predictions
    if pred_col is not None and pred_col in df.columns:
        y_pred = df[pred_col].apply(lambda x: 1 if (str(x).lower() in ['1', 'true', 't', 'yes']) or x == 1 or x is True else 0).to_numpy(dtype=int)
    elif prob_col is not None and prob_col in df.columns:
        y_scores = df[prob_col].to_numpy(dtype=float)
        y_pred = (y_scores >= threshold).astype(int)
    else:
        raise ValueError('No prediction column found (neither pred_col nor prob_col provided/found)')

    tp, fp, tn, fn = compute_confusion(y_true, y_pred)
    metrics = compute_metrics(tp, fp, tn, fn)

    # Compute AUC if scores available
    auc = None
    roc_curve = None
    if prob_col and prob_col in df.columns:
        y_scores = df[prob_col].to_numpy(dtype=float)
        auc, fprs, tprs = compute_roc_auc(y_true, y_scores)
        roc_curve = {'fprs': fprs.tolist(), 'tprs': tprs.tolist()}

    out = {'metrics': metrics}
    if auc is not None:
        out['roc_auc'] = auc
    if roc_curve is not None:
        out['roc_curve'] = roc_curve

    return out


def main():
    parser = argparse.ArgumentParser(description='Evaluate fraud detection performance from CSV')
    parser.add_argument('--input', '-i', required=True, help='Path to CSV file with columns for ground truth and predictions')
    parser.add_argument('--truth-col', default=None, help=f'Ground-truth column name (default: {DEFAULT_TRUTH_COL} or autodetect)')
    parser.add_argument('--prob-col', default=None, help='Prediction probability column name (optional)')
    parser.add_argument('--pred-col', default=None, help='Prediction label column name (optional)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold to convert probabilities to labels (default: 0.5)')
    parser.add_argument('--output-json', default=None, help='Path to write JSON summary of metrics')
    args = parser.parse_args()

    df = load_dataframe(args.input)
    truth, prob, pred = infer_columns(df, args.truth_col, args.prob_col, args.pred_col)

    results = evaluate(df, truth, prob, pred, threshold=args.threshold)

    print('\n=== Evaluation Summary ===')
    m = results['metrics']
    print(f"Total samples: {m['total']}")
    print(f"Accuracy: {m['accuracy']:.4f}")
    print(f"Precision: {m['precision']:.4f}")
    print(f"Recall: {m['recall']:.4f}")
    print(f"F1: {m['f1']:.4f}")
    print(f"TP: {m['tp']}, FP: {m['fp']}, TN: {m['tn']}, FN: {m['fn']}")
    if 'roc_auc' in results:
        print(f"ROC AUC: {results['roc_auc']:.4f}")

    if args.output_json:
        with open(args.output_json, 'w') as fh:
            json.dump(results, fh, indent=2)
        print(f"Wrote JSON summary to {args.output_json}")


if __name__ == '__main__':
    main()
