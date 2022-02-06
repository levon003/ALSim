import numpy as np
import sklearn
import sklearn.metrics
from collections import defaultdict


def compute_metrics(y_true, y_pred, y_pred_proba, compute_or=False):
    assert len(y_true) > 0
    assert len(y_pred) > 0
    true_pos_pct = np.sum(y_true) / len(y_true)
    pred_pos_pct = np.sum(y_pred) / len(y_pred)
    pos_pct_difference = pred_pos_pct - true_pos_pct
    precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    cmat = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=(0, 1))
    tn, fp, fn, tp = cmat.ravel()

    if true_pos_pct == 0.0 or true_pos_pct == 1.0:
        ap = 0.0
        roc_auc = 0.0
    else:  # at least one true instance of each class
        ap = sklearn.metrics.average_precision_score(y_true, y_pred_proba)
        roc_auc = sklearn.metrics.roc_auc_score(y_true, y_pred_proba)

    metrics_dict = {
        "n": len(y_true),
        "true_pos_count": np.sum(y_true),
        "pred_pos_count": np.sum(y_pred),
        "true_pos_pct": true_pos_pct,
        "pred_pos_pct": pred_pos_pct,
        "pos_pct_difference": pos_pct_difference,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap": ap,
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }
    if compute_or:
        if fp == 0 or tn == 0:
            odds_ratio = 0.0
        else:
            odds_ratio = (tp / fp) / (fn / tn)
        metrics_dict["odds_ratio"] = odds_ratio
    return metrics_dict


def combine_model_metrics(model_metrics_list, ignore_list=["model_ind", "n"]):
    assert len(model_metrics_list) > 0
    keys = model_metrics_list[0].keys()
    combined_metrics = {}
    for key in keys:
        if key in ignore_list:
            continue
        metric_values = [mm[key] for mm in model_metrics_list]
        value_mean = np.mean(metric_values)
        value_std = np.std(metric_values)
        combined_metrics[key + "_mean"] = value_mean
        combined_metrics[key + "_std"] = value_std
    return combined_metrics


def combine_run_metrics(run_metrics_list):
    """
    Mean pool the available metric values (and store the standard deviation as well).

    Note this can produce weird outcomes where a value is not available in some subset of the batches.
    """
    n_batches = len(run_metrics_list[0])
    combined_run_metric_values_list = [defaultdict(list) for i in range(n_batches)]
    for batch_metrics_list in run_metrics_list:
        for batch_ind, batch_metrics in enumerate(batch_metrics_list):
            metric_values = combined_run_metric_values_list[batch_ind]
            for key in batch_metrics.keys():
                value = batch_metrics[key]
                if type(value) == list:
                    raise ValueError("not implemented")
                elif type(value) == dict:
                    for subkey in value.keys():
                        assert type(value[subkey]) != dict
                        metric_values[key + "_" + subkey].append(value[subkey])
                else:
                    metric_values[key].append(value)
    combined_run_metrics_list = []
    for i in range(n_batches):
        combined_run_metrics = {}
        metric_values = combined_run_metric_values_list[i]
        for key, values in metric_values.items():
            value_mean = np.mean(values)
            value_std = np.std(values)
            combined_run_metrics[key + "_mean"] = value_mean
            combined_run_metrics[key + "_std"] = value_std
        combined_run_metrics_list.append(combined_run_metrics)
    return combined_run_metrics_list
