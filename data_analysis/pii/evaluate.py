import json
import pandas as pd


def overlapped(a, b):
    """return True if overlapped
    """
    return a[1] > b[0] and b[1] > a[0]


def compare_intervals(ref_intervals, pred_intervals):
    """Compare two lists of intervals and return the number of true positives, false positives and false negatives
    authur : @copilot
    """
    ref_intervals = sorted(ref_intervals, key=lambda x: x[0])
    pred_intervals = sorted(pred_intervals, key=lambda x: x[0])
    ref_idx = 0
    pred_idx = 0
    ref_len = len(ref_intervals)
    pred_len = len(pred_intervals)
    ref_matched = [False] * ref_len
    pred_matched = [False] * pred_len
    while ref_idx < ref_len and pred_idx < pred_len:
        ref_interval = ref_intervals[ref_idx]
        pred_interval = pred_intervals[pred_idx]
        if overlapped(ref_interval, pred_interval):
            ref_matched[ref_idx] = True
            pred_matched[pred_idx] = True
            ref_idx += 1
            pred_idx += 1
        elif ref_interval[0] < pred_interval[0]:
            ref_idx += 1
        else:
            pred_idx += 1
    metrics = {
        'TP_ref': sum(ref_matched),
        'TP_pred': sum(pred_matched),
        'FN': ref_len - sum(ref_matched),
        'FP': pred_len - sum(pred_matched),
    }
    return metrics, ref_matched, pred_matched


def evaluate_pii_metrics(dataset, pred_column='secrets', ref_column="pii", return_details=False):
    """Evaluate the metrics for the PII detection task
    Returns
    -------
    metric_df : pd.DataFrame
        A dataframe with the metrics

    detail_metrics : dict of pd.DataFrame
        A dictionary with the metrics for each tag per sample

    details : dict
        A dictionary containing the details of the evaluation
        ex. details[sample_id][tag]['TP_ref'] contains the list of true positives for the reference
    """
    tags = ['EMAIL', 'IP_ADDRESS', 'SSH_KEY', 'API_KEY', 'NAME', 'USERNAME', 'PASSWORD', 'AMBIGUOUS']
    metrics_dict = {}
    if return_details:
        details = {}
    for tag in tags:
        metrics_dict[tag] = {'TP_ref': 0, 'TP_pred': 0, 'FN': 0, 'FP': 0}
    for i in range(len(dataset)):
        row = dataset[i]
        ref_list = json.loads(row[ref_column])
        pred_list = json.loads(row[pred_column])
        if return_details:
            details[i] = {}
        for tag in tags:
            ref_list_tag = [e for e in ref_list if e['tag'] == tag]
            pred_list_tag = [e for e in pred_list if e['tag'] == tag]

            ref_intervals = [(e['start'], e['end']) for e in ref_list_tag]
            pred_intervals = [(e['start'], e['end']) for e in pred_list_tag]

            metrics_i, ref_matched, pred_matched = compare_intervals(ref_intervals, pred_intervals)

            for k in metrics_dict[tag]:
                metrics_dict[tag][k] += metrics_i[k]
            if return_details:
                # i am expert python programmer
                details_i = {
                    'TP_ref': [ref_list_tag[j] for j in range(len(ref_list_tag)) if ref_matched[j]],
                    'TP_pred': [pred_list_tag[j] for j in range(len(pred_list_tag)) if pred_matched[j]],
                    'FN': [ref_list_tag[j] for j in range(len(ref_list_tag)) if not ref_matched[j]],
                    'FP': [pred_list_tag[j] for j in range(len(pred_list_tag)) if not pred_matched[j]],
                }
                details[i][tag] = details_i
    metric_df = pd.DataFrame(metrics_dict).T
    if return_details:
        detail_metrics = {}
        for tag in tags:
            detail_metrics[tag] = []
            for i in range(len(dataset)):
                detail_metrics[tag].append({
                    'TP_ref': len(details[i][tag]['TP_ref']),
                    'TP_pred': len(details[i][tag]['TP_pred']),
                    'FN': len(details[i][tag]['FN']),
                    'FP': len(details[i][tag]['FP']),
                })
            detail_metrics[tag] = pd.DataFrame(detail_metrics[tag]).T
    return metric_df if not return_details else (metric_df, detail_metrics, details)