import json


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


def evaluate_pii_metrics(dataset, pred_column='secrets', ref_column="pii"):
    """Evaluate the metrics for the PII detection task
    """
    tags = ['EMAIL', 'IP_ADDRESS', 'SSH_KEY', 'API_KEY', 'NAME', 'USERNAME', 'PASSWORD', 'AMBIGUOUS']
    metrics_dict = {}
    for tag in tags:
        metrics_dict[tag] = {'TP_ref': 0, 'TP_pred': 0, 'FN': 0, 'FP': 0}
    for i in range(len(dataset)):
        row = dataset[i]
        ref_list = json.loads(row[ref_column])
        pred_list = json.loads(row[pred_column])
        for tag in tags:
            ref_intervals = [(e['start'], e['end']) for e in ref_list if e['tag'] == tag]
            pred_intervals = [(e['start'], e['end']) for e in pred_list if e['tag'] == tag]
            metrics_i, _, _ = compare_intervals(ref_intervals, pred_intervals)
            for k in metrics_dict[tag]:
                metrics_dict[tag][k] += metrics_i[k]
    return metrics_dict