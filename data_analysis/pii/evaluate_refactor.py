import json

TAGS = ['EMAIL', 'IP_ADDRESS', 'KEY']


def load_json(sample):
    try:
        return json.loads(sample)
    except ValueError:
        return []


def overlapped(a, b, alpha=0.8, beta=0.8):
    """Returns True if the intervals a and b overlap for more than 80% of their lengths"""
    size_overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    ref_overlap = size_overlap / (b[1] - b[0])
    pred_overlap = size_overlap / (a[1] - a[0])
    return (ref_overlap > alpha and pred_overlap > beta)


def compare_intervals(ref_intervals, pred_intervals, alpha=0.8, beta=0.8):
    """Compare two lists of intervals and return the number 
    of true positives, false positives and false negatives.
    >>> compare_intervals([(0, 7), (10, 20)], [(1,8), (99, 119)], 0, 0)[0]
    {'TP': 1, 'FN': 1, 'FP': 1}
    """
    scores = {"TP": 0, "FN": 0, "FP": 0}
    # use index so to recover the original data
    detection_indice = {"TP_pred": set(), "TP_ref": set(), "FN": set(), "FP": set()}
    for i, interval in enumerate(pred_intervals):
        for j, target in enumerate(ref_intervals):
            if overlapped(interval, target, alpha, beta):
                # the prediction is a true positive
                scores["TP"] += 1
                detection_indice['TP_pred'].add(i)
                detection_indice['TP_ref'].add(j)
                break
        else:
            # the prediction is a false positive
            scores["FP"] += 1
            detection_indice['FP'].add(i)
    # the rest of the targets that aren't detected are false negatives
    detection_indice["FN"] = set(range(len(ref_intervals))) - detection_indice["TP_ref"]
    scores["FN"] = len(detection_indice["FN"])

    return scores, detection_indice


def recall_precision(metrics_dict):
    """Compute recall and precision for each tag"""
    metrics = {}
    for tag in TAGS:
        metrics[tag] = {}
        total = metrics_dict[tag]['TP'] + metrics_dict[tag]['FN'] + metrics_dict[tag]['FP']
        if total:
            if not (metrics_dict[tag]['TP'] + metrics_dict[tag]['FN']) or not (metrics_dict[tag]['TP'] + metrics_dict[tag]['FP']):
                # handle division by zero
                metrics[tag] = {'recall': 0, 'precision': 0}
            else:
                metrics[tag]['recall'] = metrics_dict[tag]['TP'] / (metrics_dict[tag]['TP'] + metrics_dict[tag]['FN'])
                metrics[tag]['precision'] = metrics_dict[tag]['TP'] / (metrics_dict[tag]['TP'] + metrics_dict[tag]['FP'])
        else:
            # if there are no annotations, the score is 1
            metrics[tag] = {'recall': 1.0, 'precision': 1.0}
    return metrics


def recall_precision_all_tags(metrics_dict):
    """Compute recall and precision for all tags"""
    metrics = {}
    TP = sum([metrics_dict[tag]['TP'] for tag in TAGS])
    FN = sum([metrics_dict[tag]['FN'] for tag in TAGS])
    FP = sum([metrics_dict[tag]['FP'] for tag in TAGS])
    if not (TP + FN) or not (TP + FP):
        metrics = {'recall': 0, 'precision': 0}
    else:
        metrics['recall'] = TP / (TP + FN)
        metrics['precision'] = TP / (TP + FP)
    return metrics


def evaluate_pii(references, predictions, alpha=0.8, beta=0.8, return_details=False):
    """Evaluate predictions of PII against references"""
    metrics_dict = {}
    details = {}
    for tag in TAGS:
        ref_tag = [ref for ref in references if ref['tag'] == tag]
        pred_tag = [pred for pred in predictions if pred['tag'] == tag]
        ref_intervals = [(e['start'], e['end']) for e in ref_tag]
        pred_intervals = [(e['start'], e['end']) for e in pred_tag]
        metrics, detection_indices = compare_intervals(ref_intervals, pred_intervals, alpha, beta)
        metrics_dict[tag] = metrics
        details[tag] = {
            'TP_pred': [pred_tag[i] for i in detection_indices['TP_pred']],
            'TP_ref': [ref_tag[i] for i in detection_indices['TP_ref']],
            'FN': [ref_tag[i] for i in detection_indices['FN']],
            'FP': [pred_tag[i] for i in detection_indices['FP']]
        }
    if return_details:
        return metrics_dict, details
    return metrics_dict


def evaluate_pii_ds(dataset, pred_column='pii', ref_column="secrets", overall_score=False, alpha=0.8, beta=0.8, return_details=False):
    """Evaluate predictions of PII against references in a dataset
    """
    metrics_dict = {tag: {'TP': 0, 'FN': 0, 'FP': 0} for tag in TAGS}
    details = []
    for i in range(len(dataset)):
        ref_list = load_json(dataset[i][ref_column])
        pred_list = load_json(dataset[i][pred_column])
        sample_metrics, details_i = evaluate_pii(ref_list, pred_list, alpha, beta, return_details=True)
        details.append(details_i)
        for tag in TAGS:
            for metric in metrics_dict[tag]:
                metrics_dict[tag][metric] += sample_metrics[tag][metric]
    score = recall_precision(metrics_dict) if not overall_score else recall_precision_all_tags(metrics_dict)
    if return_details:
        return score, metrics_dict, details
    return score, metrics_dict