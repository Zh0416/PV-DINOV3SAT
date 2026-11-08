"""Full-resolution segmentation metrics and exact tie-aware average precision."""
from __future__ import annotations
from collections import defaultdict
import numpy as np


def confusion(target, prediction):
    t, p = np.asarray(target, dtype=bool), np.asarray(prediction, dtype=bool)
    if t.shape != p.shape:
        raise ValueError('Mask shapes differ.')
    tp = int(np.count_nonzero(t & p))
    return {'tp': tp, 'fp': int(p.sum())-tp, 'fn': int(t.sum())-tp,
            'tn': int(t.size-p.sum()-t.sum()+tp)}


def from_counts(c):
    tp, fp, fn, tn = (int(c[k]) for k in ('tp', 'fp', 'fn', 'tn'))
    ratio = lambda a, b, empty: a/b if b else empty
    fg = ratio(tp, tp+fp+fn, 1.)
    bg = ratio(tn, tn+fp+fn, 1.)
    return {'foreground_iou': fg, 'miou': (fg+bg)/2,
            'dice': ratio(2*tp, 2*tp+fp+fn, 1.),
            'precision': ratio(tp, tp+fp, 0.), 'recall': ratio(tp, tp+fn, 0.)}


def exact_ap(target, scores):
    """AP = sum(delta recall * precision), aggregating all equal scores."""
    y, s = np.asarray(target).reshape(-1), np.asarray(scores).reshape(-1)
    if y.shape != s.shape or not np.isfinite(s).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('AP requires finite scores and matching binary labels.')
    positives = int(y.sum())
    if not positives:
        return None
    order = np.argsort(s, kind='stable')[::-1]
    sorted_scores = s[order]
    ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(s)-1]
    del sorted_scores
    tp = np.cumsum(y[order], dtype=np.int64)[ends]
    increments = np.diff(np.r_[0, tp])
    return float(np.sum((increments / positives) * (tp / (ends + 1))))


def aggregate(rows):
    if not rows:
        raise ValueError('No evaluated episodes.')
    keys = ('foreground_iou', 'miou', 'dice', 'precision', 'recall')
    counts = {k:sum(r[k] for r in rows) for k in ('tp', 'fp', 'fn', 'tn')}
    groups = defaultdict(list)
    for row in rows:
        groups[row['query_group']].append(row)
    grouped = []
    for name, values in sorted(groups.items()):
        record = {'query_group':name, 'query_count':len(values)}
        record.update({k:float(np.mean([r[k] for r in values])) for k in keys})
        for k in ('ap', 'foreground_ap'):
            valid = [r[k] for r in values if r.get(k) is not None]
            if valid:record[k] = float(np.mean(valid))
        grouped.append(record)
    result = {'query_count':len(rows), 'query_group_count':len(grouped),
              'micro':from_counts(counts), 'pixel_counts':counts,
              'macro_image':{k:float(np.mean([r[k] for r in rows])) for k in keys},
              'macro_group':{k:float(np.mean([r[k] for r in grouped])) for k in keys}}
    for k in ('ap', 'foreground_ap'):
        vals=[r[k] for r in rows if r.get(k) is not None]
        if vals:
            result['macro_image'][k]=float(np.mean(vals))
            result['macro_group'][k]=float(np.mean([r[k] for r in grouped if k in r]))
            result[k+'_valid_images']=len(vals)
    return result, grouped
