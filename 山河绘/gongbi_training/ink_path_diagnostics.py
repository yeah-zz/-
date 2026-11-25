"""Measure object ownership along supplied visible runs; never infer artist strokes."""
import math
import numpy as np


def arc_cells(vertices, max_step=0.5):
    """Midpoint quadrature per original edge: keep corners and disconnected runs separate."""
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError('A run needs at least two 2D vertices')
    if not np.isfinite(points).all() or not math.isfinite(max_step) or max_step <= 0:
        raise ValueError('Finite coordinates and positive step required')
    mids, weights = [], []
    for start, end in zip(points[:-1], points[1:]):
        length = float(np.linalg.norm(end - start))
        if length == 0:
            continue
        n = max(1, math.ceil(length / max_step))
        mids.extend(start + ((np.arange(n) + 0.5) / n)[:, None] * (end - start))
        weights.extend([length / n] * n)
    if not weights:
        raise ValueError('Zero length run')
    return np.asarray(mids), np.asarray(weights)


def score_visible_runs(prediction, runs, expected_class, max_step=0.5):
    """Unknown ownership is excluded. Separate runs never add length across gaps."""
    if expected_class is None:
        return {'status': 'unknown_not_scored', 'coverage': None, 'runs': []}
    pred = np.asarray(prediction)
    if pred.ndim != 2:
        raise ValueError('Expected native-resolution class raster')
    result = []
    for run in runs:
        mids, weights = arc_cells(run, max_step)
        xy = np.floor(mids + 0.5).astype(int)
        if (xy < 0).any() or (xy[:, 0] >= pred.shape[1]).any() or (xy[:, 1] >= pred.shape[0]).any():
            raise ValueError('Reference outside native raster')
        classes = pred[xy[:, 1], xy[:, 0]]
        good = classes == expected_class
        intervals, start = [], 0
        for stop in range(1, len(good) + 1):
            if stop == len(good) or good[stop] != good[start]:
                intervals.append({'correct': bool(good[start]), 'length_px': float(weights[start:stop].sum()),
                                  'start_xy': mids[start].tolist(), 'end_xy': mids[stop-1].tolist()})
                start = stop
        result.append({'length_px': float(weights.sum()), 'correct_length_px': float(weights[good].sum()),
                       'intervals': intervals, 'longest_wrong_px': max([v['length_px'] for v in intervals if not v['correct']] or [0.0]),
                       'class_lengths_px': {str(int(c)): float(weights[classes == c].sum()) for c in np.unique(classes)}})
    if not result:
        raise ValueError('No visible runs')
    total = sum(v['length_px'] for v in result)
    return {'status': 'assistant_visible_run_diagnostic', 'length_px': total,
            'correct_length_px': sum(v['correct_length_px'] for v in result),
            'coverage': sum(v['correct_length_px'] for v in result) / total,
            'longest_wrong_px': max(v['longest_wrong_px'] for v in result), 'runs': result,
            'sampled_points_are_not_independent_examples': True}
