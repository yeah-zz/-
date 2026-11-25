"""Partial, observable morphology references; no style-to-concept relabelling.

Concepts describe an image region, not recovered individual strokes, temporal
order, brush orientation, pressure, or calibrated robot actions.
"""
import numpy as np

CONCEPTS = ('elongated_mark', 'angular_broad_mark', 'curved_parallel_group')
IGNORE = -100


def concept_targets(record):
    if record.get('automatic_prediction_as_label'):
        raise ValueError('Predictions cannot be concept labels')
    if record.get('concept_origin') in {'style_name', 'work_title', 'model_prediction'}:
        raise ValueError('Style names cannot supply morphological truth')
    values = record.get('concepts', {})
    if set(values) - set(CONCEPTS):
        raise ValueError('Unsupported observable concept')
    for value in values.values():
        if value is not None and (type(value) is not int or value not in (0, 1)):
            raise ValueError('Use explicit 0/1 or None for unknown')
    rock_or_named_practice = record.get('object_type') == 'rock' or (
        record.get('object_type') == 'teaching_mark'
        and record.get('reference_kind') == 'named_brush_mark_exercise'
        and record.get('source_evidence')
    )
    admissible = (
        rock_or_named_practice
        and record.get('object_reference_checked') is True
        and record.get('concept_reference_checked') is True
        and record.get('source_url')
        and record.get('reviewer')
        and record.get('visual_evidence')
        and record.get('label_scope') == 'observable_region_morphology'
    )
    return np.array([values.get(k) if admissible and values.get(k) is not None else IGNORE
                     for k in CONCEPTS], dtype='int64')


def validate_concept_split(rows):
    """Crops of one illustration, source image, or source group stay together."""
    seen = {k: {} for k in ('source_group', 'reference_unit', 'source_sha256')}
    for row in rows:
        if row.get('split') not in {'train', 'test', 'review_only'}:
            raise ValueError('Explicit split required')
        if row['split'] == 'review_only':
            continue
        for field, table in seen.items():
            key = row.get(field)
            if not key:
                raise ValueError('Missing source lineage')
            if key in table and table[key] != row['split']:
                raise ValueError('Source/illustration leakage across concept splits')
            table[key] = row['split']
    return True


def partial_concept_loss(logits, targets):
    import torch
    from torch.nn.functional import binary_cross_entropy_with_logits
    valid = targets.ne(IGNORE)
    if not bool(valid.any()):
        return logits.sum() * 0
    return binary_cross_entropy_with_logits(logits[valid], targets[valid].to(logits.dtype))


def permitted_cunfa_output(object_type, object_verified, role, predicted_style):
    # A reference-guided API guard, not an independently learned object detector.
    if object_type != 'rock' or not object_verified or role != 'cun':
        return 'unknown'
    return predicted_style if predicted_style in {'pima', 'fupi', 'mixed'} else 'unknown'
