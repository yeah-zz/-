"""Evidence-scoped supervision, independent of any recognition model.

An object region is not an ink mask; a visible guide is not an artist stroke.
These rules deliberately do not infer missing labels from a tutorial title.
"""
from collections import defaultdict

SCOPES = {'visible_curve_candidate', 'temporal_role_core', 'object_region',
          'named_style_example'}


def validate_evidence(record):
    required = ('id', 'scope', 'source_id', 'work_group', 'source_group',
                'evidence', 'reviewer', 'seen_before_round')
    if any(k not in record for k in required):
        raise ValueError('Missing evidence provenance')
    if record['scope'] not in SCOPES or not record['evidence']:
        raise ValueError('Unknown or empty evidence scope')
    if record.get('prediction_as_label'):
        raise ValueError('Prediction cannot be reference truth')
    if record.get('independent_test') and record['seen_before_round']:
        raise ValueError('Previously inspected material is a regression case')
    if record.get('object_type') == 'tree' and record.get('role') == 'cun':
        raise ValueError('Tree cannot provide a rock cun target')
    if record.get('instance_supervision'):
        raise ValueError('These reference scopes do not authorize complete stroke instances')
    if record.get('artist_order_verified') or record.get('physical_measurement'):
        raise ValueError('These image reference scopes do not authorize artist order or physical measurements')
    if record.get('usage') == 'independent_test' and not record.get('independent_test'):
        raise ValueError('Independent-test usage requires a matching provenance declaration')
    if record['scope'] == 'named_style_example' and record.get('localized_style_target') is not None:
        raise ValueError('Example-level style cannot become a localized stroke label')
    if record['scope'] == 'object_region' and record.get('region_is_ink_mask'):
        raise ValueError('Object region cannot be promoted to an ink mask')
    if record.get('pressure_mm') is not None and not record.get('physical_measurement'):
        raise ValueError('Image width is not physical brush pressure')
    return True


def supervision_channels(record):
    """Separate task channels instead of admitting/rejecting an entire record."""
    validate_evidence(record)
    scope = record['scope']
    return {
        'object_region': scope == 'object_region' and record.get('object_checked', False),
        'partial_role_pixels': scope == 'temporal_role_core' and record.get('role_checked', False),
        'named_example_style': scope == 'named_style_example' and record.get('source_named', False),
        'complete_stroke_instance': bool(record.get('instance_supervision', False)),
        'artist_stroke_order': bool(record.get('artist_order_verified', False)),
        'physical_brush': bool(record.get('physical_measurement', False)),
    }


def audit_lineage(records):
    groups = defaultdict(set)
    for record in records:
        validate_evidence(record)
        for field in ('work_group', 'source_group'):
            groups[(field, record[field])].add(record.get('usage', 'development'))
    conflicts = [str(k) for k, uses in groups.items()
                 if 'independent_test' in uses and len(uses) > 1]
    if conflicts:
        raise ValueError('Source/work leakage: ' + ', '.join(conflicts))
    return {'records': len(records), 'groups': len(groups),
            'independent_test_records': sum(bool(r.get('independent_test')) for r in records)}
