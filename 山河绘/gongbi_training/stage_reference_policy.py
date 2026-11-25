"""Permissions for attributed, cropped tutorial steps. No artist-stroke inference."""
IGNORE=-100


def permitted_supervision(record):
    mixed=record.get('style_evidence')=='explicit_mixed_named_object'
    if mixed and len(record.get('style_components',[]))<2:
        raise ValueError('Mixed needs explicitly named components')
    return dict(
        source_specific_stage_order=bool(record.get('artist_attribution_checked') and record.get('ordered_steps_checked')),
        knowledge_style_tag='mixed' if mixed else 'unknown',
        full_image_style_target=IGNORE,
        # Article wording locates a semantic object, not a verified image mask.
        binary_style_training=False,pixel_role_training=False,
        single_stroke_training=False,physical_parameter_training=False,
        automatic_planner_verified=False,
        aligned_temporal_difference_allowed=bool(record.get('same_view_alignment_verified',False)),
    )
