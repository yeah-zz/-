"""Checked contextual role of a given region, not an ink-pixel or stroke label."""
ROLE_NAMES=('outline_context','interior_mark_context','diffuse_tone','unmarked_surface')

def context_role_target(record):
    if record.get('automatic_prediction_as_label'):
        raise ValueError('Predictions are not role targets')
    role=record.get('context_role_target')
    if role is not None and role not in ROLE_NAMES:
        raise ValueError('Unsupported context role')
    if role is None:return -100
    if record.get('object_type') not in {'rock','paper'}:return -100
    if record.get('object_type')=='paper' and role!='unmarked_surface':return -100
    if not (record.get('object_reference_checked') and record.get('role_reference_checked')
            and record.get('source_url') and record.get('visual_evidence') and record.get('reviewer')):
        return -100
    if record.get('role_reference_scope')!='region_context_not_pixel_or_stroke_truth':return -100
    return ROLE_NAMES.index(role)

def context_box(box,size,factor=2.5):
    w,h=size;x0,y0,x1,y1=box;cx=(x0+x1)/2;cy=(y0+y1)/2
    return [max(0,int(cx-factor*(x1-x0)/2)),max(0,int(cy-factor*(y1-y0)/2)),
            min(w,int(cx+factor*(x1-x0)/2)),min(h,int(cy+factor*(y1-y0)/2))]
