"""Task-specific partial labels. Work-level cun names never become local truth."""
STYLE_NAMES=('披麻皴','斧劈皴')
IGNORE_INDEX=255
LOCAL_SOURCE_SCOPES={'named_teaching_example','named_local_source_region'}

def style_target(record):
    name=record.get('local_style')
    if name not in STYLE_NAMES or record.get('conflicting_style_labels',False):
        return -100
    if record.get('automatic_prediction_as_label',False):
        return -100
    traceable=bool(record.get('source_url') and record.get('source_evidence'))
    explicit=record.get('label_scope') in LOCAL_SOURCE_SCOPES
    expert=bool(record.get('expert_local_verified') and record.get('reviewer_id') and record.get('review_evidence'))
    return STYLE_NAMES.index(name) if traceable and (explicit or expert) else -100

def role_targets_checked(record):
    # Candidate guides are retained for review but cannot silently become truth.
    return bool(record.get('role_reference_checked') and record.get('role_mask_path') and not record.get('automatic_prediction_as_label'))

def partial_role_loss(logits,targets,weights=None):
    import torch
    from torch.nn.functional import cross_entropy
    valid=targets.ne(IGNORE_INDEX)
    w=valid.to(logits.dtype) if weights is None else weights.to(logits.dtype)*valid
    if not bool(w.gt(0).any()):
        return logits.sum()*0
    losses=cross_entropy(logits,targets.long(),ignore_index=IGNORE_INDEX,reduction='none')
    return (losses*w).sum()/w.sum()

def partial_style_loss(logits,targets):
    from torch.nn.functional import cross_entropy
    valid=targets.ne(-100)
    if not bool(valid.any()):
        return logits.sum()*0
    return cross_entropy(logits[valid],targets[valid].long())
