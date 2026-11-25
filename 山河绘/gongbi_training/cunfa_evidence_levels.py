"""Keep source-named examples separate from spatial roles and physical skills."""
IGNORE=-100


def supervision_for(e):
    if e.get('object_type')=='tree' and e.get('cunfa_target') is not None:
        raise ValueError('Vegetation must not receive cunfa supervision')
    if e.get('complete_stroke') or e.get('physical_parameters_verified'):
        raise ValueError('This evidence contract does not establish strokes or brush actions')
    result=dict(image_style_target=IGNORE,pixel_role_enabled=False,stroke_enabled=False,physical_enabled=False)
    if e['kind']=='named_example':
        if not e.get('caption_checked') or not e.get('source_group') or e.get('label_scope')!='whole_teaching_example':
            raise ValueError('Named style requires a checked caption and example scope')
        result['image_style_target']={'hemp':0,'axe':1,'known_out_of_scope':2}[e['initial_taxonomy']]
    elif e['kind']=='video_supported_band':
        if not e.get('assistant_reviewed') or not e.get('stage_pair_checked') or not e.get('partial_mask_sha256'):
            raise ValueError('Role supervision requires reviewed spatial evidence')
        if e.get('named_style') is not None:
            raise ValueError('New interior ink alone does not establish named cunfa')
        result['pixel_role_enabled']=True
    else:raise ValueError('Unsupported evidence level')
    return result
