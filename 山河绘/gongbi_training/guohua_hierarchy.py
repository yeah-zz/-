"""Contracts for object, method and ordered skill learning; no learned claims."""
from dataclasses import dataclass
from typing import Mapping

OBJECTS={'rock','tree','architecture','water','paper','tool_or_hand','unknown'}
ROLES={'rock':{'outline','plane_turn','cun','wash'},'tree':{'trunk','main_branch','twig','foliage'},'architecture':{'outline','detail'},'water':{'ripple'},'paper':set(),'tool_or_hand':set(),'unknown':set()}
CUNFA={'pima','fupi','zhedai','mixed','unknown'}
CONCEPT_AXES=('mark_family','length_distribution','curvature','width_change','edge_quality','surface_aligned_group_flow','overlap_order','dry_wet_evidence')

def validate_region(record:Mapping):
    category=record.get('object_type','unknown')
    if category not in OBJECTS:raise ValueError('Unknown object type')
    role=record.get('role','unknown')
    if role!='unknown' and role not in ROLES[category]:
        raise ValueError('Role is incompatible with object; foliage cannot be cun')
    method=record.get('cunfa','unknown')
    if method not in CUNFA:raise ValueError('Unsupported cunfa category')
    if method!='unknown' and (category!='rock' or role!='cun'):
        raise ValueError('Cunfa label requires a rock cun group')
    if record.get('prediction_as_label',False):
        raise ValueError('Automatic predictions cannot become reference labels')
    return True

def cunfa_training_target(record:Mapping):
    """Unknown, work-level titles, and unconfirmed objects have no style loss."""
    validate_region(record)
    if record.get('object_type')!='rock' or not record.get('object_verified'):
        return None
    if record.get('role')!='cun' or record.get('cunfa') in {None,'unknown','mixed'}:
        return None
    if record.get('label_scope')!='local_cun_group':return None
    if not (record.get('source_url') and record.get('label_evidence') and record.get('reviewer')):return None
    if record.get('label_status') not in {'source_checked','expert_checked'}:return None
    return record['cunfa']

def verify_split(records):
    """Fail closed on missing work lineage; report author independence separately."""
    works={};sources={};artists={};artist_complete=True
    for record in records:
        split=record.get('split');work=record.get('work_id');source=record.get('source_group')
        if split not in {'train','validation','test'} or not work or not source:
            raise ValueError('Split requires explicit work and source lineage')
        for table,key in [(works,work),(sources,source)]:
            if key in table and table[key]!=split:raise ValueError('Work/source leakage across split')
            table[key]=split
        artist=record.get('artist_id')
        if artist:
            if artist in artists and artists[artist]!=split:raise ValueError('Artist leakage across split')
            artists[artist]=split
        else:artist_complete=False
    return {'work_source_disjoint':True,'author_independence_verified':artist_complete}

@dataclass(frozen=True)
class CapabilityState:
    object_independent_validation:bool=False
    cunfa_independent_validation:bool=False
    tree_skill_independent_validation:bool=False
    stroke_geometry_validation:bool=False
    physical_brush_validation:bool=False

def allowed_phases(state:CapabilityState):
    foundation=state.object_independent_validation and state.cunfa_independent_validation
    tree=foundation and state.tree_skill_independent_validation
    offline=tree and state.stroke_geometry_validation
    return dict(object_and_cunfa_training=True,tree_skill_training=foundation,
                full_offline_creation_validation=offline,
                physical_creation_validation=offline and state.physical_brush_validation)

def validate_skill_order(steps):
    """Explicit dependencies for the selected pilot style, not a universal art rule."""
    done=set();objects={};previous_object=None
    for step in steps:
        ident=step.get('id');oid=step.get('object_id');category=step.get('object_type');role=step.get('role')
        if not ident or ident in done or not oid:raise ValueError('Missing/duplicate plan id')
        validate_region(dict(object_type=category,role=role,cunfa='unknown'))
        if category=='unknown' or role=='unknown':raise ValueError('Unknown object/role cannot be a draw action')
        if not set(step.get('depends_on',[])).issubset(done):raise ValueError('Unmet plan dependency')
        if previous_object and previous_object!=oid and not objects[previous_object].get('complete'):
            if step.get('switch_reason') not in {'drying','occlusion','dependency'}:
                raise ValueError('Finish current object before changing object, or record justified dependency')
        if oid in objects and objects[oid]['type']!=category:raise ValueError('Object changes type')
        entry=objects.setdefault(oid,dict(type=category,roles=set()))
        if category=='rock' and role=='cun' and 'outline' not in entry['roles']:
            raise ValueError('Pilot rock plan requires outline before cun')
        if category=='tree' and role=='foliage' and not {'trunk','main_branch'}.issubset(entry['roles']):
            raise ValueError('Pilot tree plan requires trunk and main branches before foliage')
        done.add(ident);entry['roles'].add(role);entry['complete']=bool(step.get('object_complete',False));previous_object=oid
    return True
