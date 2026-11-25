# -*- coding: utf-8 -*-
"""Additive whole-candidate recovery, separate core/cun/unknown layers."""
import sys
sys.dont_write_bytecode=True
from mountain_recovery_catalog_20261101 import *
from train_mountain_candidate_recovery_20261101 import predict
from v4_local_repair_candidates import graph_cycles,evidence
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
import torch,json,hashlib
POLICY=dict(version='whole_candidate_combined_policy_r1',model_threshold=.8,uncertainty_limit=.15,min_length=8,min_span_ratio=.65,min_support012=.65,min_support006=.80,min_median_contrast=.012,core_length=20,core_span_ratio=.70,optional_additions_per_mass=6,duplicate_distance=1.6,duplicate_fraction=.70,minimum_novel_pixels=4,maximum_gap=3,bridge_tangent_alignment=.95,bridge_direction_alignment=.94,maximum_bridge_imbalance=.20,bridge_min_source_fraction=1.,bridge_min_profile_fraction=.8,allow_new_bridge_cycles=False,model_alone_accepted=False,neural_pixel_extractor_changed=False)

def qualifies(row,prob,std):
    f=row['features']
    return (row['fully_interior'] and not row['protected'] and f['length']>=POLICY['min_length'] and f['span_ratio']>=POLICY['min_span_ratio'] and f['support012']>=POLICY['min_support012'] and f['support006']>=POLICY['min_support006'] and f['contrast_median']>=POLICY['min_median_contrast'] and prob>=POLICY['model_threshold'] and std<=POLICY['uncertainty_limit'])

def bridge_gaps(rgb,result,masks):
    """Only short, unique, opposite endpoint pairs on actual ink; no cycles."""
    scale=max(result.shape)/496.;paths,_=v4.pixel_chains(result);ends=[];gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255;ink,_=v4.source_ink(rgb)
    degree=cv2.filter2D(result.astype(np.uint8),-1,np.ones((3,3),np.uint8))-result.astype(np.uint8)
    for i,p in enumerate(paths):
        if v4.plen(p)<8*scale:continue
        for side in [0,1]:
            xy,t=v4.endpoint(p,side);x,y=np.rint(xy).astype(int)
            if degree[y,x]==1 and masks['interior'][y,x] and not masks['protect'][y,x]:ends.append((i,side,xy,t))
    proposals={};events=[];added=np.zeros_like(result);used=set();current=result.copy()
    if not ends:return current,added,events
    tree=cKDTree(np.stack([e[2] for e in ends]))
    for a,e in enumerate(ends):
        scores=[]
        for b in tree.query_ball_point(e[2],POLICY['maximum_gap']*scale):
            f=ends[b]
            if a==b or e[0]==f[0]:continue
            delta=f[2]-e[2];gap=float(np.linalg.norm(delta));axis=v4.unit(delta)
            # endpoint() returns tangent pointing OUTWARD from the path.
            align=float(-e[3]@f[3]);da=float(e[3]@axis);db=float(f[3]@(-axis))
            if gap<1.5 or align<POLICY['bridge_tangent_alignment'] or min(da,db)<POLICY['bridge_direction_alignment']:continue
            scores.append((align+da+db-gap*.02,b))
        scores.sort(reverse=True)
        if scores and (len(scores)==1 or scores[0][0]-scores[1][0]>.15):proposals[a]=scores[0][1]
    for a,b in proposals.items():
        if a>b or a in used or b in used or proposals.get(b)!=a:continue
        pa=ends[a][2];pb=ends[b][2];gap=float(np.linalg.norm(pb-pa));p=np.linspace(pa,pb,max(3,int(np.ceil(gap*3))));q=np.rint(p).astype(int);line=v4.raster(result.shape,[p]);novel=line&~current
        reason=None
        if not novel.any():continue
        if not masks['interior'][line].all() or masks['protect'][line].any():reason='outside_mountain_or_protected'
        elif not ink[line].all():reason='missing_source_ink'
        else:
            contrast,center=evidence(gray,p,scale)
            if float((contrast>=.008).mean())<POLICY['bridge_min_profile_fraction']:reason='weak_cross_section'
            elif graph_cycles(current|line)>graph_cycles(current):reason='would_create_cycle'
        event=dict(a=[int(ends[a][0]),int(ends[a][1])],b=[int(ends[b][0]),int(ends[b][1])],points=p.tolist(),gap=gap,accepted=reason is None,reason=reason or 'unique_mutual_aligned_endpoints_on_source_ink',artist_stroke_identity_verified=False)
        events.append(event)
        if reason is None:current|=line;added|=novel;used|={a,b}
    return current,added,events

def select_recovery(rgb,base,old,masks,rows,artifact):
    scale=max(old.shape)/496.;p,std=predict(artifact,rows);result=old.copy();context=old&(~masks['interior']|masks['protect']);optional_old=[];fixed=[]
    for r in rows:
        if not r['old_keep']:continue
        dst=optional_old if r['old_decision']['reason']=='long_image_supported_interior' else fixed;dst.append(np.asarray(r['points'],np.float32))
    fixedmask=v4.raster(old.shape,fixed)|context;oldoptional=(v4.raster(old.shape,optional_old)&old)&~fixedmask
    core=old&~oldoptional;optional=oldoptional.copy();restored=np.zeros_like(old);decisions=[];eligible=[];counts={};_,mass=cv2.connectedComponents(masks['interior'].astype(np.uint8),8)
    for r,prob,unc in zip(rows,p,std):
        d=dict(candidate=r['id'],score=float(prob),ensemble_std=float(unc),old_decision=r['old_decision']['decision'],decision='preserve_previous' if r['old_keep'] else 'pending',layer='previous' if r['old_keep'] else 'unknown',reason='previous_selection_not_deleted' if r['old_keep'] else 'uncertain_or_failed_source_geometry_gate')
        decisions.append(d)
        if r['old_keep']:continue
        if prob<=.2 and unc<=.15:d.update(decision='reject_candidate',reason='model_rejects_this_trajectory_not_background')
        if qualifies(r,prob,unc):
            f=r['features'];is_core=f['length']>=POLICY['core_length'] and f['span_ratio']>=POLICY['core_span_ratio'];score=f['length']*(.5+.5*f['support012'])*float(prob);eligible.append((int(is_core),score,r,d))
    for is_core,score,r,d in sorted(eligible,key=lambda row:(-row[0],-row[1])):
        pts=np.asarray(r['points'],np.float32);q=np.rint(pts).astype(int);y,x=q[:,1],q[:,0];dist=distance_transform_edt(~result);near=float((dist[y,x]<=POLICY['duplicate_distance']*scale).mean());pathmask=v4.raster(old.shape,[pts]);novel=pathmask&~result
        gid=int(np.bincount(mass[y,x]).argmax())
        if near>=POLICY['duplicate_fraction'] or novel.sum()<POLICY['minimum_novel_pixels']*scale:d.update(decision='pending',reason='near_duplicate_keep_unknown');continue
        if not is_core and counts.get(gid,0)>=POLICY['optional_additions_per_mass']:d.update(decision='pending',reason='optional_density_budget_keep_unknown');continue
        result|=pathmask;restored|=novel
        if is_core:core|=pathmask
        else:optional|=pathmask;counts[gid]=counts.get(gid,0)+1
        d.update(decision='restore',layer='core' if is_core else 'adjustable_cun',reason='whole_candidate_model_and_source_geometry')
    # Core takes precedence at shared pixels so the raster layers are disjoint.
    optional&=~core
    result,bridges,events=bridge_gaps(rgb,result,masks);core|=bridges;optional&=~core
    unknown=base&~result
    assert np.array_equal(core|optional,result)
    assert not (old&~result).any()
    assert np.array_equal(result&masks['protect'],old&masks['protect'])
    assert np.array_equal(result&~masks['interior'],old&~masks['interior'])
    return dict(result=result,core=core,optional=optional,unknown=unknown,restored=restored,bridges=bridges,decisions=decisions,bridge_events=events)

def run_one(ident,artifact):
    rgb,base,old,masks,rows=catalog(ident);out=ROOT/'outputs'/ident;out.mkdir(parents=True,exist_ok=True);data=select_recovery(rgb,base,old,masks,rows,artifact);bw=lambda m:np.where(m,0,255).astype(np.uint8)
    for name,mask in [('lines',data['result']),('structure',data['core']),('adjustable_cun',data['optional']),('unknown',data['unknown']),('restored',data['restored']),('bridges',data['bridges']),('tree_context',old&masks['protect'])]:Image.fromarray(bw(mask)).save(out/(name+'.png'))
    for name in ['input','baseline','lines']:
        src=OLD/'outputs'/ident/(name+'.png');dst=out/('previous.png' if name=='lines' else name+'.png');Image.open(src).save(dst)
    changes=np.repeat(bw(old)[:,:,None],3,2);changes[data['restored']]=[30,135,225];changes[data['bridges']]=[220,65,150];Image.fromarray(changes).save(out/'changes.png')
    layers=np.full_like(rgb,255);layers[data['unknown']]=[195,195,195];layers[data['optional']]=[35,135,190];layers[data['core']]=[25,25,25];Image.fromarray(layers).save(out/'layers.png')
    paths,_=whole_candidates(data['result'],max(old.shape)/496.);export=[dict(points=r['points'].tolist(),stroke_identity_verified=False) for r in paths]
    layer_paths={}
    for name,key in [('structure','core'),('adjustable_cun','optional'),('unknown','unknown')]:layer_paths[name]=[dict(points=r['points'].tolist(),stroke_identity_verified=False) for r in whole_candidates(data[key],max(old.shape)/496.)[0]]
    jsave(out/'paths.json',dict(paths=export,layers=layer_paths,coordinate_system='input_image_pixels',source_sha256=digest(OLD/'outputs'/ident/'input.png'),previous_paths_sha256=digest(OLD/'outputs'/ident/'paths.json'),model_sha256=digest(ROOT/'fit/model.pt'),policy=POLICY,physical_z=None,physical_speed=None,ready_for_robot=False,actually_drawn=False,role_labels_are_provisional=True))
    jsave(out/'decisions.json',data['decisions']);jsave(out/'bridge_events.json',data['bridge_events'])
    stats=dict(id=ident,base_pixels=int(base.sum()),previous_pixels=int(old.sum()),selected_pixels=int(data['result'].sum()),restored_pixels=int(data['restored'].sum()),bridge_pixels=int(data['bridges'].sum()),restored_whole_candidates=sum(r['decision']=='restore' for r in data['decisions']),restored_structure_candidates=sum(r['decision']=='restore' and r['layer']=='core' for r in data['decisions']),restored_optional_candidates=sum(r['decision']=='restore' and r['layer']=='adjustable_cun' for r in data['decisions']),unknown_pixels=int(data['unknown'].sum()),previous_deleted_pixels=int((old&~data['result']).sum()),protected_detail_exact=bool(np.array_equal(data['result']&masks['protect'],old&masks['protect'])),outside_interior_exact=bool(np.array_equal(data['result']&~masks['interior'],old&~masks['interior'])),baseline_cycles=graph_cycles(base),previous_cycles=graph_cycles(old),selected_cycles=graph_cycles(data['result']),new_pixels_outside_v4=int((data['result']&~base).sum()))
    jsave(out/'statistics.json',stats);poly=''.join('<polyline points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in r['points'])+'"/>' for r in export);(out/'lines.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {old.shape[1]} {old.shape[0]}"><g fill="none" stroke="black" stroke-width="1">{poly}</g></svg>',encoding='utf-8')
    return stats

def main():
    p=argparse.ArgumentParser();p.add_argument('--ids',default='G01_01,G02_01,G05_01');a=p.parse_args();artifact=torch.load(ROOT/'fit/model.pt',map_location='cpu',weights_only=False)
    ids=[r['id'] for r in read(OLD/'results_all.json')['records']] if a.ids=='all' else a.ids.split(',');policy_path=ROOT/'combined_policy.json'
    frozen=dict(policy=POLICY,model_sha256=digest(ROOT/'fit/model.pt'),code_sha256=digest(__file__),catalog_sha256=digest(Path(__file__).with_name('mountain_recovery_catalog_20261101.py')),threshold_source='Combined geometry+model gate fixed separately; isolated model grid threshold 1.01 is disabled and not used',reference_sha256=digest(ROOT/'partial_references.json'))
    if policy_path.exists():assert read(policy_path)==frozen,'Start a new version to change frozen inference'
    else:jsave(policy_path,frozen)
    records=[]
    for ident in ids:
        r=run_one(ident,artifact);records.append(r);print('RECOVERED',json.dumps(r),flush=True)
    jsave(ROOT/('results_'+a.ids.replace(',','_')+'.json'),dict(records=records,policy=POLICY))
if __name__=='__main__':main()
