# -*- coding: utf-8 -*-
"""Complete evidence selection, directional curve gaps, separate sparse view."""
import sys
sys.dont_write_bytecode=True
from mountain_complete_evidence_20261101 import *
from train_complete_candidates_20261101 import prediction,geometry_gate,FIT
from scipy.spatial import cKDTree
import torch,heapq,json,shutil
POLICY=dict(version='complete_curve_evidence_r2_bucket',complete_count_limit=None,maximum_gap=12.,minimum_novel_pixels=3,duplicate_distance=.8,duplicate_fraction=.85,maximum_near_existing_fraction=.55,near_existing_distance=1.8,bridge_turn_weight=.7,bridge_field_alignment=.55,bridge_support_fraction=.88,bridge_max_unsupported_run=1,bridge_max_detour=1.7,bridge_unique_margin=.12,protect_existing=True,sparse_optional_budget=6,contrast_bucket_boundary=.03,source_reference_scope='No manual annotations or image IDs used during selection')
DIRS=np.array([(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)],np.int32)
UNIT=DIRS/np.linalg.norm(DIRS,axis=1,keepdims=True)

def field_path(field,masks,a,b,ta,tb,scale):
    h,w=field['gray'].shape;gap=float(np.linalg.norm(b-a));pad=max(3,int(math.ceil(gap*.35)));lo=np.maximum(0,np.floor(np.minimum(a,b)-pad)).astype(int);hi=np.minimum([w,h],np.ceil(np.maximum(a,b)+pad+1)).astype(int);x0,y0=lo;x1,y1=hi
    start=tuple((np.rint(a)-lo).astype(int));target=tuple((np.rint(b)-lo).astype(int));hh,ww=y1-y0,x1-x0
    xx,yy=np.meshgrid(np.arange(x0,x1),np.arange(y0,y1));axis=v4.unit(b-a);off=np.abs((xx-a[0])*axis[1]-(yy-a[1])*axis[0]);allowed=masks['area'][y0:y1,x0:x1]&(off<=max(3*scale,gap*.35));weak=field['weak'][y0:y1,x0:x1]
    # Allow one isolated detector miss, then verify the entire path below.
    corridor=(cv2.dilate(weak.astype(np.uint8),np.ones((3,3),np.uint8))>0)&allowed;corridor[start[1],start[0]]=True;corridor[target[1],target[0]]=True
    raw=field['score'][y0:y1,x0:x1];normal=field['normal'][y0:y1,x0:x1];ridge_dist=distance_transform_edt(~field['ridge'])[y0:y1,x0:x1]/scale
    costs=1.+.8*(1.-np.clip(raw/.02,0,1))+.25*np.minimum(ridge_dist,4.)+2.*(~weak)
    initial=int(np.argmax(UNIT@ta));best={(start[0],start[1],initial):0.};parent={};heap=[(0.,start[0],start[1],initial)];goal=None;steps=0
    while heap:
        cost,x,y,last=heapq.heappop(heap);state=(x,y,last)
        if cost>best.get(state,float('inf'))+1e-8:continue
        if (x,y)==target:
            if float(UNIT[last]@(-tb))>=.55:goal=state;break
        if cost>gap*6:continue
        steps+=1
        if steps>hh*ww*8:break
        for k,(dx,dy) in enumerate(DIRS):
            nx,ny=x+int(dx),y+int(dy)
            if not (0<=nx<ww and 0<=ny<hh) or not corridor[ny,nx]:continue
            turn=float(UNIT[last]@UNIT[k])
            if turn<.3:continue
            n=normal[ny,nx];align=1.-abs(float(UNIT[k]@n));step=math.hypot(int(dx),int(dy))*(float(costs[ny,nx])+.8*(1-align)**2+POLICY['bridge_turn_weight']*(1-turn));nxt=(nx,ny,k);value=cost+step
            if value<best.get(nxt,float('inf')):
                best[nxt]=value;parent[nxt]=state;heapq.heappush(heap,(value,nx,ny,k))
    if goal is None:return None,None
    rev=[];at=goal
    while True:
        rev.append([at[0]+x0,at[1]+y0])
        if at not in parent:break
        at=parent[at]
    return np.array(rev[::-1],np.float32),best[goal]/max(gap,1.)

def longest_false(a):
    best=cur=0
    for value in a:
        cur=0 if value else cur+1;best=max(best,cur)
    return best

def bridge_curves(rgb,result,masks,field):
    scale=max(result.shape)/496.;paths,neighbors=v4.pixel_chains(result);ends=[];_,cc=cv2.connectedComponents(result.astype(np.uint8),8)
    for i,p in enumerate(paths):
        if v4.plen(p)<7*scale:continue
        for side in [0,1]:
            xy,t=v4.endpoint(p,side);x,y=np.rint(xy).astype(int)
            if len(neighbors[(y,x)])==1 and masks['area'][y,x]:ends.append((i,side,xy,t,int(cc[y,x])))
    events=[];bridges=np.zeros_like(result);current=result.copy()
    if len(ends)<2:return current,bridges,events
    tree=cKDTree(np.stack([e[2] for e in ends]));pairs=[]
    for a,b in sorted(tree.query_pairs(POLICY['maximum_gap']*scale)):
        ea,eb=ends[a],ends[b]
        if ea[0]==eb[0] or ea[4]==eb[4]:continue
        axis=v4.unit(eb[2]-ea[2]);gap=float(np.linalg.norm(eb[2]-ea[2]))
        if gap<1.5 or min(float(ea[3]@axis),float(eb[3]@(-axis)))<.55 or float(-ea[3]@eb[3])<.35:continue
        pairs.append((gap,a,b))
    # Each pair must be mutually best by source path cost, not merely closest.
    byend={};valid=[];dist_to_selected=distance_transform_edt(~result)
    for gap,a,b in sorted(pairs):
        ea,eb=ends[a],ends[b];p,cost=field_path(field,masks,ea[2],eb[2],ea[3],eb[3],scale)
        event=dict(a=[int(ea[0]),int(ea[1])],b=[int(eb[0]),int(eb[1])],gap=gap,accepted=False,reason='no_source_supported_curve',points=[])
        if p is not None:
            q=np.rint(p).astype(int);y,x=q[:,1],q[:,0];supported=field['weak'][y,x];pathmask=v4.raster(result.shape,[p]);novel=pathmask&~result;ratio=v4.plen(p)/gap;edge_distance=np.minimum(np.linalg.norm(p-ea[2],axis=1),np.linalg.norm(p-eb[2],axis=1));middle=edge_distance>2.5*scale;reason=None
            if not novel.any():reason='already_selected'
            elif ratio>POLICY['bridge_max_detour']:reason='excessive_detour'
            elif float(supported.mean())<POLICY['bridge_support_fraction'] or longest_false(supported)>POLICY['bridge_max_unsupported_run']:reason='unsupported_gap'
            elif middle.any() and (dist_to_selected[y[middle],x[middle]]<1.3*scale).any():reason='would_touch_neighboring_stroke'
            elif masks['protect'][y,x].any():reason='protected_detail'
            event.update(points=p.tolist(),source_fraction=float(supported.mean()),cost=float(cost),reason=reason or 'eligible_curve')
            if reason is None:
                j=len(valid);valid.append((float(cost),a,b,p,event));byend.setdefault(a,[]).append((float(cost),j));byend.setdefault(b,[]).append((float(cost),j))
        events.append(event)
    rankings={k:sorted(v) for k,v in byend.items()};used=set()
    for j,(cost,a,b,p,event) in sorted(enumerate(valid),key=lambda q:q[1][0]):
        ra,rb=rankings[a],rankings[b]
        if a in used or b in used or ra[0][1]!=j or rb[0][1]!=j:event['reason']='not_unique_mutual_best';continue
        if any(len(r)>1 and r[1][0]-r[0][0]<POLICY['bridge_unique_margin'] for r in [ra,rb]):event['reason']='ambiguous_connection';continue
        pm=v4.raster(result.shape,[p]);novel=pm&~current
        if graph_cycles(current|pm)>graph_cycles(current):event['reason']='would_create_cycle';continue
        current|=pm;bridges|=novel;used|={a,b};event.update(accepted=True,reason='source_supported_directional_curve')
    return current,bridges,events

def extract_complete(rgb,previous,base,rows,field,masks,artifact):
    scale=max(previous.shape)/496.;prob,std=prediction(artifact,rows);complete=previous.copy();decisions=[];accepted=[]
    for r,p,u in zip(rows,prob,std):
        gate=geometry_gate(r);bucket='faint' if r['features']['contrast_median']<POLICY['contrast_bucket_boundary'] else 'ordinary';threshold=artifact.get('bucket_thresholds',{}).get(bucket,artifact['threshold']);d=dict(key=r['key'],origin=r['origin'],score=float(p),ensemble_std=float(u),geometry_gate=gate,threshold=threshold,bucket=bucket,accepted=False,reason='source_curve_gate_failed',label_used=False)
        if gate:
            d['reason']='model_uncertain_or_reject'
            if p>=threshold and u<=CFG['model_uncertainty']:
                accepted.append((float(p)*math.expm1(r['features']['log_length']),r,d))
        decisions.append(d)
    for score,r,d in sorted(accepted,key=lambda q:-q[0]):
        p=np.asarray(r['points'],np.float32);q=np.rint(p).astype(int);y,x=q[:,1],q[:,0];pm=v4.raster(previous.shape,[p]);novel=pm&~complete;dist=distance_transform_edt(~complete);covered=float((dist[y,x]<=POLICY['duplicate_distance']*scale).mean());near=float((dist[y,x]<=POLICY['near_existing_distance']*scale).mean())
        if novel.sum()<POLICY['minimum_novel_pixels']*scale or covered>=POLICY['duplicate_fraction']:d['reason']='same_centerline_already_covered';continue
        # Avoid adding a complete second ridge beside an existing thick band.
        # Exact 1px-offset ambiguity is kept pending, not labeled background.
        if r['origin']=='adaptive_multiscale_ridge' and near>POLICY['maximum_near_existing_fraction']:d['reason']='adjacent_centerline_ambiguous';continue
        if not masks['area'][novel].all():d['reason']='outside_search_or_protected';continue
        complete|=pm;d.update(accepted=True,reason='whole_image_supported_curve_no_count_budget')
    before_gap=complete.copy();complete,bridges,events=bridge_curves(rgb,complete,masks,field);sparse=previous.copy();_,mass=cv2.connectedComponents(masks['mass'].astype(np.uint8),8);optional={};structural=[]
    boundary=distance_transform_edt(masks['mass'])
    for score,r,d in sorted(accepted,key=lambda q:-q[0]):
        if not d['accepted']:continue
        p=np.asarray(r['points'],np.float32);q=np.rint(p).astype(int);y,x=q[:,1],q[:,0];f=r['features'];length=math.expm1(f['log_length']);near_boundary=float((boundary[y,x]<=5*scale).mean())>.5;structural_curve=length>=20 or (length>=12 and near_boundary);gid=int(np.bincount(mass[y,x]).argmax())
        keep=structural_curve or optional.get(gid,0)<POLICY['sparse_optional_budget']
        d['sparse_keep']=keep
        if keep:
            sparse|=v4.raster(previous.shape,[p])
            if not structural_curve:optional[gid]=optional.get(gid,0)+1
    # Only show a bridge in the sparse view when its complete endpoint paths
    # are present there; do not leave an isolated connection floating in space.
    for e in events:
        if not e['accepted']:continue
        p=np.asarray(e['points']);q=np.rint(p[[0,-1]]).astype(int)
        if sparse[q[:,1],q[:,0]].all():sparse|=v4.raster(previous.shape,[p])
    assert not (previous&~complete).any() and not (sparse&~complete).any()
    assert np.array_equal(complete&masks['protect'],previous&masks['protect'])
    return dict(complete=complete,sparse=sparse,bridges=bridges,before_gap=before_gap,decisions=decisions,bridge_events=events)

def save(ident,rgb,base,previous,rows,field,masks,data,output=None):
    out=Path(output) if output else ROOT/'outputs'/ident;out.mkdir(parents=True,exist_ok=True);bw=lambda x:np.where(x,0,255).astype(np.uint8)
    for name,a in [('input',rgb),('baseline',bw(base)),('previous',bw(previous)),('complete',bw(data['complete'])),('sparse',bw(data['sparse'])),('before_gap',bw(data['before_gap'])),('bridges',bw(data['bridges']))]:Image.fromarray(a).save(out/(name+'.png'))
    overlay=np.repeat(bw(previous)[:,:,None],3,2);overlay[data['complete']&~previous]=[35,135,220];overlay[data['bridges']]=[205,60,160];Image.fromarray(overlay).save(out/'changes.png')
    proposals=v4.raster(previous.shape,[np.asarray(r['points']) for r in rows]);unknown=proposals&~data['complete'];Image.fromarray(bw(unknown)).save(out/'unknown.png');jsave(out/'decisions.json',data['decisions']);jsave(out/'bridge_events.json',data['bridge_events'])
    for name in ['complete','sparse']:
        paths=[dict(points=r['points'].tolist(),stroke_identity_verified=False) for r in whole_candidates(data[name],max(previous.shape)/496.)[0]];jsave(out/(name+'_paths.json'),dict(paths=paths,coordinate_system='input_image_pixels',policy=POLICY,model_sha256=digest(FIT/'model.pt'),physical_z=None,physical_speed=None,ready_for_robot=False,actually_drawn=False,roles_verified=False));poly=''.join('<polyline points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in r['points'])+'"/>' for r in paths);(out/(name+'.svg')).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {previous.shape[1]} {previous.shape[0]}"><g fill="none" stroke="black" stroke-width="1">{poly}</g></svg>',encoding='utf-8')
    stats=dict(id=ident,previous_pixels=int(previous.sum()),complete_pixels=int(data['complete'].sum()),sparse_pixels=int(data['sparse'].sum()),added_pixels=int((data['complete']&~previous).sum()),added_beyond_v4_pixels=int((data['complete']&~base).sum()),bridge_pixels=int(data['bridges'].sum()),accepted_candidates=sum(d['accepted'] for d in data['decisions']),accepted_existing=sum(d['accepted'] and d['origin']=='existing_v4_candidate' for d in data['decisions']),accepted_new_ridges=sum(d['accepted'] and d['origin']=='adaptive_multiscale_ridge' for d in data['decisions']),accepted_connections=sum(e['accepted'] for e in data['bridge_events']),previous_deleted_pixels=int((previous&~data['complete']).sum()),protected_exact=bool(np.array_equal(data['complete']&masks['protect'],previous&masks['protect'])),previous_cycles=graph_cycles(previous),complete_cycles=graph_cycles(data['complete']),scope='Complete evidence candidate, not claim of fully correct artist strokes')
    jsave(out/'statistics.json',stats);return stats

def run(ident,artifact):
    if not (ROOT/'prepared'/ident/'candidates.json').exists():prepare(ident)
    rows=read(ROOT/'prepared'/ident/'candidates.json')['rows'];z=np.load(ROOT/'prepared'/ident/'field.npz');field={k:z[k] for k in z.files if k not in ['area','protect','mass']};masks={k:z[k] for k in ['area','protect','mass']};rgb=np.asarray(Image.open(STRUCTURE/'outputs'/ident/'input.png').convert('RGB'));base=np.asarray(Image.open(STRUCTURE/'outputs'/ident/'baseline.png'))<128;previous=np.asarray(Image.open(PREVIOUS/'outputs'/ident/'lines.png'))<128;data=extract_complete(rgb,previous,base,rows,field,masks,artifact);return save(ident,rgb,base,previous,rows,field,masks,data)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ids',default='G01_01,G02_01,G03_02,G04_01,G05_01');a=p.parse_args();artifact=torch.load(FIT/'model.pt',map_location='cpu',weights_only=False);artifact['bucket_thresholds']=read(ROOT/'bucket_calibration.json')['thresholds'];ids=[r['id'] for r in read(PREVIOUS/'results_all.json')['records']] if a.ids=='all' else a.ids.split(',');frozen=dict(policy=POLICY,config=CFG,model_sha256=digest(FIT/'model.pt'),inference_code_sha256=digest(__file__),evidence_code_sha256=digest(Path(__file__).with_name('mountain_complete_evidence_20261101.py')),calibration_sha256=digest(ROOT/'bucket_calibration.json'))
    path=ROOT/'inference_protocol_r2_bucket.json'
    if path.exists():assert read(path)==frozen,'Use a new version to change frozen inference'
    else:jsave(path,frozen)
    records=[]
    for ident in ids:
        r=run(ident,artifact);records.append(r);print('COMPLETE_LINES',json.dumps(r),flush=True)
    jsave(ROOT/('results_'+a.ids.replace(',','_')+'.json'),dict(records=records))
