# -*- coding: utf-8 -*-
"""Native fine-detail routing + long faint evidence. Independent offline candidate."""
import sys
sys.dont_write_bytecode=True
from native_fine_faint_20261101 import *
from train_native_fan_router_20261101 import FanRouter,predict
from mountain_structure_v4_20261101 import whole_candidates
from scipy.spatial import cKDTree
from scipy.ndimage import maximum_filter
import torch,heapq,time,datetime

CFG3=dict(version='learned_fan_native_faint_r3',fan_probability=.65,fan_seed=.90,minimum_fine_length=2.5,faint_min_length=38.,faint_min_extent=26.,faint_support=.82,faint_contrast=.012,faint_span_ratio=.30,maximum_gap=8.,bridge_facing=.84,bridge_axis=.72,bridge_support=.90,bridge_max_unsupported_run=1,whole_replacement_distance=5.,whole_replacement_fraction=.80,old_near_fraction_for_skip=.97,neighbor_contact_distance=1.4,mountain_blue_minus_red=.012,mountain_green_minus_red=.006)
OUTPUTS=ROOT/'outputs_r3'

def same_band(gray,old,new):
    """Reject matches with a brighter separation valley between neighboring strokes."""
    bright=np.maximum(gray[np.rint(old[:,1]).astype(int),np.rint(old[:,0]).astype(int)],gray[np.rint(new[:,1]).astype(int),np.rint(new[:,0]).astype(int)])
    peak=bright.copy()
    for t in [.2,.4,.6,.8]:
        p=old*(1-t)+new*t;xy=np.rint(p).astype(int);peak=np.maximum(peak,gray[xy[:,1],xy[:,0]])
    return peak<=bright+.012

def mountain_area(rgb):
    a=cv2.GaussianBlur(rgb.astype(np.float32)/255,(0,0),5);m=((a[:,:,2]-a[:,:,0])>CFG3['mountain_blue_minus_red'])&((a[:,:,1]-a[:,:,0])>CFG3['mountain_green_minus_red']);m=cv2.morphologyEx(m.astype(np.uint8),cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31)));n,cc,stats,_=cv2.connectedComponentsWithStats(m,8);keep=np.zeros(m.shape,bool)
    for i,(x,y,w,h,area) in enumerate(stats[1:],1):
        if h>=50 and area>=400 and w/max(h,1)<4.5:keep|=cc==i
    return cv2.dilate(keep.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)))>0

def bridge_ridges(f,allowed):
    mask=f['ridge']&allowed;paths,neighbors=graph.pixel_chains(mask);coords=[];desc=[];n,cc=cv2.connectedComponents(mask.astype(np.uint8),8);h,w=mask.shape
    for i,p in enumerate(paths):
        if graph.plen(p)<12:continue
        q=np.rint(p).astype(int)
        if float(np.median(f['contrast'][q[:,1],q[:,0]]))<.009:continue
        for side in [0,1]:
            xy,t=graph.endpoint(p,side);x,y=np.rint(xy).astype(int)
            if len(neighbors[(y,x)])!=1:continue
            coords.append(xy);desc.append((i,side,t,int(cc[y,x])))
    if not coords:return mask,[]
    coords=np.asarray(coords);tree=cKDTree(coords);candidates=[];by_end={};offsets=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for a,b in sorted(tree.query_pairs(CFG3['maximum_gap'])):
        ia,sa,ta,ca=desc[a];ib,sb,tb,cb=desc[b]
        if ia==ib or ca==cb or float(ta@(-tb))<CFG3['bridge_facing']:continue
        delta=coords[b]-coords[a];gap=float(np.linalg.norm(delta))
        if gap<1.5:continue
        axis=delta/gap
        if min(float(ta@axis),float(tb@(-axis)))<CFG3['bridge_axis']:continue
        start=tuple(np.rint(coords[a][::-1]).astype(int));end=tuple(np.rint(coords[b][::-1]).astype(int));lo=np.maximum([0,0],np.minimum(start,end)-3);hi=np.minimum([h-1,w-1],np.maximum(start,end)+3);queue=[(0.,start)];cost={start:0.};parent={};visited=set()
        while queue:
            c,at=heapq.heappop(queue)
            if at in visited:continue
            visited.add(at)
            if at==end:break
            for dy,dx in offsets:
                y,x=at[0]+dy,at[1]+dx;nxt=(y,x)
                if y<lo[0] or x<lo[1] or y>hi[0] or x>hi[1] or not allowed[y,x]:continue
                step=math.hypot(dx,dy);e=float(f['contrast'][y,x]/max(f['threshold'][y,x],.001));score=float(f['score'][y,x]);value=c+step*(1+3*max(0,1-e)+.25/(1+score/.004))
                if value<cost.get(nxt,1e9):cost[nxt]=value;parent[nxt]=at;heapq.heappush(queue,(value,nxt))
        if end not in parent:continue
        seq=[end]
        while seq[-1]!=start:seq.append(parent[seq[-1]])
        seq=seq[::-1];q=np.array([(x,y) for y,x in seq],np.float32)
        if graph.plen(q)>gap*1.55+1:continue
        qq=np.rint(q).astype(int);x,y=qq[:,0],qq[:,1];support=f['contrast'][y,x]>=f['threshold'][y,x]*.9;run=bad=0
        for v in support:
            run=0 if v else run+1;bad=max(bad,run)
        if support.mean()<CFG3['bridge_support'] or bad>CFG3['bridge_max_unsupported_run']:continue
        # A source-backed gap path may touch its two endpoint components only.
        if len(q)>4:
            interior=np.rint(q[2:-2]).astype(int);touch=cv2.dilate(mask.astype(np.uint8),np.ones((3,3),np.uint8));nearlabels=set()
            for x,y in interior:
                nearlabels.update(cc[max(0,y-1):y+2,max(0,x-1):x+2].ravel().tolist())
            if nearlabels-{0,ca,cb}:continue
        score=float(ta@(-tb))+.3*min(float(ta@axis),float(tb@(-axis)))-gap*.02;k=len(candidates);candidates.append(dict(a=a,b=b,component_a=ca,component_b=cb,points=q,score=score,support=float(support.mean()),gap=gap));by_end.setdefault(a,[]).append((score,k));by_end.setdefault(b,[]).append((score,k))
    parentcc={}
    def root(x):
        while x in parentcc:x=parentcc[x]
        return x
    result=mask.copy();events=[];used=set()
    for k,r in sorted(enumerate(candidates),key=lambda a:-a[1]['score']):
        ranks=[sorted(by_end[e],reverse=True) for e in [r['a'],r['b']]]
        if any(z[0][1]!=k or (len(z)>1 and z[0][0]-z[1][0]<.06) for z in ranks) or r['a'] in used or r['b'] in used:continue
        a,b=root(r['component_a']),root(r['component_b'])
        if a==b:continue
        parentcc[a]=b;used.update([r['a'],r['b']]);q=np.rint(r['points']).astype(int);result[q[:,1],q[:,0]]=True;events.append({**{k:v for k,v in r.items() if k not in ['points','a','b']},'points':r['points'].tolist(),'same_artist_stroke_verified':False})
    return result,events

def refine(rgb,basepaths,prob,f):
    h,w=rgb.shape[:2];base=graph.raster((h,w),basepaths);fan=binary_propagation(prob>=CFG3['fan_seed'],mask=prob>=CFG3['fan_probability']);fan=cv2.morphologyEx(fan.astype(np.uint8),cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)));fan=cv2.dilate(fan,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))>0
    fine=np.zeros_like(base);detail=[];paths,_=graph.pixel_chains(f['fine'])
    for p in paths:
        q=np.rint(p).astype(int);x,y=q[:,0],q[:,1]
        if graph.plen(p)>=CFG3['minimum_fine_length'] and float(fan[y,x].mean())>=.88:
            fine[y,x]=True;detail.append(dict(role='native_detail_candidate',points=p.tolist(),stroke_identity_verified=False))
    area=mountain_area(rgb)&~cv2.dilate(fan.astype(np.uint8),np.ones((9,9),np.uint8)).astype(bool);ridge,bridges=bridge_ridges(f,area);candidates,_=whole_candidates(ridge,1.);faint=[];pending=[]
    for r in candidates:
        p=r['points'];q=np.rint(p).astype(int);x,y=q[:,0],q[:,1];length=graph.plen(p);extent=float(np.linalg.norm(np.ptp(p,axis=0)));span=float(np.linalg.norm(p[-1]-p[0]));support=float((f['contrast'][y,x]>=f['threshold'][y,x]).mean());contrast=float(np.median(f['contrast'][y,x]));valid=bool(length>=CFG3['faint_min_length'] and extent>=CFG3['faint_min_extent'] and span/max(length,1)>=CFG3['faint_span_ratio'] and contrast>=CFG3['faint_contrast'] and support>=CFG3['faint_support'])
        row=dict(points=p,role='long_faint_candidate',length=length,contrast=contrast,support=support)
        (faint if valid else pending).append(row)
    # Replace complete matching old fragments; never chop a new path into novelty bits.
    result=(base&~fan)|fine;selected=[];replaced=set();oldtree=cKDTree(np.concatenate(basepaths)) if basepaths else None;currentdistance=distance_transform_edt(~result)
    for r in sorted(faint,key=lambda r:-r['length']):
        p=r['points'];q=np.rint(p).astype(int);x,y=q[:,0],q[:,1];close=currentdistance[y,x]<CFG3['whole_replacement_distance']
        if float(close.mean())>=CFG3['old_near_fraction_for_skip']:pending.append(dict(r,reason='mostly_already_covered'));continue
        tree=cKDTree(p);remove=[]
        for i,old in enumerate(basepaths):
            if i in replaced:continue
            if np.any(old.max(0)<p.min(0)-3) or np.any(old.min(0)>p.max(0)+3):continue
            d,idx=tree.query(old);covered=(d<CFG3['whole_replacement_distance'])&same_band(f['gray'],old,p[idx])
            if covered.mean()<CFG3['whole_replacement_fraction']:continue
            if len(old)>2 and len(p)>2:
                ot=np.gradient(old,axis=0);ot/=np.maximum(np.linalg.norm(ot,axis=1,keepdims=True),1e-6);pt=np.gradient(p,axis=0);pt/=np.maximum(np.linalg.norm(pt,axis=1,keepdims=True),1e-6);alignment=np.abs((ot*pt[idx]).sum(1))
                if np.median(alignment[covered])<.80:continue
            remove.append(i)
        if close.mean()>.12 and not remove:pending.append(dict(r,reason='partial_overlap_identity_uncertain'));continue
        if remove:
            # The new complete candidate may replace complete old fragments only.
            # Any other parallel old path that would remain makes this ambiguous.
            retained=[old for i,old in enumerate(basepaths) if i not in replaced and i not in remove]
            if retained:
                dist=cKDTree(np.concatenate(retained)).query(p)[0]
                if float((dist<CFG3['whole_replacement_distance']).mean())>.12:pending.append(dict(r,reason='remaining_neighbor_overlap'));continue
        if remove:
            removed=graph.raster((h,w),[basepaths[i] for i in remove]);result[removed&~fan]=False;replaced.update(remove)
        result[y,x]=True;selected.append(dict(r,replaced_old_path_ids=remove));currentdistance=distance_transform_edt(~result)
    # Re-raster every retained old path to protect shared graph-junction pixels.
    surviving=[p for i,p in enumerate(basepaths) if i not in replaced];oldkeep=graph.raster((h,w),surviving)&~fan;result|=oldkeep
    return result,dict(fan=fan,fine=fine,mountain=area,faint_added=result&~base&~fan,pending=graph.raster((h,w),[r['points'] for r in pending])),dict(details=detail,faint=selected,pending=pending,bridges=bridges,replaced_old_path_ids=sorted(replaced))

def save_result(source,baseline_paths,output,model=None,precomputed_field=None):
    source=Path(source);output=Path(output)
    if output.drive.upper() not in ['E:','F:'] or not output.is_absolute():raise ValueError('Use absolute E/F output')
    if (output/'entry.json').exists():raise FileExistsError('Preserve completed experiment')
    rgb=np.asarray(Image.open(source).convert('RGB'));doc=read(baseline_paths);baseline_source=Path(baseline_paths).parent/'input.png'
    with Image.open(baseline_source) as im:bw,bh=im.size
    paths=[np.asarray(r['points'],np.float32)*[rgb.shape[1]/bw,rgb.shape[0]/bh] for r in doc['paths']];base=graph.raster(rgb.shape[:2],paths);prob=predict(rgb,model);f=precomputed_field if precomputed_field is not None else field(rgb);result,masks,decisions=refine(rgb,paths,prob,f);output.mkdir(parents=True,exist_ok=True);canonical,_=graph.pixel_chains(result);canonical_mask=graph.raster(result.shape,canonical);isolated=result&~canonical_mask
    for name,mask in [('lines',result),('baseline',base),('exported_lines',canonical_mask),('isolated_evidence',isolated)]+list(masks.items()):Image.fromarray(np.where(mask,0,255).astype(np.uint8)).save(output/(name+'.png'))
    Image.fromarray(np.rint(prob*255).astype(np.uint8)).save(output/'fan_probability.png');changes=np.repeat(np.where(base,0,255).astype(np.uint8)[:,:,None],3,2);changes[result&~base]=[25,135,200];changes[base&~result]=[215,65,65];Image.fromarray(changes).save(output/'changes.png')
    def clean(r):return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in r.items()}
    jsave(output/'decisions.json',dict(details=decisions['details'],faint=[clean(r) for r in decisions['faint']],pending=[clean(r) for r in decisions['pending']],bridges=decisions['bridges'],replaced_old_path_ids=decisions['replaced_old_path_ids']))
    jsave(output/'paths.json',dict(paths=[dict(points=p.tolist(),stroke_identity_verified=False) for p in canonical],coordinate_system='native_source_image_pixels',width=rgb.shape[1],height=rgb.shape[0],source_sha256=digest(source),ready_for_robot=False,actually_drawn=False,physical_z=None,speed=None))
    svg=''.join('<polyline points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in p)+'"/>' for p in canonical);(output/'lines.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {rgb.shape[1]} {rgb.shape[0]}"><g fill="none" stroke="black" stroke-width="1">'+svg+'</g></svg>',encoding='utf-8')
    stats=dict(input_size=[rgb.shape[1],rgb.shape[0]],baseline_pixels=int(base.sum()),result_pixels=int(result.sum()),added_pixels=int((result&~base).sum()),removed_pixels=int((base&~result).sum()),native_detail_pixels=int(masks['fine'].sum()),long_faint_candidates_accepted=len(decisions['faint']),whole_old_fragments_replaced=len(decisions['replaced_old_path_ids']),source_supported_bridge_hypotheses=len(decisions['bridges']),export_paths=len(canonical),isolated_evidence_pixels=int(isolated.sum()),model_semantic_accuracy_established=False,ready_for_robot=False)
    jsave(output/'statistics.json',stats);jsave(output/'entry.json',dict(source=str(source),source_sha256=digest(source),baseline_paths=str(baseline_paths),baseline_paths_sha256=digest(baseline_paths),router_model_sha256=digest(ROOT/'router/router.pt'),config=CFG3,source_field_config=CFG,annotation_files_read_at_inference=False,inference_uses_sample_ids=False,robot_connected=False,statistics=stats));return stats

def batch(ids):
    torch.set_num_threads(4);cv2.setNumThreads(2);artifact=torch.load(ROOT/'router/router.pt',map_location='cpu',weights_only=False);model=FanRouter().cuda();model.load_state_dict(artifact['model']);records=[]
    for ident in ids:
        out=OUTPUTS/ident
        if (out/'entry.json').exists():records.append(read(out/'entry.json'));continue
        source=BASE/'originals'/(ident+'.png');sc=ROOT/'scout'/ident/'evidence.npz';f=None
        if sc.exists():z=np.load(sc);f={k:z[k] for k in z.files}
        started=time.perf_counter();stats=save_result(source,BASE/'outputs'/ident/'complete/complete_paths.json',out,model,f);records.append(read(out/'entry.json'));print('NATIVE_REFINEMENT_DONE',ident,'seconds',round(time.perf_counter()-started,1),stats,flush=True)
    jsave(ROOT/'batch_results.json',dict(records=records,all_results_retained=True,configuration=CFG3))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ids');p.add_argument('--image');p.add_argument('--output');p.add_argument('--baseline-paths');a=p.parse_args()
    if a.image:
        out=Path(a.output) if a.output else ROOT/'manual_outputs'/(Path(a.image).stem+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        if a.baseline_paths:baseline=Path(a.baseline_paths)
        else:
            from extract_complete_lines_20261101 import extract
            rgb=Image.open(a.image).convert('RGB');rgb.thumbnail((496,496),Image.Resampling.LANCZOS);out.mkdir(parents=True,exist_ok=True);rgb.save(out/'evaluation_input.png');extract(out/'evaluation_input.png',out/'upstream');baseline=out/'upstream/complete/complete_paths.json'
        print(save_result(a.image,baseline,out/'refinement'))
    else:batch((a.ids or 'T01,T06,T17,T19').split(','))
