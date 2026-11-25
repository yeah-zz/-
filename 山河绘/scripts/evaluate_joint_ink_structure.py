"""Separate raw neural predictions, learned graph paths, and v4 comparison.

No test reference is opened until inference is saved. Known partial pixels only
are scored on actual paintings; reused artwork is never called a fresh blind test.
"""
import sys
sys.dont_write_bytecode=True
from joint_ink_structure_data import ROOT,R2,LOCAL,OFFSETS
from train_joint_ink_structure import JointModel,initialize,prior
from rock_v15_common import LAB,read,jsave,digest
from evaluate_simple_landscape_specialist_r2 import center_mask,pixel_metrics
import split_detail_v4_frozen as v4
import numpy as np,cv2,torch,math,time,json
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from torch.nn import functional as F
from PIL import Image

@torch.no_grad()
def predict(model,rgb):
    h,w=rgb.shape[:2];tile=256;stride=192
    padded=cv2.copyMakeBorder(rgb,20,max(20,tile-h-20),20,max(20,tile-w-20),cv2.BORDER_REFLECT_101);ph,pw=padded.shape[:2]
    starts=lambda n:sorted(set(list(range(0,n-tile+1,stride))+[n-tile]));loc=[(y,x) for y in starts(ph) for x in starts(pw)]
    features=prior.photometry(padded)[0];window=np.maximum(np.outer(np.hanning(tile),np.hanning(tile)),.05).astype(np.float32);total=np.zeros((18,ph,pw),np.float32);den=np.zeros((ph,pw),np.float32)
    for start in range(0,len(loc),8):
        group=loc[start:start+8];x=torch.from_numpy(np.stack([features[:,y:y+tile,x:x+tile] for y,x in group])).cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):o=model(x)
        o=o.float();o[:,[0,1,4]]=o[:,[0,1,4]].sigmoid();o[:,2:4]=F.normalize(o[:,2:4],dim=1);o[:,5:10]=o[:,5:10].softmax(1);o[:,10:18]=o[:,10:18].sigmoid()
        for (y,x),q in zip(group,o.cpu().numpy()):total[:,y:y+tile,x:x+tile]+=q*window;den[y:y+tile,x:x+tile]+=window
    return (total/den)[:,20:20+h,20:20+w]

def load_joint():
    model=JointModel().cuda();model.load_state_dict(torch.load(ROOT/'joint.pt',map_location='cpu',weights_only=False)['model']);return model.eval()

def reference_center(r):
    doc=read(r['reference']);m=np.zeros(np.asarray(Image.open(r['image'])).shape[:2],np.uint8)
    for s in doc['strokes']:cv2.polylines(m,[np.rint(s['points']).astype(np.int32)],False,1,1)
    return m>0

def partial_metric(mask,z):
    known=z['weight'][0]>0;truth=(z['target'][0]>.3)&known
    pred=mask&known
    return pixel_metrics(pred,truth) if truth.any() else dict(precision=None,recall=None,f1=None,background_fp_fraction=float(pred.sum()/max(1,known.sum())))

def relation_counts(pred,z,threshold):
    valid=z['relation_weight']>0;truth=z['relation']>.5;p=pred[10:18]>=threshold
    return dict(tp=int((p&truth&valid).sum()),tn=int((~p&~truth&valid).sum()),fp=int((p&~truth&valid).sum()),fn=int((~p&truth&valid).sum()))

def summarize_relation(rows):
    sums={k:sum(r[k] for r in rows) for k in ['tp','tn','fp','fn']};s=sums
    positive=s['tp']/max(1,s['tp']+s['fn']);negative=s['tn']/max(1,s['tn']+s['fp'])
    return dict(**s,same_stroke_recall=positive,separate_stroke_recall=negative,balanced_accuracy=(positive+negative)/2)

def calibrate(model):
    rows=[r for r in read(ROOT/'manifest.json')['records'] if r['split']=='calibration'];cache=[]
    for r in rows:
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));cache.append((r,predict(model,rgb)))
    trials=[]
    for method in ['skeleton','nms']:
        for threshold in [.25,.35,.45,.55,.65]:
            synth=[];partial=[];background=[]
            for r,pred in cache:
                mask=center_mask(pred,threshold,method)
                if r['kind']=='actual_partial':
                    z=np.load(r['data']);s=partial_metric(mask,z)
                    if s.get('f1') is not None:partial.append(s['f1'])
                    else:background.append(s['background_fp_fraction'])
                else:synth.append(pixel_metrics(mask,reference_center(r))['f1'])
            row=dict(method=method,threshold=threshold,synthetic_f1=float(np.mean(synth)),partial_f1=float(np.mean(partial)),background_fp_fraction=float(np.mean(background)))
            row['score']=row['synthetic_f1']+.1*row['partial_f1']-5*row['background_fp_fraction'];trials.append(row)
    chosen=max(trials,key=lambda r:r['score']);rel_trials=[]
    for threshold in [.2,.35,.5,.65,.8]:
        values=[relation_counts(pred,np.load(r['data']),threshold) for r,pred in cache if r['kind']!='actual_partial']
        rel_trials.append(dict(threshold=threshold,**summarize_relation(values)))
    relation=max(rel_trials,key=lambda r:r['balanced_accuracy']);selection=dict(center=chosen,relation=relation,center_trials=trials,relation_trials=rel_trials,source='calibration only',calibration_ids=[r['id'] for r in rows],checkpoint_sha256=digest(ROOT/'joint.pt'),test_used=False,unlabeled_artwork_used=False)
    jsave(ROOT/'selection.json',selection);print('JOINT_SELECTED',json.dumps(dict(center=chosen,relation=relation)),flush=True);return selection

def affinity(pred,a,b):
    y,x=a;ny,nx=b;dy,dx=ny-y,nx-x
    for k,(oy,ox) in enumerate(OFFSETS[:4]):
        if (dy,dx)==(oy,ox):return float(pred[10+k,y,x])
        if (dy,dx)==(-oy,-ox):return float(pred[10+k,ny,nx])
    return 1.

def trace_graph(neighbors):
    nodes={p for p,n in neighbors.items() if len(n)!=2};used=set();out=[]
    edge=lambda a,b:tuple(sorted([a,b]))
    def follow(a,b):
        points=[a];used.add(edge(a,b));previous=a;current=b
        while True:
            points.append(current)
            if current in nodes or current==a:break
            rest=[q for q in neighbors[current] if q!=previous and edge(current,q) not in used]
            if not rest:break
            q=rest[0];used.add(edge(current,q));previous,current=current,q
        return np.asarray([(x,y) for y,x in points],np.float32)
    for pool in [sorted(nodes),sorted(neighbors)]:
        for a in pool:
            for b in neighbors[a]:
                if edge(a,b) not in used:out.append(follow(a,b))
    return out

def join_supported_endpoints(neighbors,pred,rgb,threshold):
    paths=trace_graph(neighbors);ends=[]
    for p in paths:
        if v4.plen(p)<7 or np.linalg.norm(p[0]-p[-1])<2:continue
        for side in [0,1]:
            xy,tangent=v4.endpoint(p,side);point=tuple(np.rint(xy[::-1]).astype(int))
            if len(neighbors[point])==1:ends.append((point,xy,tangent))
    if len(ends)<2:return neighbors,[]
    component={};count=0
    for point in neighbors:
        if point in component:continue
        component[point]=count;stack=[point]
        while stack:
            a=stack.pop()
            for b in neighbors[a]:
                if b not in component:component[b]=count;stack.append(b)
        count+=1
    parent=list(range(count))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    evidence=(cv2.GaussianBlur(gray,(0,0),3)-gray>.008)&(cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)))>.018)
    tree=cKDTree([e[1] for e in ends]);candidates=[];ranks={}
    for a,b in sorted(tree.query_pairs(4.3)):
        pa,xa,ta=ends[a];pb,xb,tb=ends[b]
        if component[pa]==component[pb]:continue
        delta=xb-xa;gap=float(np.linalg.norm(delta))
        if gap<1.5:continue
        align=float(ta@(-tb));axis=min(float(ta@(delta/gap)),float(tb@(-delta/gap)))
        if align<.94 or axis<.93:continue
        # Radius-three relation channels are anchored at the actual endpoints.
        def relation_at(p,delta_yx):
            options=[]
            for k,(dy,dx) in enumerate(OFFSETS[4:],4):
                for sign in [1,-1]:
                    error=np.linalg.norm(np.array([dy,dx])*sign-delta_yx)
                    if error<=1.45:
                        q=p if sign==1 else tuple(np.array(p)+delta_yx)
                        options.append((error,float(pred[10+k,q[0],q[1]])))
            return min(options)[1] if options else 0.
        relation=min(relation_at(pa,np.array(pb)-pa),relation_at(pb,np.array(pa)-pb))
        if relation<max(.65,threshold):continue
        bridge=np.rint(np.linspace(xa,xb,max(2,int(math.ceil(gap*2))+1))).astype(int);bridge=np.unique(bridge,axis=0)
        # np.unique sorts coordinates, so sort by projection along start->end.
        bridge=bridge[np.argsort((bridge-xa)@delta)];middle=[(int(y),int(x)) for x,y in bridge[1:-1]]
        if not middle or any(p in neighbors for p in middle):continue
        if float(np.mean([evidence[p] for p in middle]))<.9:continue
        score=.5*relation+.3*axis+.2*align;k=len(candidates);candidates.append((a,b,score,middle,relation,gap))
        ranks.setdefault(a,[]).append((score,k));ranks.setdefault(b,[]).append((score,k))
    accepted=[]
    for k,(a,b,score,middle,relation,gap) in enumerate(candidates):
        ordered=[sorted(ranks[e],reverse=True) for e in [a,b]]
        if any(r[0][1]!=k or (len(r)>1 and r[0][0]-r[1][0]<.08) for r in ordered):continue
        pa=ends[a][0];pb=ends[b][0];ca=root(component[pa]);cb=root(component[pb])
        if ca==cb or len(neighbors[pa])!=1 or len(neighbors[pb])!=1 or any(p in neighbors for p in middle):continue
        sequence=[pa]+middle+[pb]
        for p in middle:neighbors[p]=[]
        for p,q in zip(sequence[:-1],sequence[1:]):neighbors[p].append(q);neighbors[q].append(p)
        parent[ca]=cb;accepted.append(dict(a=list(map(int,pa)),b=list(map(int,pb)),gap=gap,relation_probability=relation,image_supported=True))
    return neighbors,accepted

def vectorize(mask,pred,rgb,selection,use_relations=True):
    _,neighbors=v4.pixel_chains(mask);threshold=selection['relation']['threshold'];removed=0
    if use_relations:
        filtered={p:[] for p in neighbors}
        for p,near in neighbors.items():
            for q in near:
                if affinity(pred,p,q)>=threshold:filtered[p].append(q)
                elif p<q:removed+=1
        neighbors=filtered
    joins=[]
    if use_relations:neighbors,joins=join_supported_endpoints(neighbors,pred,rgb,threshold)
    paths=trace_graph(neighbors)
    # The ordered pixel curves are not artist stroke gold. No reference paths,
    # v4 masks, or manual ROI are loaded at inference.
    return paths,dict(removed_graph_edges=removed,endpoint_bridges=len(joins),joins=joins,direct_bezier_prediction=False,relation_offsets_for_raster_edges=OFFSETS[:4])

def save_output(folder,rgb,pred,mask,paths,meta,persist_prediction=True):
    folder.mkdir(parents=True,exist_ok=True);h,w=mask.shape
    Image.fromarray(np.where(mask,0,255).astype(np.uint8)).save(folder/'raw_center.png')
    planned=v4.raster(mask.shape,paths);Image.fromarray(np.where(planned,0,255).astype(np.uint8)).save(folder/'planned_lines.png')
    Image.fromarray(np.clip(255*(1-pred[1]),0,255).astype(np.uint8)).save(folder/'ink_probability.png')
    overlay=rgb.copy();overlay[planned]=[20,105,220];Image.fromarray(overlay).save(folder/'overlay.png')
    colored=np.full(rgb.shape,255,np.uint8)
    for i,p in enumerate(paths):
        color=tuple(int(v) for v in cv2.cvtColor(np.array([[[i*47%180,140,185]]],np.uint8),cv2.COLOR_HSV2RGB)[0,0]);cv2.polylines(colored,[np.rint(p).astype(np.int32)],False,color,1)
    Image.fromarray(colored).save(folder/'path_identity.png')
    method='neural center/ink/direction prediction and learned pixel affinity graph tracing' if folder.name=='joint' else 'previous neural center prediction; no trained relation branch used'
    jsave(folder/'paths.json',dict(paths=[p.tolist() for p in paths],meta=meta,method=method,reference_used=False,v4_fallback=False,stroke_identity_verified=False,physical_z=None,physical_speed=None,ready_for_robot=False,actually_drawn=False))
    poly=''.join('<polyline points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in p)+'"/>' for p in paths)
    (folder/'lines.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}"><g fill="none" stroke="black" stroke-width="1">{poly}</g></svg>',encoding='utf-8')
    if persist_prediction:np.savez_compressed(folder/'prediction.npz',prediction=pred.astype(np.float16))

def evaluate():
    torch.set_num_threads(4);cv2.setNumThreads(2);joint=load_joint();selection=read(ROOT/'selection.json') if (ROOT/'selection.json').exists() else calibrate(joint)
    assert selection['checkpoint_sha256']==digest(ROOT/'joint.pt')
    prior_model=initialize().eval();old_selection=read(R2/'selection.json')['selected'];manifest=read(ROOT/'manifest.json')
    test=[r for r in manifest['records'] if r['split']=='test'];art=read(LOCAL/'baseline_manifest.json')['records'];rows=[dict(r,eval_kind=r['kind']) for r in test]
    rows += [dict(id=r['id'],image=str(Path(r['baseline'])/'input.png'),eval_kind='actual_reused',source_group=r['group']) for r in art]
    # Save inference on all test samples BEFORE opening their test labels.
    for i,r in enumerate(rows):
        rgb=np.asarray(Image.open(r['image']).convert('RGB'))
        for tag,model,sel in [('previous_neural',prior_model,dict(center=old_selection,relation=selection['relation'])),('joint',joint,selection)]:
            pred=predict(model,rgb);mask=center_mask(pred,sel['center']['threshold'],sel['center']['method']);paths,meta=vectorize(mask,pred,rgb,sel,use_relations=tag=='joint')
            save_output(ROOT/'evaluation'/r['id']/tag,rgb,pred,mask,paths,meta)
        if i%10==0:print('JOINT_PREDICT',i,len(rows),flush=True)
    jsave(ROOT/'prediction_stage_complete.json',dict(images=len(rows),test_labels_opened=False,model_sha256=digest(ROOT/'joint.pt')))
    records=[]
    for r in rows:
        row=dict(id=r['id'],kind=r['eval_kind'])
        if r['eval_kind']!='actual_reused':
            truth=reference_center(r);z=np.load(r['data'])
            for tag in ['previous_neural','joint']:
                folder=ROOT/'evaluation'/r['id']/tag;mask=np.asarray(Image.open(folder/'raw_center.png'))<128;pred=np.load(folder/'prediction.npz')['prediction'].astype(np.float32)
                center=pixel_metrics(mask,truth) if truth.any() else dict(precision=None,recall=None,f1=None,blank_fp_fraction=float(mask.mean()))
                row[tag]=dict(center=center,relation=summarize_relation([relation_counts(pred,z,selection['relation']['threshold'])]))
        else:
            row['accuracy']=None;row['note']='reused unlabeled painting; do not calculate full-image accuracy'
        records.append(row)
    jsave(ROOT/'evaluation_results.json',dict(records=records,actual_accuracy=None,fresh_imagegen_test=False,new_declared_hard_test=True,prior_relation_head='random initialized control; previous neural model had no relation head',source_groups_not_independent_styles=True))
    print('JOINT_EVALUATED',len(records),flush=True)
from pathlib import Path
if __name__=='__main__':evaluate()
