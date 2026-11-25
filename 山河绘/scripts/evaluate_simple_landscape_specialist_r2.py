# -*- coding: utf-8 -*-
"""Calibration, paired synthetic test, and unlabeled imagegen domain check."""
from train_simple_landscape_specialist_r2 import Specialist,ROOT,prior
from rock_v15_common import LAB,read,jsave,digest
import split_detail_v4_frozen as baseline
import numpy as np,cv2,torch,json,time,math
from torch.nn import functional as F
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from PIL import Image

def load_model():
    state=torch.load(ROOT/'specialist.pt',map_location='cpu',weights_only=False);model=Specialist().cuda();model.load_state_dict(state['model']);return model.eval()

@torch.no_grad()
def predict(model,rgb):
    h,w=rgb.shape[:2];tile=256;stride=192
    padded=cv2.copyMakeBorder(rgb,20,max(20,tile-h-20),20,max(20,tile-w-20),cv2.BORDER_REFLECT_101);ph,pw=padded.shape[:2]
    starts=lambda n:sorted(set(list(range(0,n-tile+1,stride))+[n-tile]));loc=[(y,x) for y in starts(ph) for x in starts(pw)]
    features=prior.photometry(padded)[0];window=np.maximum(np.outer(np.hanning(tile),np.hanning(tile)),.05).astype(np.float32);total=np.zeros((10,ph,pw),np.float32);den=np.zeros((ph,pw),np.float32)
    for start in range(0,len(loc),8):
        group=loc[start:start+8];x=torch.from_numpy(np.stack([features[:,y:y+tile,x:x+tile] for y,x in group])).cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):o=model(x)
        o=o.float();o[:,[0,1,4]]=o[:,[0,1,4]].sigmoid();o[:,2:4]=F.normalize(o[:,2:4],dim=1);o[:,5:]=o[:,5:].softmax(1)
        for (y,x),q in zip(group,o.cpu().numpy()):total[:,y:y+tile,x:x+tile]+=q*window;den[y:y+tile,x:x+tile]+=window
    return (total/den)[:,20:20+h,20:20+w]

def center_mask(pred,threshold,method):
    mask=(pred[0]>=threshold)&(pred[1]>=.12)
    if method=='nms':
        h,w=mask.shape;yy,xx=np.mgrid[:h,:w].astype(np.float32);theta=.5*np.arctan2(pred[3],pred[2]);nx=-np.sin(theta);ny=np.cos(theta);p=pred[0]
        left=cv2.remap(p,xx+nx,yy+ny,cv2.INTER_LINEAR);right=cv2.remap(p,xx-nx,yy-ny,cv2.INTER_LINEAR)
        mask&=((p>=left)&(p>=right))|(pred[8]>.45)
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8);keep=stats[:,cv2.CC_STAT_AREA]>=3;keep[0]=False
    return skeletonize(keep[labels])

def reference_mask(doc,shape):
    mask=np.zeros(shape,np.uint8)
    for r in doc['strokes']:cv2.polylines(mask,[np.rint(r['points']).astype(np.int32)],False,1,1,cv2.LINE_8)
    return mask>0

def pixel_metrics(mask,ref,tol=1.5):
    ds=distance_transform_edt(~mask);dr=distance_transform_edt(~ref)
    precision=float((dr[mask]<=tol).mean()) if mask.any() else 0.;recall=float((ds[ref]<=tol).mean()) if ref.any() else 1.
    return dict(precision=precision,recall=recall,f1=2*precision*recall/max(precision+recall,1e-9))

def vectorize(mask,pred,rgb):
    paths,_=baseline.pixel_chains(mask);pine=cv2.dilate((pred[8]>.45).astype(np.uint8),np.ones((3,3),np.uint8))>0
    detail=[];structure=[]
    for p in paths:
        if baseline.near_region(p,pine).any():detail.append(p)
        elif baseline.plen(p)>=3:structure.append(p)
    structure,joins,_=baseline.stitch(structure,rgb,pred[1]>.5,pine,baseline.CFG)
    return detail+structure,joins

def qa(mask,rgb):
    _,source=baseline.source_ink(rgb);alpha=prior.photometry(rgb)[1];local=np.maximum(cv2.GaussianBlur(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255,(0,0),3)-cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255,0)
    support=float(((alpha>.012)|(local>.004))[mask].mean()) if mask.any() else 0.
    ds=distance_transform_edt(~mask);coverage=float((ds[source]<=2).mean()) if source.any() else 1.
    n,labels,stats,_=cv2.connectedComponentsWithStats((~mask).astype(np.uint8),connectivity=4);h,w=mask.shape
    loops=sum(1 for x,y,bw,bh,area in stats[1:] if area<=32 and x>0 and y>0 and x+bw<w and y+bh<h)
    return dict(image_support=support,source_pixel_coverage=coverage,small_closed_regions=int(loops),pixel_count=int(mask.sum()))

def calibrate(model):
    manifest=read(ROOT/'paired_manifest.json');rows=[r for r in manifest['records'] if r['split']=='calibration'][12:]
    cache=[]
    for r in rows:
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));pred=predict(model,rgb);doc=read(r['reference']);cache.append((r,rgb,pred,reference_mask(doc,rgb.shape[:2])))
    trials=[]
    for method in ['skeleton','nms']:
        for threshold in [.25,.35,.45,.55]:
            scores=[pixel_metrics(center_mask(pred,threshold,method),ref) for _,rgb,pred,ref in cache]
            trials.append(dict(method=method,threshold=threshold,mean_f1=float(np.mean([s['f1'] for s in scores])),mean_precision=float(np.mean([s['precision'] for s in scores])),mean_recall=float(np.mean([s['recall'] for s in scores]))))
    selected=max(trials,key=lambda r:r['mean_f1']);q=[]
    for _,rgb,pred,ref in cache:
        mask=center_mask(pred,selected['threshold'],selected['method']);q.append(qa(mask,rgb))
    # Rules screen risk only; synthetic calibration cannot certify artwork quality.
    gates=dict(min_image_support=max(.45,float(np.quantile([a['image_support'] for a in q],.05))-.03),min_source_coverage=max(.20,float(np.quantile([a['source_pixel_coverage'] for a in q],.05))-.05),max_small_closed_regions=max(3,int(np.quantile([a['small_closed_regions'] for a in q],.95))+2))
    config=dict(selected=selected,trials=trials,qa_gates=gates,retry_threshold=max(.18,selected['threshold']-.08),calibration_ids=[r['id'] for r in rows],checkpoint_sha256=digest(ROOT/'specialist.pt'),training_manifest_sha256=digest(ROOT/'paired_manifest.json'),imagegen_holdout_manifest_sha256=digest(ROOT/'imagegen_holdout_manifest.json'),test_labels_used_for_selection=False,imagegen_images_used_for_selection=False,source_code_sha256=digest(__file__),qa_pass_is_not_expert_acceptance=True)
    jsave(ROOT/'selection.json',config);print('SPECIALIST_SELECTED',json.dumps(config['selected']),flush=True);return config

def passes(q,gates):
    return q['pixel_count']>5 and q['image_support']>=gates['min_image_support'] and q['source_pixel_coverage']>=gates['min_source_coverage'] and q['small_closed_regions']<=gates['max_small_closed_regions']

def write_lines(folder,rgb,mask,paths,qa_record,method):
    folder.mkdir(parents=True,exist_ok=True);h,w=mask.shape
    Image.fromarray(rgb).save(folder/'input.png');Image.fromarray(np.where(mask,0,255).astype(np.uint8)).save(folder/'lines.png');overlay=rgb.copy();overlay[mask]=[20,105,220];Image.fromarray(overlay).save(folder/'overlay.png')
    jsave(folder/'paths.json',dict(paths=[p.tolist() for p in paths],method=method,qa=qa_record,qa_is_not_expert_review=True,reference_labels_used=False,physical_z=None,ready_for_robot=False,actually_drawn=False))
    polys=['<polyline points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in p)+'"/>' for p in paths]
    (folder/'lines.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}"><metadata>Offline image coordinates; no calibrated robot commands.</metadata><g fill="none" stroke="black" stroke-width="1">'+''.join(polys)+'</g></svg>',encoding='utf-8')

def single_path_score(paths,doc):
    if not paths:return 0.
    points=np.concatenate(paths);owner=np.concatenate([np.full(len(p),i,int) for i,p in enumerate(paths)]);tree=cKDTree(points);scores=[]
    for row in doc['strokes']:
        p=np.asarray(row['points']);length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];t=np.linspace(0,length[-1],max(2,int(length[-1])+1));q=np.stack([np.interp(t,length,p[:,k]) for k in [0,1]],1)
        dist,ids=tree.query(q,k=min(4,len(points)))
        if dist.ndim==1:dist=dist[:,None];ids=ids[:,None]
        counts={}
        for d,idx in zip(dist,ids):
            for who in set(owner[idx[d<=2.]].tolist()):counts[who]=counts.get(who,0)+1
        scores.append(max(counts.values(),default=0)/len(q))
    return float(np.mean(scores))

def evaluate():
    torch.set_num_threads(4);cv2.setNumThreads(2);model=load_model();selection=calibrate(model)
    paired=[dict(r,kind='paired') for r in read(ROOT/'paired_manifest.json')['records'] if r['split']=='test'];generated=[dict(r,kind='imagegen') for r in read(ROOT/'imagegen_holdout_manifest.json')['records']]
    rows=paired+generated
    # Stage A writes every prediction before Stage B opens any test reference.
    for i,r in enumerate(rows):
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));pred=predict(model,rgb);folder=ROOT/'predictions'/r['id'];folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/'prediction.npz',prediction=pred.astype(np.float16))
        if i%12==0:print('SPECIALIST_PREDICT',i,len(rows),flush=True)
    jsave(ROOT/'prediction_stage_complete.json',dict(count=len(rows),test_references_opened_in_this_stage=False,checkpoint_sha256=digest(ROOT/'specialist.pt')))
    records=[]
    for i,r in enumerate(rows):
        folder=ROOT/'outputs'/r['id'];rgb=np.asarray(Image.open(r['image']).convert('RGB'));pred=np.load(ROOT/'predictions'/r['id']/'prediction.npz')['prediction'].astype(np.float32)
        mask=center_mask(pred,selection['selected']['threshold'],selection['selected']['method']);paths,joins=vectorize(mask,pred,rgb);mask=baseline.raster(rgb.shape[:2],paths);q=qa(mask,rgb);retried=False
        chosen_mask=mask;chosen_paths=paths;chosen_q=q
        if not passes(q,selection['qa_gates']):
            low=center_mask(pred,selection['retry_threshold'],selection['selected']['method']);pp,jj=vectorize(low,pred,rgb);low=baseline.raster(rgb.shape[:2],pp);qq=qa(low,rgb);retried=True
            def score(v):return v['image_support']+.6*v['source_pixel_coverage']-.03*v['small_closed_regions']
            if passes(qq,selection['qa_gates']) or score(qq)>score(q):chosen_mask,chosen_paths,chosen_q=low,pp,qq
        basefolder=folder/'v4'
        if not (basefolder/'paths.json').exists():baseline.run(r['image'],basefolder)
        basedoc=read(basefolder/'paths.json');basepaths=[np.asarray(a['points'],np.float32) for a in basedoc['paths']];basemask=baseline.raster(rgb.shape[:2],basepaths)
        passed=passes(chosen_q,selection['qa_gates']);finalmask=chosen_mask if passed else basemask;finalpaths=chosen_paths if passed else basepaths
        write_lines(folder/'raw_neural',rgb,mask,paths,q,'trained specialist raw output')
        write_lines(folder/'retried_neural',rgb,chosen_mask,chosen_paths,chosen_q,'specialist with calibration-fixed retry rule')
        write_lines(folder/'selected',rgb,finalmask,finalpaths,chosen_q,'specialist' if passed else 'v4 fallback; unresolved quality flag')
        record=dict(id=r['id'],kind=r['kind'],source=r['image'],source_sha256=digest(r['image']),retry_attempted=retried,automatic_rule_pass=passed,fallback_to_v4=not passed,raw_qa=q,chosen_qa=chosen_q,automatic_pass_not_expert_accuracy=True)
        if r['kind']=='paired':
            doc=read(r['reference']);ref=reference_mask(doc,rgb.shape[:2]);record['metrics']={name:dict(pixel_metrics(mm,ref),single_path_coverage=single_path_score(pp,doc)) for name,mm,pp in [('v4',basemask,basepaths),('neural',mask,paths),('selected',finalmask,finalpaths)]}
        records.append(record);jsave(ROOT/'evaluation_progress.json',dict(records=records,complete=len(records)==len(rows),expected=len(rows)))
        print('SPECIALIST_EVAL',i+1,len(rows),r['id'],'qa',passed,'fallback',not passed,flush=True)
    paired_records=[r for r in records if r['kind']=='paired'];gen_records=[r for r in records if r['kind']=='imagegen']
    summary=dict(paired_test_count=len(paired_records),imagegen_scene_count=len(gen_records),imagegen_call_count=5,paired_metrics={name:{k:float(np.mean([r['metrics'][name][k] for r in paired_records])) for k in ['precision','recall','f1','single_path_coverage']} for name in ['v4','neural','selected']},imagegen_automatic_pass_count=sum(r['automatic_rule_pass'] for r in gen_records),imagegen_fallback_count=sum(r['fallback_to_v4'] for r in gen_records),imagegen_human_accuracy_unknown=True,procedural_metrics_do_not_establish_imagegen_accuracy=True,checkpoint_sha256=digest(ROOT/'specialist.pt'),selection_sha256=digest(ROOT/'selection.json'),test_references_used_only_after_prediction_stage_complete=True,old_version_not_overwritten=True,new_training_updates=read(ROOT/'training_summary.json')['completed_updates'])
    jsave(ROOT/'evaluation_summary.json',summary);print('SPECIALIST_EVALUATED',json.dumps(summary),flush=True)

if __name__=='__main__':evaluate()
