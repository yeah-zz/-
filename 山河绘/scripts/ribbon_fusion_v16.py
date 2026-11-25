# -*- coding: utf-8 -*-
"""Observed ink consistency for the v16 neural candidate, with an analytic ablation."""
from ink_eval_v16 import *

def observed_ink(rgb):
    g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    # A median resists bright grain and a narrow white photo frame. A local
    # maximum/closing overestimated gray paper in the noisy R02 calibration.
    paper=cv2.medianBlur(np.rint(g*255).astype(np.uint8),61).astype(np.float32)/255;paper=cv2.GaussianBlur(paper,(0,0),2)
    return np.clip((paper-g)/np.maximum(paper,.15),0,1)

def fusion_decode(pred,rgb,weak=.035,strong=.14,neural=True):
    alpha=observed_ink(rgb);candidate=alpha>weak;candidate[:4]=False;candidate[-4:]=False;candidate[:,:4]=False;candidate[:,-4:]=False;n,labels,stats,centers=cv2.connectedComponentsWithStats(candidate.astype(np.uint8),8);good=np.zeros(n,bool);strong_ids=set(np.unique(labels[alpha>strong]).tolist());seed=(pred[1]>.85)&(pred[0]>.45)&(alpha>weak) if neural else alpha>.065;soft_ids=set(np.unique(labels[seed]).tolist())
    for k in range(1,n):
        area=stats[k,cv2.CC_STAT_AREA]
        if area<12:continue
        if k in strong_ids:good[k]=True;continue
        if k not in soft_ids or area<24:continue
        ys,xs=np.nonzero(labels==k);cov=np.cov(np.stack((xs,ys)));eig=np.linalg.eigvalsh(cov);elongation=float(eig[-1]/max(eig[0],1.))
        if elongation>=3 and np.max(eig)>=25:good[k]=True
    binary=good[labels];sk=skeletonize(binary);broad=distance_transform_edt(binary)>18;sk[broad]=False;field=np.zeros((8,*sk.shape),np.float32)
    if neural:field[0]=np.maximum(pred[0],np.minimum(alpha*4,1));field[1:3]=pred[2:4]
    else:
        g=cv2.GaussianBlur(alpha,(0,0),2);a=cv2.Sobel(g,cv2.CV_32F,2,0);b=cv2.Sobel(g,cv2.CV_32F,0,2);c=cv2.Sobel(g,cv2.CV_32F,1,1);theta=.5*np.arctan2(2*c,a-b);field[0]=np.minimum(alpha*4,1);field[1:3]=np.stack((np.cos(2*theta),np.sin(2*theta)))
    # Dense grain can create one huge junction node. It has no defensible
    # unique pen continuation and otherwise makes branch pairing quadratic.
    pad=np.pad(sk,1);degree=cv2.filter2D(sk.astype(np.float32),-1,np.ones((3,3),np.float32))-sk
    h,w=sk.shape
    for dy in (-1,1):
        for dx in (-1,1):
            diag=pad[1+dy:1+dy+h,1+dx:1+dx+w];hor=pad[1:1+h,1+dx:1+dx+w];ver=pad[1+dy:1+dy+h,1:1+w];degree-=(diag&(hor|ver))
    expanded=cv2.dilate(((degree>2)&sk).astype(np.uint8),np.ones((5,5),np.uint8))*sk.astype(np.uint8);nn,ll,ss,_=cv2.connectedComponentsWithStats(expanded,8);large=np.zeros(nn,bool);large[ss[:,cv2.CC_STAT_AREA]>48]=True;large[0]=False;crowded=large[ll];sk[crowded]=False
    field[3:5]=field[1:3];paths,info=graph_paths(sk,field,True,min_length=8);return paths,sk,dict(**info,broad_ink_pixels_quarantined=int(broad.sum()),crowded_junction_pixels_quarantined=int(crowded.sum()),observed_ink_consistency=True,neural_weak_seed=neural)

def calibrate(stage):
    torch.set_num_threads(4);cv2.setNumThreads(2);model=load(stage);configs=[dict(weak=.025,strong=.14,neural=True),dict(weak=.035,strong=.14,neural=True),dict(weak=.045,strong=.16,neural=True),dict(weak=.025,strong=.14,neural=False)];pool=[]
    for i in range(20):
        rgb,t,m,meta=generated(162000+i);pool.append((rgb,predict(model,rgb),t,meta))
    r=next(r for r in reference_records() if r['id']=='R02');rgb=np.asarray(Image.open(r['image']).convert('RGB'));pred=predict(model,rgb);doc=read(r['labels']);_,_,mask,_=fields(doc);rows=[]
    for config in configs:
        scores=[]
        print('FUSION_CONFIG_START',config,flush=True)
        for pi,(im,p,t,meta) in enumerate(pool):
            paths,sk,_=fusion_decode(p,im,**config);scores.append(complete_metrics(paths,sk,t,meta))
            if pi%10==9:print('FUSION_CONFIG_PROGRESS',pi+1,flush=True)
        paths,sk,_=fusion_decode(pred,rgb,**config);real=path_metrics([s['points'] for s in doc['strokes']],paths,mask[3]);density=float(sk[mask[1]>0].mean());mean=lambda key:float(np.mean([s[key] for s in scores if s[key] is not None]));score=.4*mean('f1')+.15*mean('faint_reference_coverage')+.45*real['mean_best_single_path_coverage']-min(.8,density*40)-.1*mean('bold_multiple_center_fraction');rows.append(dict(config=config,score=score,R02_single_path_coverage=real['mean_best_single_path_coverage'],R02_background_density=density,synthetic_f1=mean('f1'),synthetic_faint_coverage=mean('faint_reference_coverage'),synthetic_double_fraction=mean('bold_multiple_center_fraction')))
    # The analytic result is retained as an ablation, not silently called DL.
    selected=max([r for r in rows if r['config']['neural']],key=lambda r:r['score']);folder=ROOT/f'stage_{stage:02d}'/'observed_fusion';folder.mkdir(exist_ok=True)
    if (folder/'calibration.json').exists() and not (folder/'closing_trial_calibration.json').exists():shutil.copy2(folder/'calibration.json',folder/'closing_trial_calibration.json')
    jsave(folder/'calibration.json',dict(selected=selected,candidates=rows,selection_data=['R02_partial','synthetic_162000_162019'],analytic_ablation_separate=True,paper_estimator='61px local median, Gaussian sigma 2',previous_failed_closing_trial_preserved=True));print('FUSION_CALIBRATED',json.dumps(selected),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',type=int);a=p.parse_args();calibrate(a.stage)
