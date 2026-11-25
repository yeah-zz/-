# -*- coding: utf-8 -*-
"""Post-hoc image-ridge verification; raw failed proposals remain preserved."""
from v32_experiment import *
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d

CFG=dict(min_ridge_score=.002,min_relative_ink=.02,max_projection_distance=2.5,min_axis_agreement=.75,min_distance_to_existing=4.2,min_run_points=13,max_projected_jump=2.6)

def runs(mask):
    d=np.diff(np.r_[False,mask,False].astype(np.int8));return list(zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)))

def verified_paths(rgb,base,proposals):
    a=observed(rgb);score,axis,_=ridge_field(a);yy,xx=np.where((score>=CFG['min_ridge_score'])&(a>=CFG['min_relative_ink']));ridge=np.c_[xx,yy].astype(np.float32)
    if not len(ridge):return [],[]
    tree=cKDTree(ridge);axes=axis[yy,xx];seen=paths_mask(base,a.shape);distance=distance_transform_edt(seen==0).astype(np.float32);output=[];audit=[]
    for pp in sorted(proposals,key=lambda r:len(r['points']),reverse=True):
        p=arc(pp['points']);d=np.gradient(p,axis=0);d/=np.maximum(np.linalg.norm(d,axis=1,keepdims=True),1e-6);dist,idx=tree.query(p,k=min(8,len(ridge)));dist=np.atleast_2d(dist) if dist.ndim==1 else dist;idx=np.atleast_2d(idx) if idx.ndim==1 else idx
        agreement=np.abs((axes[idx]*d[:,None]).sum(-1));allowed=(dist<=2.5)&(agreement>=.75);cost=np.where(allowed,dist+2*(1-agreement),1e6);k=np.argmin(cost,axis=1);q=ridge[idx[np.arange(len(p)),k]];supported=cost[np.arange(len(p)),k]<1e5
        # Every accepted run needs actual ridge support; no weak-support majority vote.
        supported[1:]&=np.linalg.norm(np.diff(q,axis=0),axis=1)<=CFG['max_projected_jump']
        supported&=sample(distance,q)>CFG['min_distance_to_existing']
        for start,end in runs(supported):
            if end-start<CFG['min_run_points']:continue
            z=arc(gaussian_filter1d(q[start:end],.55,axis=0))
            if len(z)<13:continue
            # Recheck novelty after smoothing and after earlier accepted runs.
            mask=sample(distance,z)>CFG['min_distance_to_existing']
            for left,right in runs(mask):
                if right-left<13:continue
                piece=z[left:right];output.append(piece);audit.append(dict(seed=pp['seed'],points=len(piece),every_source_point_ridge_supported=True));cv2.polylines(seen,[np.rint(piece).astype(np.int32)],False,1,1);distance=distance_transform_edt(seen==0).astype(np.float32)
    return output,audit

def main():
    folder=ROOT/'image_verified';folder.mkdir(exist_ok=True)
    protocol=folder/'protocol.json'
    if not protocol.exists():jsave(protocol,dict(config=CFG,post_hoc_revision=True,reason='raw neural proposal crosses paper despite weak alpha majority support',parameters_not_selected_on_development=True,development_already_seen_in_original_experiment=True,independent_validation=False,no_new_neural_training=True))
    mf=read(V31/'manifest.json');views={}
    for partition in ['calibration','development']:
        records=[]
        for r in mf[partition]:
            rgb=np.asarray(Image.open(r['raw']).convert('RGB'));base=read(V31/'initial_paths'/(r['id']+'.json'))['paths'];rec=dict(id=r['id'],metrics=dict(initial=metric(rgb,r,base)),additions={})
            for name in ['classical','adapted']:
                dest=folder/name/r['id'];dest.mkdir(parents=True,exist_ok=True);file=dest/'trajectory.json'
                if file.exists():doc=read(file);added=[np.asarray(p,np.float32) for p in doc['added_paths']]
                else:
                    pp=read(ROOT/'proposals'/r['id']/(name+'.json'))['proposals'];added,audit=verified_paths(rgb,base,pp);doc=dict(paths=base+[p.tolist() for p in added],added_paths=[p.tolist() for p in added],reference_labels_used_at_inference=False,physical_z=None,ready_for_robot=False,actually_drawn=False,audit=audit);jsave(file,doc)
                rec['metrics'][name]=metric(rgb,r,doc['paths']);rec['additions'][name]=len(added);images=dict(combined=line_image(rgb,doc['paths']),new=line_image(rgb,added),overlay=rgb.copy())
                if added:cv2.polylines(images['overlay'],[np.rint(p).astype(np.int32) for p in added],False,(230,35,30),1,cv2.LINE_AA)
                for key,a in images.items():Image.fromarray(a).save(dest/(key+'.png'))
                if name=='adapted':views[r['id']]=dict(rgb=rgb,initial=line_image(rgb,base),**images)
            records.append(rec);print('V32_VERIFIED',partition,r['id'],rec['additions'],flush=True)
        s=summary(records,['initial','classical','adapted']);jsave(folder/(partition+'_evaluation.json'),dict(summary=s,records=records,post_hoc=True,partial_references=True,default_replaced=False));print('V32_VERIFIED_SUMMARY',partition,json.dumps(s),flush=True)
        if partition=='calibration':jsave(folder/'gate.json',dict(passed=gate(s['initial'],s['adapted']),default_replace_allowed=False,post_hoc_not_independent=True))
    board=Image.new('RGB',(1600,1195),'#f3f0e8');d=ImageDraw.Draw(board);d.text((7,5),'v32 图像脊线核验后：原图 / 旧候选 / 核验后新增线 / 新增红线叠加（离线，非实画）',font=FONT,fill='black')
    for i,key in enumerate(['R04','H02','X03','N11']):
        for j,(title,name) in enumerate([('原图','rgb'),('旧候选','initial'),('核验后新增线','new'),('新增线红色叠加','overlay')]):
            x=j*400;y=32+i*290;d.text((x+5,y),key+' '+title,font=SMALL,fill='black');paste(board,views[key][name],x+4,y+25,392,254)
    board.save(ROOT/'RESULT_COMPARISON_GUARDED.jpg',quality=97)

if __name__=='__main__':main()
