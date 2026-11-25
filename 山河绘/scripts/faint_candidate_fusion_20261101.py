# -*- coding: utf-8 -*-
from evaluate_faint_joint_20261101 import *
from faint_context_routing_20261101 import mountain_context
from refine_native_fine_faint_20261101 import same_band
CFG_FUSION=dict(minimum_length=24.,minimum_mean_score=.65,minimum_direction_alignment=.55,maximum_mean_boundary=.35,matching_distance=4.,matching_fraction=.8,covered_fraction=.97,maximum_surviving_near_fraction=.12)

def fuse(rgb,base,protect,rows):
    paths,_=graph.pixel_chains(base);gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255.;replaced=set();accepted=[];pending=[];distance=distance_transform_edt(~base)
    for r in sorted(rows,key=lambda r:-r['length']):
        p=np.asarray(r['points'],np.float32);q=np.rint(p).astype(int);x,y=q[:,0],q[:,1];reason=None
        if r['length']<CFG_FUSION['minimum_length'] or r['mean_score']<CFG_FUSION['minimum_mean_score'] or r['direction_alignment']<CFG_FUSION['minimum_direction_alignment'] or r['mean_color_boundary_probability']>CFG_FUSION['maximum_mean_boundary']:reason='confidence_length_or_direction_pending'
        if protect[y,x].any():reason='protected_tree_detail'
        if np.linalg.norm(p[0]-p[-1])<3:reason='closed_candidate_identity_uncertain'
        if reason:pending.append(dict(r,reason=reason));continue
        nearby=distance[y,x]<CFG_FUSION['matching_distance']
        if nearby.mean()>=CFG_FUSION['covered_fraction']:pending.append(dict(r,reason='already_covered'));continue
        tree=cKDTree(p);remove=[]
        for i,old in enumerate(paths):
            if i in replaced or np.any(old.max(0)<p.min(0)-5) or np.any(old.min(0)>p.max(0)+5):continue
            ij=np.rint(old).astype(int)
            if protect[ij[:,1],ij[:,0]].any():continue
            d,idx=tree.query(old);covered=(d<CFG_FUSION['matching_distance'])&same_band(gray,old,p[idx])
            if covered.mean()<CFG_FUSION['matching_fraction']:continue
            ot=np.gradient(old,axis=0);ot/=np.maximum(np.linalg.norm(ot,axis=1,keepdims=True),1e-6);pt=np.gradient(p,axis=0);pt/=np.maximum(np.linalg.norm(pt,axis=1,keepdims=True),1e-6)
            if np.median(np.abs((ot*pt[idx]).sum(1))[covered])<.8:continue
            remove.append(i)
        # With uncertain overlap, defer the whole candidate instead of cutting it.
        remaining=[old for i,old in enumerate(paths) if i not in replaced and i not in remove]
        if remaining:
            near=cKDTree(np.concatenate(remaining)).query(p)[0]<CFG_FUSION['matching_distance']
            if near.mean()>CFG_FUSION['maximum_surviving_near_fraction']:pending.append(dict(r,reason='unresolved_overlap_with_remaining_paths'));continue
        if accepted and (cKDTree(np.concatenate([np.asarray(a['points']) for a in accepted])).query(p)[0]<2).mean()>.15:pending.append(dict(r,reason='overlap_with_new_candidate'));continue
        replaced.update(remove);accepted.append(dict(r,replaced_old_path_ids=remove));current=graph.raster(base.shape,[old for i,old in enumerate(paths) if i not in replaced]+[np.asarray(a['points']) for a in accepted]);distance=distance_transform_edt(~current)
    retained=[old for i,old in enumerate(paths) if i not in replaced];isolated=base&~graph.raster(base.shape,paths);result=graph.raster(base.shape,retained+[np.asarray(a['points']) for a in accepted])|isolated;result[protect]=base[protect]
    return result,dict(accepted=accepted,pending=pending,replaced_old_path_ids=sorted(replaced),old_paths=paths)

def local():
    selected=read(ROOT/'context_calibration.json')['selected']['config'];records=read(ROOT/'audited_reference_manifest.json')['records'];folder=ROOT/'local_repair';folder.mkdir(exist_ok=True);cache={};font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16);summary=[]
    for r in records:
        ident=r['image']
        if ident not in cache:
            rgb=np.asarray(Image.open(BASE/'originals'/(ident+'.png')).convert('RGB'));cache[ident]=(mountain_context(rgb),np.asarray(Image.open(PRIOR/'outputs_r3'/ident/'lines.png'))<128,np.asarray(Image.open(PRIOR/'outputs_r3'/ident/'fan.png'))<128)
        x,y,xx,yy=r['box'];area,baseline,protected=[z[y:yy,x:xx] for z in cache[ident]];rgb=np.asarray(Image.open(r['path']).convert('RGB'));o=np.load(ROOT/'local_predictions_r2'/r['id']/'fields.npz')['fields'];model_mask,rows,score=trace(rgb,o,selected,area&~protected);result,decisions=fuse(rgb,baseline,protected,rows);out=folder/r['id'];out.mkdir(exist_ok=True)
        for name,m in [('previous',baseline),('model_only',model_mask),('repaired',result),('protected',protected),('context',area)]:Image.fromarray(np.where(m,0,255).astype(np.uint8)).save(out/(name+'.png'))
        change=np.repeat(np.where(baseline,0,255).astype(np.uint8)[:,:,None],3,2);change[result&~baseline]=[20,135,210];change[baseline&~result]=[215,70,65];Image.fromarray(change).save(out/'changes.png');jsave(out/'decisions.json',{k:v for k,v in decisions.items() if k!='old_paths'});stats=dict(id=r['id'],image=ident,split=r['split'],previous=partial_metrics(baseline,r),model_only=partial_metrics(model_mask,r),repaired=partial_metrics(result,r),added_pixels=int((result&~baseline).sum()),removed_pixels=int((baseline&~result).sum()),accepted_whole_candidates=len(decisions['accepted']),replaced_whole_fragments=len(decisions['replaced_old_path_ids']),protected_exact=bool(np.array_equal(result[protected],baseline[protected])));jsave(out/'statistics.json',stats);summary.append(stats);board=Image.new('RGB',(1920,430),'#f3f1eb');d=ImageDraw.Draw(board)
        for j,(im,title) in enumerate([(Image.fromarray(rgb),'原图'),(Image.fromarray(np.where(baseline,0,255).astype(np.uint8)),'上轮结果'),(Image.fromarray(np.where(model_mask,0,255).astype(np.uint8)),'模型候选＋山体分区'),(Image.fromarray(np.where(result,0,255).astype(np.uint8)),'完整候选修正'),(Image.fromarray(change),'蓝新增 / 红替换')]):d.text((j*384+3,4),r['id']+' '+title,font=font,fill='black');board.paste(im.convert('RGB').resize((384,384),Image.Resampling.NEAREST),(j*384,40))
        board.save(out/'comparison.jpg',quality=97)
    jsave(ROOT/'local_repair_summary.json',dict(records=summary,fusion_config=CFG_FUSION,reference_coverage_is_biased_partial_not_accuracy=True));print('FAINT_LOCAL_REPAIR',sum(r['accepted_whole_candidates'] for r in summary),sum(r['added_pixels'] for r in summary),flush=True)
if __name__=='__main__':local()
