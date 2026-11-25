# -*- coding: utf-8 -*-
"""User reviewed mountain-mainline calibration. No robot or drawing imports."""
import sys,os,json,time,hashlib,shutil,argparse,math
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,cv2,torch
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import distance_transform_edt
from torch.nn import functional as F
from gongbi_training.stroke_train_v08 import model_new,infer
from gongbi_training.review_server_v11 import validate_document
REVIEW=LAB/'user_review_v11_20261024'
ROOT=LAB/'human_mountain_v11_20261024'
PARENT=LAB/'rock_strokes_v08_20261023'/'best.pt'
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
SMALL=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
IDS=['A01','A02','A03','A04','A05']
# The user explicitly confirmed this finished batch in chat, including A01
# whose saved UI state remains draft. Preserve the source state and record the
# confirmation basis instead of requiring another click or editing user files.
CHAT_CONFIRMED_IDS=set(IDS)

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jsave(p,data):Path(p).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def paste(board,a,x,y,w,h):
    im=Image.fromarray(a) if isinstance(a,np.ndarray) else a.copy();im=im.convert('RGB');im.thumbnail((w,h));board.paste(im,(x+(w-im.width)//2,y+(h-im.height)//2))

def fields(doc):
    h,w=doc['height'],doc['width'];center=np.zeros((h,w),np.uint8);sx=np.zeros((h,w),np.float32);sy=sx.copy();count=sx.copy();background=np.zeros_like(center);uncertain=np.zeros_like(center)
    # Keep the user's semantic intent: all these are selected mountain main
    # lines, not 300 verified historical contour-vs-cunfa class annotations.
    for stroke in doc['strokes']:
        if stroke['role'] not in {'contour','cun','unknown'}:continue
        p=np.asarray(stroke['points'],np.float32)
        for a,b in zip(p[:-1],p[1:]):
            delta=b-a;length=float(np.linalg.norm(delta))
            if length<.1:continue
            u=np.linspace(0,1,max(2,int(math.ceil(length*2))));q=np.rint(a[None]+u[:,None]*delta).astype(int);q[:,0]=np.clip(q[:,0],0,w-1);q[:,1]=np.clip(q[:,1],0,h-1);theta=math.atan2(delta[1],delta[0]);xx,yy=q.T;center[yy,xx]=1
            np.add.at(sx,(yy,xx),math.cos(2*theta));np.add.at(sy,(yy,xx),math.sin(2*theta));np.add.at(count,(yy,xx),1)
    for region in doc['regions']:
        target=background if region['kind']=='background' else uncertain;cv2.fillPoly(target,[np.rint(region['points']).astype(np.int32)],1)
    dist,idx=distance_transform_edt(center==0,return_indices=True);a,b=idx;norm=np.sqrt(sx*sx+sy*sy);angles=np.stack((sx/np.maximum(norm,1e-6),sy/np.maximum(norm,1e-6)))[:,a,b];coherence=norm/np.maximum(count,1)
    # Exclude a 2px polygon boundary margin. Where user mainlines and the
    # remaining background interior still disagree, keep the conflict unknown.
    # No user marks are moved, deleted, or converted to the opposite class.
    background=cv2.erode(background,np.ones((5,5),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
    conflicts=(dist<=1.8)&(background>0)&(uncertain==0)
    conflict_margin=cv2.dilate(conflicts.astype(np.uint8),np.ones((3,3),np.uint8))>0
    valid=(uncertain==0)&~conflict_margin
    positive=(dist<=1.8)&valid;negative=(background>0)&valid
    # Unknown is not background, including every unmarked house/tree/fall.
    mask=np.stack((positive,negative,positive&(coherence[a,b]>.75),valid)).astype(np.float32)
    return center,angles.astype(np.float32),mask,dist.astype(np.float32)

def audit():
    ROOT.mkdir(exist_ok=True);(ROOT/'audit').mkdir(exist_ok=True)
    manifest=json.loads((REVIEW/'package_manifest.json').read_text(encoding='utf-8'));records=[];board=Image.new('RGB',(1500,550*len(IDS)),'#f5f3ec');d=ImageDraw.Draw(board)
    for i,identity in enumerate(IDS):
        r=next(x for x in manifest['samples'] if x['id']==identity);path=REVIEW/'labels'/(identity+'.json');doc=json.loads(path.read_text(encoding='utf-8'));validate_document(doc,r,manifest['package_id'])
        if doc['review_state']!='checked_partial' and identity not in CHAT_CONFIRMED_IDS:raise ValueError(identity+' not confirmed by user')
        if digest(REVIEW/r['image'])!=r['image_sha256']:raise ValueError('source changed')
        center,angles,mask,dist=fields(doc);rgb=np.asarray(Image.open(REVIEW/r['image']).convert('RGB'));overlay=rgb.copy();visible=cv2.dilate(center,np.ones((2,2),np.uint8))>0;overlay[visible]=[230,55,20]
        for region in doc['regions']:
            poly=np.rint(region['points']).astype(np.int32);cv2.polylines(overlay,[poly],True,(20,170,65) if region['kind']=='background' else (150,50,190),2)
        labels=np.full_like(rgb,255);labels[center>0]=[235,70,25];labels[mask[1]>0]=[180,230,185]
        for j,(label,a) in enumerate([('原图',rgb),('你的山体主线',labels),('你的描线叠加（红）',overlay)]):d.text((j*500+10,i*550+6),identity+' '+label,font=FONT,fill='black');paste(board,a,j*500+8,i*550+40,484,500)
        Image.fromarray(overlay).save(ROOT/'audit'/(identity+'_overlay.png'))
        records.append(dict(id=identity,image=str(REVIEW/r['image']),image_sha256=r['image_sha256'],label_path=str(path),label_sha256=digest(path),strokes=len(doc['strokes']),regions=len(doc['regions']),background_regions=sum(a['kind']=='background' for a in doc['regions']),background_pixels=int(mask[1].sum()),ignored_pixels=int((mask[3]==0).sum()),saved_review_state=doc['review_state'],confirmation_basis='user_chat_confirmed_finished_batch',
          explicit_cunfa_labels=sum(s['role']=='cun' and s['cunfa']!='unknown' for s in doc['strokes']),source_group=r['source_group'],source_id=r['source_id'],source_box=r['source_box'],annotation_intent='user_selected_mountain_mainlines',independent_test=False))
    board.save(ROOT/'audit'/'user_annotations.jpg',quality=95);jsave(ROOT/'audit'/'summary.json',dict(records=records,total_strokes=sum(r['strokes'] for r in records),total_regions=sum(r['regions'] for r in records),houses_trees_falls_unmarked='unknown_not_negative',all_one_work=True))
    print('AUDIT_COMPLETE',json.dumps(dict(strokes=sum(r['strokes'] for r in records),background_regions=sum(r['background_regions'] for r in records),total_regions=sum(r['regions'] for r in records)),ensure_ascii=False),flush=True)
    return records

def snapshot():
    if (ROOT/'training_summary.json').exists():raise FileExistsError('Completed run preserved')
    records=audit();folder=ROOT/'snapshot';folder.mkdir(exist_ok=True)
    for r in records:
        shutil.copy2(r['label_path'],folder/(r['id']+'.json'));shutil.copy2(r['image'],folder/(r['id']+'.png'));r['snapshot_image']=str(folder/(r['id']+'.png'));r['snapshot_labels']=str(folder/(r['id']+'.json'))
        if digest(r['snapshot_labels'])!=r['label_sha256'] or digest(r['snapshot_image'])!=r['image_sha256']:raise RuntimeError('Review changed during snapshot; rerun preparation')
    jsave(ROOT/'manifest.json',dict(records=records,source_group_count=1,independent_test=False,unmarked='unknown',semantic_classes_not_trained=True,physical_width_not_trained=True,training_target='selected mountain mainline center + undirected tangent'))
    shutil.copy2(Path(__file__),ROOT/'source_snapshot.py');return records

def real_arrays(r,teacher):
    rgb=np.asarray(Image.open(r['snapshot_image']).convert('RGB'));doc=json.loads(Path(r['snapshot_labels']).read_text(encoding='utf-8'));center,angles,mask,dist=fields(doc)
    cache=ROOT/'preflight'/(r['id']+'_parent.npz')
    if cache.exists():
        with np.load(cache) as z:
            if str(z['image_sha256'])!=r['image_sha256'] or str(z['parent_sha256'])!=digest(PARENT):raise RuntimeError('stale parent cache')
            old=z['parent'].copy()
    else:old=infer(teacher,rgb)
    np.savez_compressed(ROOT/'snapshot'/(r['id']+'_targets.npz'),center=center,angles=angles,mask=mask,dist=dist,parent=old)
    return rgb,np.exp(-dist**2/(2*1.2**2)).astype(np.float32),angles,mask,old

def sample_real(a,rng,size=256):
    rgb,center,angles,mask,old=a;h,w=rgb.shape[:2];scale=float(rng.choice([.85,1.,1.2]));s=round(size/scale)
    chosen=mask[1] if mask[1].sum()>0 and rng.random()<.3 else ((center>.9)&(mask[3]>0))
    yy,xx=np.nonzero(chosen);k=int(rng.integers(len(xx)));x=int(np.clip(xx[k]-rng.uniform(.2,.8)*s,0,w-s));y=int(np.clip(yy[k]-rng.uniform(.2,.8)*s,0,h-s));crop=lambda a:cv2.resize(a[y:y+s,x:x+s],(size,size))
    im=crop(rgb).astype(np.float32)/255.;t=np.stack([crop(center.astype(np.float32)),crop(angles[0]),crop(angles[1])]);m=np.stack([crop(z) for z in mask]);p=np.stack([crop(z) for z in old[:3]])
    if rng.random()<.5:im=im[:,::-1].copy();t=t[:,:,::-1].copy();t[2]*=-1;m=m[:,:,::-1].copy();p=p[:,:,::-1].copy();p[2]*=-1
    im=np.clip(im*rng.uniform(.94,1.06)+rng.uniform(-.015,.015),0,1)
    return im.transpose(2,0,1).copy(),t,m,p

def objective(logits,target,mask,parent):
    out=logits.float();p=out[:,0].sigmoid();positive=mask[:,0];negative=mask[:,1];direction=mask[:,2]
    pos=(F.binary_cross_entropy_with_logits(out[:,0],target[:,0],reduction='none')*positive).sum()/positive.sum().clamp_min(1)
    neg=(F.softplus(out[:,0])*negative).sum()/negative.sum().clamp_min(1)
    angle=((1-F.cosine_similarity(out[:,1:3],target[:,1:3],dim=1))*direction).sum()/direction.sum().clamp_min(1)
    # Soft parent regularization is explicitly a model prior, never new user
    # background labels. Unmarked architectural/vegetation areas remain unknown.
    unknown=(1-positive)*(1-negative)*mask[:,3];prior=(F.binary_cross_entropy_with_logits(out[:,0],parent[:,0],reduction='none')*unknown).sum()/unknown.sum().clamp_min(1)
    return pos+neg+.12*angle+.15*prior,dict(positive=float(pos.detach()),negative=float(neg.detach()),tangent=float(angle.detach()),parent_prior=float(prior.detach()))

def train():
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(112410);torch.backends.cudnn.benchmark=False
    records=snapshot()
    if not sum(r['background_regions'] for r in records):raise RuntimeError('Awaiting the user-confirmed A04 background regions; training not started')
    teacher=model_new().cuda();teacher.load_state_dict(torch.load(PARENT,map_location='cuda',weights_only=False)['model']);teacher.eval();arrays={r['id']:real_arrays(r,teacher) for r in records}
    model=model_new().cuda();model.load_state_dict(teacher.state_dict());del teacher;torch.cuda.empty_cache()
    # There is no reliable independent validation set in this one-work batch.
    # Use a fixed short schedule, retain all checkpoints, report training fit.
    steps=600;opt=torch.optim.AdamW([dict(params=model.encoder.parameters(),lr=8e-6),dict(params=[p for name,p in model.named_parameters() if not name.startswith('encoder.')],lr=4e-5)],weight_decay=1e-4)
    rng=np.random.default_rng(112410);start=time.perf_counter();config=dict(parent=str(PARENT),parent_sha256=digest(PARENT),updates=steps,seed=112410,source_group_count=1,annotation_strokes=sum(r['strokes'] for r in records),human_negative_regions=sum(r['background_regions'] for r in records),
      unknown_pixels_not_background=True,purple_regions_excluded_from_all_losses=True,background_boundary_margin_pixels=2,positive_background_conflicts='ignore_with_1px_margin',parent_regularization='soft prior on unmarked non-purple pixels, not user labels',task='RGB -> selected mountain mainline center and undirected tangent',width_role_heads_not_trained=True,encoder_batchnorm_statistics_frozen=True,independent_validation=False,selection='fixed final step; not claimed best on independent validation',robot_connected=False)
    jsave(ROOT/'config.json',config)
    for step in range(1,steps+1):
        batch=[sample_real(arrays[IDS[int(rng.integers(len(IDS)))]],rng) for _ in range(6)];x,t,m,p=[torch.from_numpy(np.stack([a[k] for a in batch])).cuda().float() for k in range(4)];factor=.3+.7*.5*(1+math.cos(math.pi*step/steps))
        for group,base in zip(opt.param_groups,[8e-6,4e-5]):group['lr']=base*factor
        model.train()
        for layer in model.encoder.modules():
            if isinstance(layer,torch.nn.BatchNorm2d):layer.eval()
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(x);loss,parts=objective(logits,t,m,p)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
        loss.backward()
        # Auxiliary width/role outputs remain numerically inherited; their
        # shared trunk may change, so they are not used or evaluated here.
        model.head.weight.grad[3:].zero_();model.head.bias.grad[3:].zero_();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        if not torch.isfinite(norm):raise RuntimeError('nonfinite gradient')
        opt.step()
        if step%25==0:
            row=dict(step=step,loss=float(loss.detach()),elapsed_seconds=time.perf_counter()-start,**parts)
            with (ROOT/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            print('TRAIN',json.dumps(row),flush=True)
        if step%200==0:torch.save(dict(model=model.state_dict(),step=step,config=config),ROOT/(f'step_{step:04d}.pt'))
    shutil.copy2(ROOT/f'step_{steps:04d}.pt',ROOT/'final.pt');jsave(ROOT/'training_summary.json',dict(successful_updates=steps,elapsed_seconds=time.perf_counter()-start,final_sha256=digest(ROOT/'final.pt'),independent_test=False,training_images=len(records),reviewed_strokes=sum(r['strokes'] for r in records),parent_unchanged=digest(PARENT)==config['parent_sha256'],promoted_to_robot=False))
    print('TRAINING_COMPLETE',flush=True)

def preflight():
    from human_mountain_eval_v11 import extract,metrics,draw_result
    torch.set_num_threads(4);cv2.setNumThreads(2);rows=audit();folder=ROOT/'preflight';folder.mkdir(exist_ok=True);model=model_new().cuda();model.load_state_dict(torch.load(PARENT,map_location='cuda',weights_only=False)['model']);model.eval();result=[]
    for r in rows:
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));doc=json.loads(Path(r['label_path']).read_text(encoding='utf-8'));p=infer(model,rgb);np.savez_compressed(folder/(r['id']+'_parent.npz'),parent=p,image_sha256=r['image_sha256'],parent_sha256=digest(PARENT));sk,paths=extract(p);line,overlay=draw_result(rgb,sk);Image.fromarray(line).save(folder/(r['id']+'_parent.png'));_,_,mask,_=fields(doc);v=metrics(doc,p,sk,mask);v.pop('per_user_stroke');result.append(dict(id=r['id'],**v));print('PREFLIGHT',r['id'],round(v['partial_reference_coverage_3px'],3),len(paths),flush=True)
    jsave(folder/'baseline_partial_fit.json',result)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['audit','preflight','train']);args=parser.parse_args();{'audit':audit,'preflight':preflight,'train':train}[args.mode]()
