# -*- coding: utf-8 -*-
"""Narrow clean-stroke learning with explicit annotation provenance."""
import sys,json,shutil,time,math
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,cv2,torch
from PIL import Image
from clean_strokes_v12 import ROOT,jsave,digest
from clean_stroke_data_v12 import build
from human_mountain_v11 import fields,sample_real
from human_mountain_refine_v11 import objective
from gongbi_training.stroke_train_v08 import model_new,infer
from stroke_link_v12 import synthetic
PARENT=LAB/'human_mountain_v11_20261024'/'refinement_sparse'

def prepare():
    if (ROOT/'training_summary.json').exists():raise FileExistsError('Preserve completed clean run')
    build();clean=json.loads((ROOT/'clean_manifest.json').read_text(encoding='utf-8'));user=json.loads((PARENT/'manifest.json').read_text(encoding='utf-8'));records=[]
    for r in clean['records']:records.append(dict(r,group_kind='clean_draft',annotation_weight=.45))
    for r in user['records']:
        out=ROOT/'data'/('U_'+r['id']);out.mkdir(exist_ok=True);shutil.copy2(r['snapshot_image'],out/'image.png');shutil.copy2(r['snapshot_labels'],out/'labels.json')
        records.append(dict(id='U_'+r['id'],image=str(out/'image.png'),labels=str(out/'labels.json'),split='train',source_group=r['source_group'],group_kind='user_reviewed',annotation_weight=1.,image_sha256=r['image_sha256'],label_sha256=r['label_sha256'],provenance='user_reviewed_partial_mountain_mainlines'))
    # Previously inspected A06 remains a reused development comparison, never
    # promoted to a new independent test or fed into optimization.
    old=LAB/'user_review_v11_20261024';a06=json.loads((old/'labels'/'A06.json').read_text(encoding='utf-8'));out=ROOT/'data'/'U_A06';out.mkdir(exist_ok=True);shutil.copy2(old/'samples'/'A06.png',out/'image.png');shutil.copy2(old/'labels'/'A06.json',out/'labels.json');records.append(dict(id='U_A06',image=str(out/'image.png'),labels=str(out/'labels.json'),split='development',source_group='wang_ximeng_qianli_jiangshan',group_kind='user_reviewed',annotation_weight=1.,image_sha256=digest(out/'image.png'),label_sha256=digest(out/'labels.json'),provenance='user_reviewed_partial_reused_development'))
    teacher=model_new().cuda();teacher.load_state_dict(torch.load(PARENT/'final.pt',map_location='cuda',weights_only=False)['model']);teacher.eval();arrays={}
    for r in records:
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));doc=json.loads(Path(r['labels']).read_text(encoding='utf-8'));center,angle,mask,dist=fields(doc);parent=infer(teacher,rgb);path=ROOT/'data'/(r['id']+'_arrays.npz');np.savez_compressed(path,center=center,angles=angle,mask=mask,dist=dist,parent=parent);r['arrays']=str(path)
        if r['split']=='train':
            weighted=mask.copy();weighted[:3]*=r['annotation_weight'];arrays[r['id']]=(rgb,np.exp(-dist**2/(2*1.2**2)).astype(np.float32),angle,weighted,parent)
        print('PREPARED',r['id'],r['split'],len(doc['strokes']),flush=True)
    del teacher;torch.cuda.empty_cache()
    groups={}
    for r in records:groups.setdefault(r['source_group'],set()).add(r['split'])
    assert groups['artist_steps']=={'train'} and groups['wang_steps']=={'test'}
    jsave(ROOT/'manifest.json',dict(records=records,clean_test_not_used_for_selection=True,source_book_independence_fully_verified=False,style_labels_are_hints_not_expert_gold=True))
    return records,arrays

def train():
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(121224);records,arrays=prepare();pool=[]
    for seed in range(610000,610080):
        rgb,paths,p=synthetic(seed);doc=dict(width=320,height=320,sample_id='procedural',strokes=[dict(role='contour',points=a.tolist()) for a in paths],regions=[]);center,angle,mask,dist=fields(doc);mask[1]=(dist>3.5).astype(np.float32);parent=np.concatenate((p[None],angle,np.zeros((3,320,320),np.float32)))
        pool.append((rgb,np.exp(-dist**2/(2*1.2**2)).astype(np.float32),angle,mask,parent))
    model=model_new().cuda();model.load_state_dict(torch.load(PARENT/'final.pt',map_location='cuda',weights_only=False)['model']);opt=torch.optim.AdamW([dict(params=model.encoder.parameters(),lr=5e-6),dict(params=[p for name,p in model.named_parameters() if not name.startswith('encoder.')],lr=3e-5)],weight_decay=1e-4)
    clean=[r['id'] for r in records if r['split']=='train' and r['group_kind']=='clean_draft'];user=[r['id'] for r in records if r['split']=='train' and r['group_kind']=='user_reviewed'];rng=np.random.default_rng(121224);steps=1000;start=time.perf_counter()
    cfg=dict(initialization=str(PARENT/'final.pt'),initialization_sha256=digest(PARENT/'final.pt'),steps=steps,seed=121224,real_train_images=len(clean)+len(user),clean_draft_images=len(clean),user_reviewed_images=len(user),procedural_geometries=len(pool),test_images_used=False,unmarked_is_unknown=True,weak_sparsity_not_background_gold=True,purple_all_losses_excluded=True,task='centerline and tangent only; stroke linking trained separately',semantic_cunfa_classification=False,physical_pressure=False,robot_connected=False)
    jsave(ROOT/'config.json',cfg);shutil.copy2(Path(__file__),ROOT/'training_source.py')
    for step in range(1,steps+1):
        examples=[arrays[clean[int(rng.integers(len(clean)))]] for _ in range(3)]+[arrays[user[int(rng.integers(len(user)))]] for _ in range(2)]+[pool[int(rng.integers(len(pool)))]];batch=[sample_real(a,rng) for a in examples];x,t,m,p=[torch.from_numpy(np.stack([a[k] for a in batch])).cuda().float() for k in range(4)];model.train()
        for layer in model.encoder.modules():
            if isinstance(layer,torch.nn.BatchNorm2d):layer.eval()
        factor=.3+.7*.5*(1+math.cos(math.pi*step/steps))
        for group,base in zip(opt.param_groups,[5e-6,3e-5]):group['lr']=base*factor
        opt.zero_grad(set_to_none=True)
        # Keep the previous weak sparsity schedule fixed at its mid-range;
        # increasingly erasing faint strokes is not the objective of this run.
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,parts=objective(model(x),t,m,p,.45)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
        loss.backward();model.head.weight.grad[3:].zero_();model.head.bias.grad[3:].zero_();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        if not torch.isfinite(norm):raise RuntimeError('nonfinite gradient')
        opt.step()
        if step%100==0:
            row=dict(step=step,loss=float(loss.detach()),elapsed_seconds=time.perf_counter()-start,**parts)
            with (ROOT/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            print('CLEAN_TRAIN',json.dumps(row),flush=True)
        if step%500==0:torch.save(dict(model=model.state_dict(),step=step,config=cfg),ROOT/(f'step_{step:04d}.pt'))
    shutil.copy2(ROOT/'step_1000.pt',ROOT/'final.pt');jsave(ROOT/'training_summary.json',dict(successful_updates=steps,elapsed_seconds=time.perf_counter()-start,training_images=len(clean)+len(user),new_weak_draft_paths=sum(len(r.get('paths',[])) for r in records if r['split']=='train' and r['group_kind']=='clean_draft'),known_procedural_geometries=80,parent_unchanged=digest(PARENT/'final.pt')==cfg['initialization_sha256'],promoted_to_robot=False));print('CLEAN_TRAINING_COMPLETE',flush=True)
if __name__=='__main__':train()
