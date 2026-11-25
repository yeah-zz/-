# -*- coding: utf-8 -*-
"""Refine ridge localization; preserve the failed 600-step fit as evidence."""
import sys,json,shutil,time,math
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import torch,numpy as np,cv2
from PIL import Image
from torch.nn import functional as F
import human_mountain_v11 as h
from gongbi_training.stroke_train_v08 import model_new
BASE=h.ROOT;ROOT=BASE/'refinement_sparse'

def objective(out,t,m,parent,progress):
    out=out.float();p=out[:,0].sigmoid();positive=m[:,0];negative=m[:,1];valid=m[:,3];angle_mask=m[:,2];unknown=(1-positive)*(1-negative)*valid
    pos=(F.binary_cross_entropy_with_logits(out[:,0],t[:,0],reduction='none')*positive).sum()/positive.sum().clamp_min(1)
    neg=(F.softplus(out[:,0])*negative).sum()/negative.sum().clamp_min(1)
    angle=((1-F.cosine_similarity(out[:,1:3],t[:,1:3],dim=1))*angle_mask).sum()/angle_mask.sum().clamp_min(1)
    prior=(F.binary_cross_entropy_with_logits(out[:,0],parent[:,0],reduction='none')*unknown).sum()/unknown.sum().clamp_min(1)
    # An explicit weak sparsity prior, not background ground truth. Purple
    # pixels are excluded, even as off-centre samples of the ridge loss.
    sparse=(p*unknown).sum()/unknown.sum().clamp_min(1)
    b,hh,ww=p.shape;yy,xx=torch.meshgrid(torch.linspace(-1,1,hh,device=p.device),torch.linspace(-1,1,ww,device=p.device),indexing='ij');theta=.5*torch.atan2(t[:,2],t[:,1]);nx=-torch.sin(theta);ny=torch.cos(theta)
    offset=torch.stack((nx*6/(ww-1),ny*6/(hh-1)),dim=-1);grid=torch.stack((xx,yy),dim=-1)[None].expand(b,-1,-1,-1);side=[];ok=[]
    for sign in (-1,1):
        g=grid+sign*offset;side.append(F.grid_sample(p[:,None],g,align_corners=True,padding_mode='border')[:,0]);ok.append(F.grid_sample(valid[:,None],g,align_corners=True,padding_mode='zeros')[:,0]>.999)
    core=(t[:,0]>.85)*angle_mask*ok[0]*ok[1];rank=(F.relu(.18-p+(side[0]+side[1])*.5)*core).sum()/core.sum().clamp_min(1)
    loss=pos+neg+.12*angle+.03*prior+(.15+.55*progress)*sparse+.7*rank
    return loss,dict(positive=float(pos.detach()),negative=float(neg.detach()),sparsity_prior=float(sparse.detach()),local_peak_rank=float(rank.detach()),tangent=float(angle.detach()))

def run():
    if (ROOT/'training_summary.json').exists():raise FileExistsError('Completed refinement preserved')
    ROOT.mkdir(exist_ok=True);shutil.copytree(BASE/'snapshot',ROOT/'snapshot',dirs_exist_ok=True);manifest=json.loads((BASE/'manifest.json').read_text(encoding='utf-8'))
    arrays={}
    for r in manifest['records']:
        identity=r['id'];r['snapshot_image']=str(ROOT/'snapshot'/(identity+'.png'));r['snapshot_labels']=str(ROOT/'snapshot'/(identity+'.json'));z=np.load(ROOT/'snapshot'/(identity+'_targets.npz'));rgb=np.asarray(Image.open(r['snapshot_image']).convert('RGB'));arrays[identity]=(rgb,np.exp(-z['dist']**2/(2*1.2**2)).astype(np.float32),z['angles'].copy(),z['mask'].copy(),z['parent'].copy());z.close()
        if h.digest(r['snapshot_image'])!=r['image_sha256'] or h.digest(r['snapshot_labels'])!=r['label_sha256']:raise RuntimeError('snapshot changed')
    h.jsave(ROOT/'manifest.json',manifest);config=json.loads((BASE/'config.json').read_text());config.update(updates=800,initialization=str(BASE/'final.pt'),initialization_sha256=h.digest(BASE/'final.pt'),phase='ridge localization refinement after failed wide-response skeleton fit',seed=112411,weak_regularizers=['unlabeled non-purple sparsity prior, not negative labels','relative local-peak ranking at confirmed centerlines with purple-safe side samples'],normal_sparsity_weight=[.15,.70],local_peak_margin=.18,local_peak_offset_px=3)
    h.jsave(ROOT/'config.json',config);shutil.copy2(Path(__file__),ROOT/'source_snapshot.py');shutil.copy2(Path(__file__).with_name('human_mountain_v11.py'),ROOT/'data_and_sampling_source.py');shutil.copy2(Path(__file__).with_name('human_mountain_eval_v11.py'),ROOT/'evaluation_source.py')
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(112411);model=model_new().cuda();model.load_state_dict(torch.load(BASE/'final.pt',map_location='cuda',weights_only=False)['model']);opt=torch.optim.AdamW([dict(params=model.encoder.parameters(),lr=8e-6),dict(params=[p for name,p in model.named_parameters() if not name.startswith('encoder.')],lr=4e-5)],weight_decay=1e-4)
    rng=np.random.default_rng(112411);start=time.perf_counter()
    for step in range(1,801):
        batch=[h.sample_real(arrays[h.IDS[int(rng.integers(5))]],rng) for _ in range(6)];x,t,m,p=[torch.from_numpy(np.stack([a[k] for a in batch])).cuda().float() for k in range(4)];model.train()
        for layer in model.encoder.modules():
            if isinstance(layer,torch.nn.BatchNorm2d):layer.eval()
        factor=.3+.7*.5*(1+math.cos(math.pi*step/800))
        for group,base in zip(opt.param_groups,[8e-6,4e-5]):group['lr']=base*factor
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,parts=objective(model(x),t,m,p,step/800)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
        loss.backward();model.head.weight.grad[3:].zero_();model.head.bias.grad[3:].zero_();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        if not torch.isfinite(norm):raise RuntimeError('nonfinite gradient')
        opt.step()
        if step%50==0:
            row=dict(step=step,loss=float(loss.detach()),elapsed_seconds=time.perf_counter()-start,**parts)
            with (ROOT/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            print('REFINE',json.dumps(row),flush=True)
        if step%400==0:torch.save(dict(model=model.state_dict(),step=step,config=config),ROOT/(f'step_{step:04d}.pt'))
    shutil.copy2(ROOT/'step_0800.pt',ROOT/'final.pt');h.jsave(ROOT/'training_summary.json',dict(successful_updates=800,total_updates_including_initial_fit=1400,elapsed_seconds=time.perf_counter()-start,reviewed_strokes=sum(r['strokes'] for r in manifest['records']),training_images=5,independent_test=False,final_sha256=h.digest(ROOT/'final.pt'),parent_unchanged=h.digest(h.PARENT)==config['parent_sha256'],promoted_to_robot=False));print('REFINEMENT_COMPLETE',flush=True)
if __name__=='__main__':run()
