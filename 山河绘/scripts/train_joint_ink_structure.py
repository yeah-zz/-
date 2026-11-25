"""Masked multitask GPU training; independent checkpoint, no robot interface."""
import sys
sys.dont_write_bytecode=True
from joint_ink_structure_data import ROOT,R2,OFFSETS
from rock_v15_common import read,jsave,digest
from train_simple_landscape_specialist_r2 import Specialist,prior
import numpy as np,cv2,torch,time,json,functools,math
from PIL import Image
from torch import nn
from torch.nn import functional as F
INIT=R2/'specialist.pt'

class JointModel(Specialist):
    def __init__(self):
        super().__init__();self.relation_head=nn.Sequential(nn.Conv2d(16,24,3,padding=1),nn.SiLU(),nn.Conv2d(24,8,1))
    def forward(self,x):
        a=self.a(x);b=self.b(F.avg_pool2d(a,2));c=self.c(F.avg_pool2d(b,2));d=self.d(torch.cat((F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b),1));e=self.e(torch.cat((F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a),1))
        return torch.cat([self.head(e),self.role_head(e),self.relation_head(e)],1)

def initialize():
    torch.manual_seed(291021);model=JointModel().cuda();state=torch.load(INIT,map_location='cpu',weights_only=False)['model'];missing=model.load_state_dict(state,strict=False)
    assert all(k.startswith('relation_head.') for k in missing.missing_keys) and not missing.unexpected_keys
    return model

@functools.lru_cache(maxsize=24)
def arrays(image,data):
    rgb=np.asarray(Image.open(image).convert('RGB'));z=np.load(data)
    return rgb,{k:z[k].astype(np.float32 if k!='role' else np.int64) for k in ['target','weight','role','relation','relation_weight']}

def crop(r,rng,augment=True):
    rgb,z=arrays(r['image'],r['data']);h,w=rgb.shape[:2]
    x=int(rng.integers(max(1,w-255)));y=int(rng.integers(max(1,h-255)))
    rgb=rgb[y:y+256,x:x+256].copy();t=z['target'][:,y:y+256,x:x+256].copy();m=z['weight'][:,y:y+256,x:x+256].copy();role=z['role'][y:y+256,x:x+256].copy();rel=z['relation'][:,y:y+256,x:x+256].copy();rm=z['relation_weight'][:,y:y+256,x:x+256].copy()
    rm[:,:4]=0;rm[:,-4:]=0;rm[:,:,:4]=0;rm[:,:,-4:]=0
    if augment:
        paper=np.maximum(rgb.max((0,1)),240).astype(np.float32);alpha=np.clip((paper-rgb.astype(np.float32))/255,0,1);gamma=float(rng.uniform(.8,1.25));rgb=np.clip(paper-255*np.power(alpha,gamma)+rng.normal(0,.25,rgb.shape[:2])[:,:,None],0,255).astype(np.uint8)
    # No flips: relation offsets would need remapping, so do not silently alter labels.
    return prior.photometry(rgb)[0],t,m,role,rel,rm

def mean_mask(value,mask):return (value*mask).sum()/mask.sum().clamp_min(1)

def loss_fn(o,t,m,role,rel,rm):
    o=o.float();center=t[:,0];p=o[:,0].sigmoid();b=F.binary_cross_entropy_with_logits(o[:,0],center,reduction='none')
    pos=(center>.05)*m[:,0];neg=(center<.01)*m[:,0]
    lc=mean_mask(b,pos)+.6*mean_mask(b,neg)
    dice=1-(2*(p*center*m[:,0]).sum()+1)/(((p+center)*m[:,0]).sum()+1);lc+=dice
    li=mean_mask(F.binary_cross_entropy_with_logits(o[:,1],t[:,1],reduction='none'),m[:,1]*(1+2*t[:,1]))
    ld=mean_mask(1-F.cosine_similarity(o[:,2:4],t[:,2:4],dim=1),m[:,2])
    lw=mean_mask(F.smooth_l1_loss(o[:,4].sigmoid(),t[:,4],reduction='none'),m[:,3])
    rp=F.binary_cross_entropy_with_logits(o[:,10:18],rel,reduction='none');positive=(rel>.5)*rm;negative=(rel<.5)*rm
    la=mean_mask(rp,positive)+mean_mask(rp,negative)
    known=role!=255
    lr=F.cross_entropy(o[:,5:10],role,ignore_index=255,reduction='sum')/known.sum().clamp_min(1)
    total=lc+.7*li+.12*ld+.15*lw+.12*lr+.6*la
    return total,dict(center=lc,ink=li,direction=ld,width=lw,role=lr,relation=la)

@torch.no_grad()
def validate(model,rows):
    model.eval();grouped={};summaries={}
    for kind in ['synthetic','actual']:
        chosen=[r for r in rows if (r['kind']=='actual_partial')==(kind=='actual')]
        logs=[]
        for start in range(0,len(chosen),4):
            batch=[crop(r,np.random.default_rng(291029+i),False) for i,r in enumerate(chosen[start:start+4])]
            args=[torch.from_numpy(np.stack([a[k] for a in batch])).cuda() for k in range(6)]
            with torch.autocast('cuda',dtype=torch.bfloat16):value,parts=loss_fn(model(args[0]),*args[1:])
            logs.append(dict(loss=float(value),**{k:float(v) for k,v in parts.items()}))
        summaries[kind]={k:float(np.mean([r[k] for r in logs])) for k in logs[0]}
    return summaries

def train():
    if (ROOT/'joint.pt').exists():raise FileExistsError('Preserve completed joint checkpoint')
    torch.set_num_threads(4);cv2.setNumThreads(2);rng=np.random.default_rng(291029);manifest=read(ROOT/'manifest.json')
    rows=[r for r in manifest['records'] if r['split']=='train'];cal=[r for r in manifest['records'] if r['split']=='calibration'];pools=[[r for r in rows if r['kind']==kind] for kind in ['existing_declared','hard_declared','actual_partial']]
    assert len(pools[-1])==11 and len([r for r in cal if r['kind']=='actual_partial'])==7
    model=initialize();base=[p for n,p in model.named_parameters() if not n.startswith('relation_head.')];rates=[2e-5,2e-4]
    optimizer=torch.optim.AdamW([dict(params=base,lr=rates[0]),dict(params=model.relation_head.parameters(),lr=rates[1])],weight_decay=1e-4)
    steps=900;config=dict(steps=steps,batch=6,mix=['existing_declared']*2+['hard_declared']*2+['actual_partial']*2,init=str(INIT),init_sha256=digest(INIT),manifest_sha256=digest(ROOT/'manifest.json'),new_dependencies=False,robot_connected=False,offsets=OFFSETS,gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,unknown_weight_is_zero=True,test_labels_used=False,old_full_image_extractions_as_labels=False)
    jsave(ROOT/'training_config.json',config);baseline=validate(model,cal);jsave(ROOT/'calibration_initial.json',baseline)
    start=time.perf_counter();best=float('inf');best_step=0;best_state=None;logs=[];torch.cuda.reset_peak_memory_stats()
    for step in range(1,steps+1):
        batch=[]
        for pool in pools:
            for _ in range(2):batch.append(crop(pool[int(rng.integers(len(pool)))],rng))
        args=[torch.from_numpy(np.stack([a[k] for a in batch])).cuda() for k in range(6)]
        model.train();optimizer.zero_grad(set_to_none=True);factor=.3+.7*.5*(1+math.cos(math.pi*step/steps))
        for group,rate in zip(optimizer.param_groups,rates):group['lr']=rate*factor
        with torch.autocast('cuda',dtype=torch.bfloat16):value,parts=loss_fn(model(args[0]),*args[1:])
        if not torch.isfinite(value):raise RuntimeError('Nonfinite training loss')
        value.backward();grad=nn.utils.clip_grad_norm_(model.parameters(),5)
        if not torch.isfinite(grad):raise RuntimeError('Nonfinite gradient')
        optimizer.step()
        if step%100==0:
            row=dict(step=step,loss=float(value.detach()),seconds=time.perf_counter()-start)
            if step%150==0 or step==steps:
                summary=validate(model,cal);score=summary['synthetic']['loss']+.5*summary['actual']['loss'];row['calibration']=summary;row['selection_score']=score
                if score<best:best=score;best_step=step;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            logs.append(row)
            with (ROOT/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            print('JOINT_TRAIN',json.dumps(row),flush=True)
    torch.cuda.synchronize();torch.save(dict(model=best_state,config=config,retained_step=best_step,completed_updates=steps),ROOT/'joint.pt')
    torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},config=config,completed_updates=steps),ROOT/'last.pt')
    jsave(ROOT/'training_summary.json',dict(completed_updates=steps,retained_step=best_step,best_calibration_score=best,gpu=torch.cuda.get_device_name(),tensor_device=str(next(model.parameters()).device),elapsed_seconds=time.perf_counter()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),checkpoint_sha256=digest(ROOT/'joint.pt'),init_unchanged=digest(INIT)==config['init_sha256'],logs=logs))
    print('JOINT_TRAINED',best_step,str(ROOT/'joint.pt'),flush=True)
if __name__=='__main__':train()
