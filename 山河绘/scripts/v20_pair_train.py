from structure_train_v20 import *
import structure_train_v20 as base
BASE=ROOT
CURR=ROOT/'paired_curriculum'


class RoleNet(v16.RibbonNet):
    def __init__(self):super().__init__();self.head=nn.Conv2d(16,6,1)
    def initialize(self,file):
        state=torch.load(file,map_location='cpu',weights_only=False)['model'];w=state.pop('head.weight');b=state.pop('head.bias');self.load_state_dict(state,strict=False)
        with torch.no_grad():self.head.weight[:5]=w;self.head.bias[:5]=b;self.head.bias[5]=-1


@torch.no_grad()
def predict_role(model,rgb):
    tile=256;stride=192;h,w=rgb.shape[:2];pad=cv2.copyMakeBorder(rgb,20,max(20,tile-h-20),20,max(20,tile-w-20),cv2.BORDER_REFLECT_101);ph,pw=pad.shape[:2];starts=lambda n:sorted(set(list(range(0,n-tile+1,stride))+[n-tile]));loc=[(y,x) for y in starts(ph) for x in starts(pw)];window=np.maximum(np.outer(np.hanning(tile),np.hanning(tile)),.05).astype(np.float32);total=np.zeros((6,ph,pw),np.float32);den=np.zeros((ph,pw),np.float32);features=v16.photometry(pad)[0]
    for start in range(0,len(loc),8):
        group=loc[start:start+8];x=np.stack([features[:,y:y+tile,x:x+tile] for y,x in group]);out=model(torch.from_numpy(x).cuda()).float();out[:,[0,1,4,5]]=out[:,[0,1,4,5]].sigmoid();out[:,2:4]=F.normalize(out[:,2:4],dim=1)
        for (y,x),p in zip(group,out.cpu().numpy()):total[:,y:y+tile,x:x+tile]+=p*window;den[y:y+tile,x:x+tile]+=window
    return (total/den)[:,20:20+h,20:20+w]


def prepare():
    torch.set_num_threads(4);cv2.setNumThreads(2);manifest=read(BASE/'training_manifest.json');paired=read(CURR/'paired_targets_manifest.json');folder=CURR/'training_data';folder.mkdir(exist_ok=True);teacher=base.load_model('control');rows=[]
    for r in manifest['records']:
        z=np.load(r['data']);t=np.concatenate((z['target'],np.zeros((2,*z['rgb'].shape[:2]),np.float32)));t[5]=v16.predict(teacher,z['rgb'])[0];file=folder/(r['id']+'.npz');np.savez_compressed(file,rgb=z['rgb'],target=t,mask=z['mask']);rows.append(dict(id=r['id'],data=str(file),data_sha256=digest(file),kind='review_replay',source_image=r['image'],source_sha256=r['image_sha256']))
    for i,r in enumerate(paired['records']):
        if r['split']!='train':continue
        rows.append(dict(id='B'+str(i).zfill(2),data=r['data'],data_sha256=r['data_sha256'],kind='temporal_pair',group=r['group'],pair_id=r['id'],source_image=r['after_image'],source_sha256=r['after_sha256']))
    assert set(r.get('group') for r in rows if r['kind']=='temporal_pair').isdisjoint(paired['heldout_work_groups'])
    jsave(CURR/'training_manifest.json',dict(records=rows,init_weights=str(BASE/'control'/'final.pt'),init_sha256=digest(BASE/'control'/'final.pt'),training_ids=[r['id'] for r in rows],calibration_ids=['N07','N08'],heldout_work_groups=paired['heldout_work_groups'],role_semantics='candidate_added_internal_texture_vs_contour; not_named_cunfa',derived_temporal_supervision_not_expert=True,old_teacher_updated_to_current_control=True));print('PAIR_TRAINING_PREPARED',len(rows),'temporal',sum(r['kind']=='temporal_pair' for r in rows),flush=True)


def train(variant):
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(203510);rng=np.random.default_rng(203510);oldroot=base.ROOT;base.ROOT=CURR
    try:manifest,data=base.pool_data()
    finally:base.ROOT=oldroot
    lookup={r['id']:r for r in manifest['records']};replay=[r for r in data if lookup[r['id']]['kind']=='review_replay'];pairs=[r for r in data if lookup[r['id']]['kind']=='temporal_pair'];folder=CURR/variant;folder.mkdir(exist_ok=True)
    if (folder/'final.pt').exists():raise FileExistsError('Preserve checkpoint')
    model=RoleNet();model.initialize(manifest['init_weights']);model=model.cuda();opt=torch.optim.AdamW([dict(params=list(model.a.parameters())+list(model.b.parameters())+list(model.c.parameters()),lr=.00001),dict(params=list(model.d.parameters())+list(model.e.parameters())+list(model.head.parameters()),lr=.00008)],weight_decay=.0001);steps=1000;start=time.perf_counter();usage=dict(review=0,temporal=0);config=dict(variant=variant,steps=steps,seed=203510,init_sha256=manifest['init_sha256'],heads=['center','ink','cos2','sin2','width_div48','internal_texture_candidate'],role_targets_are_temporal_weak_not_expert=True,role_head_not_used_to_change_trajectory=True,temporal_groups_used=sorted({r.get('group') for r in manifest['records'] if r['kind']=='temporal_pair'}) if variant=='paired' else [],calibration_ids=['N07','N08'],robot_connected=False);jsave(folder/'config.json',config);shutil.copy2(Path(__file__),folder/'source_at_start.py')
    for step in range(1,steps+1):
        batch=[]
        for _ in range(6):
            if variant=='paired' and rng.random()<.35:group=pairs;usage['temporal']+=1
            else:group=[a for a in replay if a['id'].startswith('R')] if rng.random()<.65 else [a for a in replay if not a['id'].startswith('R')];usage['review']+=1
            a=group[int(rng.integers(len(group)))];batch.append(base.crop_features(a['variants'][int(rng.integers(3))],rng))
        x,t,m=[torch.from_numpy(np.stack([a[k] for a in batch])).cuda() for k in range(3)];factor=.4+.6*.5*(1+math.cos(math.pi*step/steps))
        for g,lr in zip(opt.param_groups,[.00001,.00008]):g['lr']=lr*factor
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            o=model(x);loss=train17.loss_fn(o[:,:5],t[:,:5],m);unknown=(m[:,0]<.1)&((t[:,5]>.9)|(t[:,5]<.1))&(t[:,6]<.5);distill=(F.binary_cross_entropy_with_logits(o[:,0].float(),t[:,5],reduction='none')*unknown).sum()/unknown.sum().clamp_min(1);role=F.binary_cross_entropy_with_logits(o[:,5].float(),t[:,11],reduction='none');positive=(t[:,11]>.5)*t[:,12]*(1-t[:,6]);negative=(t[:,11]<=.5)*t[:,12]*(1-t[:,6]);role=(role*positive).sum()/positive.sum().clamp_min(1)+(role*negative).sum()/negative.sum().clamp_min(1);loss=loss+.2*distill+.15*role
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss.backward();g=torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        if not torch.isfinite(g):raise RuntimeError('Nonfinite gradient')
        opt.step()
        if step%100==0:
            row=dict(step=step,loss=float(loss.detach()),role_loss=float(role.detach()),elapsed_seconds=time.perf_counter()-start)
            with (folder/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            print('V20_PAIRED_TRAIN',variant,json.dumps(row),flush=True)
    torch.save(dict(model=model.state_dict(),config=config),folder/'final.pt');jsave(folder/'training_summary.json',dict(successful_updates=steps,usage=usage,elapsed_seconds=time.perf_counter()-start,robot_connected=False));print('V20_PAIRED_TRAINED',variant,flush=True)


def load_role(variant):
    m=RoleNet().cuda();m.load_state_dict(torch.load(CURR/variant/'final.pt',map_location='cuda',weights_only=False)['model']);return m.eval()


def evaluate(variant,mode):
    root=prior_eval.ROOT;load=prior_eval.load;predict=prior_eval.v16.predict;prior_eval.ROOT=CURR;prior_eval.load=load_role;prior_eval.v16.predict=predict_role
    try:prior_eval.calibrate(variant) if mode=='calibrate' else prior_eval.evaluate(variant)
    finally:prior_eval.ROOT=root;prior_eval.load=load;prior_eval.v16.predict=predict

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','train','calibrate','evaluate']);p.add_argument('variant',nargs='?',choices=['replay','paired']);a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='train':train(a.variant)
    else:evaluate(a.variant,a.mode)
