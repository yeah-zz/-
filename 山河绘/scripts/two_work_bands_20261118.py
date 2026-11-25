# -*- coding: utf-8 -*-
"""Two-work foreground-only supervision experiment; no robot imports."""
import sys,json,hashlib,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from gongbi_training.partial_ink_bands import rasterize_band,merge_bands
ROOT=LAB/'two_work_bands_20261118';PRIOR=LAB/'training_ink_bands_20261118';OLD=LAB/'cunfa_roles_v2_20261112';PROJECT=Path(__file__).resolve().parents[1]
FONT='C:/Windows/Fonts/msyh.ttc'
ARMS=['core_control','band_paper'];NAMES=['仅墨迹核心对照','核心＋近邻纸面']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf8'))
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')

def prepare():
    assert not (ROOT/'draft.json').exists();ROOT.mkdir(exist_ok=True);(ROOT/'views').mkdir(exist_ok=True)
    registry=read(PROJECT/'research/source_usage_registry_20261117.json')['records']
    identity='ganjing_video:1gga38p7dgh3iKX2kYUhFzF7k1ns1c'
    entry=next(r for r in registry if r['source_identity']==identity);assert entry['usage']=='known_development_object_presence_training'
    raw=LAB/'clear_rock_sources_20261117/frames/fuchun_0360.png';box=[1030,110,1240,330];im=np.array(Image.open(raw).convert('RGB').crop(box))
    native=[[184,1188,1189,1186,1191],[190,1180,1183,1178,1187],[196,1177,1179,1175,1181],
            [202,1172,1173,1170,1178],[208,1172,1173,1170,1174],[214,1170,1171,1167,1173],[220,1167,1168,1165,1170]]
    sections=[[y-110,*[x-1030 for x in xs]] for y,*xs in native]
    b=rasterize_band(im.shape[:2],sections,1)
    overlay=im.copy();overlay[b['possible']&~b['core']]=[218,187,120];overlay[b['core']]=[192,120,40];overlay[b['paper']]=[105,172,137]
    board=Image.new('RGB',(1000,610),'#f6f3eb');d=ImageDraw.Draw(board)
    for i,(a,label) in enumerate([(im,'原生山石上下文'),(overlay,'橙=轮廓核心 绿=纸面 黄=未知')]):
        p=Image.fromarray(a).resize((420,440),Image.Resampling.NEAREST);board.paste(p,(i*500+5,45));d.text((i*500+5,8),label,font=ImageFont.truetype(FONT,21),fill='#293f35')
    d.text((10,520),'第二件既有训练作品；内部纹理、树木及真实起收笔不提供标签。',font=ImageFont.truetype(FONT,23),fill='#795238');board.save(ROOT/'views/second_source_review.png')
    write('draft.json',dict(source_identity=identity,source_url='https://www.ganjingworld.com/video/1gga38p7dgh3iKX2kYUhFzF7k1ns1c',
        source=str(raw),source_sha256=sha(raw),box=box,sections=sections,role=1,object_type='rock',
        role_evidence='visible mountain silhouette in full frame, not tutorial title',named_cunfa=None,
        complete_stroke=False,expert_verified=False,source_used_before=True,split='train_development',
        assistant_reviewed=False,training_eligible=False,independent_validation=False))
    print('Second-work draft ready for source review')

def admit():
    assert not (ROOT/'protocol.json').exists();r=read(ROOT/'reviewed.json');assert r['assistant_reviewed'] and r['training_eligible']
    assert sha(r['source'])==r['source_sha256'];im=np.asarray(Image.open(r['source']).convert('RGB').crop(r['box']))
    b=rasterize_band(im.shape[:2],r['sections'],r['role']);np.savez_compressed(ROOT/'F_band.npz',image=im,**b)
    m=read(PRIOR/'dataset_manifest.json');assert sha(m['data'])==m['data_sha256']
    rows=[dict(id='L',path=m['data'],sha256=m['data_sha256'],source_work=m['source_work']),dict(id='F',path=str(ROOT/'F_band.npz'),sha256=sha(ROOT/'F_band.npz'),source_work=r['source_url'])]
    assert len({v['source_work'] for v in rows})==2
    write('dataset.json',dict(records=rows,reviewed_sha256=sha(ROOT/'reviewed.json'),unique_works=2,partial_bands=6,
        named_cunfa=False,complete_strokes=False,expert_verified=False,independent_validation=False))
    # Fixed before any new model fit or metric inspection. No validation selection.
    write('protocol.json',dict(seed=181806,steps_per_arm=80,arms=ARMS,teacher_sha256=sha(OLD/'flat_roles.pt'),
        dataset_sha256=sha(ROOT/'dataset.json'),input='static local-contrast image channels and frozen teacher foreground; no reference or prior frame at inference',
        architecture='existing 4->12->12->1 foreground residual; conditional roles frozen',
        optimizer='AdamW lr0.001 weight_decay0.0001',loss='same legacy core replay + same new positive-core loss; second arm adds0.5 nearby-paper BCE;0.02 residual regularizer',
        sampling='same deterministic alternating L/F band observations, legacy 4 training frames replayed in same order',
        evaluation='final fixed step only; known H14 regression after checkpoint freeze; no blind-test claim',
        independent_validation=False,robot_connected=False))
    print('Two-work data and fixed protocol frozen')

def train():
    assert not (ROOT/'results.json').exists() and not (ROOT/'core_control.pt').exists()
    import torch
    from torch.nn import functional as F
    from cunfa_roles_v2_train_20261112 import Net,features,conf,metrics
    from cunfa_band_checked_train_20261112 import Refiner
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    ds=read(ROOT/'dataset.json');protocol=read(ROOT/'protocol.json');assert sha(ROOT/'dataset.json')==protocol['dataset_sha256'];assert sha(OLD/'flat_roles.pt')==protocol['teacher_sha256']
    assert all(sha(r['path'])==r['sha256'] for r in ds['records'])
    teacher=Net(False).cuda();teacher.load_state_dict(torch.load(OLD/'flat_roles.pt',map_location='cuda',weights_only=False)['model']);teacher.eval();teacher.requires_grad_(False)
    legacy=read(OLD/'dataset_manifest.json')['records'];trainrows=[r for r in legacy if r['split']=='train']
    def packet(z,t):
        x=torch.from_numpy(features(z['image']))[None].cuda()
        with torch.no_grad():p=teacher.probabilities(x)
        return x,p,torch.from_numpy(t.astype('int64'))[None].cuda()
    replays=[]
    for r in trainrows:
        assert sha(r['path'])==r['sha256'];z=dict(np.load(r['path']));t=z['targets'];replays.append(packet(z,np.where(t==255,255,(t>0).astype('uint8'))))
    bands=[packet(dict(np.load(r['path'])),np.load(r['path'])['ink_targets']) for r in ds['records']]
    def loss(prob,t,negative=True):
        fg=prob[:,1:].sum(1).clamp(1e-6,1-1e-6);value=-torch.log(fg[t==1]).mean()
        if negative and (t==0).any():value=value-torch.log(1-fg[t==0]).mean()
        return value
    models={};curves={};started=time.time()
    for arm in ARMS:
        torch.manual_seed(protocol['seed']);torch.cuda.manual_seed_all(protocol['seed']);net=Refiner().cuda();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);curve=[]
        for step in range(protocol['steps_per_arm']):
            x,p,t=replays[step%len(replays)];out,res=net(x,p);l=loss(out,t)+.02*res.square().mean()
            x,p,t=bands[step%2];out,res=net(x,p);l=l+loss(out,t,False)+.02*res.square().mean()
            if arm=='band_paper':
                fg=out[:,1:].sum(1).clamp(1e-6,1-1e-6);l=l-.5*torch.log(1-fg[t==0]).mean()
            opt.zero_grad();l.backward();opt.step();curve.append(float(l.detach()))
            if (step+1)%20==0:print(arm,step+1,round(curve[-1],4),flush=True)
        net.eval();torch.save(dict(model=net.state_dict(),protocol=protocol),ROOT/(arm+'.pt'));models[arm]=net;curves[arm]=curve
    write('checkpoint_freeze.json',{a:sha(ROOT/(a+'.pt')) for a in ARMS})
    (ROOT/'predictions').mkdir(exist_ok=True);records=[]
    # Evaluation-only H14 targets are opened after model freezing.
    evalrows=ds['records']+[dict(id='H',path=str(LAB/'cunfa_band_checked_20261112/references/H_14.npz'))]
    for r in evalrows:
        z=dict(np.load(r['path']));x=torch.from_numpy(features(z['image']))[None].cuda()
        with torch.no_grad():p=teacher.probabilities(x);probs={'teacher':p}
        for a,net in models.items():
            with torch.no_grad():new,_=net(x,p)
            delta=float(((new[:,1:]/new[:,1:].sum(1,keepdim=True))-(p[:,1:]/p[:,1:].sum(1,keepdim=True))).abs().max());assert delta<1e-5;probs[a]=new
        scores={}
        t=z['ink_targets'] if r['id']!='H' else np.where(z['band_targets']==255,255,(z['band_targets']>0).astype('uint8'))
        for a,prob in probs.items():
            pred=prob[0].argmax(0).cpu().numpy().astype('uint8');np.savez_compressed(ROOT/'predictions'/(r['id']+'_'+a+'.npz'),prediction=pred,image=z['image'])
            core=t==1;paper=t==0;scores[a]=dict(ink_retained=int(((pred>0)&core).sum()),ink_pixels=int(core.sum()),paper_false_ink=int(((pred>0)&paper).sum()),paper_pixels=int(paper.sum()))
            if r['id']=='H':scores[a]['legacy_roles']=metrics(conf(z['core_targets'],pred))
        records.append(dict(id=r['id'],scores=scores,scope='training reference' if r['id']!='H' else 'known H14 regression,not independent'))
    write('results.json',dict(records=records,loss_curves=curves,duration_seconds=time.time()-started,
        conditional_role_preserved=True,new_model_trained=True,independent_validation=False,adopted=False,robot_connected=False))
    print(json.dumps(records,ensure_ascii=False),flush=True)

if __name__=='__main__':{'prepare':prepare,'admit':admit,'train':train}[sys.argv[1]]()
