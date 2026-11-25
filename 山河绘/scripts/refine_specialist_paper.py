# -*- coding: utf-8 -*-
"""Small specificity correction with explicitly blank old-paper negative ROIs.

The native images are reused development checks after this correction; no new
blind-test claim is made. Parent checkpoints and strict evaluation stay intact.
"""
from train_simple_landscape_specialist_r2 import ROOT,Specialist,loss_fn,crop,validation,prior
from rock_v15_common import LAB,read,jsave,digest
from predict_simple_landscape_specialist import choose,write_lines,baseline
from PIL import Image,ImageDraw,ImageFont
import numpy as np,cv2,torch,json,time
from torch.nn import functional as F
OUT=ROOT/'paper_refinement'
REGIONS=[('01_rocks',(680,10,1230,410),'train'),('02_pine',(720,10,1230,390),'train'),('03_pavilion',(20,10,640,300),'calibration')]

def prepare():
    rows=[]
    for sid,box,split in REGIONS:
        source=LAB/'generated_simple_landscapes_20261028'/'originals'/(sid+'.png');im=Image.open(source).convert('RGB');image=im.crop(box);dest=OUT/(sid+'_blank.png');dest.parent.mkdir(exist_ok=True,parents=True);image.save(dest)
        rows.append(dict(id=sid,source=str(source),source_sha256=digest(source),box=list(box),split=split,image=str(dest),label='blank paper only',annotation_kind='assistant-selected visibly empty ROI, not expert stroke annotation'))
    jsave(OUT/'negative_paper_manifest.json',dict(records=rows,new_final_holdouts_used_for_training=False,unknown_painting_regions_not_labeled_background=True,reused_native_images_are_development_after_this_refinement=True))
    return rows

@torch.no_grad()
def negative_score(model,rgb):
    crops=[rgb[:256,:256],rgb[-256:,-256:]];model.eval();rates=[]
    for patch in crops:
        x=torch.from_numpy(prior.photometry(patch)[0][None]).cuda();o=model(x).float();p=o[:,0].sigmoid();rates.append(float((p>.35).float().mean()))
    return float(np.mean(rates))

def train():
    if (OUT/'specialist.pt').exists():raise FileExistsError('Preserve paper refinement')
    rows=prepare();negative=[np.asarray(Image.open(r['image']).convert('RGB')) for r in rows if r['split']=='train'];negval=np.asarray(Image.open(rows[-1]['image']).convert('RGB'))
    manifest=read(ROOT/'paired_manifest.json');positive=[r for r in manifest['records'] if r['split']=='train'];cal=[r for r in manifest['records'] if r['split']=='calibration'][:12]
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(103003);rng=np.random.default_rng(103003);model=Specialist().cuda();parent=ROOT/'specialist.pt';state=torch.load(parent,map_location='cpu',weights_only=False);model.load_state_dict(state['model'])
    baseline_loss=validation(model,cal);baseline_fp=negative_score(model,negval)
    for layer in [model.a,model.b,model.c,model.d]:
        for p in layer.parameters():p.requires_grad_(False)
    params=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=1.5e-5,weight_decay=1e-4);best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_score=baseline_fp+.001*baseline_loss;best_step=0;logs=[];start=time.perf_counter()
    for step in range(1,301):
        batch=[crop(positive[int(rng.integers(len(positive)))],rng) for _ in range(4)]
        for _ in range(2):
            rgb=negative[int(rng.integers(len(negative)))];h,w=rgb.shape[:2];x=int(rng.integers(w-255));y=int(rng.integers(h-255));patch=rgb[y:y+256,x:x+256].copy()
            if rng.random()<.5:patch=patch[:,::-1].copy()
            batch.append((prior.photometry(patch)[0],np.zeros((6,256,256),np.float32),np.zeros((256,256),np.int64)))
        x,t,role=[torch.from_numpy(np.stack([r[k] for r in batch])).cuda() for k in range(3)];model.train();optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,_=loss_fn(model(x),t,role)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite paper refinement')
        loss.backward();torch.nn.utils.clip_grad_norm_(params,5);optimizer.step()
        if step%50==0:
            cal_loss=validation(model,cal);fp=negative_score(model,negval);eligible=cal_loss<=baseline_loss+.025;score=fp+.001*cal_loss
            if eligible and score<best_score:best_score=score;best_step=step;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            row=dict(step=step,loss=float(loss.detach()),positive_calibration_loss=cal_loss,heldout_old_paper_false_positive_pixel_fraction=fp,eligible=eligible);logs.append(row);print('PAPER_REFINE',json.dumps(row),flush=True)
    torch.save(dict(model=best_state,completed_updates=300,retained_step=best_step,parent_sha256=digest(parent)),OUT/'specialist.pt')
    jsave(OUT/'training_summary.json',dict(completed_updates=300,retained_step=best_step,baseline_positive_calibration_loss=baseline_loss,baseline_old_paper_false_positive_fraction=baseline_fp,logs=logs,elapsed_seconds=time.perf_counter()-start,gpu=torch.cuda.get_device_name(),actual_device=str(next(model.parameters()).device),parent_checkpoint_unchanged=digest(parent)==digest(ROOT/'specialist.pt'),new_imagegen_holdouts_used_for_training=False,reused_native_checks_not_blind=True))
    return model,best_state

def evaluate(state):
    model=Specialist().cuda();model.load_state_dict(state);model.eval();selection=read(ROOT/'selection.json');rows=[]
    for sid in ['N01','N02','N03']:
        rgb=np.asarray(Image.open(ROOT/'native_holdouts'/(sid+'.png')).convert('RGB'));candidate=choose(model,rgb,selection);folder=OUT/'native_outputs'/sid
        write_lines(folder,rgb,candidate['raw_mask'],candidate['raw_paths'],candidate['raw_qa'],'paper-refined raw specialist; reused development image')
        rows.append(dict(id=sid,qa=candidate['raw_qa'],rule_pass=candidate['passed'],reused_development_not_blind=True))
    jsave(OUT/'native_recheck.json',dict(records=rows,new_accuracy_not_measured=True,old_v4_not_replaced=True))
    board=Image.new('RGB',(1974,2240),'#f3f1eb');draw=ImageDraw.Draw(board);f=ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc',26)
    draw.text((20,15),'复测：原图 / 第二阶段网络 / 空白纸面负样本修正后',font=f,fill='black');draw.text((20,60),'这是已观察图的复测，不是新盲测；展示模型原始输出。',font=f,fill='#444444')
    for i,sid in enumerate(['N01','N02','N03']):
        y=120+i*700;draw.text((20,y),sid,font=f,fill='black')
        for j,path in enumerate([ROOT/'native_holdouts'/(sid+'.png'),ROOT/'native_outputs'/sid/'raw_neural'/'lines.png',OUT/'native_outputs'/sid/'lines.png']):
            im=Image.open(path).convert('RGB');im.thumbnail((620,620),Image.Resampling.LANCZOS);board.paste(im,(20+654*j,y+45))
    board.save(OUT/'PAPER_RECHECK.jpg',quality=97,subsampling=0)
    print('PAPER_REFINEMENT_READY',str(OUT),flush=True)

if __name__=='__main__':
    model,state=train();evaluate(state)
