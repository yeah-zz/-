# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,hashlib,shutil,copy
import numpy as np,torch
from PIL import Image,ImageDraw
from torch import nn
from train_cunfa_pilot_20261107 import ROOT,CLASSES,FONT,SMALL,read,save,sha,PrimitiveNet,primitive,render

def real_probes():
 from gongbi_training.narrow_select import load_clip
 from gongbi_training.rock_style_v09 import ink_crop
 clip,encoder,transform=load_clip();src=LAB/'structural_roles_20261107/G03_01';plan=read(src/'plan.json');rgb=Image.open(src/'source.png').convert('RGB')
 ck=torch.load(ROOT/'classifier.pt',map_location='cpu',weights_only=False);head=nn.Sequential(nn.Linear(1024,64),nn.SiLU(),nn.Dropout(.25),nn.Linear(64,4)).cuda();head.load_state_dict(ck['model']);head.eval()
 probes=[];ims=[]
 for m in plan['modules']:
  if m['kind']!='rock' or m['planned_strokes']<15:continue
  paths=[np.asarray(r['points']) for r in plan['curves'] if r['module']==m['id']];q=np.concatenate(paths);lo=np.maximum(0,np.floor(q.min(0)-5)).astype(int);hi=np.minimum(496,np.ceil(q.max(0)+5)).astype(int);im=rgb.crop((*lo,*hi));ims.append(im);probes.append(dict(name=m['name'],bbox=[*lo.tolist(),*hi.tolist()]))
 with torch.no_grad():
  a=encoder.encode_image(torch.stack([transform(im) for im in ims]).cuda()).float();b=encoder.encode_image(torch.stack([transform(ink_crop(im)) for im in ims]).cuda()).float();a=nn.functional.normalize(a,dim=1);b=nn.functional.normalize(b,dim=1);p=head(torch.cat([a,b],1)/np.sqrt(2)).softmax(1).cpu().numpy()
 board=Image.new('RGB',(1440,760),'#f4f1e8');d=ImageDraw.Draw(board)
 for i,(im,r,prob) in enumerate(zip(ims,probes,p)):
  k=int(prob.argmax());r.update(probabilities=prob,candidate=CLASSES[k] if k!=3 and prob.max()>=.65 else '不确定',style_ground_truth=None,not_used_for_training=True)
  x=i%3*480;y=i//3*380;im=im.copy();im.thumbnail((430,270));board.paste(im,(x+15,y+80));d.text((x+10,y+8),r['name'],font=FONT,fill='#263e34');d.text((x+10,y+37),'分类候选：'+r['candidate'],font=FONT,fill='#263e34')
 board.save(ROOT/'current_painting_probes.jpg',quality=95);save(ROOT/'current_painting_probes.json',dict(records=probes,source_plan_sha256=sha(src/'plan.json'),not_used_for_training=True,expert_accuracy=None))

def verify():
 c=read(ROOT/'classification_report.json');r=read(ROOT/'reproduction_report.json')
 groups=[set(c['source_groups'][s]) for s in ('train','validation','test')];assert not groups[0]&groups[1] and not groups[0]&groups[2]
 hs={s:{a['image_sha256'] for a in c['records'] if a['split']==s} for s in ('train','validation','test')};assert not hs['train']&hs['test'] and not hs['train']&hs['validation']
 assert c['completed_updates']==1000 and r['completed_updates']==1200
 assert r['data']=={'train':1200,'validation':180,'test':180}
 for x in read(ROOT/'synthetic_predicted_strokes.json'):
  a=np.asarray(x['predicted_xy_width_opacity']);assert a.shape==(17,4) and np.isfinite(a).all() and (a>=0).all() and (a<=1).all();assert x['physical_z'] is None
 # Reload actual learned weights; verify the displayed predictions are from the
 # saved checkpoint and were not swapped with renderer ground-truth paths.
 ck=torch.load(ROOT/'primitive_model.pt',map_location='cpu',weights_only=False);m=PrimitiveNet().cuda();m.load_state_dict(ck['model']);m.eval();im,gt,_,_=primitive(400000)
 # Match the original inference batch. CUDA single/batched convolutions differ
 # here by ~3.2e-5; same-batch reloading must reproduce the export exactly.
 batch=np.stack([primitive(400000+i)[0].transpose(2,0,1) for i in range(180)])
 with torch.no_grad():out=m(torch.tensor(batch,device='cuda').float()/255).cpu().numpy()[0]
 assert np.allclose(out[:68].reshape(17,4),np.asarray(read(ROOT/'synthetic_predicted_strokes.json')[0]['predicted_xy_width_opacity']),atol=1e-5)
 assert not np.allclose(out[:68].reshape(17,4),gt,atol=1e-5)
 # The accepted practical drawing has not been touched.
 p=read(LAB/'object_hierarchy_20261107/G03_01/plan.json');assert sha(Path(p['parent']))==p['parent_sha256']
 save(ROOT/'verification.json',dict(passed=True,source_groups_and_exact_hashes_separated=True,checkpoint_prediction_verified=True,
  completed_updates=2200,accepted_drawing_parent_unchanged=True,no_real_or_physical_reproduction_verified=True))

def main():
 torch.set_num_threads(4);real_probes();verify();Image.open(ROOT/'reproduction_test.jpg').crop((0,0,1440,735)).save(ROOT/'pilot_preview.jpg',quality=95)
 c=read(ROOT/'classification_report.json');page=ROOT/'index.html';html=page.read_text('utf8')
 note='<p style="background:#ead8b8;padding:16px"><b>本轮未通过真实皴法可靠识别验收。</b> 来源留出探针9张仅4张与弱标签一致，只有1张达到预设接受阈值。数字重绘只验证了程序单笔，不能当真实皴法复刻、整组笔顺恢复或机械臂实画效果。</p>'
 html=html.replace('<h1>皴法识别与复刻：第一轮GPU实验</h1>','<h1>皴法识别与复刻：第一轮GPU实验</h1>'+note)
 html+='<h2>当前画作的山石区域探针</h2><p>这些裁切只用于推理，没有用于本轮优化，也没有皴法真值；分类候选不能视为正确识别。没有因此重画、替换或删改489笔线稿。</p><img src="current_painting_probes.jpg">'
 page.write_text(html,encoding='utf8')
 for n in ('train_cunfa_pilot_20261107.py','cunfa_pilot_finish_20261107.py'):shutil.copy2(Path(__file__).with_name(n),ROOT/'source_snapshot'/n)
 print('CUNFA_PILOT_VERIFIED',c['test']['samples'],c['test']['label_agreement'],flush=True)

if __name__=='__main__':main()
