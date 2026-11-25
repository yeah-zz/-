# -*- coding: utf-8 -*-
"""FRIDA-inspired action-to-image pipeline. Contains no robot controller.

Real labels require observed actions. Tutorial screenshots cannot supply them.
Smoke data are known procedural paths and are never current-brush calibration.
"""
import sys,json,argparse,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import torch
from torch import nn
import numpy as np
import cv2
from PIL import Image
OUT=LAB/'rock_skill_offline_20261110/brush_forward'
FIELDS=['length_mm','bend_mm','depth_mm','speed_mm_s','dip_depth_mm','dip_seconds']

class ActionToInk(nn.Module):
 def __init__(self):
  super().__init__();self.fc=nn.Sequential(nn.Linear(6,64),nn.SiLU(),nn.Linear(64,16*32),nn.SiLU())
  self.conv=nn.Sequential(nn.Conv2d(1,16,5,padding=2),nn.SiLU(),nn.Conv2d(16,1,5,padding=2))
 def forward(self,x):
  x=self.conv(self.fc(x).reshape(-1,1,16,32))
  return torch.nn.functional.interpolate(x,size=(64,128),mode='bilinear',align_corners=False)[:,0]

def load_real(path):
 rows=list(csv.DictReader(Path(path).open(encoding='utf-8-sig')))
 if not rows:raise ValueError('No measured action/image pairs; real training not started')
 x=[];y=[];split=[];groups={}
 for r in rows:
  if r.get('source_kind')!='measured_robot':raise ValueError('Only measured_robot pairs admitted')
  for name in ['session_id','brush_id','paper_id','calibration_id','mask_path','split']+FIELDS:
   if not r.get(name):raise ValueError('Unknown field '+name+'; never infer it from tutorial pixels')
  if r['split'] not in ['train','test']:raise ValueError('Explicit train/test session split required')
  group=r['session_id'];groups.setdefault(group,set()).add(r['split'])
  if len(groups[group])>1:raise ValueError('Session leakage between train/test')
  q=Path(r['mask_path']);q=q if q.is_absolute() else Path(path).parent/q
  im=np.asarray(Image.open(q).convert('L').resize((128,64)),dtype='float32')/255
  # File convention: white=deposited ink probability; black=paper.
  x.append([float(r[k]) for k in FIELDS]);y.append(im);split.append(r['split'])
 if set(split)!={'train','test'}:raise ValueError('Both independent sessions are required')
 return np.array(x,'float32'),np.array(y),np.array(split)=='train'

def procedural_smoke():
 rng=np.random.default_rng(111010);x=rng.uniform(0,1,(48,6)).astype('float32');ys=[]
 for p in x:
  m=np.zeros((64,128),'float32');t=np.linspace(0,1,50)
  q=np.column_stack([8+t*(25+p[0]*85),32+np.sin(t*np.pi)*(p[1]-.5)*35]).astype('int32')
  cv2.polylines(m,[q],False,float(.3+.6*p[4]),max(1,int(1+p[2]*4)))
  ys.append(m)
 return x,np.stack(ys),np.arange(48)<36

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest');p.add_argument('--smoke',action='store_true');p.add_argument('--steps',type=int,default=100)
 a=p.parse_args();OUT.mkdir(exist_ok=True)
 header=['source_kind','session_id','brush_id','paper_id','calibration_id','split','mask_path']+FIELDS
 if not (OUT/'measurement_template.csv').exists():
  with (OUT/'measurement_template.csv').open('w',encoding='utf-8-sig',newline='') as f:csv.writer(f).writerow(header)
 if not a.smoke and not a.manifest:p.error('Provide measured manifest, or --smoke for procedural pipeline check only')
 x,y,train=procedural_smoke() if a.smoke else load_real(a.manifest)
 mean=x[train].mean(0);std=np.maximum(x[train].std(0),1e-6);x=(x-mean)/std
 torch.set_num_threads(4);torch.manual_seed(111010);device='cuda' if torch.cuda.is_available() else 'cpu'
 xt=torch.tensor(x,device=device);yt=torch.tensor(y,device=device);tr=torch.tensor(train,device=device)
 net=ActionToInk().to(device);opt=torch.optim.Adam(net.parameters(),lr=.002);history=[]
 for step in range(a.steps):
  net.train();pred=net(xt[tr]);loss=(torch.nn.functional.binary_cross_entropy_with_logits(pred,yt[tr],reduction='none')*(1+10*yt[tr])).mean()
  opt.zero_grad();loss.backward();opt.step()
  if step%25==0:history.append(dict(step=step,loss=float(loss.detach())))
 net.eval()
 with torch.no_grad():out=net(xt).sigmoid().cpu().numpy()
 fname='pipeline_smoke_only.pt' if a.smoke else 'measured_forward_model.pt'
 torch.save(dict(model=net.cpu().state_dict(),mean=mean.tolist(),std=std.tolist(),source_kind='procedural_smoke' if a.smoke else 'measured_robot',current_brush_validated=False,action_fields=FIELDS),OUT/fname)
 result=dict(smoke_only=a.smoke,steps=a.steps,train_samples=int(train.sum()),test_samples=int((~train).sum()),
  current_brush_model=False,physical_actions_recommended=False,robot_connected=False,history=history,
  heldout_image_mae=float(np.abs(out[~train]-y[~train]).mean()),
  warning='Procedural smoke is a software pipeline check, not reproduction of FRIDA results or brush calibration' if a.smoke else 'Requires further physical repeatability and sequential-stroke validation')
 (OUT/'pipeline_result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
 board=Image.new('RGB',(512,64*6),'white')
 for j,i in enumerate(np.where(~train)[0][:6]):
  for col,v in enumerate([y[i],out[i]]):board.paste(Image.fromarray(np.uint8((1-v)*255)).resize((256,64)),(col*256,j*64))
 board.save(OUT/'procedural_smoke_pairs.png');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
