# -*- coding: utf-8 -*-
"""Fixed-budget partial role probe. No style, instance, or robot claims."""
import sys,json,hashlib,random,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import cv2,numpy as np,torch
from torch import nn
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from gongbi_training.cunfa_supervision import partial_role_loss,role_targets_checked
ROOT=LAB/'cunfa_temporal_roles_20261112'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x):(ROOT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def features(image):
 g=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY).astype('float32')
 return np.stack([g/255,np.clip((cv2.GaussianBlur(g,(0,0),4)-g)/32,-1,1)])
class Block(nn.Sequential):
 def __init__(self,a,b):super().__init__(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU(),nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU())
class RoleNet(nn.Module):
 def __init__(self):
  super().__init__();self.a=Block(2,12);self.b=Block(12,24);self.c=Block(24,48);self.d=Block(48,96);self.u=Block(144,48);self.v=Block(72,24);self.w=Block(36,12);self.out=nn.Conv2d(12,3,1)
 def forward(self,x):
  a=self.a(x);b=self.b(F.avg_pool2d(a,2));c=self.c(F.avg_pool2d(b,2));d=self.d(F.avg_pool2d(c,2))
  up=lambda x,y:F.interpolate(x,size=y.shape[-2:],mode='bilinear',align_corners=False)
  u=self.u(torch.cat([up(d,c),c],1));v=self.v(torch.cat([up(u,b),b],1));return self.out(self.w(torch.cat([up(v,a),a],1)))
def confusion(t,p):
 known=t!=255;return np.bincount((t[known]*3+p[known]).ravel(),minlength=9).reshape(3,3)
def describe(c):
 return dict(confusion=c.tolist(),known_pixels=int(c.sum()),reference_support=c.sum(1).tolist(),recall=[float(c[k,k]/c[k].sum()) if c[k].sum() else None for k in range(3)],paper_false_ink_rate=float(c[0,1:].sum()/c[0].sum()) if c[0].sum() else None,structure_vs_cun_independent_test_valid=bool(c[1].sum() and c[2].sum()))
def main():
 assert not (ROOT/'role_probe_results.json').exists(),'Evaluation already frozen; create a new experiment instead.'
 seed=1212;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
 manifest=json.loads((ROOT/'dataset_manifest.json').read_text('utf8'));records=manifest['records'];data={r['id']:dict(np.load(r['path'])) for r in records};train=[r for r in records if r['split']=='train'];test=[r for r in records if r['split']!='train']
 assert all(role_targets_checked(r) and sha(r['path'])==r['sha256'] for r in records)
 assert not ({r['source_family'] for r in train}&{r['source_family'] for r in test})
 protocol=dict(seed=seed,steps=240,optimizer='AdamW lr=0.001 weight_decay=0.0001',architecture='Small U-Net 2 -> 12/24/48/96 -> 3',model_input='Static after image luminance and local contrast only; no before/delta/guide/time/role id',reference_ignore_index=255,weights='inverse per-class training core frequency',selection='Final fixed step, no test checkpoint or threshold selection',training_work_count=1,training_observation_count=3,test_work_count=1,test_observation_count=3,test_is_new_blind=False,test_crop='Event-localized crops from previously reviewed guides; autonomous localization NOT tested',test_images='Earlier multi-frame composites, not raw single frames',test_structure_reference_available=False,wash_or_style_trained=False,individual_strokes_or_pressure_trained=False,model_adoption_allowed=False,robot_connected=False,dataset_sha256=sha(ROOT/'dataset_manifest.json'),script_sha256=sha(__file__),records=[dict(id=r['id'],split=r['split'],source_family=r['source_family'],sha256=r['sha256']) for r in records])
 write('role_probe_protocol.json',protocol)
 # Fit the simple ink baseline on training cores only. Roles 1/2 collapse to ink.
 thresholds=[2.,3.,4.,6.,8.,12.];scores=[]
 for threshold in thresholds:
  correct=np.zeros(2);total=np.zeros(2)
  for r in train:
   z=data[r['id']];t=z['targets'];p=features(z['image'])[1]*32>threshold
   for k,mask in enumerate([t==0,(t==1)|(t==2)]):correct[k]+=(p[mask]==bool(k)).sum();total[k]+=mask.sum()
  scores.append(float((correct/total).mean()))
 threshold=thresholds[int(np.argmax(scores))];write('ink_baseline_protocol.json',dict(threshold=threshold,training_balanced_ink_recall=scores,threshold_candidates=thresholds,rule='local contrast > threshold -> cun; otherwise paper',test_used_for_selection=False))
 counts=sum((np.bincount(data[r['id']]['targets'][data[r['id']]['targets']!=255],minlength=3) for r in train));classw=torch.tensor(1/counts,dtype=torch.float32,device='cuda');classw/=classw.mean()
 model=RoleNet().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);losses=[];started=time.time()
 for step in range(protocol['steps']):
  r=train[step%len(train)];z=data[r['id']];im=z['image'];target=z['targets'];h,w=target.shape
  # Reference and image receive exactly the same affine; outside remains unknown.
  angle=random.uniform(-9,9);scale=random.uniform(.88,1.08);m=cv2.getRotationMatrix2D((w/2,h/2),angle,scale);m[:,2]+=np.array([random.uniform(-8,8),random.uniform(-5,5)])
  im=cv2.warpAffine(im,m,(w,h),borderValue=(255,255,255));target=cv2.warpAffine(target,m,(w,h),flags=cv2.INTER_NEAREST,borderValue=255)
  if random.random()<.5:im=im[:,::-1].copy();target=target[:,::-1].copy()
  im=np.clip(im.astype('float32')*random.uniform(.92,1.05)+random.uniform(-6,6),0,255).astype('uint8');x=torch.from_numpy(features(im).copy())[None].cuda();t=torch.from_numpy(target.astype('int64'))[None].cuda()
  logits=model(x);weights=classw[t.clamp(0,2)]*t.ne(255);loss=partial_role_loss(logits,t,weights);opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
  if (step+1)%60==0:print('TRAIN',step+1,round(losses[-1],4),flush=True)
 checkpoint=ROOT/'role_probe.pt';torch.save(dict(model=model.state_dict(),protocol=protocol),checkpoint);model.eval();results=[];colours=np.array([[234,239,226],[192,128,40],[40,135,180]],'uint8');font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n);(ROOT/'predictions').mkdir(exist_ok=True)
 for r in records:
  z=data[r['id']];im=z['image'];t=z['targets']
  with torch.inference_mode():prob=model(torch.from_numpy(features(im))[None].cuda()).softmax(1)[0].cpu().numpy()
  p=prob.argmax(0).astype('uint8');bp=np.where(features(im)[1]*32>threshold,2,0).astype('uint8');c=confusion(t,p);bc=confusion(t,bp);np.savez_compressed(ROOT/'predictions'/(r['id']+'.npz'),probabilities=prob,prediction=p)
  raw=colours[p];Image.fromarray(raw).save(ROOT/'predictions'/(r['id']+'_raw.png'))
  board=Image.new('RGB',(1200,400),'#f4f1e8');d=ImageDraw.Draw(board)
  refs=np.array(Image.open(ROOT/'references'/(r['id']+'_targets.png')))
  for j,(a,title) in enumerate([(im,'静态输入'),(refs,'视频辅助参考 / 灰色待定'),(raw,'模型直接角色图'),(colours[bp],'对照：墨迹全部当皴笔')]):
   image=Image.fromarray(a);image.thumbnail((285,285));board.paste(image,(j*300+6,40));d.text((j*300+6,7),title,font=font(17),fill='#243e32')
  d.text((8,334),r['id']+'：橙=结构；蓝=皴笔；浅绿=纸面。模型图未用参考遮罩修饰。',font=font(18),fill='#76452e')
  d.text((8,365),'训练仅1件作品；另一来源测试缺结构真值，不证明整体皴法理解或自主找笔。',font=font(17),fill='#76452e');board.save(ROOT/'predictions'/(r['id']+'_comparison.jpg'),quality=94)
  results.append(dict(id=r['id'],split=r['split'],model=describe(c),ink_as_cun_baseline=describe(bc),mean_max_probability=float(prob.max(0).mean()),raw_prediction_class_pixels=np.bincount(p.ravel(),minlength=3).tolist()))
 aggregates={}
 for split in ['train','exploratory_other_work_test']:
  rows=[r for r in results if r['split']==split];aggregates[split]={key:describe(sum((np.array(r[key]['confusion']) for r in rows))) for key in ['model','ink_as_cun_baseline']}
 result=dict(protocol=protocol,model_sha256=sha(checkpoint),duration_seconds=time.time()-started,loss_start=losses[0],loss_end=losses[-1],loss_curve=losses,records=results,aggregates=aggregates,conclusion='Exploratory role probe only. No complete new-work structure/cun gold; no style/instance/order/robot evidence. Do not replace accepted drawing model.',adopted=False,robot_connected=False)
 write('role_probe_results.json',result);print(json.dumps(aggregates,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
