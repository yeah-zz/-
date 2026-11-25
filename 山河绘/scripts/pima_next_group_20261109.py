"""A dense next-ink-group research pilot on real temporal WEAK observations.

No fixed stroke slots, vertical templates, expert paths, pen-order invention,
physical parameters or robot API. Same-painting development is not a blind test.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,time,copy,hashlib
import numpy as np,cv2,torch
from torch import nn
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'pima_demonstration_20261109';SIZE=160
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def fit(a):
 h,w=a.shape;scale=min(SIZE/h,SIZE/w);nh,nw=round(h*scale),round(w*scale);out=np.zeros((SIZE,SIZE),np.float32);y=(SIZE-nh)//2;x=(SIZE-nw)//2
 out[y:y+nh,x:x+nw]=cv2.resize(a.astype(np.float32),(nw,nh),interpolation=cv2.INTER_AREA)
 return out
class NextGroupNet(nn.Module):
 """Image+observed canvas+object region+phase -> next observed group footprint."""
 def __init__(self):
  super().__init__()
  def block(a,b):return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU(),nn.Conv2d(b,b,3,padding=1),nn.SiLU())
  self.a=block(4,16);self.b=block(16,32);self.c=block(32,64);self.d=block(96,32);self.e=block(48,16);self.head=nn.Conv2d(16,1,1)
 def forward(self,x):
  a=self.a(x);b=self.b(F.avg_pool2d(a,2));c=self.c(F.avg_pool2d(b,2));d=self.d(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1));e=self.e(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1));return self.head(e)
def load_data():
 manifest=json.loads((ROOT/'sequence_manifest.json').read_text('utf8'));records=manifest['records'];data=[]
 # All examples from this source family remain research/development, never test.
 for r in records:
  z=np.load(r['file']);roi=z['roi'];target=z['target'];valid=z['valid'].astype(np.float32)
  if r['new_ink_pixels']<18:continue
  lesson=r['lesson'];cfg=__import__('pima_sequence_20261109').CONFIG[lesson];t=cfg['times'][-1]
  final=np.array(Image.open(ROOT/'sequence'/f'{lesson}_{t:04d}.png'));x0,y0,x1,y1=cfg['box'];final=final[y0:y1,x0:x1]
  from pima_sequence_20261109 import ink
  phase=np.full_like(roi,1. if r['phase_candidate']=='cun_candidate' else 0.,dtype=np.float32)
  x=np.stack([fit(ink(final)),fit(z['before']),fit(roi),fit(phase*roi)])
  y=fit(target)[None];v=(fit(valid)>.98).astype(np.float32)[None]
  r=dict(r,assistant_content_review='source windows inspected; masks remain automatic and weak',used=True)
  data.append((r,x,y,v))
 return data
def metrics(pred,target,valid):
 p=(pred>.3)*valid;t=(target>.2)*valid
 tp=float((p*t).sum());return dict(dice=2*tp/max(float(p.sum()+t.sum()),1),precision=tp/max(float(p.sum()),1),recall=tp/max(float(t.sum()),1))
def lossfn(logits,y,v):
 b=F.binary_cross_entropy_with_logits(logits,y,reduction='none',pos_weight=logits.new_tensor(8.));b=(b*v).sum()/v.sum().clamp_min(1)
 p=logits.sigmoid()*v;t=y*v;dice=1-(2*(p*t).sum((1,2,3))+1)/(p.sum((1,2,3))+t.sum((1,2,3))+1)
 return b+.5*dice.mean()
def main():
 torch.manual_seed(1109);np.random.seed(1109);torch.set_num_threads(4);assert torch.cuda.is_available()
 data=load_data();train=[d for d in data if d[0]['split']=='train'];dev=[d for d in data if d[0]['split']!='train'];assert train and dev
 def tensors(rows):return [torch.tensor(np.stack([r[k] for r in rows]),device='cuda') for k in [1,2,3]]
 x,y,v=tensors(train);dx,dy,dv=tensors(dev);model=NextGroupNet().cuda();opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-4);history=[];best=None;bestscore=float('inf');start=time.time()
 for step in range(401):
  if step:
   model.train();ids=torch.randint(len(train),(min(6,len(train)),),device='cuda');bx,by,bv=x[ids].clone(),y[ids].clone(),v[ids].clone()
   # Apply transformations consistently; no fixed direction or artificial stroke geometry.
   if step%2:bx=bx.flip(-1);by=by.flip(-1);bv=bv.flip(-1)
   if step%3==0:bx=bx.flip(-2);by=by.flip(-2);bv=bv.flip(-2)
   opt.zero_grad(set_to_none=True);loss=lossfn(model(bx),by,bv);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);opt.step()
  if step%40==0:
   model.eval()
   with torch.no_grad():dl=float(lossfn(model(dx),dy,dv));tl=float(lossfn(model(x),y,v))
   history.append(dict(step=step,train_loss=tl,same_work_dev_loss=dl));print('TRAIN',history[-1],flush=True)
   if dl<bestscore:bestscore=dl;best=copy.deepcopy(model.state_dict());retained=step
 model.load_state_dict(best);model.eval()
 with torch.no_grad():prediction=model(dx).sigmoid().cpu().numpy();trainpred=model(x).sigmoid().cpu().numpy()
 target=dy.cpu().numpy();valid=dv.cpu().numpy();remaining=np.clip(dx[:,0:1].cpu().numpy()-dx[:,1:2].cpu().numpy(),0,1)
 stats=metrics(prediction,target,valid);baseline=metrics(remaining,target,valid)
 torch.save(dict(model=best,architecture='dense object-conditioned next observed group; no fixed stroke slots',step=retained,training_scope='real video automatic temporal weak labels',source_group='liupengkai_qiushan_same_painting',independent_test=False),ROOT/'next_group_pilot.pt')
 np.savez_compressed(ROOT/'next_group_predictions.npz',prediction=prediction,target=target,valid=valid,inputs=dx.cpu().numpy(),train_prediction=trainpred)
 report=dict(device=torch.cuda.get_device_name(0),updates=400,retained_step=retained,seconds=time.time()-start,train_groups=len(train),same_work_dev_groups=len(dev),independent_test_groups=0,
   same_work_weak_mask_metrics=stats,all_remaining_ink_baseline=baseline,history=history,trained_on_expert_strokes=False,individual_count_or_order_learned=False,
   style_recognition_evaluated=False,complete_curve_prediction=False,stop_policy_trained=False,physical_brush_evaluated=False,adopted=False,robot_connected=False,
   decision='Only a temporal ink-group pilot. Full cunfa reproduction gate not passed: no true stroke instances, no independent works, no physical calibration.')
 save(ROOT/'next_group_report.json',report);save(ROOT/'training_manifest.json',[d[0] for d in data]);gallery(dev,prediction,target,valid,remaining)
 print('RESULT',json.dumps({k:report[k] for k in ['train_groups','same_work_dev_groups','retained_step','same_work_weak_mask_metrics','all_remaining_ink_baseline']},ensure_ascii=False),flush=True)
def gallery(dev,pred,target,valid,remaining):
 font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
 board=Image.new('RGB',(1200,320*min(5,len(dev))),'#f1eee6');d=ImageDraw.Draw(board)
 for row,(r,x,y,v) in enumerate(dev[:5]):
  ims=[1-x[0],1-x[1],1-target[row,0],1-pred[row,0]]
  titles=['最终画面局部（输入）','当前已画（输入）','真实前后增量·弱参考','模型直接预测·未通过验收']
  for col,(a,title) in enumerate(zip(ims,titles)):
   im=Image.fromarray(np.uint8(np.clip(a,0,1)*255)).convert('RGB').resize((280,280));board.paste(im,(col*300+10,row*320+30));d.text((col*300+8,row*320+5),title,font=small,fill='#222')
 board.save(ROOT/'next_group_preview.jpg',quality=95)
if __name__=='__main__':main()
