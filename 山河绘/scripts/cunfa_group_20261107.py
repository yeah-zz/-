# -*- coding: utf-8 -*-
"""Multi-stroke inverse graphics; real images have appearance-only supervision.

No claim of real stroke order/identity truth. No controller or existing drawing
modifications. Source article style labels are weak, not expert certification.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,time,copy,hashlib,shutil
from collections import Counter
import numpy as np,cv2,torch
from torch import nn
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from gongbi_training.rock_style_v09 import crop_record,ink_crop
ROOT=LAB/'cunfa_group_inverse_20261107';OLD=LAB/'rock_multitask_v09_20261024/style_classifier'
CLASSES=['披麻皴','斧劈皴','折带皴'];SLOTS=12;POINTS=9;PARAMS=40;SIZE=80
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17);SMALL=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13)
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item(),allow_nan=False),encoding='utf8')
def read(p):return json.loads(p.read_text('utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def procedural(seed):
 r=np.random.default_rng(seed);style=seed%3;n=int(r.integers(4,11));t=np.linspace(0,1,POINTS);out=np.zeros((SLOTS,PARAMS),np.float32)
 for k in range(n):
  x0=.13+.65*k/max(n-1,1)+r.uniform(-.018,.018);top=r.uniform(.12,.31);bottom=r.uniform(.68,.90)
  if style==0:
   x=x0+.025*np.sin(t*r.uniform(2,5)+r.uniform(-1,1))+.03*t;y=top+(bottom-top)*t;w=(.025+.045*np.sin(t*np.pi))*r.uniform(.7,1.3)
  elif style==1:
   x=x0+.065*t+.01*np.sin(t*4);y=top+(bottom-top)*t;w=(.02+.18*np.sin(t*np.pi)**.7)*r.uniform(.7,1.2)
  else:
   turn=r.uniform(.35,.6);x=x0+np.interp(t,[0,turn,1],[0,.085,.095]);y=top+np.interp(t,[0,turn,1],[0,.015,bottom-top]);w=(.035+.045*t)*r.uniform(.7,1.3)
  q=np.column_stack([x,y]);angle=r.uniform(-.10,.10);rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);q=(q-.5)@rot.T+.5
  opacity=np.clip(r.uniform(.3,.95)*(1-r.uniform(0,.3)*t),.12,1)
  out[k,:36]=np.column_stack([np.clip(q,.03,.97),w,opacity]).flatten();out[k,36:]=[1,r.uniform(0,.35),r.uniform(.15,.85),k/max(n-1,1)]
 return out,style

def render_numpy(a,size=SIZE):
 alpha=np.zeros((size*3,size*3),np.float32);yy,xx=np.mgrid[:size*3,:size*3]
 for k,row in enumerate(a):
  if row[36]<.5:continue
  q=row[:36].reshape(POINTS,4);layer=np.zeros_like(alpha)
  for i in range(POINTS-1):
   tangent=q[i+1,:2]-q[i,:2];normal=np.array([-tangent[1],tangent[0]])/max(np.linalg.norm(tangent),1e-6)
   for u in np.linspace(0,1,5):
    v=q[i]*(1-u)+q[i+1]*u;center=v[:2]*size*3;sigma=max(.65,float(v[2])*size*3*.38)
    # Crop each Gaussian stamp at five sigma (tail <4e-6), rather than
    # recomputing an entire 240x240 field for every sample on every stroke.
    radius=int(np.ceil(5*sigma));x0=max(0,int(center[0])-radius);x1=min(size*3,int(center[0])+radius+1);y0=max(0,int(center[1])-radius);y1=min(size*3,int(center[1])+radius+1)
    dx=xx[y0:y1,x0:x1]-center[0];dy=yy[y0:y1,x0:x1]-center[1];cross=(dx*normal[0]+dy*normal[1])/sigma
    stamp=np.exp(-(dx*dx+dy*dy)/(2*sigma*sigma))*float(v[3])
    stamp*=np.clip(1+(float(row[38])-.5)*.8*cross,.25,1.5)
    layer[y0:y1,x0:x1]=np.maximum(layer[y0:y1,x0:x1],stamp)
  texture=(np.sin(xx*.47+yy*.31+k)>-.35).astype(np.float32)
  layer*=1-float(row[37])*(1-texture);alpha=1-(1-alpha)*(1-layer*.96)
 return np.uint8(np.clip(cv2.resize(255*(1-alpha),(size,size),interpolation=cv2.INTER_AREA),0,255))

class GroupNet(nn.Module):
 def __init__(self):
  super().__init__();self.enc=nn.Sequential(nn.Conv2d(1,16,5,2,2),nn.SiLU(),nn.Conv2d(16,32,3,2,1),nn.SiLU(),nn.Conv2d(32,64,3,2,1),nn.SiLU(),nn.AdaptiveAvgPool2d((4,4)),nn.Flatten());self.head=nn.Sequential(nn.Linear(1028,384),nn.SiLU(),nn.Linear(384,SLOTS*PARAMS))
 def forward(self,x,c):return self.head(torch.cat([self.enc(x),c],1)).reshape(-1,SLOTS,PARAMS).sigmoid()

def render_torch(a,size=40):
 """Differentiable multi-stroke tone renderer, not a physical brush simulation."""
 grid=torch.linspace(0,1,size,device=a.device);yy,xx=torch.meshgrid(grid,grid,indexing='ij');pos=torch.stack([xx,yy],-1)[None,None,None]
 p=a[:,:,:36].reshape(-1,SLOTS,POINTS,4);start=p[:,:,:-1,:2,None,None] if False else p[:,:,:-1,:2]
 u=p[:,:,:-1,:2][:,:,:,None,None,:];v=p[:,:,1:,:2][:,:,:,None,None,:];d=v-u
 t=((pos-u)*d).sum(-1)/(d.square().sum(-1)+1e-6);t=t.clamp(0,1);closest=u+t[...,None]*d
 delta=pos-closest;dist=delta.square().sum(-1)
 w=p[:,:,:-1,2][:,:,:,None,None]*(1-t)+p[:,:,1:,2][:,:,:,None,None]*t
 op=p[:,:,:-1,3][:,:,:,None,None]*(1-t)+p[:,:,1:,3][:,:,:,None,None]*t
 sigma=(w*.38).clamp_min(.007);normal=torch.stack([-d[...,1],d[...,0]],-1);normal=normal/(d.square().sum(-1,keepdim=True).sqrt()+1e-6)
 cross=(delta*normal).sum(-1)/sigma
 side=a[:,:,38][:,:,None,None,None]-.5
 bands=torch.exp(-dist/(2*sigma.square()))*op*(1+side*.8*cross).clamp(.25,1.5)
 layer=bands.amax(2);texture=(torch.sin(xx*SIZE*3*.47+yy*SIZE*3*.31)[None,None]>-.35).float()
 layer=layer*(1-a[:,:,37,None,None]*(1-texture));layer=(layer*a[:,:,36,None,None]*.96).clamp(0,.995)
 return 1-torch.prod(1-layer,1)[:,None]

def supervised_loss(out,target):
 valid=target[:,:,36];geometry=F.smooth_l1_loss(out[:,:,:36].reshape(-1,SLOTS,POINTS,4),target[:,:,:36].reshape(-1,SLOTS,POINTS,4),reduction='none')
 weights=out.new_tensor([2.,2.,.35,.25]);g=(geometry*weights*valid[:,:,None,None]).sum()/((valid.sum()+1)*POINTS*4)
 presence=F.binary_cross_entropy(out[:,:,36],valid);aux=((out[:,:,37:]-target[:,:,37:]).abs()*valid[:,:,None]).sum()/((valid.sum()+1)*3)
 return g*4+presence*.2+aux*.08

def prepare_real():
 records=read(OLD/'report.json')['records'];out=[]
 for r in records:
  if r['label'] not in CLASSES:continue
  group=r['source_group'];split='validation' if group=='gray_cunfa_demonstration_book' else 'test' if group=='new_sohu_376343741' else 'train'
  im=crop_record(r);ink=ink_crop(im).convert('L');ink.thumbnail((SIZE-8,SIZE-8),Image.Resampling.LANCZOS);canvas=Image.new('L',(SIZE,SIZE),255);canvas.paste(ink,((SIZE-ink.width)//2,(SIZE-ink.height)//2))
  out.append(dict(id=r['id'],label=r['label'],style=CLASSES.index(r['label']),image=r['image'],box=r.get('box'),source_group=group,split=split,
    image_sha256=sha(Path(r['image'])),appearance=np.array(canvas),raw=im,expert_paths=False,ink_target='automatic paper normalization, weak appearance only'))
 overlap={r['image_sha256'] for r in out if r['split']=='train'}&{r['image_sha256'] for r in out if r['split']!='train'}
 out=[r for r in out if not(r['split']=='train' and r['image_sha256'] in overlap)]
 return out

def metrics(pred,target):
 n=(pred[:,:,36]>=.5).sum(1);valid=target[:,:,36]>0
 q=pred[:,:,:36].reshape(-1,SLOTS,POINTS,4);gt=target[:,:,:36].reshape(-1,SLOTS,POINTS,4)
 return dict(count_mae=float(np.abs(n-valid.sum(1)).mean()),ordered_slot_point_error_80px=float(np.linalg.norm(q[:,:,:,:2]-gt[:,:,:,:2],axis=3)[valid].mean()*SIZE),
  count_ground_truth='program declared; no artist stroke identity claim')

def main():
 ROOT.mkdir(exist_ok=True)
 if (ROOT/'group_model.pt').exists():raise FileExistsError('Preserve completed grouped experiment')
 torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(110711);rng=np.random.default_rng(110711)
 real=prepare_real();manifest=[{k:v for k,v in r.items() if k not in ('appearance','raw')} for r in real];save(ROOT/'real_manifest.json',manifest)
 save(ROOT/'protocol.json',dict(classes=CLASSES,max_slots=SLOTS,points_per_stroke=POINTS,
  synthetic='4-10 procedural strokes; exact group paths/width/tone/side bias and declared order',
  real='only normalized ink appearance: no real geometry, cardinality, width or order labels',
  reviewed_rock_curves_excluded='R01-R04 are contour labels, not verified cunfa strokes',
  test_style_condition='unknown/no class supplied; source labels are display-only for real tests',
  real_source_group_split=True,real_labels_nonexpert=True,prior_sources_previously_seen=True,not_new_blind_artworks=True,
  accepted_489_plan_modified=False,robot_connected=False))
 sets={}
 for split,base,n in [('train',0,900),('validation',200000,120),('test',400000,120)]:
  ys=[];xs=[];cs=[]
  for i in range(n):
   a,style=procedural(base+i);ys.append(a);xs.append(render_numpy(a));cs.append(style)
  sets[split]=(torch.tensor(np.array(xs)[:,None],device='cuda').float()/255,torch.tensor(np.array(ys),device='cuda'),np.array(cs))
  print('GROUP_DATA',split,n,flush=True)
 rtrain=[r for r in real if r['split']=='train'];rval=[r for r in real if r['split']=='validation'];rtest=[r for r in real if r['split']=='test']
 rx=torch.tensor(np.array([r['appearance'] for r in rtrain])[:,None],device='cuda').float()/255
 rc=torch.tensor([r['style'] for r in rtrain],device='cuda',dtype=torch.long);rv=torch.tensor(np.array([r['appearance'] for r in rval])[:,None],device='cuda').float()/255
 model=GroupNet().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.0007);x,y,classes=sets['train'];history=[];best=1e9;state=None;start=time.monotonic()
 def condition(v):return F.one_hot(v,4).float()
 for step in range(1,1801):
  ids=rng.integers(0,len(x),32);cl=torch.tensor(classes[ids],device='cuda',dtype=torch.long);cl[torch.rand(len(cl),device='cuda')<.35]=3
  optimizer.zero_grad();out=model(x[ids],condition(cl));loss=supervised_loss(out,y[ids])
  if step>1200:
   ix=rng.integers(0,len(rx),3);cc=rc[ix].clone();cc[torch.rand(len(cc),device='cuda')<.5]=3
   rp=model(rx[ix],condition(cc));rendered=render_torch(rp);target=F.interpolate(1-rx[ix],(40,40),mode='area')
   appearance=(rendered-target).abs().mean();coverage=((rendered-target).square()*(1+target*2)).mean()
   # Real appearance supervision cannot establish unique pen lifts or order.
   smooth=rp[:,:,:36].reshape(-1,SLOTS,POINTS,4)[:,:,:,:2].diff(dim=2).diff(dim=2).abs().mean()
   loss=loss*.5+appearance*.7+coverage*.7+smooth*.03
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),4);optimizer.step()
  if step%100==0:
   with torch.no_grad():
    vx,vy,vc=sets['validation'];vo=model(vx,condition(torch.full((len(vx),),3,device='cuda',dtype=torch.long)));vl=float(supervised_loss(vo,vy))
    ro=model(rv,condition(torch.full((len(rv),),3,device='cuda',dtype=torch.long)));rl=float((render_torch(ro)-F.interpolate(1-rv,(40,40),mode='area')).abs().mean())
   history.append(dict(step=step,synthetic_validation_loss=vl,real_weak_validation_appearance_l1=rl))
   if step==1200:warm=copy.deepcopy(model.state_dict())
   # Selection uses validation only, never the displayed test examples.
   score=vl+rl*.7
   if step>=1200 and score<best:best=score;state=copy.deepcopy(model.state_dict());retained=step
   print('GROUP_TRAIN',step,round(vl,4),round(rl,4),flush=True)
 model.load_state_dict(state);model.eval();torch.save(dict(model=state,max_slots=SLOTS,points=POINTS,completed_updates=1800,retained_step=retained),ROOT/'group_model.pt')
 torch.save(dict(model=warm,updates=1200),ROOT/'synthetic_warm_start.pt')
 tx,ty,tc=sets['test'];unknown=torch.full((len(tx),),3,device='cuda',dtype=torch.long)
 with torch.no_grad():sp=model(tx,condition(unknown)).cpu().numpy()
 synthetic_metrics=metrics(sp,ty.cpu().numpy());save(ROOT/'synthetic_metrics.json',synthetic_metrics)
 rt=torch.tensor(np.array([r['appearance'] for r in rtest])[:,None],device='cuda').float()/255;u=condition(torch.full((len(rt),),3,device='cuda',dtype=torch.long))
 with torch.no_grad():rp=model(rt,u).cpu().numpy();pred_appearance=render_torch(torch.tensor(rp,device='cuda')).cpu().numpy()
 warm_model=GroupNet().cuda();warm_model.load_state_dict(warm)
 with torch.no_grad():wp=warm_model(rt,u);warm_l1=float((render_torch(wp)-F.interpolate(1-rt,(40,40),mode='area')).abs().mean())
 real_l1=float(np.abs(pred_appearance-F.interpolate(1-rt,(40,40),mode='area').cpu().numpy()).mean())
 exports=[]
 for i,r in enumerate(rtest):exports.append(dict(id=r['id'],weak_article_label=r['label'],inference_style_condition='unknown',strokes=rp[i],predicted_count=int((rp[i,:,36]>=.5).sum()),true_count=None,true_order=None,expert_accuracy=None,physical_z=None))
 save(ROOT/'real_predictions.json',exports)
 np.savez_compressed(ROOT/'test_predictions.npz',synthetic=sp,synthetic_truth=ty.cpu().numpy(),real=rp,real_inputs=rt.cpu().numpy())
 gallery(rtest,rp,ROOT/'real_group_test.jpg',real_mode=True);synrows=[]
 for i in range(6):synrows.append(dict(raw=Image.fromarray(np.uint8(sets['test'][0][i,0].cpu().numpy()*255)).convert('RGB'),appearance=np.uint8(sets['test'][0][i,0].cpu().numpy()*255),label=CLASSES[int(tc[i])],id='program_'+str(i)))
 gallery(synrows,sp[:6],ROOT/'synthetic_group_test.jpg',real_mode=False)
 summary=dict(completed_updates=1800,synthetic_only_updates=1200,weak_real_joint_updates=600,retained_step=retained,seconds=time.monotonic()-start,
  device=torch.cuda.get_device_name(),synthetic_train_groups=900,synthetic_validation_groups=120,synthetic_test_groups=120,
  real_samples=dict(Counter(r['split'] for r in real)),real_labels_expert=False,
  synthetic_test_metrics=synthetic_metrics,real_test_weak_appearance_l1=real_l1,warm_test_weak_appearance_l1=warm_l1,
  real_stroke_identity_or_order_accuracy=None,real_cunfa_classification_accuracy=None,
  independent_new_artwork_test=False,robot_reproduction_tested=False,history=history)
 save(ROOT/'training_report.json',summary);write_report(summary)
 print('GROUP_DONE',json.dumps({k:v for k,v in summary.items() if k!='history'},ensure_ascii=False),flush=True)

def gallery(rows,pred,path,real_mode):
 board=Image.new('RGB',(1600,len(rows)*310),'#f4f1e8');d=ImageDraw.Draw(board)
 for i,(r,a) in enumerate(zip(rows,pred)):
  canvas=np.full((SIZE,SIZE,3),255,np.uint8)
  for slot in a:
   if slot[36]<.5:continue
   q=slot[:36].reshape(POINTS,4);cv2.polylines(canvas,[np.rint(q[:,:2]*SIZE).astype(np.int32)],False,(35,100,190),1,cv2.LINE_AA)
  frames=[(r['raw'],'真实局部 · 弱标签'+r['label'] if real_mode else '程序成组留出输入'),(Image.fromarray(r['appearance']),'弱墨迹参考 · 非逐笔真值' if real_mode else '已知成组目标'),(Image.fromarray(canvas),f'模型成组轨迹 · {int((a[:,36]>=.5).sum())}笔候选'),(Image.fromarray(render_numpy(a)),'数字重绘 · 非实画')]
  for j,(im,label) in enumerate(frames):
   d.text((j*400+8,i*310+8),label,font=SMALL,fill='#263e34');im=im.copy();ratio=min(280/im.width,250/im.height);im=im.resize((max(1,round(im.width*ratio)),max(1,round(im.height*ratio))),Image.Resampling.LANCZOS);board.paste(im,(j*400+50,i*310+40))
 board.save(path,quality=95)

def write_report(summary):
 ROOT.joinpath('source_snapshot').mkdir(exist_ok=True);shutil.copy2(Path(__file__),ROOT/'source_snapshot'/Path(__file__).name)
 html='''<!doctype html><meta charset="utf-8"><title>成组皴笔训练</title><style>body{font:16px/1.7 Microsoft YaHei,system-ui;background:#f4f1e8;color:#243e33;max-width:1600px;padding:24px;margin:auto}img{width:100%}pre{white-space:pre-wrap;background:white;padding:15px}</style><h1>从单笔改为成组笔触</h1><p>输入一组皴笔区域，输出最多12条连续笔触、各自宽度/浓淡/侧向墨色偏置/程序干笔参数及存在分数。程序成组样本提供完整路径；真实教学图仅提供自动归一化墨迹的弱外观监督，没有伪造分笔、笔数或历史笔顺真值。</p><p>真实测试时不提供类别标签。文章弱标签只用于展示；不代表本模型正确识别了皴法。输出槽位顺序只在程序数据有依据，真实笔顺没有被验证。全部旧来源曾用于其他实验，不能声称新的盲测泛化。原489笔计划没有修改，未连接机械臂。</p><h2>真实局部的固定来源留出结果（全部列出）</h2><p>弱墨迹参考可能有纸纹、轮廓、设色或漏线，图像重合不等于恢复画家的正确皴笔。若候选变成均匀排线、丢失组织或覆盖成块，也属于需要保留的失败例。下图不挑选成功样本。</p><img src="real_group_test.jpg"><h2>程序成组留出结果</h2><p>程序形态是披麻/斧劈/折带的形态类比，不是专家认可的皴法范本；它只检验多笔机制。</p><img src="synthetic_group_test.jpg"><h2>训练与限制</h2><pre>'''+json.dumps(summary,ensure_ascii=False,indent=2)+'</pre>'
 (ROOT/'index.html').write_text(html,encoding='utf8');save(ROOT/'decision.json',dict(adopted=False,group_mechanism_implemented=True,real_stroke_or_cunfa_reproduction_verified=False,robot_connected=False))

if __name__=='__main__':main()
