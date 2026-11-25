# -*- coding: utf-8 -*-
"""Narrow weak-label classification + separate known-path brush primitive pilot.

No outline fragments are used as stroke ground truth. Real style labels remain
article/assistant labels. Synthetic primitive names are morphological analogues,
NOT expert-certified cunfa demonstrations or robot control measurements.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,time,hashlib,copy,shutil
from collections import Counter
import cv2,numpy as np,torch
from torch import nn
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'cunfa_recognition_reproduction_20261107';OLD=LAB/'rock_multitask_v09_20261024/style_classifier'
CLASSES=['披麻皴','斧劈皴','折带皴','其他／不确定']
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17);SMALL=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13)

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item(),allow_nan=False),encoding='utf8')
def read(p):return json.loads(p.read_text('utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def evaluate(y,p):
 pred=p.argmax(1);cm=np.zeros((4,4),int)
 for a,b in zip(y,pred):cm[a,b]+=1
 accepted=(p.max(1)>=.65)&(pred!=3)
 return dict(samples=len(y),label_agreement=float((pred==y).mean()) if len(y) else None,confusion=cm,
  accepted_count=int(accepted.sum()),accepted_label_agreement=float((pred[accepted]==y[accepted]).mean()) if accepted.any() else None,
  expert_accuracy=False,probabilities_calibrated=False)

def classification():
 old=read(OLD/'report.json');records=old['records'];z=np.load(OLD/'features.npz');X=np.concatenate([z['raw'],z['normalized']],1)/np.sqrt(2)
 assert len(records)==len(X)
 y=np.array([CLASSES.index(r['label']) if r['label'] in CLASSES[:3] else 3 for r in records])
 # Repartition entire source groups. No crop from the same tutorial/book group
 # may occur in both optimization and validation/test.
 group=[r['source_group'] for r in records]
 tr=np.array([g not in ('gray_cunfa_demonstration_book','new_sohu_376343741') for g in group])
 va=np.array([g=='gray_cunfa_demonstration_book' for g in group]);te=np.array([g=='new_sohu_376343741' for g in group])
 assert set(np.array(group)[tr]).isdisjoint(set(np.array(group)[va|te]))
 # Also prevent byte-identical originals crossing source groups.
 hashes=[sha(Path(r['image'])) for r in records]
 overlap=set(np.array(hashes)[tr])&set(np.array(hashes)[va|te])
 if overlap:
  tr&=np.array([h not in overlap for h in hashes])
 assert set(np.array(hashes)[tr]).isdisjoint(set(np.array(hashes)[va|te]))
 torch.manual_seed(110701);x=torch.tensor(X,dtype=torch.float32,device='cuda');target=torch.tensor(y,dtype=torch.long,device='cuda')
 model=nn.Sequential(nn.Linear(X.shape[1],64),nn.SiLU(),nn.Dropout(.25),nn.Linear(64,4)).cuda()
 opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.05);weight=torch.bincount(target[tr],minlength=4).float().clamp_min(1).rsqrt();weight/=weight.mean()
 best=1e9;state=None;history=[];start=time.monotonic();step_saved=0
 for step in range(1,1001):
  model.train();opt.zero_grad();logits=model(x[tr]+torch.randn_like(x[tr])*.008)
  loss=F.cross_entropy(logits,target[tr],weight=weight,label_smoothing=.1);loss.backward();opt.step()
  if step%50==0:
   model.eval()
   with torch.no_grad():v=float(F.cross_entropy(model(x[va]),target[va]))
   history.append(dict(step=step,validation_ce=v))
   if v<best:best=v;state=copy.deepcopy(model.state_dict());step_saved=step
 model.load_state_dict(state);model.eval()
 with torch.no_grad():p=model(x).softmax(1).cpu().numpy()
 torch.save(dict(model=state,classes=CLASSES,encoder='cached frozen CLIP raw + normalized',completed_updates=1000,retained_step=step_saved),ROOT/'classifier.pt')
 entries=[]
 for i,r in enumerate(records):
  split='train' if tr[i] else 'validation' if va[i] else 'test' if te[i] else 'excluded_exact_duplicate'
  pred=int(p[i].argmax());candidate=CLASSES[pred] if pred!=3 and p[i].max()>=.65 else '不确定'
  entries.append(dict(id=r['id'],image=r['image'],box=r.get('box'),label=r['label'],target=CLASSES[y[i]],split=split,
   group=group[i],image_sha256=hashes[i],probabilities=p[i],candidate=candidate,expert_ground_truth=False))
 report=dict(classes=CLASSES,completed_updates=1000,retained_step=step_saved,seconds=time.monotonic()-start,device=torch.cuda.get_device_name(),
  optimization_scope='new MLP head; CLIP backbone frozen, cached features reused',split_counts=dict(Counter(r['split'] for r in entries)),
  source_groups={s:sorted({r['group'] for r in entries if r['split']==s}) for s in ('train','validation','test')},
  original_source_group_and_exact_file_hash_split=True,near_duplicate_and_artist_family_leakage_not_ruled_out=True,
  validation=evaluate(y[va],p[va]),test=evaluate(y[te],p[te]),
  legacy_sources_and_test_probes_already_seen_in_previous_experiments=True,not_new_blind_test=True,expert_style_labels=0,
  cached_feature_origin=str(OLD/'features.npz'),cached_feature_sha256=sha(OLD/'features.npz'),records=entries,history=history)
 save(ROOT/'classification_report.json',report)
 board=Image.new('RGB',(1440,3*380),'#f4f1e8');d=ImageDraw.Draw(board)
 testrows=[r for r in entries if r['split']=='test']
 for i,r in enumerate(testrows):
  im=Image.open(r['image']).convert('RGB')
  if r['box']:
   w,h=im.size;im=im.crop(tuple(round(v*(w if k%2==0 else h)) for k,v in enumerate(r['box'])))
  im.thumbnail((450,280));xx=i%3*480;yy=i//3*380;board.paste(im,(xx+10,yy+65))
  d.text((xx+10,yy+8),'文章弱标签：'+r['label'],font=FONT,fill='#263e34');d.text((xx+10,yy+34),'模型候选：'+r['candidate'],font=FONT,fill='#263e34')
 board.save(ROOT/'classification_test.jpg',quality=94)
 print('CLASSIFIER_DONE',report['split_counts'],report['validation']['label_agreement'],report['test']['label_agreement'],flush=True)

def primitive(seed):
 rng=np.random.default_rng(seed);style=seed%3;t=np.linspace(0,1,17)
 if style==0:
  x=.40+.07*np.sin(t*rng.uniform(2,5))+rng.uniform(-.04,.04)*t;y=.12+.75*t
  width=(.06+.07*np.sin(np.pi*t))*rng.uniform(.7,1.3)
 elif style==1:
  x=.3+.26*t+.025*np.sin(t*4);y=.16+.67*t
  width=(.03+.46*np.sin(np.pi*t)**.6)*rng.uniform(.7,1.1)
 else:
  k=rng.uniform(.35,.6);x=np.interp(t,[0,k,1],[.15,.72,.70]);y=np.interp(t,[0,k,1],[.30,.32,.82]);width=(.06+.09*t)*rng.uniform(.7,1.2)
 q=np.column_stack([x,y]);angle=rng.uniform(-.25,.25);rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
 q=(q-.5)@rotation.T+.5+rng.uniform(-.08,.08,(1,2));q=np.clip(q,.04,.96)
 density=np.clip(rng.uniform(.35,.95)*(1-rng.uniform(0,.35)*t)+.05*np.sin(t*5),.1,1)
 dry=float(rng.uniform(0,.35));target=np.column_stack([q,width,density]).astype(np.float32)
 im=render(target,dry,seed);noise=rng.normal(0,rng.uniform(.1,1.4),im.shape);im=np.clip(im+noise,0,255).astype(np.uint8)
 return im,target,style,dry

def render(target,dry=0.,seed=0):
 scale=3;alpha=np.zeros((288,288),np.float32);rng=np.random.default_rng(seed)
 for i in range(len(target)-1):
  for v in np.linspace(0,1,8):
   p=target[i]*(1-v)+target[i+1]*v;layer=np.zeros(alpha.shape,np.float32)
   cv2.circle(layer,tuple(np.rint(p[:2]*96*scale).astype(int)),max(1,round(p[2]*12*scale/2)),float(p[3]),-1)
   alpha=np.maximum(alpha,layer)
 if dry:alpha*=np.where(rng.random(alpha.shape)<dry,.15,1)
 out=cv2.resize(255*(1-alpha),(96,96),interpolation=cv2.INTER_AREA)
 return np.repeat(out[:,:,None],3,2).astype(np.uint8)

class PrimitiveNet(nn.Module):
 def __init__(self):
  super().__init__();self.encoder=nn.Sequential(nn.Conv2d(3,16,5,2,2),nn.SiLU(),nn.Conv2d(16,32,3,2,1),nn.SiLU(),nn.Conv2d(32,48,3,2,1),nn.SiLU(),nn.AdaptiveAvgPool2d((4,4)),nn.Flatten());self.head=nn.Sequential(nn.Linear(768,160),nn.SiLU(),nn.Linear(160,69))
 def forward(self,x):return self.head(self.encoder(x)).sigmoid()

def reproduction():
 torch.manual_seed(110702);rng=np.random.default_rng(110702);start=time.monotonic()
 sets={}
 for split,offset,count in [('train',0,1200),('validation',200000,180),('test',400000,180)]:
  rows=[primitive(offset+i) for i in range(count)];xx=np.stack([r[0].transpose(2,0,1) for r in rows]);yy=np.stack([np.r_[r[1].flatten(),r[3]] for r in rows])
  sets[split]=(torch.tensor(xx,device='cuda').float()/255,torch.tensor(yy,device='cuda'),rows)
 model=PrimitiveNet().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0007);state=None;best=1e9;history=[]
 weight=torch.tensor([1.,1.,.4,.3]*17+[.2],device='cuda');x,y,_=sets['train']
 for step in range(1,1201):
  ids=torch.tensor(rng.integers(0,len(x),48),device='cuda');opt.zero_grad();out=model(x[ids]);loss=((out-y[ids]).abs()*weight).mean();loss.backward();opt.step()
  if step%100==0:
   with torch.no_grad():v=float(((model(sets['validation'][0])-sets['validation'][1]).abs()*weight).mean())
   history.append(dict(step=step,validation_loss=v))
   if v<best:best=v;state=copy.deepcopy(model.state_dict());retained=step
 model.load_state_dict(state);model.eval();xx,yy,rows=sets['test']
 with torch.no_grad():pred=model(xx).cpu().numpy()
 truth=yy.cpu().numpy();p=pred[:,:68].reshape(-1,17,4);target=truth[:,:68].reshape(-1,17,4)
 metrics=dict(center_mean_error_96px=float(np.linalg.norm(p[:,:,:2]-target[:,:,:2],axis=2).mean()*96),width_mae_96px=float(np.abs(p[:,:,2]-target[:,:,2]).mean()*12),
  opacity_mae=float(np.abs(p[:,:,3]-target[:,:,3]).mean()),dry_parameter_mae=float(np.abs(pred[:,-1]-truth[:,-1]).mean()))
 torch.save(dict(model=state,completed_updates=1200,retained_step=retained,output='17 continuous ordered xy,width,opacity samples + procedural dry parameter',not_pressure_or_water_measurements=True),ROOT/'primitive_model.pt')
 save(ROOT/'reproduction_report.json',dict(completed_updates=1200,retained_step=retained,seconds=time.monotonic()-start,device=torch.cuda.get_device_name(),
  data={s:len(v[2]) for s,v in sets.items()},split_seed_offsets=[0,200000,400000],exact_program_paths_not_skeleton_pseudolabels=True,
  style_analogues=CLASSES[:3],procedural_shapes_not_artist_gold=True,real_cunfa_reproduction_tested=False,robot_reproduction_tested=False,
  synthetic_test_metrics=metrics,history=history))
 board=Image.new('RGB',(1440,9*245),'#f4f1e8');d=ImageDraw.Draw(board);exports=[]
 for i in range(9):
  im,gt,style,dry=rows[i];prediction=p[i];rendered=render(prediction,float(pred[i,-1]),400000+i)
  pathim=np.full((96,96,3),255,np.uint8);cv2.polylines(pathim,[np.rint(prediction[:,:2]*96).astype(np.int32)],False,(30,100,190),1,cv2.LINE_AA)
  for j,(image,label) in enumerate([(im,'程序留出输入 · '+CLASSES[style]+'式基元'),(pathim,'模型预测连续中心路径'),(rendered,'预测宽度/浓淡的数字重绘')]):
   xx=j*480;yy=i*245;d.text((xx+10,yy+8),label,font=FONT,fill='#263e34');board.paste(Image.fromarray(image).resize((200,200)),(xx+100,yy+40))
  exports.append(dict(id=i,style_analogue=CLASSES[style],predicted_xy_width_opacity=prediction,dry_parameter=float(pred[i,-1]),physical_z=None,robot_ready=False))
 board.save(ROOT/'reproduction_test.jpg',quality=95);board.crop((0,0,1440,735)).save(ROOT/'pilot_preview.jpg',quality=95);save(ROOT/'synthetic_predicted_strokes.json',exports)
 print('REPRODUCTION_DONE',metrics,flush=True)

def main():
 ROOT.mkdir(exist_ok=True);torch.set_num_threads(4);cv2.setNumThreads(2)
 if not torch.cuda.is_available():raise RuntimeError('GPU required for this isolated training pilot')
 if (ROOT/'classifier.pt').exists() or (ROOT/'primitive_model.pt').exists():raise FileExistsError('Preserve completed experiment')
 save(ROOT/'protocol.json',dict(task='separate weak-real style classification and synthetic stroke inversion',classes=CLASSES,
  classification_sources=str(OLD/'report.json'),classification_features=str(OLD/'features.npz'),expert_style_labels=0,
  reconstruction_supervision='procedural exact ordered paths only',no_real_brush_dynamics_ground_truth=True,
  adoption_into_accepted_489_plan=False,robot_connected=False,initial_classifier_seed=110701,initial_primitive_seed=110702))
 classification();reproduction();c=read(ROOT/'classification_report.json');r=read(ROOT/'reproduction_report.json')
 html='''<!doctype html><meta charset="utf-8"><title>皴法识别与复刻：第一轮</title><style>body{font:16px/1.7 Microsoft YaHei,system-ui;background:#f4f1e8;color:#243e33;max-width:1450px;margin:auto;padding:24px}img{width:100%}pre{white-space:pre-wrap;background:white;padding:15px}</style><h1>皴法识别与复刻：第一轮GPU实验</h1><p>三类：披麻皴、斧劈皴、折带皴；另设其他/不确定。类别是整组纹理的弱标签，不能直接套到每条细线。原489笔绘画方案未修改，未连接机械臂。</p><h2>真实图片分类</h2><p>新训练MLP分类头1000次，使用既有冻结CLIP特征；非CLIP骨干微调。按教程/书籍来源隔离并排除跨组同文件哈希；这些旧来源和探针此前已被实验看过，不是新的盲测。无专家皴法标注，数字仅表示与文章/助手标签一致，非真实皴法准确率。0.65阈值只是研究拒识阈值，概率未校准。</p><pre>'''
 html+=json.dumps({k:c[k] for k in ('split_counts','source_groups','validation','test')},ensure_ascii=False,indent=2)
 html+='</pre><img src="classification_test.jpg"><h2>完整笔触数字重绘</h2><p>单独训练1200次：输入程序渲染的单笔，输出连续17点路径、宽度、浓淡与程序干笔参数。程序形状只是三类皴法的形态类比，不是画家示范；未证明真实图片分笔、整组皴法复刻或机械臂复刻。以下固定取测试前9笔，未挑选好例。</p><pre>'+json.dumps(r['synthetic_test_metrics'],ensure_ascii=False,indent=2)+'</pre><img src="reproduction_test.jpg"><p>图像像素笔宽和程序干笔参数没有换算成毛笔下压、速度、含墨量。需要真实单笔及成组示范的配对资料和实机标定，才能验证这一步。</p><p>参考：<a href="https://education.asianart.org/wp-content/uploads/sites/6/2019/09/Brushstrokes.pdf">Asian Art Museum笔触教学资料</a>；<a href="https://arxiv.org/abs/2308.11187">数字笔触与实机控制分阶段的机器人绘画研究</a>。这些参考不是本实验的专家真值。</p>'
 (ROOT/'index.html').write_text(html,encoding='utf8');(ROOT/'source_snapshot').mkdir(exist_ok=True);shutil.copy2(Path(__file__),ROOT/'source_snapshot'/Path(__file__).name)
 save(ROOT/'decision.json',dict(adopted=False,real_style_accuracy_established=False,real_cunfa_reproduction_established=False,robot_connected=False,new_classification_head_updates=1000,new_primitive_model_updates=1200))

if __name__=='__main__':main()
