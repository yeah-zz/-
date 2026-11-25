# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,shutil,numpy as np,torch
from torch.nn import functional as F
from PIL import Image
from cunfa_group_20261107 import ROOT,GroupNet,prepare_real,gallery,read,save,sha,SLOTS,POINTS,PARAMS

def main():
 torch.set_num_threads(4);report=read(ROOT/'training_report.json');z=np.load(ROOT/'test_predictions.npz');pred=z['real'];inputs=z['real_inputs']
 real=prepare_real();test=[r for r in real if r['split']=='test'];train=[r for r in real if r['split']=='train'];val=[r for r in real if r['split']=='validation']
 assert not {r['source_group'] for r in train}&{r['source_group'] for r in test+val}
 assert not {r['image_sha256'] for r in train}&{r['image_sha256'] for r in test+val}
 assert all(r['expert_paths'] is False for r in real)
 ck=torch.load(ROOT/'group_model.pt',map_location='cpu',weights_only=False);model=GroupNet().cuda();model.load_state_dict(ck['model']);model.eval()
 u=F.one_hot(torch.full((len(inputs),),3,dtype=torch.long,device='cuda'),4).float()
 with torch.no_grad():replayed=model(torch.tensor(inputs,device='cuda'),u).cpu().numpy()
 assert np.allclose(pred,replayed,atol=1e-5)
 assert pred.shape==(6,SLOTS,PARAMS) and np.isfinite(pred).all() and (pred>=0).all() and (pred<=1).all()
 counts=(pred[:,:,36]>=.5).sum(1);assert min(counts)>1
 truth=z['synthetic_truth'];assert truth[:,:,36].sum(1).min()>=4 and truth[:,:,36].sum(1).max()<=10
 gallery(test,pred,ROOT/'real_group_test.jpg',True)
 syn=[]
 for i in range(6):
  from cunfa_group_20261107 import procedural,render_numpy,CLASSES
  a,style=procedural(400000+i);im=render_numpy(a);syn.append(dict(raw=Image.fromarray(im).convert('RGB'),appearance=im,label=CLASSES[style]))
 gallery(syn,z['synthetic'][:6],ROOT/'synthetic_group_test.jpg',False)
 Image.open(ROOT/'real_group_test.jpg').crop((0,0,1600,930)).save(ROOT/'real_group_preview.jpg',quality=95)
 report['blank_real_test_weak_appearance_l1']=float((1-inputs).mean());report['best_checkpoint_before_real_finetuning']=ck['retained_step']==1200
 save(ROOT/'training_report.json',report)
 page=ROOT/'index.html';html=page.read_text('utf8');status='<p id="quality-result" style="padding:16px;background:#ead9ba"><b>真实成组皴法复刻：未通过。</b> 候选虽已从单笔改为多笔，但真实局部仍大多退化为排线，未还原山体上的皴笔组织。600次真实弱监督微调未超过预训练；最终保留第1200步程序预训练权重，未采用微调版本。没有改变489笔绘画方案。</p>'
 if 'id="quality-result"' not in html:html=html.replace('<h1>从单笔改为成组笔触</h1>','<h1>从单笔改为成组笔触</h1>'+status)
 html+='<p>补充弱外观对照：空白重绘误差 '+str(round(report['blank_real_test_weak_appearance_l1'],4))+'，当前候选 '+str(round(report['real_test_weak_appearance_l1'],4))+'。该指标不证明皴法、笔顺或分笔正确。</p>'
 page.write_text(html,encoding='utf8')
 original=read(LAB/'object_hierarchy_20261107/G03_01/plan.json');assert sha(Path(original['parent']))==original['parent_sha256']
 save(ROOT/'verification.json',dict(passed=True,saved_model_replay_verified=True,real_style_condition_unknown=True,
  group_and_hash_splits_disjoint=True,real_order_and_paths_not_used_as_labels=True,real_predicted_stroke_counts=counts,
  completed_updates=1800,retained_step=ck['retained_step'],real_reproduction_passed=False,accepted_drawing_unchanged=True,robot_connected=False))
 for name in ('cunfa_group_20261107.py','finish_cunfa_group_20261107.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/'source_snapshot'/name)
 save(ROOT/'decision.json',dict(adopted=False,group_output_implemented=True,real_reproduction_passed=False,
  real_finetuned_weights_adopted=False,retained_step=ck['retained_step'],robot_connected=False))
 print('GROUP_VERIFIED',counts.tolist(),'retained',ck['retained_step'],flush=True)

if __name__=='__main__':main()
