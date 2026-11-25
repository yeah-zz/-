# -*- coding: utf-8 -*-
"""Fixed-budget, new-work-only curve training. Source-disjoint old work is held out.

Not a cunfa classifier. Candidate ink ownership is not reliable enough to train
an instance segmentation head. No temporal delta, guide or oracle seed at the
automatic discovery evaluation. Training diagnostics with known seeds are
reported separately and cannot establish autonomous extraction performance.
"""
import sys,json,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pima_rock_reference_20261109 import ROOT,LAB
from pima_static_model_20261109 import StaticTracer,features,training_rows,batch,lossfn,trace_many
from pima_static_discover_20261109 import proposals,evaluate_reference
from pima_single_train_20261109 import assess,unit
import numpy as np,cv2,torch
from PIL import Image,ImageDraw,ImageFont
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 torch.manual_seed(11093);torch.set_num_threads(4);rng=np.random.default_rng(11093)
 manifest=json.loads((ROOT/'reference_manifest.json').read_text('utf8'));admitted=[r for r in manifest['records'] if r['status']=='reviewed_candidate'];data=[]
 for r in admitted:
  z=np.load(ROOT/'references'/(r['id']+'.npz'));q=z['curve'].astype(np.float32)
  data.append(dict(id=r['id'],q=q,images=features(z['gray']),width=np.zeros(len(q)),width_valid=np.zeros(len(q))))
 rows=training_rows(data);end=np.array([i for i,r in enumerate(rows) if r['stop']]);normal=np.array([i for i,r in enumerate(rows) if not r['stop']]);net=StaticTracer().cuda();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);hist=[];started=time.time()
 for step in range(601):
  if step:
   ids=np.r_[rng.choice(normal,48),rng.choice(end,16)];net.train();b=batch(rows,ids,rng,True);opt.zero_grad();loss=lossfn(net,b);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),3);opt.step()
  if step%100==0:
   net.eval();scores=[]
   for r in data:
    q=r['q'];seed=dict(point=q[0].tolist(),direction=unit(q[2]-q[0]).tolist());p=trace_many(net,r['images'],[seed],max_steps=150)[0];scores.append(dict(id=r['id'],**assess(p['curve'],q),stop_reason=p['stop_reason']))
   hist.append(dict(step=step,known_seed_training_only=scores));print('TRAIN',step,scores,flush=True)
 torch.save(dict(model={k:v.cpu() for k,v in net.state_dict().items()},step=600,seed=11093,training_work=admitted[0]['source_work'],training_events=[r['id'] for r in admitted],checkpoint_selection='fixed 600 updates; no external selection',candidate_reference_not_expert_truth=True),ROOT/'new_work_static_tracer.pt')
 write(ROOT/'training_report.json',dict(real_training_events=len(data),train_source_works=1,original_artist_verified=False,history=hist,seconds=time.time()-started,width_head_trained=False,instance_ownership_head_trained=False,named_cunfa_classifier_trained=False,unlabelled_areas_treated_as_negative=False,adopted=False))
 # Only these final static images enter discovery; source references are read
 # afterwards for evaluation and never provide seeds, ROIs or tracing guides.
 old_manifest=json.loads((LAB/'pima_single_stroke_20261109'/'single_reference_manifest.json').read_text('utf8'))
 source_old=LAB/'pima_single_stroke_20261109'/'single_reference'/'S03_after.png'
 oldrgb=np.array(Image.open(source_old).convert('RGB'));newrgb=np.array(Image.open(ROOT/'references'/'group_after.png').convert('RGB'))
 oldrefs=[dict(id=r['id'],points=r['reference_curve'],annotation_uncertainty_px=5) for r in old_manifest['records'] if r['status']=='admitted_candidate']
 newrefs=[dict(id=r['id'],points=r['reference_curve'],annotation_uncertainty_px=5) for r in admitted]
 oldck=torch.load(LAB/'pima_static_discovery_20261109'/'static_tracer.pt',map_location='cpu',weights_only=False);oldnet=StaticTracer().cuda();oldnet.load_state_dict(oldck['model']);oldnet.eval();net.eval();results=[]
 for name,rgb,refs,role in [('new_work_training_scene',newrgb,newrefs,'training scene diagnostic'),('liupengkai_other_work',oldrgb,oldrefs,'source-disjoint from new weights; method development, not blind test')]:
  ims,sk,jnear,seeds,stats=proposals(rgb,max_seeds=80);Image.fromarray(rgb).save(ROOT/(name+'.jpg'))
  for label,model in [('old_model',oldnet),('new_model',net)]:
   raw=trace_many(model,ims,seeds,max_steps=250);curves=[dict(id=f'{name}_{i}',curve=r['curve'].tolist(),stop_reason=r['stop_reason']) for i,r in enumerate(raw)]
   matches=[evaluate_reference(curves,r) for r in refs];result=dict(scene=name,model=label,scene_role=role,stats=stats,curves=curves,reference_matches=matches,
     manual_seed_or_curve_at_inference=False,temporal_difference_at_inference=False,manual_rock_crop_at_inference=False,author_disjointness_verified=False,global_precision=None)
   results.append(result);preview(rgb,curves,ROOT/f'{name}_{label}.jpg',label);print('DISCOVERY',name,label,stats,matches,flush=True)
 write(ROOT/'discovery_report.json',dict(results=results,new_model_training_works=1,new_model_source_disjoint_evaluation_works=1,independent_author_test_works=0,
  evaluation_note='Old Liu work was used in previous method development. It is held out from these new weights only; no blind generalization claim.',adopted=False,robot_connected=False))
def preview(rgb,curves,path,label):
 im=Image.fromarray(rgb);d=ImageDraw.Draw(im);colors=['#237b98','#b27f35','#23815c','#815b95']
 for i,r in enumerate(curves):
  if len(r['curve'])>=5:d.line([tuple(p) for p in r['curve']],fill=colors[i%4],width=2)
 im.save(path,quality=95)
if __name__=='__main__':main()
