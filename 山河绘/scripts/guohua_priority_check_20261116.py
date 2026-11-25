# -*- coding: utf-8 -*-
import sys,json,hashlib,re,datetime,concurrent.futures
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,torch
from PIL import Image
from guohua_hierarchy_source_audit_20261112 import fetch
from gongbi_training.scene_objects import SceneObjects,scene_metrics
from gongbi_training.source_scene_mapping import source_to_scene
from gongbi_training.rock_candidate_gate import rock_candidate_mask,cunfa_candidate_route
from guohua_objects_train_20261112 import predict
ROOT=LAB/'guohua_priority_check_20261116';SEG=LAB/'guohua_scene4_mapping_fix_20261116';FRESH=LAB/'cunfa_fresh_validation_20261111';POOL=LAB/'guohua_faint_sources_20261116'
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def test_gate():
    p=np.zeros((4,3,4),dtype='float32');p[0]=1;p[:,0,0]=[.02,.9,.04,.04];p[:,0,1]=[.01,.49,.49,.01];p[:,0,2]=[.01,.01,.97,.01]
    m=rock_candidate_mask(p);assert m.sum()==1 and m[0,0];assert not m[0,1] and not m[0,2]
    r=cunfa_candidate_route(p,'pima');assert not r['eligible_for_exploratory_cunfa'] and r['style_proposal'] is None and r['cunfa']=='unknown' and not r['object_verified']
    q=np.tile(np.array([.02,.9,.04,.04],dtype='float32')[:,None,None],(1,3,4));r=cunfa_candidate_route(q,'pima');assert r['eligible_for_exploratory_cunfa'] and r['style_proposal']=='pima' and r['cunfa']=='unknown' and not r['object_verified']
    try:rock_candidate_mask(q*2)
    except ValueError:pass
    else:raise AssertionError('Invalid probability accepted')
    return dict(tree_and_ties_not_admitted=True,candidate_never_becomes_verified=True,style_proposal_never_becomes_reference=True,invalid_probabilities_rejected=True)
def prepare():
 ROOT.mkdir(exist_ok=True);assert not (ROOT/'protocol.json').exists(),'Frozen protocol';sel=read(POOL/'selection_before_download.json');groups={}
 for path in sel['reserved_paths']:groups.setdefault(re.sub(r'_\d+$','',Path(path).stem),[]).append(path)
 selected=[]
 for g,paths in sorted(groups.items()):
  paths=sorted(paths);selected.extend(dict(path=paths[i],work_group='filename:'+g) for i in [0,len(paths)-1])
 assert len(groups)==2 and len(selected)==4
 p=dict(frozen_before_download_and_prediction=True,prepared_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),segmentation_sha256=sha(SEG/'mapping_fixed.pt'),segmentation_results_sha256=sha(SEG/'results.json'),frozen_style_predictions_sha256=sha(FRESH/'predictions.json'),style_manifest_sha256=sha(FRESH/'test_manifest.json'),script_sha256=sha(__file__),gate_module_sha256=sha(Path(__file__).resolve().parents[1]/'gongbi_training/rock_candidate_gate.py'),scene_revision=sel['revision'],selected=selected,selection_rule='First and last lexicographic path in each of2 previously metadata-only whole work candidate groups; no preview',gate=dict(confidence=.75,margin=.20,minimum_rock_fraction=.65),gate_not_probability_calibration=True,style_scope='Reuse13 known source-backed label regions and unchanged frozen class proposals; partition input is2.5x context with provided local box; not autonomous stroke-group detection',no_new_model_training=True,formal_acceptance=False,expert_truth=False,opened_work_groups_including_unselected_crops=sorted(groups),reserved_pool_status='Both whole work groups now opened for test; remaining crops are not a new independent test in future',robot_connected=False);write('protocol.json',p);write('gate_checks.json',test_gate());tasks=[];rows=[];base='https://huggingface.co/datasets/Rane7/CLP_dataset/resolve/'+sel['revision']+'/'
 for q in selected:
  ident=Path(q['path']).stem;split=q['path'].split('/')[-2];image=ROOT/'raw'/(ident+'.jpg');mask=ROOT/'raw'/(ident+'_source.png');mp='clp2k/annotations/'+split+'/'+ident+'.png';tasks.extend([(base+q['path'],image),(base+mp,mask)]);rows.append(dict(id=ident,work_group=q['work_group'],image=str(image),source_mask=str(mask),image_url=base+q['path'],mask_url=base+mp))
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
  fs=[ex.submit(fetch,u,t,3_000_000) for u,t in tasks]
  for f in fs:f.result()
 old=read(LAB/'guohua_objects_v2_20261113/dataset_manifest.json')['records']
 for r in rows:
  im=Image.open(r['image']).convert('RGB');raw=np.array(Image.open(r['source_mask']));target=source_to_scene(raw);assert im.size==(target.shape[1],target.shape[0]);path=ROOT/'raw'/(r['id']+'_target.png');Image.fromarray(target).save(path);r.update(target=str(path),image_sha256=sha(r['image']),source_sha256=sha(r['source_mask']),decoded_pixel_sha256=hashlib.sha256(np.array(im).tobytes()).hexdigest(),target_sha256=sha(path),size=list(im.size))
 assert not ({r['decoded_pixel_sha256'] for r in rows}&{r['decoded_pixel_sha256'] for r in old});assert not ({r['work_group'] for r in rows}&{r['work_group'] for r in old})
 write('test_manifest.json',dict(records=rows,protocol_sha256=sha(ROOT/'protocol.json'),source_reference='Coarse public source masks with corrected project mapping, no expert certification',author_or_original_work_identity_authenticated=False,new_for_this_object_experiment=True,not_new_project_wide_author_certification=True,robot_connected=False));print('NEW_WORK_CHECK_READY',len(rows),flush=True)
def gate_counts(target,pred,admit):
 known=target!=255;rock=target==1;tree=target==2;chosen=admit&known;n=int(chosen.sum());tp=int((chosen&rock).sum());return dict(selected_known_pixels=n,rock_precision=float(tp/n) if n else None,rock_recall=float(tp/rock.sum()) if rock.any() else None,tree_admitted_fraction=float((chosen&tree).sum()/tree.sum()) if tree.any() else None,known_pixel_coverage=float(n/known.sum()) if known.any() else None,confident_map_is_not_truth=True)
def evaluate():
 assert not (ROOT/'results.json').exists(),'No repeat on new test';p=read(ROOT/'protocol.json');assert sha(__file__)==p['script_sha256'];assert sha(SEG/'mapping_fixed.pt')==p['segmentation_sha256'];assert sha(FRESH/'predictions.json')==p['frozen_style_predictions_sha256'];ds=read(ROOT/'test_manifest.json');assert ds['protocol_sha256']==sha(ROOT/'protocol.json');torch.set_num_threads(4);net=SceneObjects().cuda();net.load_state_dict(torch.load(SEG/'mapping_fixed.pt',map_location='cuda',weights_only=False)['model']);net.eval();(ROOT/'predictions').mkdir(exist_ok=True);tests=[]
 for r in ds['records']:
  assert sha(r['image'])==r['image_sha256'] and sha(r['target'])==r['target_sha256'];im=np.array(Image.open(r['image']).convert('RGB'));target=np.array(Image.open(r['target']));prob,pred=predict(net,im);admit=rock_candidate_mask(prob);np.savez_compressed(ROOT/'predictions'/(r['id']+'.npz'),prediction=pred,probability=prob.astype('float16'),rock_candidate=admit);tests.append(dict(id=r['id'],work_group=r['work_group'],scores=scene_metrics(target,pred),raw_rock=gate_counts(target,pred,pred==1),candidate_rock=gate_counts(target,pred,admit)))
 # Same13 already-viewed classifier cases; no classifier refit, masking or logits changed.
 proposals={r['id']:r for r in read(FRESH/'predictions.json')['records']};rows=[r for r in read(FRESH/'test_manifest.json')['records'] if r['admission']=='source_label_test'];assert len(rows)==13;style=[]
 for r in rows:
  assert sha(r['source_image'])==r['source_sha256'] and sha(r['image'])==r['crop_sha256'];im=Image.open(r['source_image']).convert('RGB');x0,y0,x1,y1=r['source_box'];cx=(x0+x1)/2;cy=(y0+y1)/2;box=[max(0,int(cx-(x1-x0)*1.25)),max(0,int(cy-(y1-y0)*1.25)),min(im.width,int(cx+(x1-x0)*1.25)),min(im.height,int(cy+(y1-y0)*1.25))];context=im.crop(box);prob,pred=predict(net,np.array(context));local=prob[:,y0-box[1]:y1-box[1],x0-box[0]:x1-box[0]];assert local.shape[1:]==(y1-y0,x1-x0);assert np.array_equal(np.array(im.crop(tuple(r['source_box']))),np.array(Image.open(r['image']).convert('RGB')))
  proposal=proposals[r['id']]['predictions']['local_plus_context'];route=cunfa_candidate_route(local,proposal['predicted'],**p['gate']);context_path=ROOT/'predictions'/(r['id']+'_context.png');context.save(context_path);np.savez_compressed(ROOT/'predictions'/(r['id']+'.npz'),prediction=pred,rock_candidate=rock_candidate_mask(prob));style.append(dict(id=r['id'],work_id=r['work_id'],source_label=r['label'],source_url=r['source_url'],image=r['image'],context=str(context_path),context_box=box,provided_local_box=r['source_box'],raw_style_prediction=proposal['predicted'],raw_style_margin=proposal['signed_margin'],source_label_agreement=proposal['predicted']==r['label'],route=route))
 accepted=[r for r in style if r['route']['eligible_for_exploratory_cunfa']];labels=sorted({r['source_label'] for r in style});summary=dict(original_regions=len(style),retained_regions=len(accepted),coverage=len(accepted)/len(style),raw_agreement=sum(r['source_label_agreement'] for r in style)/len(style),retained_agreement=sum(r['source_label_agreement'] for r in accepted)/len(accepted) if accepted else None,by_class={label:dict(total=sum(r['source_label']==label for r in style),retained=sum(r['source_label']==label for r in accepted)) for label in labels},not_new_classifier_improvement=True,not_independent_style_test=True,style_predictions_unchanged=True,all_rejected_cases_preserved=True,not_full_frame_stroke_detection=True)
 write('results.json',dict(protocol=p,new_work_partition=tests,style_records=style,style_summary=summary,model_adopted=False,object_formal_validation=False,cunfa_formal_validation=False,tree_skill_training=False,robot_connected=False));print(json.dumps(dict(new_work_partition=tests,style_summary=summary),ensure_ascii=False),flush=True)
if __name__=='__main__':
 if '--prepare' in sys.argv:prepare()
 else:evaluate()
