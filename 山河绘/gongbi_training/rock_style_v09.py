# -*- coding: utf-8 -*-
"""Regularized, region-level non-expert cunfa prototype; frozen local CLIP."""
import sys,json,math,hashlib
from pathlib import Path
import torch,numpy as np
from PIL import Image
from .runtime import LAB
from .cunfa_prepare import ink_target

CLASSES=['披麻皴','斧劈皴','荷叶皴','折带皴','解索皴','卷云皴','米点皴','雨点皴']
def crop_record(r):
    im=Image.open(r['image']).convert('RGB')
    if 'box' in r:
        w,h=im.size;im=im.crop(tuple(round(v*(w if k%2==0 else h)) for k,v in enumerate(r['box'])))
    return im
def ink_crop(im):
    t,_=ink_target(np.asarray(im));a=t[0];yy,xx=np.where(a>45)
    if len(xx):a=a[max(0,int(yy.min())-6):min(a.shape[0],int(yy.max())+7),max(0,int(xx.min())-6):min(a.shape[1],int(xx.max())+7)]
    return Image.fromarray(255-a).convert('RGB')
def metrics(y,p):
    pred=p.argmax(1);confusion=np.zeros((8,8),int)
    for a,b in zip(y,pred):confusion[a,b]+=1
    f1=[];present=[]
    for k in range(8):
        tp=confusion[k,k];fp=confusion[:,k].sum()-tp;fn=confusion[k].sum()-tp;v=2*tp/max(1,2*tp+fp+fn);f1.append(v)
        if confusion[k].sum():present.append(v)
    accepted=p.max(1)>=.65
    return dict(top1_agreement=float((pred==y).mean()),macro_f1_all8=float(np.mean(f1)),macro_f1_present_classes=float(np.mean(present)),confusion=confusion.tolist(),abstention_threshold=.65,accepted_fraction=float(accepted.mean()),agreement_on_accepted=float((pred[accepted]==y[accepted]).mean()) if accepted.any() else None,probabilities_calibrated=False)
def train(root,old_root):
    sys.path.append(r'E:\Codex_Workspace_Data\CLIPascene_20261021\environment\deps');sys.path.insert(1,r'E:\Codex_Workspace_Data\CLIPascene_20261021\source\assets\CLIP-main')
    import clip
    torch.set_num_threads(4);torch.manual_seed(9024)
    folder=root/'style_classifier';folder.mkdir(exist_ok=True)
    if (folder/'report.json').exists():raise FileExistsError('Preserve classifier experiment')
    old=json.loads((old_root/'cunfa_classifier_normalized'/'evaluation.json').read_text(encoding='utf-8'));records=old['records'];fresh=json.loads((root/'frozen_manifest.json').read_text(encoding='utf-8'))['new_style_probes']
    rows=[dict(r,label=r['label'],cohort='train' if r['split']=='train' else 'legacy_validation') for r in records]
    rows +=[dict(r,label=r['method'],cohort='new_source_probe') for r in fresh]
    model,preprocess=clip.load(r'E:\Codex_Workspace_Data\CLIPascene_20261021\source\assets\ViT-B-32.pt',device='cuda',jit=False);model.eval();raw=[];normal=[]
    with torch.no_grad():
        for start in range(0,len(rows),6):
            batch=rows[start:start+6];original=[crop_record(r) for r in batch];normalized=[ink_crop(im) for im in original]
            a=model.encode_image(torch.stack([preprocess(im) for im in original]).cuda()).float();b=model.encode_image(torch.stack([preprocess(im) for im in normalized]).cuda()).float();raw.append(torch.nn.functional.normalize(a,dim=1).cpu());normal.append(torch.nn.functional.normalize(b,dim=1).cpu());print('STYLE_FEATURES',min(start+6,len(rows)),len(rows),flush=True)
    del model;torch.cuda.empty_cache();raw=torch.cat(raw);normal=torch.cat(normal);y=torch.tensor([CLASSES.index(r['label']) for r in rows]);tr=torch.tensor([r['cohort']=='train' for r in rows]);va=torch.tensor([r['cohort']=='legacy_validation' for r in rows]);te=torch.tensor([r['cohort']=='new_source_probe' for r in rows])
    # The new source images are encoded without labels influencing optimization.
    # Every model choice uses only the previously used legacy validation set.
    candidates=[];best=None;best_value=float('inf')
    for variant,x in [('normalized_only',normal),('raw_plus_normalized',torch.cat((raw,normal),1)/math.sqrt(2))]:
        for decay in (.02,.2,2.):
            torch.manual_seed(9024);head=torch.nn.Linear(x.shape[1],8);opt=torch.optim.AdamW(head.parameters(),lr=.012,weight_decay=decay);counts=torch.bincount(y[tr],minlength=8).float();class_weight=torch.sqrt(counts.sum()/counts.clamp_min(1));class_weight/=class_weight.mean();selected=None;score=float('inf')
            for step in range(1,1001):
                opt.zero_grad();jitter=x[tr]+torch.randn_like(x[tr])*.006;loss=torch.nn.functional.cross_entropy(head(jitter),y[tr],weight=class_weight,label_smoothing=.08);loss.backward();opt.step()
                if step%50==0:
                    with torch.no_grad():v=float(torch.nn.functional.cross_entropy(head(x[va]),y[va]))
                    if v<score:score=v;selected={k:t.detach().clone() for k,t in head.state_dict().items()};selected_step=step
            candidates.append(dict(variant=variant,weight_decay=decay,selected_step=selected_step,legacy_validation_ce=score,updates=1000))
            if score<best_value:best_value=score;best=dict(head=selected,variant=variant,weight_decay=decay,selected_step=selected_step,dimension=x.shape[1])
    x=normal if best['variant']=='normalized_only' else torch.cat((raw,normal),1)/math.sqrt(2);head=torch.nn.Linear(best['dimension'],8);head.load_state_dict(best['head'])
    with torch.no_grad():prob=head(x).softmax(1).numpy()
    baseline_data=torch.load(old_root/'cunfa_classifier_normalized'/'classifier.pt',map_location='cpu',weights_only=False);baseline=torch.nn.Linear(512,8);baseline.load_state_dict(baseline_data['head'])
    with torch.no_grad():base_prob=baseline(normal).softmax(1).numpy()
    truth=y.numpy();a=va.numpy();b=te.numpy()
    report=dict(classes=CLASSES,training_examples=int(tr.sum()),legacy_validation_examples=int(va.sum()),new_source_probe_examples=int(te.sum()),legacy_validation_reused_for_development=True,new_source_probe_groups=1,expert_labels=0,candidates=candidates,selection={k:v for k,v in best.items() if k!='head'},previous_legacy_validation=metrics(truth[a],base_prob[a]),new_legacy_validation=metrics(truth[a],prob[a]),previous_new_source_probe=metrics(truth[b],base_prob[b]),new_source_probe=metrics(truth[b],prob[b]),interpretation='agreement with teaching article / assistant labels only; NOT expert cunfa accuracy; coarse local image type, not per-stroke identity')
    for r,p,q in zip(rows,prob,base_prob):r.update(predicted=CLASSES[int(p.argmax())],candidate=CLASSES[int(p.argmax())] if p.max()>=.65 else '不确定',score=float(p.max()),previous_predicted=CLASSES[int(q.argmax())],probabilities=p.tolist())
    report['records']=rows;(folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');torch.save(dict(**best,classes=CLASSES,encoder='frozen local CLIP ViT-B/32',expert_labels=0,unknown_threshold=.65),folder/'classifier.pt');np.savez_compressed(folder/'features.npz',raw=raw.numpy(),normalized=normal.numpy(),labels=truth)
    print('STYLE_COMPLETE',json.dumps({k:report[k] for k in ('new_legacy_validation','new_source_probe')},ensure_ascii=True),flush=True)
