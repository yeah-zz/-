# -*- coding: utf-8 -*-
"""Report fixed-checkpoint results, including object-confusion regressions."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from guohua_objects_train_20261112 import pack,confusion,metrics
from guohua_objects_v2_train_20261113 import line_view
from gongbi_training.cunfa_supervision import partial_role_loss
ROOT=LAB/'guohua_objects_v2_20261113';OLD=LAB/'guohua_objects_pilot_20261112'
ARMS=['previous_frozen','expanded_control','line_aware_aug']
NAMES=['上一轮对象模型','扩充数据对照','线描／淡墨增强']
COLORS=np.array([[145,157,170],[183,134,75],[65,149,91]],'uint8')
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def font(n):return ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
def board(rows,out,notes):
    height=350*len(rows)+90;im=Image.new('RGB',(1440,height),'#f4f1e8');d=ImageDraw.Draw(im)
    for i,r in enumerate(rows):
        paths=[r['input']]+[r['predictions'][a] for a in ARMS]
        for j,(p,title) in enumerate(zip(paths,['原图']+NAMES)):
            x=Image.open(ROOT/p).convert('RGB');x.thumbnail((346,280));im.paste(x,(j*360+7,i*350+40));d.text((j*360+7,i*350+8),title,font=font(19),fill='#294735')
        label='已知树叶错例（非新测试）' if r['id']=='H_14' else r['id']
        d.text((8,i*350+322),label,font=font(17),fill='#70543d')
    for k,n in enumerate(notes):d.text((8,350*len(rows)+7+k*30),n,font=font(18),fill='#855137')
    im.save(ROOT/out,quality=94)
def main():
    ds=read(ROOT/'dataset_manifest.json');res=read(ROOT/'results.json');p=res['protocol'];(ROOT/'views').mkdir(exist_ok=True)
    assert sha(ROOT/'dataset_manifest.json')==p['dataset_sha256'];assert sha(ROOT/'selection_before_download.json')==p['selection_sha256'];assert sha(Path(__file__).with_name('guohua_objects_v2_train_20261113.py'))==p['script_sha256'];assert sha(OLD/'results.json')==p['old_result_sha256'];assert sha(OLD/'context_object_head.pt')==p['initial_model_sha256']
    initial=torch.load(OLD/'context_object_head.pt',map_location='cpu',weights_only=False)['model']
    for a,h in res['checkpoint_hashes'].items():
        assert sha(ROOT/(a+'.pt'))==h
        state=torch.load(ROOT/(a+'.pt'),map_location='cpu',weights_only=False)['model'];assert all(torch.equal(v,state[k]) for k,v in initial.items() if k.startswith('encoder.'));assert any(not torch.equal(v,state[k]) for k,v in initial.items() if k.startswith('head.'))
    rows=[]
    for r in ds['records']:
        assert sha(r['image'])==r['image_sha256'] and sha(r['target'])==r['target_sha256']
        if r['split'] not in ['new_test','old_named_regression']:continue
        im=np.array(Image.open(r['image']).convert('RGB'));t=np.array(Image.open(r['target']));ref=np.full_like(im,[229,223,210]);known=t!=255;ref[known]=COLORS[t[known]]
        Image.fromarray(im).save(ROOT/'views'/(r['id']+'_input.jpg'),quality=92);Image.fromarray(ref).save(ROOT/'views'/(r['id']+'_reference.png'));paths={};scores=next(x for x in res['records'] if x['id']==r['id'])['scores']
        for a in ARMS:
            z=np.load(ROOT/'predictions'/(r['id']+'_'+a+'.npz'));pred=z['prediction'];assert np.array_equal(confusion(t,pred),np.array(scores[a]['confusion']));Image.fromarray(COLORS[pred]).save(ROOT/'views'/(r['id']+'_'+a+'.png'));paths[a]='views/'+r['id']+'_'+a+'.png'
        rows.append(dict(id=r['id'],split=r['split'],work_group=r['work_group'],input='views/'+r['id']+'_input.jpg',reference='views/'+r['id']+'_reference.png',predictions=paths,scores=scores))
    h=np.array(Image.open(LAB/'cunfa_roles_v2_20261112/references/H_14_input.png').convert('RGB'));Image.fromarray(h).save(ROOT/'views/H_14_input.png');paths={}
    for a in ARMS:
        z=np.load(ROOT/'predictions'/('H_14_'+a+'.npz'));Image.fromarray(COLORS[z['prediction']]).save(ROOT/'views'/('H_14_'+a+'.png'));paths[a]='views/H_14_'+a+'.png'
    hr=dict(id='H_14',split='known_user_regression',work_group='previously_seen_tutorial',input='views/H_14_input.png',reference=None,predictions=paths,scores=None)
    rows.insert(0,hr);board([hr],'foliage_regression.jpg',['棕=山石，绿=树木，灰蓝=其他；完整原始预测，未用参考遮罩改色。','线描增强使树叶核心识别退步；色块是粗对象区域，不是皴笔、墨带或优化线稿。'])
    tests=[r for r in rows if r['split']=='new_test'];worst=sorted(tests,key=lambda r:r['scores']['line_aware_aug']['mean_iou'])[:3];board(worst,'new_test_failures.jpg',['新测试中线描增强模型平均IoU最低的3张；展示失败，未据此重训或挑检查点。','所有指标来自源粗区域标注，不是专家准确率；图中空白未标区域也可能误分。'])
    # Work-level aggregation prevents treating four crops of one work as four independent works.
    perwork=[]
    for work in sorted({r['work_group'] for r in tests}):
        perwork.append(dict(work_group=work,images=sum(r['work_group']==work for r in tests),scores={a:metrics(sum(np.array(r['scores'][a]['confusion']) for r in tests if r['work_group']==work)) for a in ARMS}))
    # Display a training-only approximation without implying new stroke ground truth.
    tr=next(r for r in ds['records'] if r['split']=='train' and r['id'].startswith('wang-shimin'))
    original=np.array(Image.open(tr['image']).convert('RGB'));Image.fromarray(original).save(ROOT/'views/augmentation_input.jpg');Image.fromarray(line_view(original)).save(ROOT/'views/augmentation_line.png')
    tiny=np.full((31,63,3),200,'uint8');ta=np.full((31,63),255,'uint8');ta[8:12,10:15]=2;tb=np.zeros_like(ta);x,t,_=pack(tiny,ta);assert np.array_equal(x,pack(tiny,tb)[0]);assert set(np.unique(t))=={2,255};logits=torch.randn(1,3,256,256,requires_grad=True);partial_role_loss(logits,torch.from_numpy(t.astype('int64'))[None]).backward();assert torch.count_nonzero(logits.grad[0,:,t==255])==0
    assert confusion(np.array([[255,1,2]],'uint8'),np.array([[2,1,2]],'uint8')).sum()==2
    splits=sorted({r['split'] for r in ds['records']})
    for i,s in enumerate(splits):
        for ss in splits[i+1:]:
            for key in ['work_group','artist_group','decoded_pixel_sha256']:
                assert not {r[key] for r in ds['records'] if r['split']==s}&{r[key] for r in ds['records'] if r['split']==ss}
    oldids={r['id'] for r in read(OLD/'dataset_manifest.json')['records']};assert all(r['id'] not in oldids for r in tests)
    promoted=[r for r in ds['records'] if r.get('previous_split')=='test_collection'];assert len(promoted)==8 and all(r['split']=='train' for r in promoted)
    assert not any(res[k] for k in ['model_adopted','object_formal_acceptance','cunfa_formal_acceptance','tree_skill_training_enabled','robot_connected'])
    checks=dict(hashes_match=True,old_artifacts_unchanged=True,frozen_encoders_unchanged=True,decoder_actually_changed=True,all_images_and_targets_match=True,displayed_confusions_recomputed_from_raw_predictions=True,work_artist_candidate_and_pixel_splits_disjoint=True,new_test_absent_from_old_dataset=True,old_collection_test_promoted_to_training_disclosed=True,packing_input_independent_of_target=True,void_preserved_and_loss_gradients_zero=True,void_excluded_from_metrics=True,raw_predictions_not_reference_masked=True,cunfa_trained=False,tree_skill_training_enabled=False,robot_connected=False,model_adopted=False)
    write('verification.json',checks);write('report_data.json',dict(records=rows,aggregates=res['aggregates'],user_case=res['user_case'],perwork=perwork,split_counts={s:sum(r['split']==s for r in ds['records']) for s in splits},concepts=read(ROOT/'source_checked_cunfa_concepts.json')))
    template=Path(__file__).with_name('guohua_objects_v2_report_20261113.html').read_text('utf8');(ROOT/'index.html').write_text(template,encoding='utf8');print(json.dumps(checks,ensure_ascii=False))
if __name__=='__main__':main()
