# -*- coding: utf-8 -*-
"""Autonomous full-image experimental readout. Existing models, no new training."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,hashlib,shutil,colorsys,time,argparse,copy,gc
from collections import Counter
import numpy as np,cv2,torch
from PIL import Image,ImageDraw,ImageFont
from full_image_ownership_core_20261105 import CONFIG,source_ink,ink_graph,automatic_queries,junction_votes,read_paths,curve_metrics,compare_geometry
from hierarchical_paint_core_20261105 import object_ownership,structural_roles,hierarchy_schedule,order_metrics
from drawing_budget_core_20261105 import select_budgets
from module_stroke_core_20261104 import samples,length
import whole_stroke_instance_20261105 as instance

ROOT=LAB/'autonomous_full_image_20261105';BASE=LAB/'drawing_budgets_20261105'
SCENES=('G03_01','G05_03','T02','G01_01','G02_02','G04_02')
INPUTS={i:LAB/'simple_landscape_specialist_20261029/round2/imagegen_holdouts'/(i+'.png') for i in SCENES}
INPUTS['T02']=LAB/'stability_brush_preview_20261101/originals/T02.png'
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18);SMALL=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)

def read(p):return json.loads(p.read_text('utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def serial(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=serial,allow_nan=False),encoding='utf8')
def color(i):return tuple(round(v*190) for v in colorsys.hsv_to_rgb((i*.61803398875)%1,.8,1))

def module_masks(ident,rgb,sam):
    folder=ROOT/ident
    if (folder/'sam_manifest.json').exists():
        manifest=read(folder/'sam_manifest.json');assert manifest['source_sha256']==sha(INPUTS[ident])
        return np.load(folder/'sam_masks.npz'),manifest['records']
    old=LAB/'module_stroke_planner_20261104/outputs'/ident
    if (old/'sam_manifest.json').exists():
        m=read(old/'sam_manifest.json');oldim=np.array(Image.open(m['source']).convert('RGB').resize((496,496),Image.Resampling.LANCZOS))
        newim=cv2.resize(rgb,(496,496),interpolation=cv2.INTER_AREA)
        if np.abs(oldim.astype(float)-newim).mean()<1.5:
            shutil.copy2(old/'sam_masks.npz',folder/'sam_masks.npz')
            m.update(source_sha256=sha(INPUTS[ident]),native_source=str(INPUTS[ident]),cache_reused=True,no_manual_prompts=True)
            save(folder/'sam_manifest.json',m);return np.load(folder/'sam_masks.npz'),m['records']
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        generated=sam.generate(rgb)
    np.savez_compressed(folder/'sam_masks.npz',**{f'm{i:03}':r['segmentation'] for i,r in enumerate(generated)})
    records=[dict(key=f'm{i:03}',area=r['area'],bbox=r['bbox'],predicted_iou=r['predicted_iou'],stability_score=r['stability_score']) for i,r in enumerate(generated)]
    save(folder/'sam_manifest.json',dict(source=str(INPUTS[ident]),source_sha256=sha(INPUTS[ident]),records=records,
        cache_reused=False,no_manual_prompts=True,pretrained_not_finetuned=True,seed_grid='24x24 RGB only',
        device=str(sam.predictor.device),precision='fp16 autocast',runtime_note='Float32 SAM allocation failed; reduced precision and batch=4'))
    return np.load(folder/'sam_masks.npz'),records

def draw(rows,colored=False):
    im=np.full((992,992,3),255,np.uint8)
    for r in rows:
        cv2.polylines(im,[np.rint(np.asarray(r['points'])*2).astype(np.int32)],False,color(r['module']) if colored else (30,41,35),1,cv2.LINE_AA)
    return Image.fromarray(im)

def plain_control(graph,modules,factor):
    r,p=read_paths(graph,{},modules,factor)
    for a in r+p:a['points']=a['points']/factor
    return r,p

def module_records(modules,rows,order):
    out=[]
    for m in modules:
        r={k:v for k,v in m.items() if k!='mask'};contours,_=cv2.findContours(m['mask'].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        r['outlines']=[c.reshape(-1,2).astype(float).tolist() for c in contours if len(c)>2]
        r.update(name=f'对象候选 {m["id"]+1:02}' if m['sam_key'] else f'归属待定 {m["id"]+1:02}',
            planned_strokes=sum(a['module']==m['id'] for a in rows),order=order.index(m['id'])+1 if m['id'] in order else 0)
        out.append(r)
    return out

def make_plan(ident,rows,modules,control,metadata):
    route,order=hierarchy_schedule(rows,modules)
    original=[dict(module=r['module'],air_travel_px=0) for r in rows]
    st=dict(before=curve_metrics(control),after=curve_metrics(rows),accepted_continuations=sum(r.get('learned_junctions',0) for r in rows),
        object_proposals=sum(m['sam_key'] is not None for m in modules),unresolved_regions=sum(m['sam_key'] is None for m in modules),
        baseline_order=order_metrics(original),hierarchy_order=order_metrics(route))
    return dict(schema='hierarchical-image-plan-v1',id=ident,width=496,height=496,source=str(INPUTS[ident]),source_sha256=sha(INPUTS[ident]),
        curves=rows,modules=module_records(modules,rows,order),actions=route,module_order=order,stats=st,
        ready_for_robot=False,adopted=False,new_weights_trained=False,model_predicts_masks_not_coordinates=True,
        reference_files_read_at_extraction=False,old_baseline_used_at_extraction=False,semantic_object_labels_verified=False,
        traditional_cun_style_verified=False,**metadata)

def build(ident,net,sam):
    start=time.monotonic();folder=ROOT/ident;folder.mkdir(exist_ok=True)
    rgb=np.array(Image.open(INPUTS[ident]).convert('RGB'));factor=max(rgb.shape[:2])/496.
    assert rgb.shape[0]==rgb.shape[1], 'Viewer uses square comparison coordinates'
    Image.fromarray(rgb).resize((496,496),Image.Resampling.LANCZOS).save(folder/'source.png')
    binary=source_ink(rgb);Image.fromarray(np.uint8(binary)*255).save(folder/'native_ink.png')
    graph=ink_graph(binary);chains=graph['chains']
    normalized=[r['points']/factor for r in chains]
    masks,recs=module_masks(ident,rgb,sam)
    rgb496=np.array(Image.open(folder/'source.png'))
    modules,ownership=object_ownership(rgb496,normalized,masks,recs);masks.close()
    owners=[o['module'] for o in ownership]
    queries=automatic_queries(graph,factor)
    print('FULL_IMAGE',ident,'native',rgb.shape[:2],'chains',len(chains),'queries',len(queries),'modules',len(modules),flush=True)
    cache=folder/'raw_masks.npz';queryfile=folder/'automatic_queries.json'
    if cache.exists():
        meta=read(queryfile);assert meta['source_sha256']==sha(INPUTS[ident]);assert meta['count']==len(queries)
        zz=np.load(cache);pred=[]
        for i,q in enumerate(queries):pred.append(dict(mask=zz['masks'][i].astype(np.float32),presence=float(zz['presence'][i]),query=q,origin=np.asarray(q['center'])-q['side']/2,side=q['side']))
    else:
        # Bound host memory: infer_queries otherwise retains every RGB crop.
        pred=[]
        for begin in range(0,len(queries),96):
            pred.extend(instance.infer_queries(net,rgb,queries[begin:begin+96]))
        np.savez_compressed(cache,masks=np.stack([p['mask'] for p in pred]).astype(np.float16),presence=np.array([p['presence'] for p in pred],np.float32))
        save(queryfile,dict(queries=queries,count=len(queries),source_sha256=sha(INPUTS[ident]),model_sha256=sha(instance.ROOT/'instance_curve.pt'),
            manual_or_reference_prompts=False,old_extraction_used=False))
    links,diag=junction_votes(graph,pred,owners,factor)
    learned,pending=read_paths(graph,links,owners,factor)
    for a in learned+pending:a['points']=a['points']/factor
    raw,raw_pending=plain_control(graph,owners,factor)
    structural_roles(rgb496,learned,modules)
    # Stable IDs refer to raw native ink graph chains, not old baseline segments.
    for r in learned:r['complete_curve_id']='ink-'+','.join(str(i) for i in r['source_curve_ids'])
    save(folder/'pending.json',dict(curves=pending,unresolved_junctions=[d for d in diag if d['status']=='unresolved_junction'],
        note='Pending texture is not labelled false ink; no guessed connectors across gaps'))
    save(folder/'junction_diagnostics.json',diag)
    metadata=dict(native_size=rgb.shape[1::-1],native_to_viewer_scale=factor,algorithm='native source ink graph + frozen learned instance ownership at junctions',
                  raw_query_count=len(queries),raw_graph_chain_count=len(chains),pending_count=len(pending),parameter_protocol_sha256=sha(ROOT/'protocol.json'))
    plan=make_plan(ident,learned,modules,raw,metadata);save(folder/'plan.json',plan)
    draw(learned).save(folder/'extraction.png');draw(learned,True).save(folder/'modules.png')
    draw(raw).save(folder/'source_control.png')
    baselines={}
    # Old plans are first read here, after every new curve has been finalized.
    for budget in (500,700):
        p=BASE/f'{ident}_{budget}'/'plan.json'
        if p.exists():
            old=read(p);source=np.array(Image.open(p.with_name('source.png')).convert('RGB'))
            assert np.abs(source.astype(float)-rgb496.astype(float)).mean()<1.5
            baselines[budget]=old
            metrics=compare_geometry(old['curves'],learned);metrics['baseline_plan_sha256']=sha(p)
            metadata[f'baseline_{budget}_comparison']=metrics
    save(folder/'plan.json',plan)
    outputs=select_budgets(learned,(500,700))
    scenes=[dict(id=ident,label=f'{ident} · 完整提取{len(learned)}笔候选',count=len(learned),input_count=len(raw),new_candidate=True)]
    for budget,selection in outputs.items():
        key=f'{ident}_{budget}';dest=ROOT/key;dest.mkdir(exist_ok=True)
        selected=[copy.deepcopy(learned[i]) for i in selection['selected_ids']]
        bp=make_plan(ident,selected,modules,raw,{**metadata,'drawing_detail_selection':selection})
        bp['id']=key;save(dest/'plan.json',bp)
        for file in ('source.png','modules.png'):shutil.copy2(folder/file,dest/file)
        scenes.append(dict(id=key,label=f'{ident} · 新提取绘画候选{len(selected)}笔',count=len(selected),input_count=len(raw),new_candidate=True))
    panels=[(Image.fromarray(rgb),'原图 · 自动提示，不用参考框'),
        (draw(baselines[500]['curves']) if 500 in baselines else draw(raw), '已认可500笔基线' if 500 in baselines else '源图骨架对照 · 未经模型配对'),
        (draw(baselines[700]['curves']) if 700 in baselines else draw(learned), '已认可700笔基线' if 700 in baselines else '新整幅提取 · 全部确定候选'),
        (draw(raw),f'源图墨迹骨架对照 · {len(raw)}段'),
        (draw(learned),f'模型归属辅助读出 · {len(learned)}笔候选'),
        (draw(learned,True),'对象候选分块 · 块内轮廓先行')]
    board=Image.new('RGB',(1560,1150),'#f4f1e8');d=ImageDraw.Draw(board)
    for i,(im,title) in enumerate(panels):
        x=(i%3)*520;y=(i//3)*555;d.text((x+10,y+10),title,font=FONT,fill='#263e34');board.paste(im.resize((496,496),Image.Resampling.LANCZOS),(x+10,y+43))
    d.text((10,1119),'本轮无新训练；中心线为分割后图结构读出，非实画。自动对象尚未验证为独立山石，皴线不等于皴法类别识别。',font=SMALL,fill='#263e34')
    board.save(folder/'comparison.jpg',quality=95)
    for budget in (500,700):
        dest=ROOT/f'{ident}_{budget}';shutil.copy2(folder/'comparison.jpg',dest/'comparison.jpg');shutil.copy2(folder/'comparison.jpg',dest/'sequence_comparison.jpg')
    summary=dict(id=ident,native_size=rgb.shape[1::-1],queries=len(queries),source_graph=curve_metrics(raw),new_full=curve_metrics(learned),
        learned_pairs=len(links)//2,unresolved_junctions=sum(d['status']=='unresolved_junction' for d in diag),pending=len(pending),
        modules=len(modules),object_proposals=sum(m['sam_key'] is not None for m in modules),module_reentries=plan['stats']['hierarchy_order']['module_reentries'],
        baselines={str(b):compare_geometry(o['curves'],learned) for b,o in baselines.items()},
        old_baselines_unchanged=True,new_weights_trained=False,elapsed_seconds=time.monotonic()-start,
        source_control_is_not_ground_truth=True,autonomous_full_image_accuracy_not_established=True)
    save(folder/'summary.json',summary);print('FULL_DONE',ident,json.dumps(summary,ensure_ascii=False),flush=True)
    return scenes,summary

def report(results):
    rows=[]
    for r in results:
        i=r['id'];a=r['source_graph'];b=r['new_full']
        rows.append(f'<section><h2>{i}</h2><p>原图 {r["native_size"]}；源图骨架 {a["count"]}段 → 模型归属辅助 {b["count"]}笔；模型支持配对 {r["learned_pairs"]}处；待定交汇 {r["unresolved_junctions"]}处。分块回访 {r["module_reentries"]}次。</p><img src="/{i}/comparison.jpg"><p><a href="/?scene={i}_500">回放500笔绘画候选</a> · <a href="/?scene={i}_700">回放700笔绘画候选</a></p><pre>{json.dumps(r,ensure_ascii=False,indent=2)}</pre></section>')
    html='''<!doctype html><meta charset="utf-8"><title>整幅自动提取与笔序对照</title><style>body{font:16px/1.7 Microsoft YaHei,system-ui;background:#f4f1e8;color:#243d33;max-width:1560px;margin:auto;padding:24px}img{width:100%}section{margin:40px 0}pre{white-space:pre-wrap;font-size:13px;background:white;padding:15px}a{color:#285c47}.note{padding:16px;background:#eee0bf}</style><h1>整幅自动提取 · 冻结模型实验</h1><p class="note">本轮在RGB墨迹图上自动产生全部短提示和对象候选，不读取人工线稿、参考框或旧500／700笔路径来提取。模型预测墨迹归属，后续图结构读出中心线；未新训练权重，未上机。三幅固定对照与另外三幅既有生成画均列出，后者只对本轮流程新增，并非从未见过的独立测试集。</p><p>重要限制：源图墨迹阈值、SAM对象与轮廓/皴线角色仍可能出错；连续性改善不等于正确分笔。旧线稿覆盖只能表示相对保留，不能当提取准确率。各画全部候选保留，500／700只是可选绘画简化，不作为完整提取上限。不会自动替换认可版本。</p><p><a href="/">打开分块笔序回放</a></p>'''+''.join(rows)
    (ROOT/'report.html').write_text(html,encoding='utf8')

def main(ids):
    ROOT.mkdir(exist_ok=True);(ROOT/'source_snapshot').mkdir(exist_ok=True)
    checkpoint=LAB/'whole_stroke_instance_20261105/ownership_attention/instance_curve.pt'
    protocol=dict(config=CONFIG,created_before_full_image_run=True,model_sha256=sha(checkpoint),new_training=False,
        stages=['RGB source ink graph','unseeded automatic SAM object proposals','source-derived multi-scale short hints','frozen instance mask inference','mutual ownership junction decisions','connected skeleton readout','separate detail budgets and module-locked scheduling'],
        inputs=[dict(id=i,source=str(INPUTS[i]),sha256=sha(INPUTS[i]),prior_experiment_seen=i in ('G03_01','G05_03','T02')) for i in SCENES],
        no_manual_reference_or_old_path_at_extraction=True,not_independent_artwork_test=True,not_neural_curve_head=True,
        semantic_rock_tree_pavilion_classification_trained=False,robot_connected=False,adopted=False)
    if (ROOT/'protocol.json').exists():assert read(ROOT/'protocol.json')==json.loads(json.dumps(protocol)),'Do not mix parameter protocols in one experiment'
    else:save(ROOT/'protocol.json',protocol)
    for name in ('full_image_ownership_core_20261105.py','build_full_image_ownership_20261105.py'):
        shutil.copy2(Path(__file__).with_name(name),ROOT/'source_snapshot'/name)
    torch.set_num_threads(4);cv2.setNumThreads(2);instance.ROOT=checkpoint.parent;net=instance.load(True)
    sys.path.insert(0,str(LAB/'module_stroke_planner_20261104/runtime'))
    from segment_anything import sam_model_registry,SamAutomaticMaskGenerator
    # Float32 SAM and CPU trials exceeded runtime resources. Frozen SAM runs
    # under FP16 autocast; the ownership network remains its original precision.
    sm=sam_model_registry['vit_b'](checkpoint=str(LAB/'module_stroke_planner_20261104/weights/sam_vit_b_01ec64.pth')).cuda().eval()
    sam=SamAutomaticMaskGenerator(sm,points_per_side=24,points_per_batch=4,pred_iou_thresh=.85,stability_score_thresh=.90,crop_n_layers=0,min_mask_region_area=0)
    catalog=[];results=[]
    for ident in ids:
        if (ROOT/ident/'summary.json').exists():
            result=read(ROOT/ident/'summary.json');scenes=[]
            for key in (ident,ident+'_500',ident+'_700'):
                existing=read(ROOT/key/'plan.json');scenes.append(dict(id=key,label=f'{key} · 新提取{len(existing["curves"])}笔候选',new_candidate=True))
        else:scenes,result=build(ident,net,sam)
        catalog+=scenes;results.append(result);gc.collect();torch.cuda.empty_cache()
        save(ROOT/'catalog.json',dict(scenes=catalog,default=catalog[0]['id'],robot_connected=False))
        save(ROOT/'results.json',results);report(results)
    save(ROOT/'execution_binding.json',dict(scene=None,journal=None,plan_sha256=None))
    page=Path(__file__).with_name('hierarchical_paint_platform_20261105.html').read_text('utf8')
    page=page.replace('o.textContent=({G03_01:','o.textContent=s.label||({G03_01:')
    page=page.replace('plan=p;scene=id;sourceImage',"plan=p;scene=id;$('scene').value=id;sourceImage")
    page=page.replace('await load(c.default);',"const requested=new URLSearchParams(location.search).get('scene');await load(c.scenes.some(s=>s.id===requested)?requested:c.default);")
    page=page.replace('本轮保留全部输入的来源记录','本轮分割后的曲线是实验候选，未替换认可版本。保留源图骨架来源记录')
    (ROOT/'index.html').write_text(page,encoding='utf8')
    save(ROOT/'decision.json',dict(adopted=False,new_training=False,robot_connected=False,reason='Full-image correctness requires audit; neither fewer strokes nor old-line coverage establishes accuracy'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ids',nargs='+',default=list(SCENES));a=p.parse_args();main(a.ids)
