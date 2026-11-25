# -*- coding: utf-8 -*-
"""v09 source-grouped, non-expert multi-task experiment. No robot imports."""
import sys,json,re,hashlib,io,math,time,shutil
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'rock_multitask_v09_20261024';V08=LAB/'rock_strokes_v08_20261023';V07=LAB/'rock_strokes_v07_20261023';V06=LAB/'rock_cunfa_v06_20261023';INK=LAB/'runs'/'mobilenet_unet_rock_cunfa_v06_expanded'
for p in ('raw','html','review','crops','labels','runs','evaluation','source_snapshot'):(ROOT/p).mkdir(parents=True,exist_ok=True)
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
def write(name,obj):(ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def sheet(rows,name):
    for start in range(0,len(rows),48):
        batch=rows[start:start+48];board=Image.new('RGB',(1440,math.ceil(len(batch)/6)*190),'#f4f4ed');d=ImageDraw.Draw(board)
        for i,r in enumerate(batch):
            p=Image.open(r['image']).convert('RGB');p.thumbnail((232,155));x=i%6*240;y=i//6*190;board.paste(p,(x+(232-p.width)//2,y));d.text((x+4,y+158),f'{start+i} '+r.get('method','')+' '+r['id'][-6:],font=FONT,fill='black')
        board.save(ROOT/'review'/f'{name}_{start//48+1:02d}.jpg',quality=91)

def collect():
    from gongbi_training.cunfa_collect import fetch,Images
    from concurrent.futures import ThreadPoolExecutor,as_completed
    from urllib.parse import urljoin
    previous=[]
    for p in (LAB/'rock_cunfa_v04'/'web_sources.json',V06/'new_sources.json',V06/'new_sources_phase2.json',V07/'new_sources.json'):previous+=json.loads(p.read_text(encoding='utf-8-sig'))
    old_pages={r.get('page',r.get('source_page')) for r in previous};old_urls={r.get('url',r.get('image_url')) for r in previous};old_hashes={hashlib.sha256(np.asarray(Image.open(r['image']).convert('RGB')).tobytes()).hexdigest() for r in previous}
    methods=['披麻皴','斧劈皴','荷叶皴','折带皴','解索皴','卷云皴','米点皴','雨点皴']
    archives=[f'https://www.kuhw.com/hhjc/index_{i}.html' for i in (20,21,22,23,24,25,26,27,28,29,30,40,50,60,80,100)]
    archives+=['https://www.kuhw.com/hhjc/sshjc/','https://www.kuhw.com/hhjc/hnhjc/']
    def read(page):
        file=ROOT/'html'/(hashlib.sha256(page.encode()).hexdigest()[:16]+'.html');raw=file.read_bytes() if file.exists() else fetch(page);file.write_bytes(raw);html=re.sub(r'<!--.*?-->','',raw.decode('utf-8','replace'),flags=re.S);parser=Images();parser.feed(html);return page,html,parser
    discovered={};errors=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs={pool.submit(read,p):p for p in archives}
        for f in as_completed(jobs):
            try:
                page,html,p=f.result()
                for u,title in re.findall(r'<a[^>]*href=[\"\']([^\"\']+)[\"\'][^>]*>(.*?)</a>',html,re.S):
                    title=re.sub('<[^>]+>','',title);url=urljoin(page,u)
                    if re.search('皴|山石|画石|石法',title) and '/hhjc/' in url and url.endswith('.html') and url not in old_pages:discovered.setdefault(url,title)
            except Exception as exc:errors.append(dict(page=jobs[f],error=str(exc)))
    # Public new tutorial sources; old sources are not relabelled as fresh.
    pages=list(discovered)[:60];tasks={};source_pages=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs={pool.submit(read,p):p for p in pages}
        for f in as_completed(jobs):
            try:
                page,html,parser=f.result()
                if 'class="entry"' not in html:continue
                body=html.split('class="entry"',1)[1].split('class="page5"',1)[0];p=Images();p.feed(body);found=[m for m in methods if m in parser.title];method=found[0] if len(found)==1 else ''
                for u in p.images:
                    url=urljoin(page,u)
                    if '/d/file/' in url and url not in old_urls:tasks.setdefault(url,dict(page=page,url=url,title=parser.title,method=method,method_provenance='single-method article title only; not expert verified'))
                source_pages.append(dict(page=page,title=parser.title,method=method));print('NEW_PAGE',parser.title,len(tasks),flush=True)
            except Exception as exc:errors.append(dict(page=jobs[f],error=str(exc)))
    def download(r):
        identity='v09_'+hashlib.sha256(r['url'].encode()).hexdigest()[:14];p=ROOT/'raw'/(identity+'.png')
        if not p.exists():
            im=Image.open(io.BytesIO(fetch(r['url'],r['page']))).convert('RGB')
            if min(im.size)<110 or max(im.size)<250:raise ValueError('small')
            im.thumbnail((1700,1700));im.save(p)
        im=Image.open(p).convert('RGB');return dict(r,id=identity,image=str(p),rgb_sha256=hashlib.sha256(np.asarray(im).tobytes()).hexdigest(),width=im.width,height=im.height)
    result=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs={pool.submit(download,r):r for r in list(tasks.values())[:450]}
        for f in as_completed(jobs):
            try:result.append(f.result())
            except Exception as exc:errors.append(dict(url=jobs[f]['url'],error=str(exc)))
    rows=[]
    for r in sorted(result,key=lambda r:(r['page'],r['id'])):
        if r['rgb_sha256'] not in old_hashes:rows.append(r);old_hashes.add(r['rgb_sha256'])
    write('new_sources.json',rows);write('collection_audit.json',dict(new_pages=source_pages,attempted=len(tasks),downloaded=len(result),new_exact_unique=len(rows),errors=errors));sheet(rows,'new');print('COLLECTION_COMPLETE',len(rows),flush=True)

def collect_style():
    from gongbi_training.cunfa_collect import fetch
    from concurrent.futures import ThreadPoolExecutor,as_completed
    page='https://www.sohu.com/a/376343741_271454';base='https://5b0988e595225.cdn.sohucs.com/images/20200227/'
    ids=[('3e9d7f357a084659a8aa53f69d73e906','披麻皴'),('b21112cb76d94e94a466759e0e027afb','披麻皴'),('d11d8c57688b4417aff480922592b33f','斧劈皴'),('7341a6fb216e420184ab10e5d47ab184','斧劈皴'),('7d5c5c512b6c4d56805dafe363a2ae4c','折带皴'),('a1d6ebd13b5f41b895e31da514f4539a','折带皴'),('f14fad57288240589b696ab5468c3115','荷叶皴'),('c2d590dd3a8c4b759bd3c5865811e563','解索皴'),('7a3e850fe56b4f71a8a9379d05a749d9','米点皴')]
    def job(pair):
        key,label=pair;path=ROOT/'raw'/('style_'+key+'.png')
        if not path.exists():Image.open(io.BytesIO(fetch(base+key+'.jpeg',page))).convert('RGB').save(path)
        return dict(id='style_'+key,image=str(path),page=page,url=base+key+'.jpeg',method=label,provenance='explicit paragraph/image association on teaching article; not expert ground truth')
    rows=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for f in as_completed([pool.submit(job,p) for p in ids]):
            try:rows.append(f.result())
            except Exception as exc:print('STYLE_DOWNLOAD_ERROR',str(exc),flush=True)
    rows.sort(key=lambda r:r['id']);write('new_style_probes.json',rows);sheet(rows,'style');print('STYLE_PROBES',len(rows),flush=True)

def prepare():
    from gongbi_training.cunfa_prepare import phash,ink_target
    from gongbi_training.rock_multitask_v09 import synthetic
    if any((ROOT/'runs').glob('*/training.jsonl')):raise FileExistsError('Freeze inputs before training; preserve existing experiment')
    old=json.loads((V07/'manifest.json').read_text(encoding='utf-8'));parent_rows=old['real'];old_phash=np.array([phash(np.asarray(Image.open(r['image']).convert('RGB'))) for r in parent_rows]);train_groups={r['group'] for r in parent_rows if r['split']=='train'}
    raw=json.loads((ROOT/'new_sources.json').read_text(encoding='utf-8'));fresh=[];rejected=[];seen=[]
    for i,r in enumerate(raw):
        if i==9:rejected.append(dict(id=r['id'],reason='dense full painting, outside initial isolated-rock task'));continue
        im=Image.open(r['image']).convert('RGB');w,h=im.size;im=im.crop((round(w*.025),round(h*.02),round(w*.975),round(h*.91)));im.thumbnail((800,800));rgb=np.asarray(im);p=phash(rgb);nearest=int(np.count_nonzero(old_phash!=p,axis=1).min())
        if nearest<=5 or any(np.count_nonzero(p!=q)<=5 for q in seen):rejected.append(dict(id=r['id'],reason='near duplicate of prior or current crop',nearest_parent_phash=nearest));continue
        seen.append(p);identity=r['id']+'_heldout';image=ROOT/'crops'/(identity+'.png');im.save(image);t,c=ink_target(rgb);label=ROOT/'labels'/(identity+'_automatic.npz');np.savez_compressed(label,targets=t,weights=c)
        fresh.append(dict(r,id=identity,image=str(image),labels=str(label),split='test',source_group='new_page_'+hashlib.sha256(r['page'].encode()).hexdigest()[:10],nearest_parent_phash=nearest,expert_ground_truth=False,labels_provenance='automatic paper normalized ink; weak agreement only',sha256=hashlib.sha256(image.read_bytes()).hexdigest()))
    # These source images are fixed now and never passed to a training sampler.
    style_raw=json.loads((ROOT/'new_style_probes.json').read_text(encoding='utf-8'));style_old=json.loads((V06/'cunfa_classifier_normalized'/'evaluation.json').read_text(encoding='utf-8'))['records']
    style_parent_hash=[]
    for r in style_old:
        im=Image.open(r['image']).convert('RGB');w,h=im.size;b=r['box'];a=np.asarray(im.crop(tuple(round(v*(w if k%2==0 else h)) for k,v in enumerate(b))));style_parent_hash.append(phash(a))
    style=[]
    for r in style_raw:
        im=Image.open(r['image']).convert('RGB');w,h=im.size;im=im.crop((round(w*.10),round(h*.025),round(w*.92),round(h*.86)));rgb=np.asarray(im);p=phash(rgb);nearest=min(int(np.count_nonzero(p!=h)) for h in style_parent_hash)
        path=ROOT/'crops'/(r['id']+'_type_probe.png');im.save(path)
        style.append(dict(r,image=str(path),split='test',source_group='new_sohu_376343741',nearest_known_classifier_phash=nearest,near_duplicate_warning=nearest<=7,occluding_watermark=True,expert_ground_truth=False,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    # Freeze source-grouped geometry manifest and candidly retain classification
    # duplicate warnings: they cannot be called new independent test accuracy.
    frozen=dict(created_before_training=True,new_geometry_test=fresh,new_style_probes=style,rejected=rejected,training_real_records=len([r for r in parent_rows if r['split']=='train']),training_groups=sorted(train_groups),expert_annotations=0,source_disjoint_geometry=True,strict_book_family_independence_verified=False,user_authorized_nonexpert_experiment=True)
    write('frozen_manifest.json',frozen);write('data_audit.json',dict(downloaded_geometry=len(raw),retained_new_geometry=len(fresh),new_geometry_source_groups=len({r['source_group'] for r in fresh}),style_probes=len(style),style_probe_duplicate_warnings=sum(r['near_duplicate_warning'] for r in style),parent_training_real_records=frozen['training_real_records'],expert_annotations=0,rejected=rejected))
    sheet(fresh,'heldout_geometry');sheet(style,'heldout_style')
    sample=synthetic(np.random.default_rng(543210));rgb,t,c,ids=sample
    preview=Image.new('RGB',(1280,256),'white');arrays=[rgb,np.uint8(255*(1-t[0])),np.uint8(255*(1-t[6])),np.uint8(t[9]*255),cv2.applyColorMap(np.uint8(ids*17),cv2.COLORMAP_TURBO)]
    for i,a in enumerate(arrays):preview.paste(Image.fromarray(a).convert('RGB'),(i*256,0))
    preview.save(ROOT/'review'/'known_target_preview.jpg',quality=94)
    print('FROZEN_DATA_READY',len(fresh),'geometry',len(style),'style',flush=True)

def checks():
    import torch
    from gongbi_training.rock_multitask_v09 import model_new,synthetic,loss_fn
    torch.set_num_threads(4);model=model_new(V08/'best.pt').cuda();rgb,t,c,ids=synthetic(np.random.default_rng(777))
    x=torch.from_numpy(rgb.transpose(2,0,1).copy())[None].cuda().float()/255;t=torch.from_numpy(t)[None].cuda();c=torch.from_numpy(c)[None].cuda();ids=torch.from_numpy(ids)[None].cuda().long()
    out=model(x);assert out.shape==(1,14,256,256);value,parts=loss_fn(out,t,c,ids);assert torch.isfinite(value);value.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5);assert torch.isfinite(norm)
    assert ids.max()>3 and t[:,9].max()<=1 and t[:,6].max()<=1
    frozen=json.loads((ROOT/'frozen_manifest.json').read_text(encoding='utf-8'));old=json.loads((V07/'manifest.json').read_text(encoding='utf-8'));train_paths={r['image'] for r in old['real'] if r['split']=='train'}
    assert not any(r['image'] in train_paths for r in frozen['new_geometry_test']);assert frozen['created_before_training']
    write('training_checks.json',dict(output_shape=list(out.shape),loss=float(value),gradient_norm=float(norm),finite_forward_backward=True,instance_labels_present=True,test_image_paths_excluded=True,hardware_used=False))
    print('TRAINING_CHECKS_PASS',float(value),parts,flush=True)

def protocol():
    frozen=json.loads((ROOT/'frozen_manifest.json').read_text(encoding='utf-8'));rows=frozen['new_geometry_test'];groups=sorted({r['source_group'] for r in rows});ids=[next(r['id'] for r in rows if r['source_group']==g) for g in groups]
    for r in rows:
        if len(ids)>=3:break
        if r['id'] not in ids:ids.append(r['id'])
    p=dict(fixed_before_model_test_predictions=True,preview_ids=ids,preview_rule='first retained image per source group, then first remaining retained crop',ink_probability_threshold=.5,weak_label_coverage_threshold=.3,partial_reference_tolerance_pixels=4,synthetic_test_seeds=[500000,500031],classification_unknown_threshold=.65,classification_selection='legacy validation CE only; never new source probe',real_metrics='agreement with automatic weak labels; no expert accuracy',book_family_independence_verified=False)
    write('evaluation_protocol.json',p);print('PROTOCOL_FIXED',ids,flush=True)

def evaluate():
    from gongbi_training.rock_evaluate_v09 import evaluate
    evaluate(ROOT,V08,INK)

def finalize():
    import torch
    training=json.loads((ROOT/'training_summary.json').read_text(encoding='utf-8'));summary=json.loads((ROOT/'evaluation'/'geometry_summary.json').read_text(encoding='utf-8'));style=json.loads((ROOT/'style_classifier'/'report.json').read_text(encoding='utf-8'));audit=json.loads((ROOT/'data_audit.json').read_text(encoding='utf-8'));frozen=json.loads((ROOT/'frozen_manifest.json').read_text(encoding='utf-8'))
    # No selective adoption based on the new small test. All old versions and
    # frozen-protocol outputs remain available for review.
    known=summary['known_synthetic'];weak=summary['new_source_weak'];legacy=summary['legacy_partial']
    def legacy_avg(stage,key):return float(np.mean([r[stage][key] for r in legacy]))
    report=f'''# v09 非专家标注多任务实验（2026-10-24）

用户已明确选择“暂无，先做非专家标注实验”。本轮没有画家或人工专家真值。

## 数据与划分

原有591个训练裁图继续作为低权重自动标签来源；逐笔监督仍使用v08的4张图49条助手初始描线。原验证图1张11条描线用于选择权重，但该图曾被父模型训练过，所以不是全流程独立验证。原3张30条助手描线仅作开发回归检查。

本轮下载20张新山石教程图，剔除1张复杂全景后保留19张裁图，来自2个新的网页来源组；它们未进入任何训练采样器。另有9张图文对应皴法名称的新网页图用于分类探针，来自同一个网页来源组，均有遮挡水印。图像数量不等于独立画作数量，19张中存在同书步骤/相似变体；网页来源互斥不能证明同一画谱、作者或跨站转载关系全部排除。源文件、清单和预览选择已在测试预测前冻结。

程序生成240个山石训练几何，具有已知笔画编号、中心线、宽度、透明墨色、短断续及飞白外观；其笔画数量不能计作人工描线或真实画作。另用不重叠种子的验证/测试程序样本。程序物理定义与真实毛笔存在差异。

## 实际训练

三阶段各800次有效更新，共{training['successful_updates']}次，跳过{training['skipped_updates']}次，耗时{training['elapsed_seconds']:.1f}秒。按固定验证损失选择第{training['selected']['phase']}阶段第{training['selected']['step']}步。保留各阶段best/last，不覆盖v08。

网络从v08初始化，新增墨色、可见墨迹、已知程序笔画端点、干笔外观代理和4维笔画区分表征；中心线与方向及轮廓/内部线候选继续训练。加入适配的soft-clDice连通性损失与笔画区分损失，目标是同时减少断裂与误接。它们不保证恢复真实人类笔画；端点和干笔外观只在程序样本上有明确监督。参考：[clDice](https://arxiv.org/abs/2003.07311)。

## 已知程序几何测试（32个样本，非真实画作准确率）

| 指标 | 旧版 | v09 |
|---|---|---|
| 中心线像素F1 | {known['previous_center_f1']:.4f} | {known['continued_center_f1']:.4f} |
| 宽度MAE（像素，程序定义） | {known['previous_width_mae_pixels']:.3f} | {known['continued_width_mae_pixels']:.3f} |
| 墨色MAE（0..1） | {known['previous_density_mae']:.4f} | {known['continued_density_mae']:.4f} |

干笔外观代理MAE={known['continued_dry_appearance_mae']:.4f}。这里的“干笔”仅指程序定义的局部缺墨比例，不是纸面含水量。笔画表征的同笔余弦均值={known['within_stroke_embedding_cosine']:.3f}，异笔={known['between_stroke_embedding_cosine']:.3f}；不能当作真实笔画实例识别率。

## 新来源真实图的自动弱标签对照

19张/2来源组，以下为按来源组平均；目标由纸色归一化自动生成，仍不是专家准确率。

| 指标 | 旧墨迹模型 | v09 |
|---|---|---|
| 与自动弱标签F1一致性 | {weak['previous']['f1']:.4f} | {weak['continued']['f1']:.4f} |
| 精确率一致性 | {weak['previous']['precision']:.4f} | {weak['continued']['precision']:.4f} |
| 召回率一致性 | {weak['previous']['recall']:.4f} | {weak['continued']['recall']:.4f} |
| 墨色MAE | {weak['previous']['density_mae']:.4f} | {weak['continued']['density_mae']:.4f} |

所有新图的原图/旧墨迹/v09墨迹均保存在all_new_ink_*.jpg。新来源流水线图还单列“新模型＋旧墨迹”与“新模型＋自身墨迹”，用于区分轨迹网络和墨迹输出的影响。不得仅以画面变黑、更密或更少来判断更好。

## 原30条助手初始描线回归

| 指标 | v08＋原墨迹 | v09＋原墨迹 | v09＋自身墨迹 |
|---|---|---|---|
| 已描线点覆盖（4px容差） | {legacy_avg('previous','coverage'):.4f} | {legacy_avg('new_model_old_ink','coverage'):.4f} | {legacy_avg('continued','coverage'):.4f} |
| 最佳单路径覆盖 | {legacy_avg('previous','best_single_path'):.4f} | {legacy_avg('new_model_old_ink','best_single_path'):.4f} | {legacy_avg('continued','best_single_path'):.4f} |
| 每条参考线对应片段 | {legacy_avg('previous','fragments'):.2f} | {legacy_avg('new_model_old_ink','fragments'):.2f} | {legacy_avg('continued','fragments'):.2f} |

这些是反复查看过的开发图，标注不完整且未经专家确认，不能当作独立泛化结论。

## 皴法候选分类

冻结本地CLIP图像编码器，以原图/自动白底墨迹双视图训练正则化分类头；这是局部图像候选类别，不是对每条笔画作可靠命名。训练{style['training_examples']}个旧示例，{style['legacy_validation_examples']}个旧开发验证示例用于选择正则化和检查点。新9张探针不用于选择。类别：{', '.join(style['classes'])}；最大概率低于0.65输出“不确定”，概率未经校准。

旧开发验证集标签一致性：旧头{style['previous_legacy_validation']['top1_agreement']:.3f}，本轮{style['new_legacy_validation']['top1_agreement']:.3f}。
新来源9张探针标签一致性：旧头{style['previous_new_source_probe']['top1_agreement']:.3f}，本轮{style['new_source_probe']['top1_agreement']:.3f}；本轮有样本类别的macro-F1={style['new_source_probe']['macro_f1_present_classes']:.3f}，非拒识比例={style['new_source_probe']['accepted_fraction']:.3f}。这些标签来自教程段落和助手判断，水印遮挡、样本量小、来源只有1组、同书关系未完全排除，不可称为专家皴法识别准确率。完整混淆矩阵、所有预测和模型候选均保留在style_classifier/report.json。

## 机械臂与使用限制

没有连接、使能或移动机械臂。相对下压0..1没有标定成毫米或真实力；墨色图像量不是墨液浓度，干笔外观代理不是实际含水量。physical_z、physical_speed、paper_transform为空，ready_for_robot=false。回放是像素几何检查，不是实画或经过验证的毛笔物理仿真。本轮模型保留为实验版本，不能自动认为所有任务都改善或替换旧版。
'''
    (ROOT/'REPORT.md').write_text(report,encoding='utf-8')
    sources={r['page']:r.get('title',r['page']) for r in frozen['new_geometry_test']};sources.update({r['page']:r['page'] for r in frozen['new_style_probes']});(ROOT/'SOURCES.md').write_text('# 本轮新来源\n\n'+'\n'.join(f'- [{title}]({url})' for url,title in sources.items()),encoding='utf-8')
    for path in [Path(__file__),Path(__file__).with_suffix('.ps1')]+[Path(__file__).resolve().parents[1]/'gongbi_training'/name for name in ('rock_multitask_v09.py','rock_style_v09.py','rock_plan_v09.py','rock_evaluate_v09.py')]:shutil.copy2(path,ROOT/'source_snapshot'/path.name)
    # Classification visual audit: every new probe, including low confidence.
    candidates=[r for r in style['records'] if r['cohort']=='new_source_probe'];board=Image.new('RGB',(1500,math.ceil(len(candidates)/3)*300),'#f4f4ed');d=ImageDraw.Draw(board)
    for i,r in enumerate(candidates):
        x=i%3*500;y=i//3*300;im=Image.open(r['image']).convert('RGB');im.thumbnail((475,225));board.paste(im,(x+(475-im.width)//2,y));d.text((x+6,y+228),'教程标签：'+r['label']+'（非专家）',font=FONT,fill='black');d.text((x+6,y+251),'本轮：'+r['candidate']+f"；最高分 {r['score']:.2f}",font=FONT,fill='black');d.text((x+6,y+274),'闭集首选：'+r['predicted'],font=FONT,fill='black')
    board.save(ROOT/'evaluation'/'cunfa_candidates.jpg',quality=93)
    verify()
    print('FINAL_REPORT_READY',flush=True)

def verify(geometry_only=False):
    import torch
    from gongbi_training.rock_plan_v09 import repair_instances
    # Same ink/direction, different embeddings: prevent an otherwise plausible
    # connection. Same embeddings with visible ink: allow it. Blank stays blank.
    d=np.zeros((80,130),np.float32);cv2.line(d,(10,30),(105,30),.7,3);pred=np.zeros((14,80,130),np.float32);pred[0]=1;pred[1]=1;pred[10]=1
    a=np.stack((np.arange(10,44),np.full(34,30)),1).astype(float);b=np.stack((np.arange(49,106),np.full(57,30)),1).astype(float)
    paths,events,_=repair_instances([a,b],d,pred);assert len(paths)==1
    pred[10,:,47:]=0;pred[11,:,47:]=1;paths,events,_=repair_instances([a,b],d,pred);assert len(paths)==2
    pred[10]=1;pred[11]=0;d[:,44:49]=0;paths,events,_=repair_instances([a,b],d,pred);assert len(paths)==2
    write('geometry_checks.json',dict(embedding_disagreement_blocks_join=True,ink_supported_same_embedding_can_join=True,blank_gap_not_joined=True,hardware_used=False))
    if geometry_only:
        print('GEOMETRY_CHECKS_PASS',flush=True);return
    hashes={}
    for path in [ROOT/'best.pt']+[ROOT/'runs'/f'phase_{i}'/'best.pt' for i in (1,2,3)]:
        ck=torch.load(path,map_location='cpu',weights_only=False);assert all(torch.isfinite(t).all() for t in ck['model'].values());hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    ck=torch.load(ROOT/'best.pt',map_location='cpu',weights_only=False);assert ck['config']['frozen_manifest_sha256']==hashlib.sha256((ROOT/'frozen_manifest.json').read_bytes()).hexdigest()
    frozen=json.loads((ROOT/'frozen_manifest.json').read_text(encoding='utf-8'))
    for record in frozen['new_geometry_test']+frozen['new_style_probes']:assert record['sha256']==hashlib.sha256(Path(record['image']).read_bytes()).hexdigest()
    classifier=torch.load(ROOT/'style_classifier'/'classifier.pt',map_location='cpu',weights_only=False);assert all(torch.isfinite(v).all() for v in classifier['head'].values())
    files=list((ROOT/'evaluation').glob('*/planned_trajectory.json'))
    for path in files:
        info=json.loads(path.read_text(encoding='utf-8'));assert not info['ready_for_robot'] and not info['actually_drawn'] and info['physical_z'] is None
        for s in info['strokes']:assert len(s['xy_pixels'])==len(s['width_pixels'])==len(s['ink_density'])==len(s['relative_pressure']);assert np.isfinite(s['xy_pixels']).all()
    images=list((ROOT/'evaluation').rglob('*.png'))+list((ROOT/'evaluation').glob('*.jpg'))
    for path in images:
        with Image.open(path) as im:im.verify()
    html=(ROOT/'trajectory_viewer.html').read_text(encoding='utf-8');(ROOT/'viewer_check.js').write_text(html.split('<script>',1)[1].split('</script>',1)[0],encoding='utf-8')
    write('verification.json',dict(frozen_manifest_unchanged=True,frozen_test_images_unchanged=True,weights_finite=True,classifier_head_finite=True,hashes=hashes,trajectory_files=len(files),verified_images=len(images),embedding_disagreement_blocks_join=True,ink_supported_same_embedding_can_join=True,blank_gap_not_joined=True,expert_annotations=0,robot_connected=False))
    print('VERIFIED',flush=True)

if __name__=='__main__':
    if sys.argv[1]=='collect':collect()
    elif sys.argv[1]=='collect_style':collect_style()
    elif sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='train':
        from gongbi_training.rock_multitask_v09 import train
        train(ROOT,V08/'best.pt',V07,V08)
    elif sys.argv[1]=='check':checks()
    elif sys.argv[1]=='train_style':
        from gongbi_training.rock_style_v09 import train
        train(ROOT,V06)
    elif sys.argv[1]=='protocol':protocol()
    elif sys.argv[1]=='evaluate':evaluate()
    elif sys.argv[1]=='finalize':finalize()
    elif sys.argv[1]=='plan_check':verify(geometry_only=True)
