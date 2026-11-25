# -*- coding: utf-8 -*-
"""Build an auditable reference pack, never relabel it as model output."""
import sys, json, hashlib, html, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure, LAB
configure()
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from gongbi_training.evidence_packet import supervision_channels, audit_lineage

ROOT = LAB/'whole_rock_evidence_20261117'
PROJECT = Path(__file__).resolve().parents[1]
FONT = 'C:/Windows/Fonts/msyh.ttc'
PALETTE = {'contour': (173,111,35), 'structure': (59,122,77), 'cun': (37,131,174)}

def read(p): return json.loads(Path(p).read_text('utf8'))
def write(p,v): Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding='utf8')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def base(ident,scope,source_id,work,source,evidence,**kw):
    return dict(id=ident,scope=scope,source_id=source_id,work_group=work,source_group=source,
                evidence=evidence,reviewer='assistant_evidence_audit_20261117',seen_before_round=True,
                independent_test=False,usage='known_development',prediction_as_label=False,
                artist_order_verified=False,pressure_mm=None,**kw)

def panel(image,title,subtitle='',size=(500,470)):
    board=Image.new('RGB',size,'#f5f2e9'); d=ImageDraw.Draw(board)
    d.text((12,10),title,font=ImageFont.truetype(FONT,22),fill='#213e35')
    im=image.copy();scale=min((size[0]-24)/im.width,(size[1]-100)/im.height)
    im=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.NEAREST)
    board.paste(im,((size[0]-im.width)//2,50+(size[1]-100-im.height)//2))
    d.text((12,size[1]-37),subtitle,font=ImageFont.truetype(FONT,15),fill='#775440')
    return board

def main():
    ROOT.mkdir(exist_ok=True);(ROOT/'views').mkdir(exist_ok=True);(ROOT/'data').mkdir(exist_ok=True)
    sources=[];records=[]
    def source(ident,path,url,work,group):
        im=Image.open(path).convert('RGB')
        sources.append(dict(id=ident,path=str(path),sha256=sha(path),size=list(im.size),url=url,
                            work_group=work,source_group=group,seen_before_round=True))
        return im
    legacy=read(LAB/'rock_skill_offline_20261110/visible_reference.json')
    full=source('tutorial_final',Path(legacy['source']),legacy['source_url'],'mixiaolu_pima_work','mixiaolu_publisher')
    crop=full.crop(legacy['box']);native=np.array(crop);h,w=native.shape[:2]
    crop.save(ROOT/'views/rock_native.png')
    # Always use the native 185x165 crop. The old preview was enlarged 4x.
    assert [w,h]==[185,165]
    masks=[]; rows=[]
    for r in legacy['records']:
        q=np.array(r['curve'],float)
        assert np.isfinite(q).all() and (q>=0).all() and (q[:,0]<w).all() and (q[:,1]<h).all()
        m=np.zeros((h,w),'uint8');cv2.polylines(m,[np.round(q).astype('int32')],False,1,3);masks.append(m>0)
    count=np.stack(masks).sum(0)
    guide=native.copy();overlap=native.copy();fragment=np.full_like(native,255)
    occupied=np.any(masks,axis=0)
    # These are original source pixels inside candidate corridors, not generated strokes.
    fragment[occupied]=native[occupied]
    for i,r in enumerate(legacy['records']):
        q=np.round(r['curve']).astype('int32');cv2.polylines(guide,[q],False,PALETTE[r['role']],1)
        conflict=masks[i]&(count>1)
        rec=base(r['id'],'visible_curve_candidate','tutorial_final','mixiaolu_pima_work','mixiaolu_publisher',
                 '继承11月10日静态可见曲线；本轮审计坐标、交叠和标签权限，未确认画家真实分笔。',
                 object_type='rock',object_id=r['object_id'],role=r['role'],group_id=r['group_id'],
                 source_crop_box=legacy['box'],coordinate_space='native_crop_xy',native_size=[w,h],
                 curve=r['curve'],source_curve_manifest=str(LAB/'rock_skill_offline_20261110/visible_reference.json'),
                 instance_supervision=False,artist_identity_verified=False,
                 endpoint_uncertainty_source_px=r['endpoint_uncertainty_source_px'],
                 candidate_corridor_radius_px=1.5,conflicting_corridor_pixels=int(conflict.sum()),
                 corridor_pixel_count=int(masks[i].sum()),local_cunfa='unknown',training_admitted=False)
        records.append(rec); rows.append({k:rec[k] for k in ['id','object_id','role','group_id','conflicting_corridor_pixels','corridor_pixel_count']})
    overlap[count>1]=(197,54,143)
    Image.fromarray(guide).save(ROOT/'views/rock_guides.png');Image.fromarray(overlap).save(ROOT/'views/rock_conflicts.png')
    Image.fromarray(fragment).save(ROOT/'views/source_corridor_fragments.png')
    np.savez_compressed(ROOT/'data/candidate_corridors.npz',corridors=np.stack(masks),overlap=count>1,
                        instance_targets=np.full((h,w),255,'uint8'))
    # Reuse the actual before/after role evidence, without moving it to the final view.
    temporal=read(LAB/'cunfa_temporal_roles_20261112/dataset_manifest.json')
    temporal_views=[]
    for ident in ['T_structure','T_cun_group1']:
        r=next(x for x in temporal['records'] if x['id']==ident);z=np.load(r['path'])
        target=z['targets'];image=z['image'];before=z['before']
        overlay=image.copy()
        for k,col in [(0,(186,210,167)),(1,PALETTE['contour']),(2,PALETTE['cun'])]:
            overlay[target==k]=(.35*image[target==k]+.65*np.array(col)).astype('uint8')
        for suffix,im in [('before',before),('after',image),('roles',overlay)]:
            Image.fromarray(im).save(ROOT/f'views/{ident}_{suffix}.png')
        copied=ROOT/f'data/{ident}.npz';shutil.copyfile(r['path'],copied)
        sources.append(dict(id='tutorial_phase_'+ident,path=r['path'],sha256=sha(r['path']),
                            size=[int(image.shape[1]),int(image.shape[0])],
                            url=temporal['source_url'],work_group='mixiaolu_pima_work',
                            source_group='mixiaolu_publisher',seen_before_round=True,
                            original_array_keys=['before','image','targets','delta'],
                            raw_frame_manifest=str(LAB/'cunfa_temporal_roles_20261112/frame_manifest.json')))
        rec=base(ident,'temporal_role_core','tutorial_phase_'+ident,'mixiaolu_pima_work','mixiaolu_publisher',
                 r['phase_role_basis'],object_type='rock',role_checked=True,object_checked=True,
                 source_npz=r['path'],source_npz_sha256=sha(r['path']),packet_npz=str(copied),
                 before_seconds=r['before_seconds'],after_seconds=r['after_seconds'],
                 registration=r['registration'],role_counts={str(k):int((target==k).sum()) for k in [0,1,2,255]},
                 role_scope='partial_structure_and_cun_core_only',instance_supervision=False,
                 complete_object_covered=False,style_target=None,unknown_value=255,
                 cannot_register_to_final_without_additional_evidence=True)
        records.append(rec);temporal_views.append((ident,Image.fromarray(image),Image.fromarray(overlay)))
    # Original user-corrected foliage core. A polygon is an object-region label, not ink truth.
    hmeta=next(x for x in read(LAB/'cunfa_roles_v2_20261112/dataset_manifest.json')['records'] if x['id']=='H_14')
    himage=LAB/'cunfa_roles_v2_20261112/references/H_14_input.png'
    hf=source('H14',himage,hmeta['source_url'],'howto_repost_work','howto_repost_artist_unverified')
    corr=read(LAB/'cunfa_object_correction_20261112/correction.json');poly=corr['polygon_native_context_xy']
    mask=np.zeros((hf.height,hf.width),'uint8');cv2.fillPoly(mask,[np.array(poly,'int32')],1)
    targets=np.full(mask.shape,255,'uint8');targets[mask>0]=1
    np.savez_compressed(ROOT/'data/H14_object_region.npz',object_targets=targets,
                        rock_role_targets=np.full(mask.shape,255,'uint8'))
    hover=np.array(hf);hover[mask>0]=(.45*hover[mask>0]+.55*np.array([137,64,155])).astype('uint8')
    Image.fromarray(hover).save(ROOT/'views/H14_foliage_core.png');hf.save(ROOT/'views/H14_original.png')
    records.append(base('H14_foliage','object_region','H14','howto_repost_work','howto_repost_artist_unverified',
        '用户指出底部为树叶；助手沿用已复核保守核心。树石交界、外缘和像素墨迹均未验证。',
        object_type='tree',role='foliage',object_checked=True,region_is_ink_mask=False,polygon=poly,
        user_object_correction=True,local_cunfa='unknown',source_correction_sha256=sha(LAB/'cunfa_object_correction_20261112/correction.json')))
    # Keep surrounding context so leaf shape cannot silently become rock texture.
    candidates=read(LAB/'cunfa_boundary_relations_20261117/candidate_references.json')['records']
    negatives=[]
    for ident in ['qingpu_20_0_region_6','shen_attributed_7','nizan_8']:
        r=next(x for x in candidates if x['id']==ident)
        path=Path(r['reference_image']);fullneg=source(ident,path,r['source_url'],r.get('role_fold_group',r.get('source_unit',ident)),
                        r.get('source_group',r.get('role_fold_group',ident)))
        box=r['box'];assert 0<=box[0]<box[2]<=fullneg.width and 0<=box[1]<box[3]<=fullneg.height
        im=fullneg.crop(box);im.save(ROOT/f'views/{ident}.png')
        marked=fullneg.copy();d=ImageDraw.Draw(marked);d.rectangle(box,outline='#9d3e9f',width=max(2,fullneg.width//250))
        marked.thumbnail((520,520));marked.save(ROOT/f'views/{ident}_context.jpg')
        rec=base(ident,'object_region',ident,sources[-1]['work_group'],sources[-1]['source_group'],r['visual_evidence'],
                 object_type='tree',role='foliage' if ident!='qingpu_20_0_region_6' else 'unknown',
                 source_crop_box=box,object_checked=True,region_is_ink_mask=False,
                 local_cunfa='unknown',label_scope='regional_object_only_not_every_pixel',original_reference=r)
        records.append(rec);negatives.append((ident,im))
    # Named illustrations corroborate style at example level only.
    named=read(LAB/'cunfa_role_reference_20261112/source_references.json')['records']
    namedviews=[]
    for ident in ['aam_long_hemp','aam_small_axe']:
        r=next(x for x in named if x['id']==ident);im=source(ident,Path(r['image']),r['source_url'],ident,r['source_group'])
        im.save(ROOT/f'views/{ident}.png');namedviews.append((ident,im))
        records.append(base(ident,'named_style_example',ident,ident,r['source_group'],r['source_evidence'],
            source_named=True,style=r['local_style'],object_type='rock',role='unknown',
            localized_style_target=None,role_checked=False,instance_supervision=False))
    channels={r['id']:supervision_channels(r) for r in records};lineage=audit_lineage(records)
    groups=[dict(id=f'{o}_cun',object_id=o,member_candidates=[r['id'] for r in records if r.get('group_id')==f'{o}_cun'],
                 basis='继承静态形态分组；不是视频确认的真实笔组、局部皴法或画家笔顺',
                 verified_as_artist_group=False) for o in 'ABC']
    packet=dict(version=1,goals=[1,2],sources=sources,records=records,groups=groups,supervision_channels=channels,
        native_crop_box=legacy['box'],native_crop_size=[w,h],unlabelled_is_background=False,
        old_reference_reused_count=27,new_complete_artist_strokes=0,new_independent_works=0,
        named_examples_do_not_label_other_images=True,expert_verified=False,new_model_trained=False,
        robot_connected=False,production_replaced=False)
    write(ROOT/'packet.json',packet);write(ROOT/'curve_audit.json',rows)
    lock=read(LAB/'local_rock_patch_20261117/baseline_lock.json')
    baseline={k:sha(lock[k])==lock[k+'_sha256'] for k in ['approved500','working489']};assert all(baseline.values())
    summary=dict(records=len(records),visible_curve_candidates=27,temporal_role_units=2,foliage_negative_units=4,
        named_style_examples=2,curve_candidates_with_corridor_conflicts=sum(r['conflicting_corridor_pixels']>0 for r in rows),
        conflict_pixels=int((count>1).sum()),new_complete_strokes=0,new_independent_works=0,
        new_model_trained=False,baseline_hash_checks=baseline,independent_validation_passed=False,
        lineage=lineage,robot_connected=False,production_replaced=False)
    write(ROOT/'summary.json',summary)
    readiness=dict(whole_object_context=True,visible_curve_candidates_available=True,
        partial_role_supervision_available=True,object_negative_regions_available=True,
        complete_stroke_instance_training_ready=False,artist_order_training_ready=False,
        physical_training_ready=False,independent_validation_ready=False,
        blocking_gaps=['完整山石逐笔归属未核清','交叠和干笔实例边界仍未知',
                       '跨来源的完整角色参考不足','未封存新的逐区域复核测试作品'],
        no_training_of_complete_instances_from_27_guides=True,recognition_improvement_claimed=False)
    write(ROOT/'training_readiness.json',readiness)
    write(ROOT/'independent_validation_protocol.json',dict(
        status='specified_not_armed_no_new_test_assets_opened_this_round',
        known_packet_usage='development_or_regression_only',
        lock_before_test=['完整作品与作者来源清单','原图与参考哈希','逐区域复核与未知范围',
                          '训练配置和权重哈希','类别映射及拒识阈值','冻结旧版与同条件基线'],
        split_unit='whole_work_and_source_group; artist_disjoint_when_identity_verified',
        correlated_crops_are_not_independent_samples=True,
        report_required=['逐作品混淆矩阵及样本数','披麻与斧劈分别召回','混合与未知误拒/漏拒',
                         '主轮廓与皴笔角色','树叶误入皴笔','完整笔触遗漏/误接/重复',
                         '置信区间及来源局限','全部失败而非只展示最好候选'],
        training_test_labels_from_predictions=False,
        acceptance='先封存带可靠局部参考的新作品；在同一批测试比较冻结基线，验证收益且不损伤主轮廓/树叶；未满足则不接入',
        new_test_count=0,independent_pass=False))
    write(ROOT/'agent_tool_contracts.json',dict(
        status='contracts_only_no_agent_runtime',agent_implemented=False,robot_tools=[],
        tools=[dict(name='lookup_reference',input=['object_type','role','source_group'],
                    output=['source_url','evidence_scope','unknown_fields','allowed_supervision'],
                    cannot_claim='检索命中不代表图像识别'),
               dict(name='get_capability_state',input=[],output=['validated_modules','unready_modules'],
                    cannot_claim='候选模型不冒充已验证技能'),
               dict(name='validate_reference_phase_plan',input=['given_object_ids','given_role_candidates','dependencies'],
                    output=['phase_conflicts','object_switch_reasons','unknowns'],
                    cannot_claim='人工对象输入下的规划不代表自主对象辨识'),
               dict(name='replay_reference',input=['packet_id','plan_id'],
                    output=['reference_canvas','object','phase','provenance'],
                    cannot_claim='参考回放不代表机械臂实时画面')],
        rule_baseline='本证据包固定A/B/C对象，轮廓→结构→皴笔候选；未实现LLM决策',
        independent_planning_test_separate_from_perception=True))
    # Explicitly separate the source, manual guide, source-pixel fragments and uncertainty.
    board=Image.new('RGB',(1500,940),'#f5f2e9')
    for i,(im,title,sub) in enumerate([(crop,'原生来源：三块山石','185×165像素；已看过的旧作品'),
       (Image.fromarray(guide),'27段静态助手候选','棕：轮廓  绿：结构  蓝：皴笔候选'),
       (Image.fromarray(overlap),'紫色：候选走廊交叠','不宣称是实际重复笔画；不提供实例真值'),
       (temporal_views[1][1],'教程加皴阶段：真实截图','51.6–54.5秒；不同视角不硬拼坐标'),
       (temporal_views[1][2],'继承的局部角色参考','未标像素保留未知；不是模型效果'),
       (Image.fromarray(hover),'树叶保护参考','用户纠正的已知错例；紫色不是模型识别')]):
        board.paste(panel(im,title,sub),(i%3*500,i//3*470))
    board.save(ROOT/'overview.jpg',quality=94)
    replay=[]
    for oid in 'ABC':
        for role in ['contour','structure','cun']:
            rs=[r for r in records if r.get('object_id')==oid and r.get('role')==role]
            replay.append(dict(object_id=oid,role=role,curves=[r['curve'] for r in rs],ids=[r['id'] for r in rs]))
    write(ROOT/'reference_replay.json',dict(steps=replay,kind='assistant_reference_planning_only',artist_order=False,object_completion_verified=False))
    data=json.dumps(replay,ensure_ascii=False)
    negativehtml=''.join(f'<article><h3>{html.escape(ident)}</h3><img src="views/{ident}_context.jpg"><img src="views/{ident}.png"><p>已知对象区域参考；不把区域内留白、岩面或染墨自动变成树叶墨迹。</p></article>' for ident,_ in negatives)
    namedhtml=''.join(f'<article><h3>{html.escape(ident)}</h3><img src="views/{ident}.png"><p>教学示例级名称；无真实分笔或局部像素皴法标签。</p></article>' for ident,_ in namedviews)
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>完整山石证据与树叶反例</title>
<style>body{font:17px/1.7 system-ui;max-width:1500px;margin:auto;padding:24px;color:#243d34;background:#f5f2e9}h1{font-size:30px}img{max-width:100%;height:auto}section{margin:30px 0} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(300px,100%),1fr));gap:20px}article{background:white;padding:16px}article img{max-height:420px;object-fit:contain}canvas{max-width:100%;background:white;border:1px solid #ccc}button,input{font:inherit;margin:8px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #bdbdb2}.notice{background:#e8dfcc;padding:16px}a{color:#215d77}</style>
<h1>完整山石证据包：参考、未知与树叶反例</h1><p class="notice">本页是参考整理与监督权限审计，未训练新模型。没有新的识别效果图，未替换500笔/489笔旧版，未连接机械臂。</p>
<p>推进G1皴法证据与G2树石区分。37个参考单位全部来自已检查资料：27条旧静态候选、2个视频阶段核心、4个树木区域、2个具名示例；另有2个阶段图像不计新增单位。实际记录总数以清单为准。</p>
<img src="overview.jpg" alt="原图、助手候选、交叠、视频参考和树叶错例">
<section><h2>按对象和阶段检查参考</h2><p>下方顺序由规则安排：一块石头的轮廓→结构→皴笔候选，再换下一块。不是画家真实笔顺、模型自主规划或机械臂实时画面；该石块未标部分不表示已经画完。</p><button id="play">播放参考</button><button id="reset">重置</button><input id="step" type="range" min="0" max="9" value="0"><span id="label"></span><br><canvas id="canvas" width="740" height="660"></canvas></section>
<section><h2>树木困难反例：保留整幅定位</h2><div class="grid">NEGATIVES</div></section>
<section><h2>具名皴法对照</h2><div class="grid">NAMED</div></section>
<section><h2>实际开放的监督</h2><ul><li>视频核心：仅已有明确角色像素；未知位置无损失。</li><li>树叶区域：只给对象归属和禁止岩石皴笔的区域约束，不给逐像素墨迹答案。</li><li>教学示例：只给本例的名称，不传播到其他山石。</li><li>27条静态曲线：不用于完整笔触实例、真实笔顺或下压学习。</li></ul><p>当前没有新增可靠单笔、独立测试作品或皴法识别能力。下一步需要补可确认的完整山石逐笔过程，先验证墨带与分笔参考，再训练。</p>
<p><a href="packet.json">全部来源、坐标与监督权限</a> · <a href="curve_audit.json">逐曲线交叠审计</a> · <a href="summary.json">本轮统计</a> · <a href="verification.json">验证记录</a></p></section>
<script>const stages=REPLAY; const cols={contour:'#ad6f23',structure:'#3b7a4d',cun:'#2583ae'};const roles={contour:'轮廓候选',structure:'结构候选',cun:'皴笔候选'};const slider=document.querySelector('#step'),canvas=document.querySelector('#canvas'),ctx=canvas.getContext('2d');let timer=null;function render(){let n=+slider.value;ctx.clearRect(0,0,740,660);ctx.fillStyle='white';ctx.fillRect(0,0,740,660);for(const stage of stages.slice(0,n)){ctx.strokeStyle=cols[stage.role];ctx.lineWidth=2;for(const q of stage.curves){ctx.beginPath();q.forEach((p,i)=>{if(i)ctx.lineTo(p[0]*4,p[1]*4);else ctx.moveTo(p[0]*4,p[1]*4)});ctx.stroke()}}document.querySelector('#label').textContent=n?`阶段 ${n}/9 · 山石${stages[n-1].object_id} · ${roles[stages[n-1].role]}`:'尚未播放参考'}slider.oninput=render;document.querySelector('#play').onclick=()=>{if(timer){clearInterval(timer);timer=null;return}timer=setInterval(()=>{if(+slider.value===9){clearInterval(timer);timer=null;return}slider.value=+slider.value+1;render()},700)};document.querySelector('#reset').onclick=()=>{clearInterval(timer);timer=null;slider.value=0;render()};render();</script></html>'''
    doc=doc.replace('NEGATIVES',negativehtml).replace('NAMED',namedhtml).replace('REPLAY',data)
    doc=doc.replace('37个参考单位','35个参考单位').replace('；另有2个阶段图像不计新增单位','')
    doc=doc.replace('<p><a href="packet.json">','<p><a href="training_readiness.json">训练就绪与缺口</a> · <a href="independent_validation_protocol.json">独立验证协议</a> · <a href="agent_tool_contracts.json">Agent工具契约（未实现）</a></p><p><a href="packet.json">')
    (ROOT/'index.html').write_text(doc,encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
