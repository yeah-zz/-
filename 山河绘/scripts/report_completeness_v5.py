# -*- coding: utf-8 -*-
"""Interactive offline paired report; no gold labels or accuracy claims."""
import stroke_completeness_v5 as implementation
from stroke_completeness_v5 import *
from rock_v15_common import read
from PIL import ImageDraw,ImageFont
import xml.etree.ElementTree as ET
import statistics,shutil

def font(size):return ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc',size)
def bw(folder,name):return np.asarray(Image.open(folder/name).convert('L'))<128

def generate():
    result=read(ROOT/'validation_results.json');manifest=read(ROOT/'validation_manifest.json');protocol=read(ROOT/'fixed_protocol.json')
    assert result['complete'] and len(result['records'])==24
    assert digest(ROOT/'validation_manifest.json')==protocol['selection_manifest_sha256']
    for r in protocol['code']:assert digest(r['path'])==r['sha256']
    rows=result['records'];detail_identical=[];sections=[];source_lookup={r['id']:r for r in manifest['records']}
    for r in rows:
        sid=r['id'];folder=ROOT/'validation_outputs'/sid;rgb=np.asarray(Image.open(folder/'v5'/'input.png').convert('RGB'))
        a=bw(folder/'v4','lines.png');b=bw(folder/'v5','lines.png')
        same=digest(folder/'v4'/'detail_lines.png')==digest(folder/'v5'/'detail_lines.png');detail_identical.append(same)
        for version in ['v4','v5']:
            doc=read(folder/version/'paths.json')
            assert doc['meta']['sha256']==r['source_sha256'] and not doc['reference_labels_used'] and not doc['ready_for_robot']
            assert doc['meta']['new_training_steps']==0
            tree=ET.parse(folder/version/'lines.svg');assert len(tree.findall('.//{http://www.w3.org/2000/svg}polyline'))==len(doc['paths'])
        # Antialiased raw centerline masks become display overlays only.
        for version in ['v4','v5']:
            mask=bw(folder/version,'lines.png');alpha=np.where(mask,255,0).astype(np.uint8);rgba=np.zeros((*alpha.shape,4),np.uint8);rgba[:,:,3]=alpha
            Image.fromarray(rgba,'RGBA').save(folder/(version+'_transparent.png'))
        dista=distance_transform_edt(~a);distb=distance_transform_edt(~b);added=b&(dista>1.5);removed=a&(distb>1.5)
        changes=rgb.copy();changes[added]=[25,100,240];changes[removed]=[230,110,10];Image.fromarray(changes).save(folder/'changes.png')
        stats4=r['v4'];stats5=r['v5'];src=source_lookup[sid]
        r['detail_output_bitwise_unchanged']=same
        r['new_center_pixels_farther_than_1_5px_from_v4']=int(added.sum());r['removed_center_pixels_farther_than_1_5px_from_v5']=int(removed.sum())
        title=html.escape(src['title']);source=html.escape(src['source_page'],quote=True)
        gallery=''.join(f'<figure><figcaption>{label}</figcaption><a href="validation_outputs/{sid}/{name}"><img loading="lazy" src="validation_outputs/{sid}/{name}"></a></figure>' for name,label in [('v5/input.png','同一原图'),('v4/lines.png','v4：用户认可基线'),('v5/lines.png','v5：本轮结果'),('changes.png','蓝：新增；橙：减少（位置差异）'),('v5/all_evidence.png','灰：原图待定骨架'),('v5/topology_ink.png','派生拓扑墨迹，原始墨迹另存')])
        sections.append(f'<details id="{sid}"><summary>{sid} — {title}</summary><p><a href="{source}">馆藏原始来源</a> · <a href="validation_outputs/{sid}/v5/index.html">完整导出</a> · <a href="validation_outputs/{sid}/v5/paths.json">路径与续接记录</a></p><p>结构路径 {stats4["structure_path_count"]} → {stats5["structure_path_count"]}；续接假设 {stats4["accepted_joins"]} → {stats5["accepted_joins"]}；细节层像素一致：{same}。这些数字不是识别准确率。</p><div class="grid">{gallery}</div><p>拖动控制新线稿在原图上的不透明度：<input type="range" min="0" max="1" step="0.01" value="0.8" oninput="document.getElementById(\'overlay_{sid}\').style.opacity=this.value"></p><div class="stack"><img loading="lazy" src="validation_outputs/{sid}/v5/input.png"><img id="overlay_{sid}" class="top" loading="lazy" style="opacity:.8" src="validation_outputs/{sid}/v5_transparent.png"></div></details>')
    assert all(detail_identical)
    # Prespecified showcase cases, selected by source content before inference.
    selected=protocol['showcase_ids_prespecified'];board=Image.new('RGB',(1944,2250),'#f3f1eb');d=ImageDraw.Draw(board)
    d.text((20,14),'新真实图片：原图 / v4 固定基线 / v5 本轮结果',font=font(30),fill='black')
    d.text((20,62),'展示样本在新图推理前确定；全部 24 张结果见报告。统一细线显示，无人工补线。',font=font(22),fill='#444444')
    for i,sid in enumerate(selected):
        y=105+i*700;d.text((20,y),sid,font=font(23),fill='black')
        for j,name in enumerate(['v5/input.png','v4/lines.png','v5/lines.png']):
            with Image.open(ROOT/'validation_outputs'/sid/name) as im:
                im=im.convert('RGB');im.thumbnail((620,640),Image.Resampling.LANCZOS);board.paste(im,(20+j*642+(620-im.width)//2,y+40+(640-im.height)//2))
    board.save(ROOT/'NEW_REAL_COMPARISON.jpg',quality=96,subsampling=0)
    # Preserve the accepted pine detail crop and examine a fixed long-line crop.
    crop=Image.new('RGB',(1765,1030),'#f3f1eb');d=ImageDraw.Draw(crop);d.text((15,10),'旧图局部复测：原图 / v4 / v5',font=font(28),fill='black');y=60
    for sid,label,box in [('02_pine','松针细节保持',(130,240,705,565)),('01_rocks','山石长线与浅墨',(330,450,905,920))]:
        d.text((15,y),label,font=font(23),fill='black');y+=36
        folders=[LAB/'generated_simple_landscapes_20261028'/'originals'/(sid+'.png'),LAB/'split_detail_structure_20261028'/'v4'/sid/'lines.png',ROOT/'development_final'/sid/'lines.png']
        for i,p in enumerate(folders):
            with Image.open(p) as im:crop.paste(im.convert('RGB').crop(box),(15+i*585,y))
        y+=box[3]-box[1]+24
    crop.save(ROOT/'DEVELOPMENT_LOCAL_COMPARISON.jpg',quality=97,subsampling=0)
    checks=read(ROOT/'behavior_checks.json');assert checks['all_passed']
    old_lock=read(LAB/'stable_line_extractor_20261028'/'baseline_lock.json');assert all(digest(r['path'])==r['sha256'] for r in old_lock['files'])
    assert digest(LAB/'split_detail_structure_20261028'/'v4'/'source_snapshot.py')==digest(Path(implementation.__file__).with_name('split_detail_v4_frozen.py'))
    totals={k:sum(r['v5']['statistics'].get(k,0) for r in rows) for k in ['weak_anchor_candidates','tiny_void_hypotheses']}
    summary=dict(real_images=24,detail_outputs_identical_count=sum(detail_identical),behavior_check_count=len(checks['checks']),behavior_checks_passed=True,totals_are_hypotheses_not_correctly_recognized_strokes=totals,mean_source_skeleton_retention_v4=float(np.mean([r['v4']['source_skeleton_retention_within_2px'] for r in rows])),mean_source_skeleton_retention_v5=float(np.mean([r['v5']['source_skeleton_retention_within_2px'] for r in rows])),median_structure_length_v4=float(np.median([r['v4']['median_structure_length'] for r in rows])),median_structure_length_v5=float(np.median([r['v5']['median_structure_length'] for r in rows])),small_enclosed_regions_outside_detail_v4=sum(r['v4']['small_enclosed_regions_outside_detail'] for r in rows),small_enclosed_regions_outside_detail_v5=sum(r['v5']['small_enclosed_regions_outside_detail'] for r in rows),fixed_protocol_verified=True,previous_weights_and_v4_verified=True,new_training_steps=0,automatic_semantic_cunfa_recognition=False,metrics_are_no_gold_proxies=True,ready_for_robot=False)
    summary.update(accepted_v4_replaced=False,candidate_promoted=False,small_cycle_nonincrease_check_passed=summary['small_enclosed_regions_outside_detail_v5']<=summary['small_enclosed_regions_outside_detail_v4'],decision='retain accepted v4; v5 remains a diagnostic experiment without verified overall quality gain')
    jsave(ROOT/'final_summary.json',summary);jsave(ROOT/'paired_review_records.json',rows)
    summarytext=f'共完成 24 张真实图的同图对照；{sum(detail_identical)}/24 张细节层与 v4 像素一致，{len(checks["checks"])} 项行为检查通过。它们验证的是程序行为，不是画作识别精度。'
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>线条完整性 v5 对照</title><style>body{font:16px system-ui,"Microsoft YaHei",sans-serif;background:#f3f1eb;color:#222;margin:24px;line-height:1.7}h1{font-size:26px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}figure{margin:0}img{max-width:100%;background:white}figure img{width:100%}details{padding:14px 0;border-top:1px solid #ccc}summary{cursor:pointer;font-weight:bold}.stack{position:relative;max-width:1000px}.stack img{display:block;width:100%}.stack .top{position:absolute;top:0;left:0;background:transparent}a{color:#075ea2}p{max-width:1100px}@media(max-width:850px){.grid{grid-template-columns:1fr}}</style><h1>v5：断口、淡线与粗墨拓扑实验</h1><p>冻结用户认可的 v4，在旧开发图上完成修改后，固定参数处理 24 张新图片；新图未用于参数或性能选择。第 19 张图片触发边界索引错误，随后只增加越界拒绝检查并完整复跑 v5，修订前文件保留，详见冻结协议。来源为克利夫兰艺术博物馆与大都会艺术博物馆，含清晰册页、画谱以及复杂绢本。同册页有关联，24 张不等于 24 位作者或 24 个独立风格。纯文字页、书封和已知旧作品已排除。历史训练裁剪的完整作品溯源仍不完备，不称为严格统计盲测。</p><p>改动：浅墨候选须邻接已有强笔画且方向相符；短断口在窄走廊内寻找双侧墨带证据；有方向性的粗墨带中仅试验处理很小的封闭空洞，保留原始墨迹。树冠细节区保持不变。所有续接和微孔修正都是几何假设，未证明真实笔顺。CPU 图像处理，未新增训练权重。</p><p>'''+summarytext+'''</p><p>整体变化有限，仍存在碎线、漏提、粗墨交汇和密集区域网状线。恢复更多浅线也可能带来误检；当前不覆盖 v4。标题、印章、装裱边框尚未自动排除。没有完整人工真值，不提供准确率或泛化提升百分比。</p><p><a href="NEW_REAL_COMPARISON.jpg">预先指定的三张新图对照</a> · <a href="DEVELOPMENT_LOCAL_COMPARISON.jpg">旧图局部对照</a> · <a href="validation_manifest.json">全部来源及重复检查</a> · <a href="fixed_protocol.json">冻结协议</a> · <a href="final_summary.json">结果记录</a></p>'''+''.join(sections)+'''<h2>下一阶段依据</h2><p>灰色待定层不能自动当背景；墨迹骨架覆盖率不是人类笔画召回率；路径数与续接数不等于正确分笔。粗墨网状部分仍需要更可靠的中心和交汇解释；应先在明确失败区域建立独立对照，再决定是否训练新的神经模型。</p></html>'''
    (ROOT/'index.html').write_text(page,encoding='utf-8')
    print('V5_REPORT_READY',str(ROOT/'index.html'),json.dumps(summary),flush=True)

if __name__=='__main__':generate()
