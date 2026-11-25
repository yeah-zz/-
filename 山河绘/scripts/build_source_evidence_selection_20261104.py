# -*- coding: utf-8 -*-
"""Archive-preserving whole-curve selection and explicit uncertainty viewer."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,hashlib,shutil
from collections import Counter
import numpy as np,cv2
from PIL import Image,ImageFont,ImageDraw
from source_evidence_filter_20261104 import filter_curves,CONFIG
from evaluate_source_evidence_filter_20261104 import evaluate
from module_stroke_core_20261104 import samples,length

ROOT=LAB/'source_evidence_selection_20261104'
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
def read(p):return json.loads(p.read_text('utf8'))
def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,default=serial,allow_nan=False),encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def scene(ident):
    previous=LAB/'landscape_stroke_refinement_20261104'
    if ident=='G03_01':
        p=LAB/'ribbon_topology_refinement_20261104/G03_01_candidates.json';d=read(p)
        return Path(d['source']),d['curves'],[p]
    if ident=='G05_03':
        p=previous/ident/'offline_candidates.json';d=read(p)
        return Path(d['source']),d['curves'],[p]
    p=LAB/'ribbon_topology_refinement_20261104/T02_native_candidates.json';d=read(p)
    old=read(previous/ident/'offline_candidates.json')
    rows=[{**r,'points':q} for r,q in zip(old['curves'],d['original_curves_native'])]
    rows += [{**r,'module':-1} for r in d['new_fine_candidates']]
    return Path(d['source']),rows,[p,previous/ident/'offline_candidates.json']

def drawing(rows,ids,shape,origin=(0,0),color=(0,0,0),image=None):
    scale=3;origin=np.asarray(origin)
    out=np.full((shape[0]*scale,shape[1]*scale,3),255,np.uint8) if image is None else image.copy()
    for i in ids:
        q=(np.asarray(rows[i]['points'])-origin)*scale
        cv2.polylines(out,[np.rint(q).astype(np.int32)],False,color,2,cv2.LINE_AA)
    return out

def panels(folder,key,title,rgb,rows,result,box):
    dest=folder/key;dest.mkdir(exist_ok=True);x0,y0,x1,y1=box;shape=(y1-y0,x1-x0);origin=(x0,y0)
    source=Image.fromarray(rgb[y0:y1,x0:x1]);source.save(dest/'source.png')
    allids=list(range(len(rows)));certain=[d['curve'] for d in result['decisions'] if d['disposition']=='keep']
    unknown=[d['curve'] for d in result['decisions'] if d['disposition']=='uncertain']
    before=drawing(rows,allids,shape,origin);after=drawing(rows,result['selected'],shape,origin)
    reliable=drawing(rows,certain,shape,origin)
    removed=drawing(rows,result['removed'],shape,origin,color=(210,48,30),image=after)
    pending=drawing(rows,unknown,shape,origin,color=(225,151,35),image=reliable)
    overlay=np.asarray(source.resize((shape[1]*3,shape[0]*3),Image.Resampling.LANCZOS)).copy()
    overlay=drawing(rows,result['removed'],shape,origin,color=(225,40,30),image=overlay)
    for name,im in [('before.png',before),('selected.png',after),('strong_only.png',reliable),('removed_overlay.png',removed),('pending_overlay.png',pending),('source_removed.png',overlay)]:Image.fromarray(im).save(dest/name)
    board=Image.new('RGB',(1640,455),'#f4f1e8');d=ImageDraw.Draw(board)
    for k,(im,label) in enumerate([(source,'原图'),(Image.fromarray(before),'完整候选'),(Image.fromarray(after),'保守筛选（待定线保留）'),(Image.fromarray(removed),'红色：排除路径')]):
        factor=min(400/im.width,335/im.height);im=im.resize((max(1,round(im.width*factor)),max(1,round(im.height*factor))),Image.Resampling.LANCZOS)
        board.paste(im,(k*410+3,42));d.text((k*410+3,8),label,font=FONT,fill='black')
    d.text((8,392),title+'｜所有中心线使用相同显示宽度；保留曲线不改形，不按笔数预算、长短或浓淡单独删除。',font=FONT,fill='black')
    d.text((8,423),'红色为算法排除候选，不是专家确认的噪声；待定路径默认保留。仅离线图像结果，非毛笔实画。',font=FONT,fill='black')
    board.save(dest/'comparison.jpg',quality=96)

def build(ident):
    source,rows,parents=scene(ident);rgb=np.array(Image.open(source).convert('RGB'))
    folder=ROOT/ident;folder.mkdir(exist_ok=True);result=filter_curves(rgb,rows)
    count=Counter(d['disposition'] for d in result['decisions'])
    total=sum(length(np.asarray(r['points'],float)) for r in rows)
    retained=sum(length(np.asarray(rows[i]['points'],float)) for i in result['selected'])
    stats=dict(id=ident,input_curves=len(rows),selected_curves=len(result['selected']),strong_curves=count['keep'],
        uncertain_retained=count['uncertain'],rejected_source_evidence=count['reject'],duplicate_curves=count['duplicate'],
        selected_length_fraction=retained/max(1e-8,total),
        protected_contour_candidates=sum(d['contour_guard'] for d in result['decisions']),
        protected_texture_candidates=sum(d['texture_guard'] for d in result['decisions']),
        guarded_candidates_rejected_as_noise=sum((d['contour_guard'] or d['texture_guard']) and d['disposition']=='reject' for d in result['decisions']),
        geometry_of_selected_curves_unchanged=True,raw_input_archive_preserved=True,uncertain_retained_by_default=True,
        semantic_roles_verified=False,all_true_cun_preserved_not_established=True,
        input_seen_before=True,new_blind_test=False,model_weights_trained=False,ready_for_robot=False,robot_connected=False)
    archive=dict(source=str(source),source_sha256=sha(source),parents=[dict(path=str(p),sha256=sha(p)) for p in parents],
        input_curves=rows,selection=result,statistics=stats,coordinate_system='image pixels only',
        physical_pressure_mm=None,ready_for_robot=False)
    save(folder/'candidates.json',archive);save(folder/'statistics.json',stats)
    panels(folder,'full',ident,rgb,rows,result,[0,0,rgb.shape[1],rgb.shape[0]])
    if ident=='G03_01':
        panels(folder,'rock','山石轮廓与内部线',rgb,rows,result,[178,285,326,389])
        panels(folder,'pavilion','亭子结构',rgb,rows,result,[108,198,231,299])
        panels(folder,'tree','树枝局部',rgb,rows,result,[278,121,488,240])
    if ident=='T02':panels(folder,'tree','原生松枝',rgb,rows,result,[859,766,1254,961])
    print(ident,dict(count),'selected',len(result['selected']),flush=True);return stats

def main():
    ROOT.mkdir(exist_ok=True);cv2.setNumThreads(2)
    # The reference check is separate from inference; annotations are never
    # passed into filter_curves and do not change the frozen parameter set.
    if '--report-only' in sys.argv:
        reference=read(ROOT/'reference_regression.json');assert reference['configuration']==CONFIG
        stats=[read(ROOT/i/'statistics.json') for i in ['G03_01','G05_03','T02']]
    else:
        reference=evaluate();stats=[build(i) for i in ['G03_01','G05_03','T02']]
    uncertain_reference=sum(r['positive_dispositions'].get('uncertain',0) for r in reference['records'])
    save(ROOT/'results.json',dict(records=stats,reference_positive_curves=reference['user_positive_curves'],
        reference_positive_kept=reference['user_positive_kept'],parameter_fitting_performed=False,
        real_reference_curves_in_uncertain_layer=uncertain_reference,strong_only_variant_rejected=True,
        deployment='offline_candidates_only',robot_connected=False))
    data=json.dumps(stats,ensure_ascii=False)
    table=''.join(f'<tr><td>{r["id"]}</td><td>{r["input_curves"]}</td><td>{r["selected_curves"]}</td><td>{r["rejected_source_evidence"]}</td><td>{r["duplicate_curves"]}</td><td>{r["uncertain_retained"]}</td><td>{r["selected_length_fraction"]:.1%}</td></tr>' for r in stats)
    html='''<!doctype html><meta charset="utf-8"><title>保留主轮廓与皴线的源图筛选</title>
<style>body{font:17px/1.65 system-ui;color:#263c33;background:#f4f1e8;max-width:1700px;margin:auto;padding:24px}img{width:100%;background:white}.note{padding:16px;background:#e1e8dd}button,select{font:inherit;padding:7px}td,th{padding:9px;border:1px solid #bdc7ba}table{border-collapse:collapse}a{color:#155e83}</style>
<h1>减少无关候选，保护主轮廓与可见皴线</h1><p class="note">按完整候选检查自适应墨迹证据、方向性、轮廓边缘和同墨带重复；没有线条数量上限。真实平行线有亮缝时保留，短松针和苔点不因长度被删除。几何保护规则不等于已识别皴法类别。</p>
<table><tr><th>图</th><th>完整候选</th><th>默认保留</th><th>源图证据不足</th><th>墨带重复</th><th>待定仍保留</th><th>保留总长度</th></tr>__TABLE__</table>
<p>图像 <select id="scene"><option>G03_01</option><option>G05_03</option><option>T02</option></select>　视图 <select id="mode"><option value="comparison.jpg">同图对照</option><option value="before.png">完整候选</option><option value="selected.png">保守筛选：默认结果</option><option value="removed_overlay.png">红色：排除候选</option><option value="source_removed.png">原图上的排除位置</option><option value="pending_overlay.png">橙色：待定路径（默认保留）</option></select></p>
<p id="count"></p><img id="board"><p><a id="audit">每条候选的保留理由与完整档案</a></p>
<h2>山石与树木局部</h2><img src="G03_01/rock/comparison.jpg"><img src="T02/tree/comparison.jpg">
<h2>保留参考回查</h2><p>__REF__。只检查明确的用户描线和背景，未标区域不当作负例。这些图片同属旧来源，检查不等于独立整图准确率。</p>
<p class="note">背景区探针是随机生成的测试曲线，并不是实际检测出的错误线。其排除率仍有限，说明这层筛选偏保守。待定路径可能包含真实淡线、细皴线或噪声，不能统一称为无关线。未采用关闭待定线的清爽版：参考中有__UNKNOWN__条已检查的真实描线属于这一层，关闭会造成漏线。</p>
<p><a href="reference_regression.json">完整参考检查</a> · <a href="results.json">本轮统计</a></p>
<p>正式候选与原始轨迹均保留；被排除路径可恢复。所示为相同宽度的中心线，不是实际墨迹、笔宽或机械臂实画。未建立当前纸笔墨的压力映射，本轮不连接机械臂。</p>
<script>const DATA=__DATA__;const scene=document.getElementById('scene'),mode=document.getElementById('mode');function render(){const id=scene.value,r=DATA.find(x=>x.id===id);document.getElementById('board').src=id+'/full/'+mode.value;document.getElementById('audit').href=id+'/candidates.json';document.getElementById('count').textContent=`${id}：完整 ${r.input_curves}，默认保留 ${r.selected_curves}，其中待定 ${r.uncertain_retained}；排除不代表全部是专家确认噪声。`;}scene.onchange=render;mode.onchange=render;render();</script>'''
    bgremoved=sum(r['background_probe_removed'] for r in reference['records']);bgcount=sum(r['explicit_background_probe_curves'] for r in reference['records'])
    ref=f'此前明确描绘的{reference["user_positive_curves"]}条曲线，本轮保留{reference["user_positive_kept"]}条；在明确背景区生成{bgcount}条测试探针，排除{bgremoved}条'
    html=html.replace('__TABLE__',table).replace('__REF__',ref).replace('__UNKNOWN__',str(uncertain_reference)).replace('__DATA__',data)
    (ROOT/'index.html').write_text(html,encoding='utf8')
    snap=ROOT/'source_snapshot';snap.mkdir(exist_ok=True)
    for name in ['source_evidence_filter_20261104.py','evaluate_source_evidence_filter_20261104.py','build_source_evidence_selection_20261104.py']:
        shutil.copy2(Path(__file__).with_name(name),snap/name)
    print('REPORT',ROOT/'index.html',flush=True)

if __name__=='__main__':main()
