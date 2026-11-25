# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,hashlib,colorsys
import numpy as np,cv2
from PIL import Image,ImageFont,ImageDraw
from scipy.spatial import cKDTree
from source_cycle_repair_20261104 import apply_cycles
from landscape_stroke_refine_20261104 import deduplicate,join_curves
from module_stroke_core_20261104 import samples,length
from source_stroke_width_20261104 import width_profile,render_profiles
from native_tree_restore_20261104 import restore

ROOT=LAB/'ribbon_topology_refinement_20261104';OLD=LAB/'landscape_stroke_refinement_20261104'
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=serial,allow_nan=False),encoding='utf8')
def read(p):return json.loads(p.read_text('utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def scout_tree():
    base=LAB/'native_fine_faint_20261101/outputs_r3/T02';entry=read(base/'entry.json')
    source=Path(entry['source']);assert sha(source)==entry['source_sha256']
    rgb=np.array(Image.open(source).convert('RGB'));doc=read(OLD/'T02/offline_candidates.json')
    factor=np.array(rgb.shape[:2][::-1])/496
    paths=[np.asarray(r['points'],float)*factor for r in doc['curves']]
    fine,diag=restore(rgb,paths,np.array(Image.open(base/'fan_probability.png'))/255,[859,766,1254,961])
    write(ROOT/'native_tree_scout.json',dict(source=str(source),curves=fine,diagnostics=diag,
        frozen_route_sha256=sha(base/'fan_probability.png'),source_sha256=sha(source)))
    print('NATIVE_FINE',len(fine),diag['source_filtered_candidates'],flush=True)

def lines(paths,shape,origin,scale=4):
    out=np.full((shape[0]*scale,shape[1]*scale,3),255,np.uint8)
    for p in paths:cv2.polylines(out,[np.rint((np.asarray(p)-origin)*scale).astype(np.int32)],False,(0,0,0),2,cv2.LINE_AA)
    return out

def view(name,title,rgb,before,after,box,provenance):
    dest=ROOT/name;dest.mkdir(exist_ok=True);x0,y0,x1,y1=box;origin=np.array([x0,y0]);shape=(y1-y0,x1-x0)
    def intersects(p):
        q=samples(p,.4)
        return ((q[:,0]>=x0)&(q[:,0]<x1)&(q[:,1]>=y0)&(q[:,1]<y1)).any()
    old=[p for p in before if intersects(p)];new=[p for p in after if intersects(p)]
    a=lines(old,shape,origin);b=lines(new,shape,origin)
    profiles=[width_profile(rgb,p,recenter=False) for p in new]
    local=[{**p,'points':p['points']-origin} for p in profiles]
    target,count,_,unknown=render_profiles(shape,local)
    native=Image.fromarray(rgb[y0:y1,x0:x1]);native.save(dest/'source.png')
    for name2,im in [('before.png',a),('after.png',b),('width_target.png',target),('uncertain_width.png',unknown)]:Image.fromarray(im).save(dest/name2)
    changes=native.resize((shape[1]*4,shape[0]*4),Image.Resampling.LANCZOS).copy();d=ImageDraw.Draw(changes)
    if provenance.get('kind')=='native_tree':
        for p in provenance['added_paths']:
            q=(np.asarray(p)-origin)*4;d.line([tuple(v) for v in q],fill='#d62d26',width=3)
    else:
        for pos in provenance.get('junctions',[]):
            x,y=(np.asarray(pos)-origin)*4
            if 0<=x<shape[1]*4 and 0<=y<shape[0]*4:d.ellipse([x-8,y-8,x+8,y+8],outline='#d62d26',width=2)
    changes.save(dest/'changes.png')
    board=Image.new('RGB',(1640,460),'#f4f1e8');d=ImageDraw.Draw(board)
    for i,(im,label) in enumerate([(native,'原图'),(Image.fromarray(a),'上轮中心线'),(Image.fromarray(b),'本轮修正候选'),(Image.fromarray(target),'图像相对笔宽目标')]):
        f=min(400/im.width,335/im.height);im=im.resize((round(im.width*f),round(im.height*f)),Image.Resampling.LANCZOS)
        board.paste(im,(410*i+4,43));d.text((410*i+4,9),label,font=FONT,fill='black')
    d.text((8,393),f'{title}：穿过视窗的候选 {len(old)} → {len(new)}；中心线前后同宽显示；笔宽目标不等于下压量。',font=FONT,fill='black')
    d.text((8,425),'原输入和修改记录均保留；新增细线、交汇收拢未经过逐笔专家验证；本轮未接机械臂。',font=FONT,fill='black')
    board.save(dest/'comparison.jpg',quality=96)
    stat=dict(title=title,roi=box,source_size=list(rgb.shape[:2][::-1]),before_count=len(old),after_count=len(new),
        observable_width_fraction=sum(int(p['observed'].sum()) for p in profiles)/max(1,sum(len(p['observed']) for p in profiles)),
        observed_fraction_is_not_accuracy=True,physical_pressure_mm=None,ready_for_robot=False)
    write(dest/'local_targets.json',dict(statistics=stat,paths=new,width_profiles=profiles,provenance=provenance));return stat

def main():
    ROOT.mkdir(exist_ok=True);cv2.setNumThreads(2)
    scout_tree()
    olddoc=read(OLD/'G03_01/offline_candidates.json');rgb=np.array(Image.open(olddoc['source']).convert('RGB'))
    oldrows=[{**r,'points':np.asarray(r['points'],float)} for r in olddoc['curves']]
    repaired,cycles=apply_cycles(rgb,oldrows)
    rows,duplicates=deduplicate(rgb,[r['points'] for r in repaired],[r['module'] for r in repaired])
    for r in rows:r['source_curve_ids']=[v for i in r['source_curve_ids'] for v in repaired[i]['source_curve_ids']]
    rows,joins=join_curves(rgb,rows)
    before=[r['points'] for r in oldrows];after=[r['points'] for r in rows]
    assert sorted(v for r in rows for v in r['source_curve_ids'])==list(range(len(olddoc['original_curves'])))
    _,remaining=__import__('source_cycle_repair_20261104').inspect_cycles(rgb,after)
    write(ROOT/'G03_01_candidates.json',dict(source=olddoc['source'],source_sha256=sha(Path(olddoc['source'])),
        parent_file=str(OLD/'G03_01/offline_candidates.json'),parent_sha256=sha(OLD/'G03_01/offline_candidates.json'),
        before=before,curves=rows,cycle_diagnostics=cycles,duplicate_diagnostics=duplicates,join_diagnostics=joins,
        source_id_partition_verified=True,coordinate_system='image pixels only',physical_pressure_mm=None,ready_for_robot=False,
        new_center_network_applied=False))
    junctions=[r['center'] for r in cycles['applied_junctions']]
    panels=[view('rock','山石交汇与皴线',rgb,before,after,[178,285,326,389],dict(kind='cycle',junctions=junctions)),
        view('pavilion','亭子屋檐与梁柱',rgb,before,after,[108,198,231,299],dict(kind='cycle',junctions=junctions))]
    # Native source is preserved. New details are outside the old provenance IDs.
    scout=read(ROOT/'native_tree_scout.json');tree=np.array(Image.open(scout['source']).convert('RGB'))
    previous=read(OLD/'T02/offline_candidates.json');factor=np.array(tree.shape[:2][::-1])/496
    oldtree=[np.asarray(r['points'],float)*factor for r in previous['curves']]
    added=[np.asarray(r['points'],float) for r in scout['curves']];newtree=oldtree+added
    panels.append(view('pine','原生分辨率松枝',tree,oldtree,newtree,[859,766,1254,961],dict(kind='native_tree',added_paths=added)))
    write(ROOT/'T02_native_candidates.json',dict(source=scout['source'],source_sha256=sha(Path(scout['source'])),
        original_native_size=list(tree.shape[:2][::-1]),original_curves_native=oldtree,new_fine_candidates=scout['curves'],
        fine_diagnostics=scout['diagnostics'],new_details_are_proposals=True,physical_pressure_mm=None,ready_for_robot=False,
        frozen_tree_router_reused=True,new_center_network_applied=False))
    train=read(ROOT/'training_summary.json');evaluation=read(ROOT/'evaluation.json');protocol=read(ROOT/'protocol.json')
    summary=dict(G03_input=len(before),G03_output=len(after),applied_small_junctions=len(junctions),
        graph_absorbed_junction_only_curves=cycles['collapsed_zero_length_curves_absorbed'],
        remaining_eligible_small_cycles=len(remaining),native_tree_added_candidates=len(added),
        network_training=train,network_calibration_passed=train['center_proposal_calibration_gate_passed'],
        network_applied_to_outputs=False,views=panels,robot_connected=False,expert_stroke_accuracy=None,physical_improvement_verified=False)
    write(ROOT/'results.json',summary)
    cards=''.join(f'<h2>{r["title"]}</h2><img src="{key}/comparison.jpg"><p><a href="{key}/changes.png">红色标出改动位置</a> · <a href="{key}/uncertain_width.png">笔宽不确定区域</a> · <a href="{key}/local_targets.json">曲线与相对笔宽数据</a></p>' for key,r in zip(['rock','pavilion','pine'],panels))
    cal=evaluation['reviewed_calibration'];test=evaluation['reviewed_test']
    page=f'<meta charset="utf-8"><title>原图支持的闭环与松针修正</title><style>body{{font:17px/1.6 system-ui;max-width:1700px;padding:24px;margin:auto;background:#f4f1e8;color:#273c32}}img{{width:100%;background:white}}.note{{padding:16px;background:#e4e9de}}td,th{{padding:10px;border:1px solid #bbb}}table{{border-collapse:collapse}}a{{color:#145b82}}</style><h1>小闭环、粗墨交汇与原生松针 · 离线候选</h1><p class="note">G03：{len(before)} → {len(after)}条候选；{len(junctions)}处极小交汇收拢。原图内有亮缝的闭合形状保留，重叠或看不清的修正暂挂。T02：回到1254×1254原图，在固定开发窗口新增{len(added)}条源图支持的细线候选，未将粗分辨率放大当作新细节。</p>'
    page+='<h2>训练结果与采用决定</h2><p>本轮新训横截面中心定位网络：34,402参数，RTX 5060 Laptop GPU，600次更新。只使用已检查的确定横截面标签和已知渲染中心的合成横截面，不确定标签排除；未标偏移不作为真实背景。</p>'
    page+=f'<table><tr><th>明确复核横截面</th><th>现有解析中心平均误差</th><th>新网络平均误差</th></tr><tr><td>N05校准 · {cal["stroke_sections"]}条</td><td>{cal["analytic_center_mae_px"]:.2f}px</td><td>{cal["learned_center_mae_px"]:.2f}px</td></tr><tr><td>N06/N09模块留出 · {test["stroke_sections"]}条</td><td>{test["analytic_center_mae_px"]:.2f}px</td><td>{test["learned_center_mae_px"]:.2f}px</td></tr></table>'
    page+='<p class="note">新网络未通过校准，因此没有用于本页修正结果。上述误差按统一±8px中心搜索范围统计，含低置信漏检和范围外参考，不能解释为整图准确率。留出图片可能同属一个出版来源，也不代表独立画风泛化。本页改动来自原图亮孔检查、几何去重续接和既有树木路由约束，不宣称新网络改善了提取。</p>'
    page+=cards+'<p class="note">三个窗口都是已见图的开发对照；新增细线和交汇收拢没有专家逐笔真值。亮孔检查、树木路由仍会有误判。相对笔宽预览为黑色几何墨带，不模拟真实墨色、含墨量或晕染。当前毛笔未重新测得细线能力，未替换上机任务。</p><p><a href="protocol.json">监督来源与划分</a> · <a href="evaluation.json">完整训练评估</a> · <a href="results.json">本轮统计</a></p>'
    (ROOT/'index.html').write_text(page,encoding='utf8');print(json.dumps({k:v for k,v in summary.items() if k not in ['views','network_training']}),flush=True)
    print('REPORT',ROOT/'index.html',flush=True)

if __name__=='__main__':main()
