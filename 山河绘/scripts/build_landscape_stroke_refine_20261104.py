# -*- coding: utf-8 -*-
"""Full-image and local same-width comparisons, offline candidates only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,hashlib,time,colorsys,html
import numpy as np,cv2
from scipy.spatial import cKDTree
from PIL import Image,ImageFont,ImageDraw
from landscape_stroke_refine_20261104 import CONFIG,refine
from module_stroke_core_20261104 import samples,length
from source_stroke_width_20261104 import width_profile,render_profiles

ROOT=LAB/'landscape_stroke_refinement_20261104'
BASE=LAB/'module_stroke_planner_20261104/outputs'
PRIOR=LAB/'pavilion_source_strokes_20261104/pavilion/candidate_strokes.json'
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)

def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)
def write(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=serial,allow_nan=False),encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def drawing(paths,shape,origin=(0,0),colors=False):
    # Identical display width on both sides, independent from contact targets.
    scale=4;out=np.full((shape[0]*scale,shape[1]*scale,3),255,np.uint8)
    for i,p in enumerate(paths):
        c=tuple(round(v*190) for v in colorsys.hsv_to_rgb(i*.618%1,.8,1)) if colors else (0,0,0)
        q=(np.asarray(p)-np.array(origin))*scale
        cv2.polylines(out,[np.rint(q).astype(np.int32)],False,c,2,cv2.LINE_AA)
    return out

def order(rows,parent_actions):
    modules=list(dict.fromkeys(r['module'] for r in parent_actions))
    modules+=sorted(set(r['module'] for r in rows)-set(modules))
    cursor=np.array([0.,495.]);out=[]
    for m in modules:
        left=[i for i,r in enumerate(rows) if r['module']==m]
        while left:
            starts=np.array([rows[i]['points'][0] for i in left]);ends=np.array([rows[i]['points'][-1] for i in left])
            d0=np.linalg.norm(starts-cursor,axis=1);d1=np.linalg.norm(ends-cursor,axis=1)
            lens=np.array([length(rows[i]['points']) for i in left]);s=lens/(1+np.minimum(d0,d1)/20)
            k=int(s.argmax());i=left.pop(k);reverse=bool(d1[k]<d0[k]);p=rows[i]['points']
            out.append(dict(curve=i,module=m,reverse=reverse,air_travel_px=float(min(d0[k],d1[k]))));cursor=p[0] if reverse else p[-1]
    return out

def curves_stats(paths):
    v=np.array([length(p) for p in paths])
    return dict(count=len(paths),length_total_px=float(v.sum()),length_median_px=float(np.median(v)),
        under_5px=int((v<5).sum()),over_25px=int((v>25).sum()))

def local_view(folder,key,title,box,image,old,new,profiles):
    x0,y0,x1,y1=box;shape=(y1-y0,x1-x0);origin=np.array([x0,y0])
    def contained(p):
        q=samples(p,.5);return bool(((q[:,0]>=x0)&(q[:,0]<x1)&(q[:,1]>=y0)&(q[:,1]<y1)).any())
    oi=[i for i,p in enumerate(old) if contained(p)]
    ni=[i for i,p in enumerate(new) if contained(p)]
    before=drawing([old[i] for i in oi],shape,origin)
    after=drawing([new[i] for i in ni],shape,origin)
    units=drawing([new[i] for i in ni],shape,origin,colors=True)
    selected=[{**profiles[i],'points':profiles[i]['points']-origin} for i in ni]
    target,counts,_,unknown=render_profiles(shape,selected)
    dest=folder/key;dest.mkdir(exist_ok=True)
    source=Image.fromarray(image[y0:y1,x0:x1]);source.save(dest/'source.png')
    for name,im in [('before.png',before),('after.png',after),('units.png',units),('width_target.png',target),('uncertain_width.png',unknown)]:Image.fromarray(im).save(dest/name)
    board=Image.new('RGB',(1680,480),'#f4f1e8');d=ImageDraw.Draw(board)
    for k,(im,label) in enumerate([(source,'原图'),(Image.fromarray(before),'修改前中心线'),(Image.fromarray(after),'修改后中心线'),(Image.fromarray(units),'连续候选（颜色分笔）')]):
        factor=min(405/im.width,355/im.height)
        im=im.resize((max(1,round(im.width*factor)),max(1,round(im.height*factor))),Image.Resampling.LANCZOS)
        board.paste(im,(k*420+5,46));d.text((k*420+5,10),label,font=FONT,fill='black')
    d.text((8,416),f'{title}｜穿过视窗的候选 {len(oi)} → {len(ni)}；前后同一显示线宽；跨边界曲线只裁切显示，不删改。',font=FONT,fill='black')
    d.text((8,449),'颜色仅区分计划中的笔画，不表示皴法类别。几何对照，不是实画，也不是墨水仿真。',font=FONT,fill='black')
    board.save(dest/'comparison.jpg',quality=95)
    return dict(key=key,title=title,roi=box,input_curves=len(oi),output_curves=len(ni),manual_development_roi=True)

def build(ident):
    folder=ROOT/ident;folder.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    doc=json.loads((BASE/ident/'offline_plan.json').read_text('utf8'))
    src=BASE/ident/'source.png';rgb=np.array(Image.open(src).convert('RGB'))
    original=[np.array(r['points'],float) for r in doc['curves']]
    staged=[dict(points=p,source_curve_ids=[i],module=doc['curves'][i]['module']) for i,p in enumerate(original)]
    flags=[True]*len(staged);arch_count=0;architecture_excluded=0
    if ident=='G03_01':
        prior=json.loads(PRIOR.read_text('utf8'));ids=prior['original_curve_indices'];lookup=set(ids)
        # Previous architectural rules remain explicit development proposals.
        # Keep only same-owner replacements; do not force mask conflicts together.
        replace=[];consumed=set()
        for row in prior['new_candidates']:
            sources=[ids[k] for k in row['source_curve_ids']]
            owners=set(doc['curves'][k]['module'] for k in sources)
            if len(owners)>1:architecture_excluded+=1;continue
            replace.append(dict(points=np.array(row['points'],float),source_curve_ids=sources,module=next(iter(owners))))
            consumed.update(sources)
        staged=[r for i,r in enumerate(staged) if i not in consumed]+replace
        flags=[True]*(len(staged)-len(replace))+[False]*len(replace);arch_count=len(consumed)-len(replace)
    rows,diag=refine(rgb,[r['points'] for r in staged],[r['module'] for r in staged],recenter=flags)
    for r in rows:r['source_curve_ids']=[k for i in r['source_curve_ids'] for k in staged[i]['source_curve_ids']]
    new=[r['points'] for r in rows]
    assert sorted(k for r in rows for k in r['source_curve_ids'])==list(range(len(original)))
    profiles=[width_profile(rgb,p,recenter=False) for p in new]
    actions=order(rows,doc['actions'])
    payload=json.dumps(dict(curves=rows,actions=actions),default=serial,ensure_ascii=False).replace('</','<\\/')
    viewer=(Path(__file__).with_name('landscape_stroke_sequence_20261104.html')).read_text('utf8')
    (folder/'sequence.html').write_text(viewer.replace('__DATA__',payload).replace('__SOURCE__',src.as_uri()).replace('__ID__',ident),encoding='utf8')
    tree=cKDTree(np.concatenate([samples(p,.25) for p in new]));deviation=tree.query(np.concatenate([samples(p,.25) for p in original]))[0]
    oldstats=curves_stats(original);newstats=curves_stats(new)
    stats=dict(id=ident,original=oldstats,candidate=newstats,architecture_joins=arch_count,
        architecture_cross_owner_replacements_rejected=architecture_excluded,
        contained_duplicates=len(diag['deduplication']['events']),new_joins=len(diag['continuation']['accepted']),
        pending_continuations=len(diag['continuation']['pending']),
        original_sample_distance_to_new_px_quantiles=np.quantile(deviation,[.5,.95,.99,1.]).tolist(),
        source_sample_fraction_within_1p5px=float((deviation<=1.5).mean()),
        input_provenance_partition_verified=True,geometry_distance_is_not_line_accuracy=True,
        short_curves_not_pruned=True,source_image_seen_before=True,new_blind_test=False,
        model_weights_trained=False,artist_stroke_accuracy=None,cun_recognition_accuracy=None,
        physical_improvement_verified=False,physical_pressure_mm=None,ready_for_robot=False,robot_connected=False,
        module_boundaries_unverified=True,elapsed_seconds=time.monotonic()-start)
    write(folder/'statistics.json',stats);write(folder/'diagnostics.json',diag)
    write(folder/'offline_candidates.json',dict(id=ident,source=str(src),source_sha256=sha(src),configuration=CONFIG,
        original_curves=original,curves=rows,width_profiles=profiles,actions=actions,
        image_size=list(rgb.shape[:2][::-1]),pressure_mm=None,ready_for_robot=False,robot_connected=False,
        coordinate_system='image pixels only',manual_architecture_development_roi=ident=='G03_01',
        artist_stroke_truth_unavailable=True))
    views=[local_view(folder,'full','整幅 '+ident,[0,0,rgb.shape[1],rgb.shape[0]],rgb,original,new,profiles)]
    if ident=='G03_01':
        for key,title,box in [('pavilion','亭子',[108,198,231,299]),('pine','树干与松针',[278,121,488,240]),('rock','山石皴线',[178,285,326,389])]:
            views.append(local_view(folder,key,title,box,rgb,original,new,profiles))
    print(ident,oldstats['count'],'->',newstats['count'],'duplicates',stats['contained_duplicates'],'joins',stats['new_joins'],flush=True)
    return stats,views

def report_saved(ident):
    folder=ROOT/ident;doc=json.loads((folder/'offline_candidates.json').read_text('utf8'))
    stats=json.loads((folder/'statistics.json').read_text('utf8'))
    rgb=np.array(Image.open(doc['source']).convert('RGB'))
    template=(Path(__file__).with_name('landscape_stroke_sequence_20261104.html')).read_text('utf8')
    payload=json.dumps(dict(curves=doc['curves'],actions=doc['actions']),ensure_ascii=False).replace('</','<\\/')
    (folder/'sequence.html').write_text(template.replace('__DATA__',payload).replace('__SOURCE__',Path(doc['source']).as_uri()).replace('__ID__',ident),encoding='utf8')
    old=doc['original_curves'];new=[r['points'] for r in doc['curves']];profiles=doc['width_profiles']
    views=[local_view(folder,'full','整幅 '+ident,[0,0,rgb.shape[1],rgb.shape[0]],rgb,old,new,profiles)]
    if ident=='G03_01':
        for key,title,box in [('pavilion','亭子',[108,198,231,299]),('pine','树干与松针',[278,121,488,240]),('rock','山石皴线',[178,285,326,389])]:
            views.append(local_view(folder,key,title,box,rgb,old,new,profiles))
    return stats,views

def main():
    ROOT.mkdir(exist_ok=True);cv2.setNumThreads(2)
    results=[];views=[]
    for ident in ['G03_01','G05_03','T02']:
        stat,local=report_saved(ident) if '--report-only' in sys.argv else build(ident)
        results.append(stat);views.extend([dict(**r,id=ident) for r in local])
    write(ROOT/'results.json',dict(records=results,views=views,model_weights_trained=False,robot_connected=False))
    cards=''.join(f'<section><h2>{html.escape(v["title"])}</h2><img src="{v["id"]}/{v["key"]}/comparison.jpg"><p><a href="{v["id"]}/{v["key"]}/width_target.png">原图笔宽目标（几何）</a> · <a href="{v["id"]}/{v["key"]}/uncertain_width.png">橙色：测量不确定区域</a></p></section>' for v in views)
    table=''.join(f'<tr><td>{r["id"]}</td><td>{r["original"]["count"]} → {r["candidate"]["count"]}</td><td>{r["contained_duplicates"]}</td><td>{r["new_joins"]}</td><td>{r["pending_continuations"]}</td><td><a href="{r["id"]}/sequence.html">逐笔预览</a> · <a href="{r["id"]}/statistics.json">统计</a> · <a href="{r["id"]}/offline_candidates.json">轨迹与追溯</a></td></tr>' for r in results)
    page='<meta charset="utf-8"><title>山石、树木与亭子离线修正</title><style>body{font:17px/1.6 system-ui;background:#f4f1e8;color:#273b32;max-width:1700px;margin:auto;padding:24px}img{width:100%;background:white}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #bbc6bc}.note{background:#e2e8de;padding:16px}a{color:#175c80}</style><h1>山石、树木、亭子 · 去重与连续笔画候选</h1><p class="note">同一位置和走向的包含重复仅在原图没有亮缝时合并。断段仅沿有墨迹支持的方向续接；竞争连接保留待定。山石不套用建筑直线规则，松针不按短线长度删除。前后中心线采用相同显示宽度；笔宽目标另行查看。</p><table><tr><th>图</th><th>候选数量</th><th>包含重复合并</th><th>新增续接</th><th>待定连接</th><th>数据</th></tr>'+table+cards
    page+='<p class="note">这些是已见图的离线开发对照，不是新盲测，也没有进行新一轮神经网络权重训练。颜色不表示皴法识别。细松针漏提、粗墨交汇的真实分笔仍未被标注真值验证。有限笔宽预览不是墨水扩散模拟；没有压力映射或上机指令，不替换原正式任务。</p>'
    page+='<p>全部原输入都保留来源对应；被包含的重复由代表曲线承接，不意味着像素完全相同，也不意味着真实细节已经完整。模块为旧的未验证候选，仅用于计划顺序。最终实画改善需要当前纸、笔、墨上的局部验证。</p>'
    (ROOT/'index.html').write_text(page,encoding='utf8')
    print('REPORT',ROOT/'index.html',flush=True)

if __name__=='__main__':main()
