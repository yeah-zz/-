"""Full fixed-set report. Numeric retention measures are not semantic accuracy."""
import sys,json,html,shutil
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import distance_transform_edt
from audit_v4_local_repairs import OUT,dump,sha
import v4_local_repair_candidates as repair

def graph_cycles(m):
    # Pixel graph uses the same no-diagonal-shortcut convention as v4.
    a=m[1:,1:];b=m[:-1,:-1];c=m[1:,:-1];d=m[:-1,1:]
    edges=int((m[:,1:]&m[:,:-1]).sum()+(m[1:]&m[:-1]).sum()+(a&b&~c&~d).sum()+(c&d&~a&~b).sum())
    components=cv2.connectedComponents(m.astype(np.uint8),connectivity=8)[0]-1
    return edges-int(m.sum())+components

def main():
    result=json.loads((OUT/'results_groups_1_2_3_4_5.json').read_text(encoding='utf-8'))
    refs=json.loads((OUT/'actual_local_partial_references.json').read_text(encoding='utf-8'))
    measurements=[]
    for r in refs['records']:
        if r['status']!='assistant_partial_positive':continue
        q=np.rint(r['points']).astype(int);row=dict(id=r['id'],image_id=r['image_id'],expert_verified=False,selection_biased_diagnostic=True)
        for name,path in [('v4',OUT/'baseline'/r['image_id']/'lines.png'),('candidate',OUT/'candidate'/r['image_id']/'lines.png')]:
            mask=np.asarray(Image.open(path).convert('L'))<128;dist=distance_transform_edt(~mask)
            row[name+'_covered_fraction_at_1_5px']=float((dist[q[:,1],q[:,0]]<=1.5).mean())
        measurements.append(row)
    background=[]
    for r in refs['background_regions']:
        x0,y0,x1,y1=r['box'];values={}
        for name in ['baseline','candidate']:
            mask=np.asarray(Image.open(OUT/name/r['image_id']/'lines.png').convert('L'))<128
            values[name+'_line_pixels']=int(mask[y0:y1,x0:x1].sum())
        background.append(dict(**r,**values))
    inventory=[]
    for r in result['records']:
        ident=r['id'];old=np.asarray(Image.open(OUT/'baseline'/ident/'lines.png').convert('L'))<128;new=np.asarray(Image.open(OUT/'candidate'/ident/'lines.png').convert('L'))<128
        raw=np.asarray(Image.open(OUT/'baseline'/ident/'raw_skeleton.png').convert('L'))<128
        inventory.append(dict(id=ident,split=r['split'],baseline_unresolved_source_pixels=int((raw&~old).sum()),candidate_unresolved_source_pixels=int((raw&~new).sum()),baseline_pixel_graph_cycles=graph_cycles(old),candidate_pixel_graph_cycles=graph_cycles(new),geometry_counts_are_not_error_labels=True))
    dump(OUT/'failure_risk_inventory.json',dict(records=inventory,note='Unresolved evidence and graph loops locate audit risks; neither quantity is a human-stroke accuracy measure. Genuine pine structures may have loops; all old mesh errors remain.'))
    check=json.loads((OUT/'behavior_checks.json').read_text(encoding='utf-8'))
    summary=dict(images=len(result['records']),development=18,calibration=6,reused_regression=6,parent_sheets=5,added_pixels=sum(r['added_pixels'] for r in result['records']),removed_pixels=sum(r['removed_pixels'] for r in result['records']),all_original_paths_preserved=all(r['original_paths_exactly_preserved'] for r in result['records']),new_neural_training_steps=0,device='CPU OpenCV/SciPy',expert_accuracy=None,checks_passed=sum(r['passed'] for r in check['checks']),config=repair.CFG,code_sha256=sha(Path(repair.__file__)),partial_reference_diagnostics=measurements,background_diagnostics=background,promotion='experimental optional candidate; original v4 default unchanged',limitations=['Only tested on this existing 496px style set; native images outside range retain v4','All changes are additions of existing source-skeleton chains. This preserves old errors too.','Some faint ink absent from source skeleton remains absent.','Thick ink mesh and artist stroke identity remain unresolved.','Local reference positives were selected among additions: not a representative accuracy estimate.','Five parent sheets: crops are correlated; final group was consumed for cycle diagnosis. None is a pristine blind test.'])
    summary['images_with_more_graph_cycles']=sum(r['candidate_pixel_graph_cycles']>r['baseline_pixel_graph_cycles'] for r in inventory)
    summary['graph_cycle_counts_are_not_error_rates']=True
    dump(OUT/'summary.json',summary)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    # Fixed examples: first scene of groups 1, 4, 5. Full gallery includes all.
    examples=['G01_01','G04_01','G05_01']
    panels=[]
    for ident in examples:panels.append(Image.open(OUT/'candidate'/ident/'comparison.jpg').convert('RGB'))
    combined=Image.new('RGB',(1600,450*len(panels)),'white')
    for i,im in enumerate(panels):combined.paste(im,(0,i*450))
    combined.save(OUT/'FIXED_COMPARISON.jpg',quality=94)
    # Large honest local view; line widths equal in baseline and candidate.
    box=(60,75,320,210);zoom=Image.new('RGB',(1560,330),'#f3f1eb');draw=ImageDraw.Draw(zoom)
    for j,(folder,name,label) in enumerate([('baseline','input.png','原图'),('baseline','lines.png','v4'),('candidate','lines.png','局部补回')]):
        im=Image.open(OUT/folder/'G01_01'/name).convert('RGB').crop(box).resize((520,270))
        zoom.paste(im,(j*520,38));draw.text((j*520+5,5),label,font=font,fill='black')
    zoom.save(OUT/'PINE_DETAIL.jpg',quality=96)
    cards=[]
    for r in result['records']:
        ident=r['id'];url='candidate/'+ident+'/'
        split='reused_regression_cycle_diagnosis' if ident.startswith('G05') else r['split']
        cards.append('<section><h2>'+ident+' · '+html.escape(split)+'</h2><p>新增 '+str(r['added_pixels'])+' 像素；删除 '+str(r['removed_pixels'])+' 像素。原有路径逐坐标保留。</p><a href="'+url+'comparison.jpg"><img src="'+url+'comparison.jpg"></a><p><a href="'+url+'additions_on_source.png">新增线叠加原图</a> · <a href="'+url+'bundle_detection.png">几何细节区</a> · <a href="'+url+'paths.json">像素路径</a> · <a href="'+url+'decisions.json">逐候选筛选记录</a></p></section>')
    content='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v4 局部短线补回：固定30张对照</title><style>body{font:17px system-ui;line-height:1.7;background:#f3f1eb;color:#222;margin:24px auto;max-width:1800px;padding:0 16px}img{width:100%;background:white}section{border-top:1px solid #ccc;margin:30px 0}aside{background:#fff2d7;padding:16px}a{color:#175f86}</style><h1>v4 局部短线补回：固定 30 张对照</h1><p>每行依次为原图、冻结 v4、本轮候选、变化图。黑色保持；青色新增；红色删除（本轮没有删除）。所有图用同一线宽显示。</p><aside>本轮是 CPU 图像处理修正，没有训练新神经网络权重。结果只作为可选候选，原 v4 及已认可的大图未替换。补回的是原图骨架已有、被固定长度筛掉的短线；不存在跨空白补画。新增墨迹证据不等于已确认画家的一笔。</aside><h2>本轮发现与处理</h2><p>496 像素小图中的松针和短皴线常达不到原 v4 的固定长度门槛。本轮根据局部短线束、方向分散、原图两侧灰度和线段直顺性选择补回，排除闭合环与明显弯回候选。全部原有路径保持，因此旧的粗墨网状线仍可能存在。</p><h2>核对范围</h2><p>30 张图来自 5 张生成母图，不是 30 幅独立画作。按母图分为开发18张、校准6张、本轮参数未使用的验证6张；它们是已有图片，不能作为全新盲测。保留全部30张结果，未删除失败样本。</p><p>已对16处实际原图局部逐一查看：8条细线作为非专家局部正例，8处粗墨/交汇保留未知；另记录5处空白背景。正例来自新增候选，具有选择偏差，只用于具体错例核对，不报告整幅准确率，也不把未标区域当背景。</p><p>原图骨架未检出的淡线、本来就存在的网状线、单笔连续性和机械臂笔触尚未解决。本轮只验证图片像素线条；没有实机绘画。</p><p><a href="summary.json">汇总与限制</a> · <a href="baseline_manifest.json">冻结基线清单与哈希</a> · <a href="behavior_checks.json">行为检查</a> · <a href="actual_local_partial_references.json">局部参考及未知标记</a> · <a href="local_reference_review_cards.jpg">逐线核对卡片</a></p><h2>松针局部</h2><img src="PINE_DETAIL.jpg"><h2>全部固定对照</h2>'''+''.join(cards)
    content=content.replace('按母图分为开发18张、校准6张、本轮参数未使用的验证6张；它们是已有图片，不能作为全新盲测。','最初按母图分为开发18张、校准6张、保留6张；首版在30张上发现新增闭环后加入统一闭环约束，因此末组也已用于问题诊断，最终全部仅作为固定回归对照，不能称作独立盲测。')
    content=content.replace('排除闭合环与明显弯回候选。','排除闭合环与明显弯回候选；若加入整条候选会增加像素图闭环数，则整条留在待定层。')
    content=content.replace('<h2>核对范围</h2>','<p>首版30张均增加几何闭环，已归档到 iterations/r1_before_cycle_guard，未采用。最终版本在固定30张上验证闭环数不增加；这项几何约束不等于已识别或修复全部错误皴线。</p><h2>核对范围</h2>')
    content=content.replace('<a href="behavior_checks.json">行为检查</a>','<a href="behavior_checks.json">行为检查</a> · <a href="failure_risk_inventory.json">逐图待定与闭环风险</a>')
    (OUT/'index.html').write_text(content,encoding='utf-8')
    snap=OUT/'source_snapshot';snap.mkdir(exist_ok=True)
    names=['audit_v4_local_repairs.py','v4_local_repair_candidates.py','run_v4_local_repair_round.py','prepare_actual_local_reference_review.py','curate_v4_partial_references.py','check_v4_local_repairs.py','report_v4_local_repairs.py']
    for n in names:shutil.copy2(Path(__file__).with_name(n),snap/n)
    dump(OUT/'source_snapshot_manifest.json',{n:sha(snap/n) for n in names})
    print(json.dumps({k:v for k,v in summary.items() if k not in ['partial_reference_diagnostics','background_diagnostics','config','limitations']},ensure_ascii=False))
if __name__=='__main__':main()
