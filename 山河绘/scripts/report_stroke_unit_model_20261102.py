# -*- coding: utf-8 -*-
"""Honest offline report: learned relations, geometry checks, unresolved cases."""
import json,hashlib,html
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from stroke_unit_core_20261102 import deduplicate_exact
from prepare_full_ink_trial_20261102 import join_existing,order_paths

ROOT=LAB/'stroke_unit_model_20261102'
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)


def read(p):return json.loads(Path(p).read_text('utf-8'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def aggregate(rows):
    old=sum(r['old_routed_curve_units'] for r in rows);new=sum(r['output_curve_units'] for r in rows)
    a=sum(r['old_under_2mm'] for r in rows);b=sum(r['new_under_2mm'] for r in rows)
    return dict(images=len(rows),old_curve_units=old,new_curve_units=new,unit_reduction_fraction=1-new/max(1,old),
        old_under_2mm=a,new_under_2mm=b,short_reduction_fraction=1-b/max(1,a),
        learned_joins=sum(r['learned_joins'] for r in rows),source_band_duplicate_records_removed=sum(r['source_band_duplicates_removed'] for r in rows),
        minimum_candidate_coverage_at_2_normalized_px=min(r['original_candidate_coverage_at_2_normalized_px'] for r in rows),
        changed_closed_unit_counts=[dict(id=r['id'],before=r['input_closed_paths'],after=r['output_closed_paths']) for r in rows if r['input_closed_paths']!=r['output_closed_paths']])


def colors(shape,paths):
    canvas=np.full((*shape,3),255,np.uint8);scale=max(shape)/496
    for i,p in enumerate(paths):
        color=tuple(int(v) for v in cv2.cvtColor(np.uint8([[[i*47%180,190,175]]]),cv2.COLOR_HSV2RGB)[0,0])
        cv2.polylines(canvas,[np.rint(p).astype(np.int32)],False,color,max(1,int(round(scale))),cv2.LINE_AA)
    return Image.fromarray(canvas)


def case_page(row,old,new,shape):
    folder=ROOT/'outputs'/row['id'];h,w=shape
    paths=json.dumps([[p.tolist() for p in old],[p.tolist() for p in new]],separators=(',',':'))
    template='''<!doctype html><meta charset="utf-8"><title>ID 笔画单元对照</title><style>body{font:17px system-ui;background:#f5f3eb;color:#252925;margin:24px auto;max-width:1450px;padding:0 15px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:15px}canvas,img{width:100%;background:white}button,input{margin:8px}a{color:#17694c}@media(max-width:650px){.grid{grid-template-columns:1fr}}</style>
<a href="../../index.html">返回总报告</a><h1>ID · 笔画单元与落笔点</h1><p>SUMMARY</p>
<p>颜色表示连续曲线单元；两列都保留整幅构图。新版本同时采用长线优先，因此“前N段覆盖”包含排序贡献，不能全部归因于神经模型。</p>
<button id="all">显示整幅</button><button id="first">显示前51段</button><label><input type="checkbox" id="starts">显示落笔点</label><input type="range" id="progress" min="1" max="MAXIMUM" value="MAXIMUM"><span id="n"></span>
<div class="grid"><div>旧规则分组<canvas id="old" width="WIDTH" height="HEIGHT"></canvas></div><div>新关系模型＋墨带检查<canvas id="new" width="WIDTH" height="HEIGHT"></canvas></div></div>
<p>当前笔宽没有新的实测值；本图不是物理墨水仿真。短线仍原样保留并标记待定，不能直接当作机械臂一笔。该图新结果仍有 SHORT 个短于2mm的单元（按画幅186.8mm折算）。</p>
<div class="grid"><div>原图<img src="source.png"></div><div>蓝为补线，红为替换的像素位置<img src="changes.png"></div></div>
<p>神经分类器提出：CONT 个连续关系、DUP 个重复关系；通过模型置信度与几何/原图检查后：JOINS 次接续、REMOVED 条同墨带重复候选替换。未经人工逐笔复核，不表示画家真实笔顺。</p>
<p><a href="relation_probabilities.npz">原始关系概率</a> · <a href="decisions.json">保留/替换/连接记录</a> · <a href="curves.json">新曲线（离线，非实机任务）</a></p>
<script>const ps=PATHS,r=document.getElementById('progress'),starts=document.getElementById('starts');function paint(){const count=+r.value;['old','new'].forEach((id,k)=>{const c=document.getElementById(id),ctx=c.getContext('2d');ctx.fillStyle='white';ctx.fillRect(0,0,c.width,c.height);const scale=c.width/496;ps[k].slice(0,count).forEach((p,i)=>{ctx.strokeStyle=`hsl(${i*137.5%360},62%,37%)`;ctx.lineWidth=scale;ctx.beginPath();ctx.moveTo(...p[0]);p.slice(1).forEach(q=>ctx.lineTo(...q));ctx.stroke();if(starts.checked){ctx.fillStyle='#d43b32';ctx.beginPath();ctx.arc(...p[0],2.0*scale,0,Math.PI*2);ctx.fill();}})});document.getElementById('n').textContent=`旧 ${Math.min(count,ps[0].length)} / 新 ${Math.min(count,ps[1].length)}`;}r.oninput=paint;starts.onchange=paint;document.getElementById('all').onclick=()=>{r.value=r.max;paint()};document.getElementById('first').onclick=()=>{r.value=51;paint()};paint();</script>'''
    replacements={'ID':row['id'],'SUMMARY':f"旧 {row['old_routed_curve_units']} → 新 {row['output_curve_units']}；短于2mm：{row['old_under_2mm']} → {row['new_under_2mm']}。",'MAXIMUM':str(max(len(old),len(new))),'WIDTH':str(w),'HEIGHT':str(h),'SHORT':str(row['new_under_2mm']),'CONT':str(row['raw_model_continuation_candidates']),'DUP':str(row['raw_model_duplicate_candidates']),'JOINS':str(row['learned_joins']),'REMOVED':str(row['source_band_duplicates_removed']),'PATHS':paths}
    # Replace long tokens first: ID is also a substring of WIDTH.
    for key in sorted(replacements,key=len,reverse=True):template=template.replace(key,replacements[key])
    (folder/'units.html').write_text(template,encoding='utf-8')


def report():
    locked=read(ROOT/'frozen_experiment.json');catalog={r['id']:r for r in locked['inference_catalog']}
    rows=[read(ROOT/'outputs'/ident/'statistics.json') for ident in sorted(catalog)]
    assert len(rows)==53
    summaries={name:aggregate([r for r in rows if r['split']==name]) for name in sorted({r['split'] for r in rows})}
    summaries['all']=aggregate(rows);summaries['non_development_50']=aggregate([r for r in rows if r['split']!='development'])
    save(ROOT/'batch_summary.json',summaries)
    hashes=[dict(path=r['path'],matches=hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256']) for r in locked['files']]
    assert all(r['matches'] for r in hashes),'Frozen experiment changed during test'
    assert all(hashlib.sha256(Path(r['image']).read_bytes()).hexdigest()==r['image_sha256'] and hashlib.sha256(Path(r['paths']).read_bytes()).hexdigest()==r['paths_sha256'] for r in catalog.values())
    parents=[]
    for p in [LAB/'faint_center_direction_20261101'/'frozen_experiment.json',LAB/'native_fine_faint_20261101'/'r3_lock.json']:
        doc=read(p)
        for record in doc.get('files',[]):
            if isinstance(record,dict) and 'path' in record and 'sha256' in record:
                parents.append(dict(path=record['path'],matches=hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()==record['sha256']))
    assert all(r['matches'] for r in parents),'Prior frozen extractor changed'
    save(ROOT/'verification.json',dict(images=53,unique_image_hashes=len({r['image_sha256'] for r in catalog.values()}),
        new_model_frozen_checks=hashes,prior_extractor_checks=parents,new_geometry_tests_passed=8,
        all_case_outputs_ready_for_robot=False,robot_connected=False,actually_drawn=False,
        new_neural_image_encoder_training_steps=0,new_relation_model_training_steps=1500,
        evaluation_cases_not_independent_whole_system_blind_tests=True))
    for row in rows:
        record=catalog[row['id']];rgb=Image.open(record['image']).convert('RGB');shape=(rgb.height,rgb.width)
        raw_doc=read(record['paths']);raw=[np.array(p['points'] if isinstance(p,dict) else p,float) for p in raw_doc['paths']]
        old,_=deduplicate_exact(raw);old,_=join_existing(old)
        if row['id']=='G03_01':old=order_paths(old)
        old=[r['points'] for r in old];new=[np.array(p['points']) for p in read(ROOT/'outputs'/row['id']/'curves.json')['paths']]
        assert len(old)==row['old_routed_curve_units']
        case_page(row,old,new,shape)
        if row['id'] in ['G03_01','G01_04','G05_03','T02','T12','F_N02']:
            board=Image.new('RGB',(1536,550),'#f5f3eb');d=ImageDraw.Draw(board)
            for i,(im,title) in enumerate([(rgb,'原图'),(colors(shape,old),f'旧分组 {len(old)}'),(colors(shape,new),f'新分组 {len(new)}')]):
                im.thumbnail((496,496),Image.Resampling.LANCZOS);d.text((i*512+4,6),row['id']+' '+title,font=FONT,fill='black');board.paste(im,(i*512,45))
            board.save(ROOT/'outputs'/row['id']/'unit_comparison.jpg',quality=95)
    a=summaries['all'];t=summaries['non_development_50'];target=next(r for r in rows if r['id']=='G03_01')
    table=''.join(f"<tr><td><a href='outputs/{r['id']}/units.html'>{r['id']}</a></td><td>{r['split']}</td><td>{r['old_routed_curve_units']}</td><td>{r['output_curve_units']}</td><td>{r['old_under_2mm']}</td><td>{r['new_under_2mm']}</td><td>{'闭合单元变化，待核对' if r['input_closed_paths']!=r['output_closed_paths'] else '仍有短段待核对'}</td></tr>" for r in rows)
    training=read(ROOT/'training_summary.json');protocol=read(ROOT/'protocol.json')
    page=f'''<!doctype html><meta charset="utf-8"><title>笔画单元模型：53图离线测试</title><style>body{{font:18px system-ui;max-width:1450px;margin:24px auto;padding:0 20px;background:#f5f3eb;color:#252925}}section{{background:white;padding:20px;margin:18px 0;border-radius:10px}}img{{width:100%}}table{{border-collapse:collapse;width:100%;font-size:15px}}th,td{{border-bottom:1px solid #ddd;padding:9px;text-align:left}}th{{position:sticky;top:0;background:#eee}}a{{color:#17694c}}.warn{{color:#a33b2d}}li{{margin:10px 0}}</style>
<h1>笔画单元模型 · 53张已有画作离线回归</h1><section><b>本轮解决什么</b><p>针对实画中短碎段密集落笔、同墨带重复覆盖的问题，新训练“连续关系／同墨带重复／分开”关系模块。上游图像提取器和缓存候选保持冻结；它不是重新训练后的整套图像编码器。所有结果均为离线候选，机械臂保持停止。</p></section>
<section><b>测试结果</b><ul><li>共53张不同图片：3张开发诊断、27张旧批次小景、20张原生整图、3张锁定迁移测试。均为已有图片，本轮没有新生成或下载图片；部分已用于旧上游开发，不是全系统新图盲测。</li><li>53张合计：曲线单元 {a['old_curve_units']:,} → {a['new_curve_units']:,}，减少{a['unit_reduction_fraction']:.1%}；短于2mm单元 {a['old_under_2mm']:,} → {a['new_under_2mm']:,}，减少{a['short_reduction_fraction']:.1%}。</li><li>扣除3张开发图后的50张：曲线单元减少{t['unit_reduction_fraction']:.1%}，短单元减少{t['short_reduction_fraction']:.1%}。</li><li>相对旧候选，在2个标准化像素容差下覆盖未下降。此数不是对原画的准确率，不能证明无误接或全部主线正确。</li></ul></section>
<section><b>刚才实画的G03_01</b><p>旧1670 → 新{target['output_curve_units']}个单元；短于2mm：1024 → {target['new_under_2mm']}。额外接续{target['learned_joins']}次，替换{target['source_band_duplicates_removed']}条同墨带重复候选。下图颜色反映分组，黑白线稿形状变化较小。</p><a href="outputs/G03_01/units.html"><img src="outputs/G03_01/unit_comparison.jpg"></a><p><a href="outputs/G03_01/units.html">交互查看整幅、前51个单元和落笔点</a>。新版本长线优先也会改变初期覆盖，不能把排序收益全算作模型收益。</p></section>
<section class="warn"><b>仍未解决</b><p>G03_01还剩714个短单元，直接逐段落笔仍可能积墨。T12新增1个闭合单元（30→31），可能是闭合轮廓接全，也可能误接，已保留为待核对案例。原画漏检、笔宽与实际含墨量的关系没有在本轮得到完整验证，因此尚不适合恢复实画。</p><a href="outputs/T12/units.html">查看T12待核对例子</a></section>
<section><b>训练与评价边界</b><p>GPU：{training['device']}；新关系模块{training['parameters']:,}个参数，执行1500步，保留第{training['retained_step']}步。使用已知路径的程序合成关系样本，以及R01–R04中明确由用户描绘的局部曲线增广；未标区域保持未知、草稿排除。合成关系分类成绩不等于真实整画提取准确率；局部人工样本来自同一来源页，测试也不能当跨来源专家评测。</p><p>新模块判断关系后，还要经过原图墨带、方向、白色间隙和歧义检查。原始概率与所有接续/替换记录按图保存。细短线保留并标记待定，没有统一删掉换取更低笔数。</p><p><a href="protocol.json">数据协议</a> · <a href="training_summary.json">GPU训练记录</a> · <a href="evaluation.json">关系分类测试</a> · <a href="batch_summary.json">批量统计</a> · <a href="verification.json">冻结校验与离线检查</a></p></section>
<section><b>逐图对照（点击编号）</b><p>“短于2mm”按186.8mm方形画幅折算，仅用于比较；当前模型没有输出机械臂Z或速度。</p><table><thead><tr><th>图片</th><th>分组</th><th>旧单元</th><th>新单元</th><th>旧短段</th><th>新短段</th><th>状态</th></tr></thead><tbody>{table}</tbody></table></section>'''
    (ROOT/'index.html').write_text(page,encoding='utf-8')
    (ROOT/'RESULTS.md').write_text(f'''# 本轮结果

新训练的是16,515参数的笔画关系模块；上游图像模型冻结。GPU训练1500步，保留1350步。
53张已有画作回归：单元{a['old_curve_units']}→{a['new_curve_units']}，短单元{a['old_under_2mm']}→{a['new_under_2mm']}。
50张非开发图：单元减少{t['unit_reduction_fraction']:.1%}，短单元减少{t['short_reduction_fraction']:.1%}。
G03_01：1670→1345；短单元1024→714。T12闭合单元30→31，未在看过测试结果后调参，保留待核对。
旧候选覆盖按2标准像素容差未降；不是原画准确率。
53张都是旧图片，部分用于旧上游开发，不是全系统新图盲测。
训练标签为已知程序路径及局部人工曲线增广，未标区域不作负例，草稿排除。
剩余短单元仍多，没有生成可用实机任务，没有连接或移动机械臂。
''',encoding='utf-8')
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


if __name__=='__main__':report()
