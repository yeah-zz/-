# -*- coding: utf-8 -*-
"""Report every fixed arm, raw full-image outputs, and known-case failures."""
import sys,json,hashlib,html
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'regional_presence_trial_20261117';FOLLOW=LAB/'regional_presence_headonly_20261117'
PROJECT=Path(__file__).resolve().parents[1]
COLORS=np.array([[241,237,222],[196,140,62],[70,143,105],[110,112,163]],'uint8')
NAMES={'frozen':'冻结基线','control':'解码器续训对照','presence':'解码器＋弱区域',
       'head_control':'仅分类层对照','head_presence':'仅分类层＋弱区域'}
def read(p):return json.loads(Path(p).read_text('utf8'))
def write(name,v):(ROOT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def model_paths():return {'frozen':(ROOT,'frozen_mapping'),'control':(ROOT,'continuation_control'),
    'presence':(ROOT,'regional_presence'),'head_control':(FOLLOW,'continuation_control'),
    'head_presence':(FOLLOW,'regional_presence')}
def main():
    base=read(ROOT/'results.json');follow=read(FOLLOW/'results.json');assert read(FOLLOW/'freeze_checks.json')['only_final_classifier_changed']
    models=model_paths();(ROOT/'views').mkdir(exist_ok=True)
    hcase=next(x for x in read(LAB/'guohua_boundary_reference_20261116/references.json')['cases'] if x['id']=='H_14')
    hdiagnostic={}
    for r in hcase['regions']:
        if r['kind']!='core_polygon':continue
        mask=Image.new('L',tuple(hcase['size']),0);ImageDraw.Draw(mask).polygon([tuple(p) for p in r['points']],fill=1);mask=np.array(mask)>0
        cls=1 if r['object']=='rock' else 2
        hdiagnostic[r['name']]={k:float((np.load(folder/f'predictions/H_14_{arm}.npz')['prediction'][mask]==cls).mean()) for k,(folder,arm) in models.items()}
    write('known_region_diagnostics.json',dict(source=str(LAB/'guohua_boundary_reference_20261116/references.json'),
        source_sha256=sha(LAB/'guohua_boundary_reference_20261116/references.json'),
        scope='previously reviewed assistant core polygons, no new expert or independent truth',regions=hdiagnostic))
    training={}
    for key,(result,arm) in {'frozen':(base,'frozen_mapping'),'control':(base,'continuation_control'),
        'presence':(base,'regional_presence'),'head_control':(follow,'continuation_control'),
        'head_presence':(follow,'regional_presence')}.items():
        rs=[r for f in result['training_work_presence'] for r in f['regions']]
        training[key]=dict(correct=sum(r['values'][arm]['presence_prediction']==r['target'] for r in rs),total=len(rs))
    summary=[]
    for key,(result,arm) in {'frozen':(base,'frozen_mapping'),'control':(base,'continuation_control'),
        'presence':(base,'regional_presence'),'head_control':(follow,'continuation_control'),
        'head_presence':(follow,'regional_presence')}.items():
        summary.append(dict(key=key,name=NAMES[key],known14_mean_iou=result['aggregates']['new_test'][arm]['mean_iou'],
            old12_mean_iou=result['aggregates']['old_named_regression'][arm]['mean_iou'],
            user_foliage_recall=result['user_foliage_recall'][arm],foreground_rock_recall=hdiagnostic['foreground_rock'][key],
            training_work_presence=training[key]))
    failure_rows=[]
    follow_by_id={r['id']:r for r in follow['records']}
    for r in base['records']:
        if not r['scores']:continue
        per={}
        for key,(result,arm) in {'frozen':(base,'frozen_mapping'),'control':(base,'continuation_control'),
            'presence':(base,'regional_presence'),'head_control':(follow,'continuation_control'),
            'head_presence':(follow,'regional_presence')}.items():
            row=r if result is base else follow_by_id[r['id']]
            iou=row['scores'][arm]['iou'];per[key]=float(np.mean([v for v in iou if v is not None]))
        failure_rows.append(dict(id=r['id'],historical_split=r['historical_split'],known_regression=True,
            scores=per,decoder_presence_delta=per['presence']-per['frozen'],
            head_presence_delta=per['head_presence']-per['frozen']))
    write('all_case_changes.json',failure_rows)
    cases=[dict(id=r['id'],image=r['image'],group='已知回归／'+r['historical_split']) for r in base['records']]
    for f in base['training_work_presence']:
        cases.append(dict(id='F'+Path(f['frame']).stem[-4:],image=f['frame'],group='训练作品内，不能验收'))
    catalog=[]
    for c in cases:
        ident=c['id'];im=Image.open(c['image']).convert('RGB');im.thumbnail((800,800));im.save(ROOT/f'views/{ident}_source.jpg',quality=93)
        paths={}
        for key,(folder,arm) in models.items():
            pred=np.load(folder/f'predictions/{ident}_{arm}.npz')['prediction']
            im=Image.fromarray(COLORS[pred]);im.thumbnail((800,800));im.save(ROOT/f'views/{ident}_{key}.png')
            paths[key]=f'views/{ident}_{key}.png'
        catalog.append(dict(**c,source=f'views/{ident}_source.jpg',predictions=paths))
    write('catalog.json',dict(records=catalog,models=NAMES,raw_argmax_only=True,no_reference_mask_correction=True))
    # Six columns show the reference and all five frozen arms; no best-column splice.
    board=Image.new('RGB',(1800,930),'#f5f2e9');d=ImageDraw.Draw(board);font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
    worst=sorted([r for r in failure_rows if r['historical_split']=='new_test'],key=lambda r:r['decoder_presence_delta'])[0]['id']
    for row,ident in enumerate(['F1500','H_14',worst]):
        c=next(c for c in catalog if c['id']==ident)
        for col,(key,title) in enumerate([('source','原图')]+list(NAMES.items())):
            im=Image.open(ROOT/c['source']) if key=='source' else Image.open(ROOT/c['predictions'][key])
            scale=min(288/im.width,250/im.height);im=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.NEAREST)
            x=col*300;y=row*310;d.text((x+6,y+5),title,font=font(17),fill='#253e35');board.paste(im,(x+6,y+40))
        d.text((8,row*310+284),f'{ident} · 对象分区原始输出，非笔触重绘',font=font(16),fill='#78523c')
    board.save(ROOT/'comparison.jpg',quality=94)
    columns=['方法','已知14图平均IoU','旧12图平均IoU','已知树叶核心召回','同图前景岩面召回','训练内区域对应']
    table='<table><thead><tr>'+''.join('<th>'+x+'</th>' for x in columns)+'</tr></thead><tbody>'
    for r in summary:
        table+='<tr>'+''.join('<td>'+x+'</td>' for x in [r['name']]+[f'{r[k]*100:.1f}%' for k in ['known14_mean_iou','old12_mean_iou','user_foliage_recall','foreground_rock_recall']]+[f'{r["training_work_presence"]["correct"]}/{r["training_work_presence"]["total"]}'])+'</tr>'
    table+='</tbody></table>'
    # Adoption is deliberately separate from ranking known-case metrics.
    decision=dict(adopted=False,independent_validation=False,production_replaced=False,
        reason='只在已知资料开发；解码器弱约束明显损伤岩面，仅分类层对照另列，仍无独立作品验收',
        core_capability_flags_unchanged=True,all_fixed_arms_reported=True,
        no_further_step_or_threshold_sweep_this_round=True,robot_connected=False)
    write('decision.json',decision);write('report_data.json',dict(summary=summary,worst_known14=worst,
        full_decoder_failure=hdiagnostic['foreground_rock'],decision=decision,new_independent_works=0,
        independent_validation=False,expert_accuracy=False))
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>弱区域对象训练与回归</title><style>body{font:17px/1.7 system-ui;max-width:1800px;margin:auto;padding:24px;background:#f5f2e9;color:#243f34}h1{font-size:30px}.notice{padding:16px;background:#e9dfc9}.scroll{overflow-x:auto}table{border-collapse:collapse;white-space:nowrap;width:100%}th,td{padding:10px;border:1px solid #bfbbaa}img{max-width:100%;height:auto}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(330px,100%),1fr));gap:16px}article{background:white;padding:12px}article img{width:100%;object-fit:contain;max-height:540px}select{font:inherit;max-width:100%;padding:8px}a{color:#265d76}</style><h1>对象在场约束：实际训练、全部对照与退步</h1><p class="notice">本轮完成四组候选训练，均未采用。以下为对象分区原始输出，不是完整笔触、皴法辨识或机械臂画面。全部是已知资料检查，独立验收尚未进行。</p><p>米白：空景；棕：山石；绿：树木；紫：其他。新参考只提供区域中存在山石／树木的弱约束，不把区域内每个像素都当墨迹。8处正区域，2处交叠剔除。</p><div class="scroll">TABLE</div><p>平均IoU根据同一套四类原始粗标签计算；树叶与岩面为既有助手核心参考。训练内区域对应只说明在已训练作品上的拟合，不是测试准确率。“仅分类层”是看到前组失败后的开发对照，也不算独立验证。</p><img src="comparison.jpg" alt="原图和所有对照，含退步案例"><h2>逐张查看全部原始分区</h2><select id="case"></select><p id="scope"></p><div id="gallery" class="grid"></div><h2>结论与限制</h2><p>解码器弱约束在训练来源中找回树木，但会把邻近岩面扩大标为树木。未用人工框修饰输出，未仅挑一个指标认定通过。限制更新范围的两组结果完整列在表中。皴法、真实完整分笔、树木技能与实画均未通过。</p><p>此次富春示范作品已用于开发训练，以后全部帧与清晰度都不能再称独立测试。历史数据和协议保留，新用途另作记录。</p><p><a href="protocol.json">首组协议</a> · <a href="followup_protocol.json">后续分类层协议</a> · <a href="admission.json">仅对象在场监督准入</a> · <a href="split_retirement.json">来源用途变更</a> · <a href="all_case_changes.json">全部逐图变化</a> · <a href="known_region_diagnostics.json">同图岩面与树叶诊断</a> · <a href="verification.json">完整性检查</a></p><script>fetch('catalog.json').then(r=>r.json()).then(data=>{window.catalog=data;const select=document.querySelector('#case');for(const r of data.records){const option=document.createElement('option');option.value=r.id;option.textContent=r.id+' · '+r.group;select.append(option)}function render(){const r=data.records.find(r=>r.id===select.value);document.querySelector('#scope').textContent=r.group+'；所有输出来自完整原图推理，未用参考框裁剪或纠正。';const grid=document.querySelector('#gallery');grid.innerHTML='';for(const [name,path] of [['原图',r.source],...Object.entries(data.models).map(([k,n])=>[n,r.predictions[k]])]){const article=document.createElement('article'),title=document.createElement('strong'),image=document.createElement('img');title.textContent=name;image.src=path;image.alt=name;article.append(title,image);grid.append(article)}}select.onchange=render;select.value='H_14';render()});</script></html>'''
    (ROOT/'index.html').write_text(doc.replace('TABLE',table),encoding='utf8')
    write('followup_protocol.json',read(FOLLOW/'protocol.json'))
    print(json.dumps(dict(summary=summary,worst=worst,cases=len(catalog)),ensure_ascii=False))
if __name__=='__main__':main()
