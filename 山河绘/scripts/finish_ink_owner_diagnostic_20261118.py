# -*- coding: utf-8 -*-
import sys,json,hashlib,html
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'ink_owner_diagnostic_20261118';PROJECT=Path(__file__).resolve().parents[1];SOURCE=LAB/'object_prospective_20261118'
ARMS=['global256','detail512','fixed_mean','learned_gate'];NAMES=['冻结256','冻结512','固定平均','学习融合'];LABELS={0:'空景',1:'山石',2:'树木',3:'其他'}
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def main():
    result=read(ROOT/'results.json');reference=read(ROOT/'reviewed_references.json');freeze=read(ROOT/'reference_freeze.json')
    assert sha(ROOT/'reviewed_references.json')==freeze['reviewed_reference_sha256']
    assert freeze['prior_full_prediction_inspection'] and result['posthoc_diagnostic']
    registry=read(PROJECT/'research/source_usage_registry_20261117.json')['records'];entry=next(r for r in registry if r['source_identity']==reference['source_work_group']);assert entry['training_allowed'] is False
    protocol=read(SOURCE/'protocol.json');assert sha(protocol['checkpoint'])==protocol['checkpoint_sha256'] and sha(protocol['gate'])==protocol['gate_sha256']
    (ROOT/'views').mkdir(exist_ok=True);font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
    board=Image.new('RGB',(1280,1100),'#f5f2e9');d=ImageDraw.Draw(board)
    for i,r in enumerate(result['records']):
        assert sha(r['image'])==r['image_sha256'];im=Image.open(r['image']).convert('RGB');x,y=r['xy'];assert 1<=x<im.width-1 and 1<=y<im.height-1
        for arm in ARMS:
            pred=np.load(SOURCE/f'predictions/{r["image_id"]}_{arm}.npz')['prediction'];p=r['predictions'][arm];assert int(pred[y,x])==p['object_id']
            if r['object_type']=='unknown':assert p['correct'] is None and p['neighborhood_fraction'] is None
            else:
                cls=1 if r['object_type']=='rock' else 2;assert p['correct']==bool(pred[y,x]==cls)
                assert abs(p['neighborhood_fraction']-float((pred[y-1:y+2,x-1:x+2]==cls).mean()))<1e-12
        assert not r['training_allowed'] and not r['expert_verified'] and r['stroke_identity'] is None and r['cunfa']=='unknown'
        tile=im.crop((x-20,y-20,x+21,y+21)).resize((164,164),Image.Resampling.NEAREST);draw=ImageDraw.Draw(tile);draw.ellipse((77,77,87,87),outline='#f2473b',width=2)
        tile.save(ROOT/f'views/{r["id"]}_patch.png');cx=i%4*320;cy=i//4*220;board.paste(tile,(cx+10,cy+36));d.text((cx+8,cy+5),f'{r["id"]} · {r["object_type"]} · {x},{y}',font=font(16),fill='#243f35')
        context=im.crop((max(0,x-50),max(0,y-50),min(512,x+51),min(512,y+51)));context.thumbnail((130,160));board.paste(context,(cx+180,cy+38))
    board.save(ROOT/'reviewed_contact.jpg',quality=95)
    # Full-image location is independent of prediction visualization.
    for image_id in sorted({r['image_id'] for r in result['records']}):
        r=next(r for r in result['records'] if r['image_id']==image_id);im=Image.open(r['image']).convert('RGB');draw=ImageDraw.Draw(im)
        for r in result['records']:
            if r['image_id']!=image_id:continue
            x,y=r['xy'];color='#aa4091' if r['object_type']=='unknown' else '#d45e32' if r['object_type']=='rock' else '#21875b'
            draw.ellipse((x-3,y-3,x+3,y+3),outline=color,width=2);draw.text((x+4,y-10),r['id'],font=font(12),fill=color)
        im.save(ROOT/f'views/{image_id}_locations.png')
    summary='<table id="summary"><tr><th>配置</th><th>岩面墨迹核心</th><th>树木墨迹核心</th><th>相邻3×3仍全部判错</th></tr>'+''.join('<tr><td>'+n+'</td>'+''.join(f'<td>{result["summary"][a][obj]["correct"]}/{result["summary"][a][obj]["core_points"]}</td>' for obj in ['rock','tree'])+f'<td>{sum(result["summary"][a][obj]["stable_wrong"] for obj in ["rock","tree"])}</td></tr>' for a,n in zip(ARMS,NAMES))+'</table>'
    table='<table><tr><th>点位／源图局部</th><th>墨迹归属</th>'+''.join('<th>'+n+'</th>' for n in NAMES)+'<th>复核备注</th></tr>'
    for r in result['records']:
        table+='<tr><td><a href="views/'+r['id']+'_patch.png">'+r['id']+'</a> '+str(r['xy'])+'</td><td>'+r['object_type']+'</td>'
        for a in ARMS:
            v=r['predictions'][a];table+='<td>'+LABELS[v['object_id']]+('（待定不计分）' if v['correct'] is None else ' ✓' if v['correct'] else ' ×')+'</td>'
        table+='<td>'+html.escape(r['review_note'])+'</td></tr>'
    table+='</table>'
    locations=''.join('<img src="views/'+i+'_locations.png" alt="原图中的核心点位置">' for i in sorted({r['image_id'] for r in result['records']}))
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>细笔迹对象归属诊断</title><style>body{font:17px/1.7 system-ui;max-width:1400px;margin:auto;padding:24px;background:#f5f2e9;color:#243e35}h1{font-size:30px}.notice{padding:16px;background:#e9dfc8}.scroll{overflow-x:auto}table{border-collapse:collapse}td,th{padding:9px;border:1px solid #bbb;white-space:nowrap}img{max-width:100%;height:auto}.locations{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(400px,100%),1fr));gap:20px}</style><h1>实际墨迹属于山石还是树木？</h1><p class="notice">本轮新增的是事后诊断参考与评估接口，没有训练新模型。2张图来自已打开的同一来源候选组；不是新作品测试或专家真值。10个墨迹核心计分，10处留白／交叠／不确定位置不计分。</p><p>岩面内部、树干内部的留白可以属于相应对象，但不能作为“可见墨迹”参考。初始候选、调整及降为待定的记录全部保留；标记点不是完整笔画、端点或画家笔顺。3×3只检查位置敏感性，不能算9份真值。</p><div class="scroll">SUMMARY</div><p>固定平均与学习融合在这10个助手核心均对应正确，不能据此宣称完整提取或识别通过。点位是在先前预测已被查看后整理的，只适合错例诊断。</p><div class="locations">LOCATIONS</div><h2>逐点原生墨迹核查</h2><img src="reviewed_contact.jpg"><div class="scroll">TABLE</div><p><a href="draft_references.json">初始候选</a> · <a href="reviewed_references.json">复核记录与未知</a> · <a href="reference_freeze.json">计分前冻结及事后诊断声明</a> · <a href="results.json">完整结果</a> · <a href="verification.json">完整性核查</a></p></html>'''
    (ROOT/'index.html').write_text(text.replace('SUMMARY',summary).replace('LOCATIONS',locations).replace('TABLE',table),encoding='utf8')
    lock=read(LAB/'local_rock_patch_20261117/baseline_lock.json');bc={k:sha(lock[k])==lock[k+'_sha256'] for k in ['approved500','working489']};assert all(bc.values())
    checks=dict(reference_hash_matches_before_point_scoring=True,source_pixels_unchanged=True,
        point_and_neighborhood_scores_recomputed=True,unknown_not_counted_as_negative=True,
        posthoc_status_explicit=True,heldout_source_training_still_forbidden=True,
        references_not_full_strokes_or_cunfa=True,model_hashes_unchanged=True,
        evaluated_points=result['evaluated_points'],unknown_points=result['unknown_points'],
        baseline_hashes=bc,new_model_trained=False,independent_validation=False,adopted=False,robot_connected=False)
    write(ROOT/'verification.json',checks)
    report=PROJECT/'reports/20261118_ink_owner_diagnostic.md'
    report.write_text('''# 可见墨迹对象归属诊断

## 1. 推进哪项目标

推进G2的实际墨迹归属，支持G1防止岩面纹理与枝叶混淆。停止追加融合训练，将“区域属于什么对象”和“选中的像素是不是墨迹核心”分开复核。

## 2. 实际新增什么能力

新增墨迹核心参考与逐点／邻域诊断流程，没有新增经过验证的识别模型。整理20处候选，经源图放大及上下文复核，保留10个可见墨迹核心（岩面4、树木6），另外10处保持未知。部分初始点落在留白或淡痕，调整记录和降级原因全部保留。

4种既有输出在核心点上的对应数：256为岩面3/4、树木6/6；512为4/4、3/6；固定平均为4/4、6/6；学习融合为4/4、6/6。它们是10个助手标记点，不是10条已恢复笔画，也不是专家准确率。

## 3. 还有哪些失败

256将R04岩面墨迹判为树木；512将T05、T06、T10树木墨迹判为山石。这4处错误在各自3×3邻域内也完全错误，不只是单个像素舍入。这里只验证相邻位置敏感性，不将9个像素算成9份标注。

初始点有一部分落在岩面／树干留白、淡痕或交叠处；这种位置仍可能属于已知对象，但不能冒充墨迹监督。因此本轮只修复参考范围，没有把待定点当负例。墨带边界、完整曲线、分笔、笔顺和皴法类别仍未恢复；岩面墨迹也不自动等于某种皴法。

## 4. 是否通过独立验证

没有。本轮参考来自已查看过预测的2张图、同一个文件名来源候选组，是明确的事后错例诊断。即使固定平均和学习融合在10点上全部对应，也不能推翻上一轮新图留出检查未通过的结论。

所有参考保持禁止训练，不回灌上轮留出作品。复核及计分前冻结文件、原图哈希、模型哈希、未知剔除及点位敏感性已核查，500/489笔基线不变。没有连接机械臂。

## 采用决定及下一步

本轮只增加可核查的细笔迹诊断，没有模型替换。下一步数据需要覆盖更多独立作品中的整段墨迹、相邻留白与交汇归属；只有这些参考可确认后，才可评估完整笔触恢复，不能继续用少量点对应代替完整画法能力。

实验目录：`E:\\Codex_Workspace_Data\\Gongbi_Line_Model_20261022\\ink_owner_diagnostic_20261118`。
''',encoding='utf8')
    statepath=LAB/'guohua_hierarchy_20261112/current_milestones.json';s=read(statepath);assert not any(s['capability_state'].values())
    s.update(latest_round_report=str(report),latest_ink_owner_diagnostic=str(ROOT/'results.json'),
        new_model_trained_this_round=False,production_replaced=False,robot_connected=False);write(statepath,s)
    for name in ['research/project_plan_20261117.json','research/guohua_four_priorities_execution_20261116.json']:
        path=PROJECT/name;v=read(path);v.update(latest_round_report=str(report),latest_ink_owner_diagnostic=str(ROOT/'results.json'))
        if 'execution' in name:v['current_round']='Ink-core ownership posthoc diagnostic only:10reviewed points and10unknown, no new model or independent acceptance'
        write(path,v)
    print(json.dumps(dict(report=str(report),checks=checks),ensure_ascii=False))
if __name__=='__main__':main()
