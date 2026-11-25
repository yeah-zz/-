# -*- coding: utf-8 -*-
import sys,json,hashlib,html
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from gongbi_training.source_scene_mapping import source_to_scene
ROOT=LAB/'object_prospective_20261118';PROJECT=Path(__file__).resolve().parents[1]
ARMS=['global256','detail512','fixed_mean','learned_gate'];NAMES=['冻结256','冻结512','固定平均','学习融合'];COLORS=np.array([[241,237,222],[196,140,62],[70,143,105],[110,112,163]],'uint8')
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def main():
    p=read(ROOT/'protocol.json');result=read(ROOT/'results.json');ds=read(ROOT/'dataset_manifest.json');freeze=read(ROOT/'reference_freeze.json');refs=read(ROOT/'local_references.json')['records']
    assert sha(p['checkpoint'])==p['checkpoint_sha256'] and sha(p['gate'])==p['gate_sha256']
    assert sha(ROOT/'selection_before_download.json')==p['selection_sha256']
    assert sha(ROOT/'dataset_manifest.json')==freeze['dataset_manifest_sha256'] and sha(ROOT/'local_references.json')==freeze['reference_sha256']
    assert not freeze['predictions_existed'] and len(ds['records'])==8 and len({r['work_group'] for r in ds['records']})==2
    assert not ds['duplicate_pixels_with_known_object_material']
    (ROOT/'views').mkdir(exist_ok=True);catalog=[]
    for r in ds['records']:
        assert sha(r['image'])==r['image_sha256'] and sha(r['mask'])==r['mask_sha256']
        im=Image.open(r['image']).convert('RGB');im.save(ROOT/f'views/{r["id"]}_source.png')
        target=source_to_scene(np.array(Image.open(r['mask'])));colored=np.full((*target.shape,3),205,'uint8');valid=target!=255;colored[valid]=COLORS[target[valid]];Image.fromarray(colored).save(ROOT/f'views/{r["id"]}_reference.png')
        paths={}
        for arm in ARMS:
            pred=np.load(ROOT/f'predictions/{r["id"]}_{arm}.npz')['prediction'];assert pred.shape==target.shape
            paths[arm]=f'views/{r["id"]}_{arm}.png';Image.fromarray(COLORS[pred]).save(ROOT/paths[arm])
        marked=im.copy();d=ImageDraw.Draw(marked)
        for ref in refs:
            if ref['image_id']==r['id']:d.rectangle(ref['box'],outline='#a54091' if ref['object_type']=='unknown' else '#d4432c',width=2)
        marked.save(ROOT/f'views/{r["id"]}_local_reference.png')
        catalog.append(dict(id=r['id'],source=f'views/{r["id"]}_source.png',public_reference=f'views/{r["id"]}_reference.png',
            local_reference=f'views/{r["id"]}_local_reference.png',predictions=paths,group=r['work_group']))
    write(ROOT/'catalog.json',dict(records=catalog,models=dict(zip(ARMS,NAMES))))
    # Recompute every score from stored confusion. Supported-class mean is supplemental.
    supplementary={}
    for group,arms in result['aggregates'].items():
        supplementary[group]={}
        for arm,metric in arms.items():
            c=np.array(metric['confusion']);u=c.sum(0)+c.sum(1)-c.diagonal();ious=[c[k,k]/u[k] for k in range(4) if u[k]]
            assert abs(float(np.mean(ious))-metric['mean_iou'])<1e-12
            present=np.where(c.sum(1)>0)[0]
            supplementary[group][arm]=float(np.mean([c[k,k]/u[k] for k in present]))
    write(ROOT/'supplementary_supported_class_mean.json',dict(scope='post-evaluation supplementary, original metric unchanged',values=supplementary))
    table='<table id="summary"><thead><tr><th>方法</th><th>8图原协议平均IoU</th><th>补充：有参考支持类别均值</th></tr></thead><tbody>'+''.join(f'<tr><td>{name}</td><td>{result["aggregates"]["all"][arm]["mean_iou"]*100:.1f}%</td><td>{supplementary["all"][arm]*100:.1f}%</td></tr>' for arm,name in zip(ARMS,NAMES))+'</tbody></table>'
    perwork='<table><tr><th>文件名来源候选组</th>'+''.join('<th>'+n+'</th>' for n in NAMES)+'</tr>'+''.join('<tr><td>'+html.escape(g)+'</td>'+''.join(f'<td>{result["aggregates"][g][a]["mean_iou"]*100:.1f}%</td>' for a in ARMS)+'</tr>' for g in result['aggregates'] if g!='all')+'</table>'
    rows=read(ROOT/'reference_source_comparison.json')
    localtable='<table><tr><th>预测前固定的助手核心</th><th>与源对象标签一致</th>'+''.join('<th>'+n+'</th>' for n in NAMES)+'</tr>'+''.join('<tr><td>'+r['id']+'</td><td>'+f'{r["source_matching_fraction"]*100:.1f}%</td>'+''.join(f'<td>{r["predictions"][a]*100:.1f}%</td>' for a in ARMS)+'</tr>' for r in rows)+'</table>'
    board=Image.new('RGB',(1800,1125),'#f5f2e9');d=ImageDraw.Draw(board);font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
    for i,ident in enumerate(['zouzhe_xiangjiaoqiuwutu_43','zouzhe_xiangjiaoqiuwutu_37','zouzhe_xiangjiaoqiuwutu_11']):
        c=next(c for c in catalog if c['id']==ident)
        for j,(path,name) in enumerate([(c['source'],'原图'),(c['public_reference'],'公开粗对象标签')]+[(c['predictions'][a],n) for a,n in zip(ARMS,NAMES)]):
            im=Image.open(ROOT/path);im.thumbnail((290,305));board.paste(im,(j*300+5,i*375+38));d.text((j*300+5,i*375+5),name,font=font(18),fill='#243e35')
        d.text((8,i*375+347),ident,font=font(17),fill='#78523c')
    board.save(ROOT/'comparison.jpg',quality=94)
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>冻结模型新图留出检查</title><style>body{font:17px/1.7 system-ui;max-width:1800px;margin:auto;padding:24px;color:#243f35;background:#f5f2e9}h1{font-size:30px}.notice{padding:16px;background:#e8dfc9}img{max-width:100%;height:auto}table{border-collapse:collapse;white-space:nowrap}td,th{padding:9px;border:1px solid #bbb}.scroll{overflow-x:auto}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(290px,100%),1fr));gap:14px}article{padding:12px;background:white}select{font:inherit;max-width:100%}</style><h1>冻结模型后的新图留出检查</h1><p class="notice">模型和名单先冻结，再下载8张裁片，跨2个文件名作品候选组。完成同一公开数据来源的新图检查；作者／完整作品身份和跨馆藏同作未外部认证，不能称专家验收或独立来源验证。未训练新权重，未接入绘画。</p><p>公开粗标签：米白为空景，棕为山石，绿为树木，紫为其他，灰为未知。原协议按有预测／参考并集的类别计算IoU；青绿组没有树木及其他类参考，不能据此验证树木召回。补充支持类别均值另列，不改原指标。不同测试集的数字不能直接当作提升。</p><div class="scroll">TABLE</div><div class="scroll">WORK</div><h2>预测前固定的8处对象核心与2处待定</h2><p>这是助手区域上下文参考，不是逐像素墨迹或精确边界。8处对象核心均与公开源标签一致；仍不等于专家真值。2处交叠未参与计分，也未用于训练。没有从作品名称推断皴法。</p><div class="scroll">LOCAL</div><img src="comparison.jpg"><h2>所有8张新图：完整输入与四种输出</h2><select id="case"></select><p id="scope"></p><div id="gallery" class="grid"></div><p><a href="selection_before_download.json">下载前名单</a> · <a href="protocol.json">冻结模型协议</a> · <a href="reference_freeze.json">预测前参考冻结</a> · <a href="results.json">全部结果及类别支持数</a> · <a href="local_references.json">局部参考与待定位置</a> · <a href="verification.json">核查记录</a></p><script>fetch('catalog.json').then(r=>r.json()).then(data=>{const select=document.querySelector('#case');for(const r of data.records){const o=document.createElement('option');o.value=r.id;o.textContent=r.id;select.append(o)}function render(){const r=data.records.find(r=>r.id===select.value);document.querySelector('#scope').textContent=r.group+'；完整图推理，未按参考框修正输出';const grid=document.querySelector('#gallery');grid.innerHTML='';for(const[n,path]of [['原图',r.source],['公开粗对象标签',r.public_reference],...Object.entries(data.models).map(([k,n])=>[n,r.predictions[k]])]){const a=document.createElement('article'),t=document.createElement('strong'),i=document.createElement('img');t.textContent=n;i.src=path;i.alt=n;a.append(t,i);grid.append(a)}}select.onchange=render;select.value='zouzhe_xiangjiaoqiuwutu_43';render()})</script></html>'''
    (ROOT/'index.html').write_text(doc.replace('TABLE',table).replace('WORK',perwork).replace('LOCAL',localtable),encoding='utf8')
    lock=read(LAB/'local_rock_patch_20261117/baseline_lock.json');bc={k:sha(lock[k])==lock[k+'_sha256'] for k in ['approved500','working489']};assert all(bc.values())
    checks=dict(model_hashes_match_download_precommit=True,selection_frozen_before_download=True,
        local_references_frozen_before_predictions=True,source_image_hashes_and_lfs_checked=True,
        no_pixel_duplicate_with_known_object_material=True,local_references_source_concordant=True,
        eight_crops_two_filename_candidate_groups=True,metrics_recomputed=True,baseline_hashes=bc,
        new_model_trained=False,strict_independent_acceptance=False,adopted=False,robot_connected=False)
    write(ROOT/'verification.json',checks)
    report=PROJECT/'reports/20261118_object_prospective.md'
    metrics='\n'.join(f'|{n}|{result["aggregates"]["all"][a]["mean_iou"]*100:.1f}%|{supplementary["all"][a]*100:.1f}%|' for a,n in zip(ARMS,NAMES))
    report.write_text('''# 冻结模型后的新图留出检查

## 1. 推进哪项目标

主目标G2树石区分，支持G1避免把树叶和染墨误作皴笔。本轮停止在熟悉图片上调参，检查冻结的全局、细节、固定平均和学习融合在项目内新图上的表现。

## 2. 实际新增什么能力

新增8张裁片、2个文件名来源候选组，固定名单和模型哈希后下载。核查与既有对象任务图片无字节或像素重复；完整来源／作者归属和跨馆藏同作未外部认证。新增10处助手区域参考，8处对象核心、2处交叠待定；在模型预测前冻结，未参与训练。8处核心均与源粗对象标签一致，仍非专家边界真值。

完成四种冻结配置的新图比较，没有训练新权重，也没有新增通过严格验收的识别或皴法能力。

|配置|原协议平均IoU|补充：有参考支持类别均值|
|---|---:|---:|
'''+metrics+'''

原协议按有预测／参考并集的类别平均，缺失类别若发生误报也计入。新图无“其他类”参考，青绿候选组也无树木参考，因此补充支持类别均值单列；没有追溯更改原指标。不能与旧14图的数值直接相减称提升。

## 3. 还有哪些失败

学习融合仍低于固定平均，没有证明额外收益。新图中淡山体核心：冻结256为70.2%，512为21.5%，固定平均39.7%，学习融合55.3%；树干核心：100%、7.6%、54.9%、95.0%。学习融合缓解部分512退步，但仍未超过原全局判断。

青绿组无法检验树木召回。源粗标签和助手区域均不能证明单笔归属、精确树石边界或皴法类别。不能把8个相关裁片当8件独立作品。

## 4. 是否通过独立验证

完成了模型冻结后的项目内新图留出检查；未达到严格独立验收。两组使用与训练相同的公开数据来源，文件名作者／作品身份及跨馆藏重复尚未独立认证，专家区域标注缺失，规模只有两组。

全过程无再训练、调阈值或按预测换图。保留下载前选择、原始图和标签哈希、预测前参考冻结、所有结果与逐组类别支持数。未替换任何绘画组件。

## 采用决定及下一步

学习融合不采用；认可500笔和489笔工作基线哈希不变，机械臂离线。新来源组今后应按“已打开留出检查”记录，若用于训练须明确退役留出身份，不能再次称新盲测。

下一步数据重点是淡山体、细树干及交叠边界的可靠监督；这批留出参考继续保留作检查，不能直接回灌训练后继续报告同一组测试成绩。皴法辨识、真实单笔与笔组数据仍需单独补齐，树木画法训练尚未解锁。
''',encoding='utf8')
    statepath=LAB/'guohua_hierarchy_20261112/current_milestones.json';state=read(statepath);assert not any(state['capability_state'].values())
    state.update(latest_round_report=str(report),latest_prospective_object_check=str(ROOT/'results.json'),
        latest_prospective_object_check_status='two new filename work candidates, same public source; strict independent acceptance not met',
        new_model_trained_this_round=False,production_replaced=False,robot_connected=False);write(statepath,state)
    registry=PROJECT/'research/source_usage_registry_20261117.json';v=read(registry)
    for group in result['aggregates']:
        if group=='all':continue
        ident='clpd_filename_group:'+group
        v['records']=[r for r in v['records'] if r.get('source_identity')!=ident]
        v['records'].append(dict(source_identity=ident,usage='opened_prospective_check_not_training',training_allowed=False,
            eligible_for_fresh_blind_test=False,attribution_unverified=True,record=str(ROOT/'protocol.json')))
    write(registry,v)
    for name in ['research/project_plan_20261117.json','research/guohua_four_priorities_execution_20261116.json']:
        path=PROJECT/name;v=read(path);v.update(latest_round_report=str(report),latest_prospective_object_check=str(ROOT/'results.json'));write(path,v)
    # Browser checker reused with actual current page counts, not model metrics.
    text=(PROJECT/'scripts/check_regional_presence_view_20261117.py').read_text('utf8')
    replacements=[("ROOT=LAB/'regional_presence_trial_20261117'","ROOT=LAB/'object_prospective_20261118'"),('length===43','length===8'),('document.querySelectorAll("tbody tr").length','document.querySelectorAll("#summary tbody tr").length'),("')==5","')==4"),("['F1500','H_14','wen-zhengming_peach-blossom-spring_10']","['wangximen_101','zouzhe_xiangjiaoqiuwutu_37','zouzhe_xiangjiaoqiuwutu_43']"),('five_model_rows=True','four_model_rows=True'),('all43_case_options=True','all8_case_options=True')]
    for a,b in replacements:text=text.replace(a,b)
    (PROJECT/'scripts/check_object_prospective_view_20261118.py').write_text(text,encoding='utf8')
    print(json.dumps(dict(report=str(report),checks=checks),ensure_ascii=False))
if __name__=='__main__':main()
