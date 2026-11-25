# -*- coding: utf-8 -*-
"""Expose reference coverage, raw failure, and a bounded next-data queue."""
import sys,json,hashlib,html
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import cv2,numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from cunfa_temporal_probe_20261112 import RoleNet,features
ROOT=LAB/'cunfa_temporal_roles_20261112'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,x):(ROOT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 ds=json.loads((ROOT/'dataset_manifest.json').read_text('utf8'));res=json.loads((ROOT/'role_probe_results.json').read_text('utf8'));protocol=res['protocol'];assert sha(ROOT/'dataset_manifest.json')==protocol['dataset_sha256'];assert sha(ROOT/'role_probe.pt')==res['model_sha256']
 counts={s:{str(k):sum(r['confirmed_partial_pixel_counts'][str(k)] for r in ds['records'] if r['split']==s) for k in [0,1,2]} for s in ['train','exploratory_other_work_test']}
 # Diagnostics use the frozen checkpoint only, never update data/model/threshold.
 torch.set_num_threads(4);model=RoleNet().cuda();model.load_state_dict(torch.load(ROOT/'role_probe.pt',map_location='cuda',weights_only=False)['model']);model.eval();diagnostics=[]
 for r in ds['records']:
  z=np.load(r['path']);im=z['image'];t=z['targets'];g=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY).astype('float32');local=cv2.GaussianBlur(g,(0,0),4)-g;ink=local>3;dist=cv2.distanceTransform(ink.astype('uint8'),cv2.DIST_L2,5)*2
  info=dict(id=r['id'],reference_width_pixels={str(k):dict(median=float(np.median(dist[t==k])),p25=float(np.quantile(dist[t==k],.25)),p75=float(np.quantile(dist[t==k],.75))) if (t==k).any() else None for k in [1,2]},width_is_image_proxy_not_physical=True,transformed_inputs=[])
  if r['split']!='train':
   for scale in [.5,1.,2.]:
    changed=cv2.resize(im,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR);ht,wt=changed.shape[:2]
    if min(ht,wt)<8:continue
    with torch.inference_mode():log=model(torch.from_numpy(features(changed))[None].cuda());p=torch.nn.functional.interpolate(log,size=t.shape,mode='bilinear',align_corners=False).argmax(1)[0].cpu().numpy()
    support=t==2;info['transformed_inputs'].append(dict(scale=scale,cun_reference_predicted_as=[int((p[support]==k).sum()) for k in range(3)],support=int(support.sum())))
  diagnostics.append(info)
 write('frozen_role_diagnostics.json',dict(records=diagnostics,posthoc_diagnostic_only=True,not_new_test_samples=True,no_tuning_or_retraining=True,model_sha256=res['model_sha256'],width_proxy_not_semantic_cause=True))
 # Collect precise data gaps without turning proposals into training labels.
 queue=[]
 for r in ds['records']:
  queue.append(dict(id=r['id'],source_family=r['source_family'],existing_role_cores=r['confirmed_partial_pixel_counts'],needs=['distributed paper/weak ink boundary review','complete visible structure vs cun relationship review','crossing/occluded area ignore decisions'],status='needs_review_not_training_gold',expert_verified=False,automatic_predictions_as_label=False))
 write('next_reference_queue.json',dict(records=queue,minimum_next_gate=['At least two independent teaching works with both structure and cun core references in each static scene','Reserve a third independent work before training, with both roles and representative paper references','Keep source/author groups separate; never turn current predictions into labels','Review curves and ownership separately only after role recognition passes'],style_mixed_unknown_required=True,robot_connected=False))
 font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n);board=Image.new('RGB',(1200,900),'#f4f1e8');d=ImageDraw.Draw(board)
 d.text((16,12),'视频辅助角色参考已经建立；新模型跨来源仍失败',font=font(27),fill='#183f32')
 for i,ident in enumerate(['T_cun_group1','V_S01']):
  panel=Image.open(ROOT/'predictions'/(ident+'_comparison.jpg'));board.paste(panel,(0,60+i*410))
 board.save(ROOT/'overview.jpg',quality=94)
 rows=[]
 for r in ds['records']:
  rr=next(z for z in res['records'] if z['id']==r['id']);rows.append(dict(**r,metrics=rr,reference_preview='previews/'+r['id']+'.jpg',comparison='predictions/'+r['id']+'_comparison.jpg',input='references/'+r['id']+'_after.png',target='references/'+r['id']+'_targets.png',raw='predictions/'+r['id']+'_raw.png'))
 phases=[dict(id=r['id'],image='references/'+r['id']+'_after.png',reference='references/'+r['id']+'_cores.png',time=str(r['after_seconds'])+'秒',label=l) for r,l in zip(ds['records'][:3],['早期结构阶段','第一组加皴后','第二组加皴后'])]
 data=dict(records=rows,phases=phases,counts=counts,aggregates=res['aggregates'],structure_transfer=json.loads((ROOT/'structure_transfer.json').read_text('utf8')),diagnostics=diagnostics)
 write('report_data.json',data)
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>皴法角色参考与失败复核 · 11月12日</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f4f1e8;color:#263c32;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1400px;margin:auto;padding:28px}h1{font-size:30px;line-height:1.35}h2{font-size:23px;margin-top:34px}p{max-width:1060px}.note{background:#fff9e8;border-left:5px solid #bc7f3d;padding:16px}.bad{color:#9c3e2f}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}.card{background:#fffdf7;border:1px solid #ddd8c9;border-radius:10px;padding:14px}.metric{font-size:29px;font-weight:650}select,button{padding:9px 14px;font:inherit;max-width:100%;border:1px solid #a1b2a3;border-radius:6px;background:#fffdf7;color:#263c32}button{cursor:pointer}img{max-width:100%;height:auto;display:block}.image{background:#faf9f4;border:1px solid #e1dece;padding:8px}.image img{max-height:500px;margin:auto}.legend{display:flex;flex-wrap:wrap;gap:18px;margin:12px 0}.dot{display:inline-block;width:14px;height:14px;margin-right:7px}table{border-collapse:collapse;width:100%;margin-top:12px}td,th{text-align:left;padding:10px;border-bottom:1px solid #d7d7c9}small{color:#627269}a{color:#236b61}.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}.strip{margin:12px 0;display:flex;gap:12px;align-items:center;flex-wrap:wrap}.scroll{overflow:auto}details{margin:16px 0}code{overflow-wrap:anywhere}@media(max-width:800px){main{padding:16px}.grid,.two{grid-template-columns:1fr 1fr}h1{font-size:25px}}@media(max-width:480px){.grid,.two{grid-template-columns:1fr}.metric{font-size:24px}td,th{padding:6px;font-size:14px}}</style>
<main><small>离线研究 · 2026-11-12 · 未连接机械臂</small><h1>把“墨迹在哪里”与“它承担什么作用”分开验证</h1>
<p class="note"><b>本轮新增了视频支持的局部角色参考，并实际训练了一个角色模型；模型仍未通过跨来源检查。</b>训练画面上能拟合参考，另一教程的皴笔却几乎全部被判为结构。原始输出也有明显的空白误检，不能作为新绘画效果或完整笔画。</p>
<div class="grid"><div class="card"><div class="metric">1 → 1</div>训练作品 → 另一来源作品<br><small>各3次观察；不是6件独立作品</small></div><div class="card"><div class="metric">2种角色</div>结构核心 / 皴笔核心<br><small>未训练皴法类别、染墨、笔顺和压力</small></div><div class="card"><div class="metric">0.17%</div>另一来源皴笔核心召回<br><small>2 / 1,182个参考像素，不是整笔准确率</small></div><div class="card"><div class="metric">未通过</div>不替换已认可绘画版本<br><small>另一来源还缺结构参考和完整分笔</small></div></div>
<h2>1. 真正补上的监督是什么</h2><p>从同一视频固定镜头内比较加笔前后，保留可核查的新增墨迹核心；交叉、已有墨迹邻域和无法确认的区域继续忽略。早期结构只做几何对齐后转移到放大镜头，<b>没有跨镜头切换直接做像素差分</b>。结构包含外轮廓与山面转折，并不等于所有主轮廓。</p>
<div class="legend"><span><i class="dot" style="background:#c08028"></i>结构核心</span><span><i class="dot" style="background:#2887b4"></i>皴笔核心</span><span><i class="dot" style="background:#5a965a"></i>检查过的纸面核心</span><span><i class="dot" style="background:#dad4c7"></i>待定，不参与损失</span></div>
<p>跨镜头几何对齐：171个内点，重投影均方根误差0.79像素；转移核心约96.7%位于可见墨迹附近。这是对齐检查，<b>不是语义标注正确率</b>。参考经过助手局部检查，未经专家逐像素审核，也不是完整笔触归属。</p>
<h2>2. 分阶段证据回放</h2><p>下面是教程实际画面的三个阶段快照，供检查参考依据；不是模型生成过程、逐笔回放或机械臂实时画面。第一幅与后两幅镜头倍率不同。</p><div class="strip"><button id="prev">上一步</button><button id="play">播放阶段</button><button id="next">下一步</button><span id="phaseTitle"></span></div><div class="two"><div class="image"><b>实际教程画面</b><img id="phaseInput"></div><div class="image"><b>本轮局部参考核心</b><img id="phaseReference"></div></div>
<h2>3. 同图检查：参考与模型直接输出</h2><div class="strip"><label>观察 <select id="case"></select></label><span id="split"></span></div><p id="scope"></p><div class="grid"><div class="image"><b>静态输入</b><img id="input"></div><div class="image"><b>局部参考 · 灰色待定</b><img id="target"></div><div class="image"><b>模型直接角色图</b><img id="raw"></div><div class="card"><b>参考覆盖</b><div id="coverage"></div><p>模型角色图显示所有像素的直接分类，没有用参考遮罩修饰。空白误检在训练画面上也仍然可见。</p></div></div><p><b>这不是线稿：</b>彩色图只预测角色，尚未输出完整实例、中心曲线、宽度和笔组。给角色图换色或隐藏错误区域，都不能算完成笔触提取。</p>
<h2>4. 换来源后为什么判定失败</h2><div class="scroll"><table><thead><tr><th>方法 / 另一来源参考</th><th>纸面核心召回</th><th>结构核心召回</th><th>皴笔核心召回</th></tr></thead><tbody id="metrics"></tbody></table></div><p>对照方法只用训练参考选择墨迹阈值，把所有检测到的墨迹都叫“皴笔”。它在这里也能取得100%皴笔核心召回，说明<b>这批测试只含皴笔正例，无法单独证明结构与皴笔的区分能力</b>。模型几乎把皴笔全判为结构，则已足以说明迁移失败。纸面参考只覆盖少量已核查位置；其中不误检，不代表整幅没有背景误检。</p>
<details><summary>训练与检查的边界</summary><ul><li>训练3个阶段观察全部来自同一教程作品；另一来源3个局部也属于同一件作品。</li><li>测试图沿用之前项目检查过的多帧合成图，并按已知笔迹附近裁剪；不是新盲测，也没有验证自动从整幅中找笔。</li><li>模型输入只有静态图的灰度与局部对比度；训练损失只用有依据的核心像素，未知为255。未输入前帧、差分、引导曲线、阶段时间或角色编号。</li><li>固定种子与240步预算，使用最后一步；没有用另一来源选择阈值或检查点。训练上的高拟合不作为通过证据。</li><li>缩放仅用于冻结后的诊断，不增加样本、不调参、不重新训练。像素宽度也不对应真实毛笔下压。</li><li>全部结果保留；旧13例分类失败测试和当前已认可绘画版本均未替换。</li></ul></details>
<h2>5. 下一轮怎么改，才有验收意义</h2><ol><li><b>补同画面中的完整角色对照。</b>每件示范都需要同时有结构、皴笔和分布在山面内外的纸面参考，避免只在图角监督背景。先做两个独立教程作品，另留第三件封存。</li><li><b>先判断墨迹，再判断角色。</b>为墨迹前景设单独约束与验证；当前未知区域不能直接补成背景标签。角色模型必须在新作品中区分结构和皴笔，不能只把所有深色都算皴笔。</li><li><b>角色稳定后再恢复整笔。</b>另外核查交叉的墨迹归属、端点和干笔断续，以完整曲线和墨带监督；不把细骨架或这轮预测当真实分笔。</li><li><b>最后接对象和笔序。</b>一块石头内部先结构、后成组皴笔，再切换下一对象；树木按主干、分枝、枝叶。当前还不能称为学会了这些规划规则。</li></ol>
<p>下一轮门槛是：封存新作品上，两类真实笔迹都有参考；比“全当结构 / 全当皴笔”对照更好，同时检查背景误检、误接和笔形。达到后再训练披麻、斧劈等类别以及对应笔组生成。</p>
<p><a href="https://www.ganjingworld.com/video/1gc2lo9n27j5ZShwAja69iQ3N12d1c" target="_blank" rel="noreferrer">训练参考来源：米小鹿画画 · 短披麻皴示范</a><br><a href="https://www.ganjingworld.com/video/1h5vi07sks9KFNZ72hjMX9VrP1331c" target="_blank" rel="noreferrer">另一来源：刘鹏凯 · 秋山问道临摹研习37</a></p><p><a href="role_probe_protocol.json">冻结训练协议</a> · <a href="role_probe_results.json">逐例结果</a> · <a href="next_reference_queue.json">待补参考清单</a> · <a href="verification.json">核查记录</a></p></main>
<script>let D,phase=0,timer=null;const $=s=>document.getElementById(s);function showPhase(){let p=D.phases[phase];$('phaseTitle').textContent=(phase+1)+'/3 '+p.label+' · '+p.time;$('phaseInput').src=p.image;$('phaseReference').src=p.reference}function step(n){phase=(phase+n+3)%3;showPhase()}function show(){let r=D.records[$('case').selectedIndex];$('input').src=r.input;$('target').src=r.target;$('raw').src=r.raw;$('split').textContent=r.split==='train'?'训练作品内观察':'另一来源局部检查';$('scope').textContent=r.split==='train'?'视频支持的局部参考；同一训练作品，不能视为独立泛化验证。':'此前已查看作品；已知笔迹附近裁剪，只能检查有参考的皴笔核心，不代表自主定位。';let c=r.confirmed_partial_pixel_counts;$('coverage').textContent='纸面 '+c['0']+' / 结构 '+c['1']+' / 皴笔 '+c['2']+' 像素；未知 '+r.unknown_pixels}fetch('report_data.json').then(r=>r.json()).then(d=>{D=d;for(let r of D.records){let o=document.createElement('option');o.textContent=r.id;$('case').append(o)}$('case').onchange=show;show();showPhase();$('next').onclick=()=>step(1);$('prev').onclick=()=>step(-1);$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='播放阶段'}else{timer=setInterval(()=>step(1),1700);$('play').textContent='停止播放'}};let a=D.aggregates.exploratory_other_work_test;for(let [key,name] of [['model','角色模型'],['ink_as_cun_baseline','墨迹全部当皴笔']]){let tr=document.createElement('tr');tr.innerHTML='<td>'+name+'</td>'+a[key].recall.map(v=>'<td>'+(v===null?'无参考，不能评估':(v*100).toFixed(2)+'%')+'</td>').join('');$('metrics').append(tr)}window.reportReady=true});</script></html>'''
 (ROOT/'index.html').write_text(page,encoding='utf8')
 # Values are unchanged; the old internal flag name in training aggregates means
 # only that both role supports exist. Do not interpret it as independence.
 checks=dict(dataset_sha_matches_frozen_protocol=True,checkpoint_hash_unchanged=True,all_record_hashes_match=all(sha(r['path'])==r['sha256'] for r in ds['records']),train_test_work_groups_disjoint=True,test_is_new_blind=False,test_structure_reference_pixels=0,test_cun_reference_pixels=counts['exploratory_other_work_test']['2'],unlabelled_ignored=True,no_automatic_predictions_used_as_targets=True,geometry_transfer_not_cross_cut_pixel_difference=True,raw_output_not_reference_masked=True,model_adopted=False,complete_strokes_trained=False,robot_connected=False,aggregate_flag_note='structure_vs_cun_independent_test_valid in training means both reference roles present, NOT independent validation')
 write('verification.json',checks);print(json.dumps(checks,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
