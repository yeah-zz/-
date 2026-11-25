# -*- coding: utf-8 -*-
import sys,json,hashlib,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from PIL import Image,ImageDraw,ImageFont
from gongbi_training.cunfa_input_scale import input_canvas
ROOT=LAB/'cunfa_concept_supervision_20261116';OLD=LAB/'cunfa_role_reference_20261112'
PROJECT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text('utf8'))
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 ds=read(ROOT/'reference_manifest_corrected.json');res=read(ROOT/'results.json');byid={r['id']:r for r in res['records']}
 refs=read(OLD/'source_references.json')['records'];font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
 cases=[];units=[]
 for ref in refs:
  rows=[r for r in ds['records'] if r['reference_unit']==ref['id']]
  if not rows:continue
  im=Image.open(ref['image']).convert('RGB');dr=ImageDraw.Draw(im)
  for i,r in enumerate(rows):dr.rectangle(r['box'],outline='#cf7330',width=2);dr.text((r['box'][0]+2,r['box'][1]+2),str(i+1),font=font(15),fill='#d75923')
  p=ROOT/'views'/(ref['id']+'_review_windows.jpg');im.save(p,quality=94)
  units.append(dict(id=ref['id'],image='views/'+p.name,source_url=ref['source_url'],split=rows[0]['split'],named_style=ref['local_style']))
  for r in rows:
   canvas,meta=input_canvas(Image.open(r['image']),fill_extent=True);p=ROOT/'views'/(r['id']+'_network.png');canvas.save(p)
   cases.append(dict(id=r['id'],image='views/'+Path(r['image']).name,network_input='views/'+p.name,context='views/'+ref['id']+'_review_windows.jpg',
    split=r['split'],targets=byid[r['id']]['targets'],scores=byid[r['id']]['scores'],predicted=byid[r['id']]['predicted'],
    errors=byid[r['id']]['errors'],visual_evidence=r['visual_evidence'],source_url=r['source_url'],source_unit=ref['id'],object_type=r['object_type'],role='unknown'))
 # A compact, actual-source illustration of co-occurring concepts and failures.
 selected=['aam_small_axe_4','hkma_axe_2','hkma_axe_3','hkma_hemp_1','hkma_hemp_2']
 board=Image.new('RGB',(1500,540),'#f3f0e7');d=ImageDraw.Draw(board)
 d.text((18,12),'属性监督与真实错例：图像未重画，参考与模型输出分开',font=font(28),fill='#243f35')
 for i,ident in enumerate(selected):
  r=next(z for z in cases if z['id']==ident);im=Image.open(ROOT/r['image']);native_size=im.size
  scale=min(270/im.width,195/im.height);im=im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))),Image.Resampling.NEAREST)
  x=i*300+12;board.paste(im,(x,68));d.text((x,276),'原生 '+str(native_size[0])+'×'+str(native_size[1])+'，仅展示缩放',font=font(15),fill='#805333')
  d.text((x,301),ident,font=font(17),fill='#243f35')
  t=' / '.join('?' if v is None else str(v) for v in r['targets']);p=' / '.join(str(v) for v in r['predicted'])
  d.text((x,337),'参考：'+t,font=font(19),fill='#805333');d.text((x,370),'预测：'+p,font=font(19),fill='#28667d')
  d.text((x,411),'训练参考' if r['split']=='train' else ('留出：有错例' if r['errors'] else '留出：本处对应'),font=font(19),fill='#a04a38' if r['errors'] else '#243f35')
 d.text((18,468),'属性顺序：细长墨迹 / 宽角墨块 / 弯曲笔组。1=可见，0=未见，?=待定。',font=font(21),fill='#243f35')
 d.text((18,505),'这是助手局部形态参考与小样模型预测，不是完整笔触恢复、国画创作或机械臂实画。',font=font(20),fill='#805333')
 board.save(ROOT/'concept_comparison.jpg',quality=94)
 # Existing higher-resolution assets are re-used, not newly acquired examples.
 native=read(OLD/'native_manifest.json');native_cases=[]
 for r in native['records']:
  assert hashlib.sha256(Path(r['native_image']).read_bytes()).hexdigest()==r['native_sha256']
  for kind,key in [('native','native_image'),('preview','old_image')]:
   dest=ROOT/'views'/(r['id']+'_'+kind+'.png');shutil.copyfile(r[key],dest)
  native_cases.append(dict(id=r['id'],native='views/'+r['id']+'_native.png',preview='views/'+r['id']+'_preview.png',native_size=r['native_size'],preview_size=r['old_size'],local_named_style=None,role=None,source_url=r['source_url']))
 write('native_asset_reconciliation.json',dict(records=native_cases,old_manifest_sha256=hashlib.sha256((OLD/'native_manifest.json').read_bytes()).hexdigest(),
  existing_native_assets_reused=True,new_downloads=0,new_independent_samples=0,new_style_targets=0,
  previous_new_url_probe_not_a_complete_local_asset_inventory=True,robot_connected=False))
 write('report_data.json',dict(cases=cases,units=units,concept_metrics=res['concept_metrics'],style_probe=res['style_probe'],style_summary=res['style_summary'],native=native_cases))
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>皴法中间监督 · 形态参考与模型直接输出</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f0e7;color:#243f35;font:17px/1.65 "Microsoft YaHei",sans-serif}main{max-width:1500px;margin:auto;padding:25px}h1{font-size:29px}.note{padding:16px;border-left:5px solid #b46f35;background:#fff8e7}.warn{color:#994c37}a{color:#246c69}img{max-width:100%;height:auto;display:block}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:15px}.card{min-width:0;padding:12px;background:#fffdf8;border:1px solid #d1d6c9}.scroll{overflow:auto}table{width:100%;border-collapse:collapse}th,td{padding:9px;text-align:left;border-bottom:1px solid #d1d6c9}select{max-width:100%;width:650px;padding:8px;font:inherit}.scores{display:grid;grid-template-columns:1fr 1fr;gap:15px}.bar{height:12px;background:#d6dbd1;position:relative}.fill{height:12px;background:#397d91}.cut{position:absolute;left:50%;top:0;height:12px;border-left:2px solid #9b4c31}small{color:#78664f}@media(max-width:720px){main{padding:14px}.grid,.scores{grid-template-columns:1fr}h1{font-size:24px}}</style><main>
<small>2026-11-16 · 第一轮可观察形态监督 · 离线实验</small><h1>从猜皴法名称，推进到可核查的笔形与笔组属性</h1>
<p class="note"><b>已实际训练3个属性头，并将属性接入一个不直接读取图像特征的皴法分类头；本轮未通过。</b>在7处留出形态参考中，宽角墨块对应5/6处，弯曲笔组对应5/7处；细长线的7处全为正例，尚未检验误报。最终皴法的2个示例中，直接分类对应2个、属性瓶颈对应1个。样本来自旧资料，不能宣称新增独立验收或优于旧版。</p>
<h2>1. 本轮的训练链路</h2><p>原始山石示例 → 已检查的局部窗口 → 冻结图像编码器 → 细长墨迹、宽角墨块、弯曲笔组 → 概念分类头。对象归属由参考确认，本轮未重新训练对象分割；窗口也由助手指定，尚未实现从整幅自动找笔触。所有局部角色仍是未知，属性不能自动变成皴笔、轮廓、笔顺或下压量。</p>
<p>原有8个具名示例中，6个可支持部分形态参考，形成25个区域视图；它们不是25件独立作品。AAM的4幅示例、18个区域用于训练；HKMA同一教学照片中的2个示例、7个区域用于留出检查。两个NPM低清局部未添加属性标签。训练与测试的来源图和来源组完全分开，测试结果未用于选阈值或调参。</p>
<h2>2. 什么才算中间监督</h2><p>细长线与宽角块可以同时存在。斧劈教学示例中也有细长边界，不能把所有区域都标成宽块；弯曲边线可能是轮廓，不自动等于披麻皴。25处参考由助手看图复核，不是专家真值，也没有自动生成完整笔画。深墨交叠、模糊宽部等目标留空，未标区域不作为负例。</p><img src="concept_comparison.jpg" alt="助手形态参考与模型直接输出及错例">
<h2>3. 逐区域检查：原图、窗口与真实网络输入</h2><select id="case"></select><p id="scope"></p><div class="grid"><div class="card"><b>原始局部墨迹</b><img id="source"></div><div class="card"><b>助手窗口位置，非模型检测</b><img id="context"></div><div class="card"><b>真实网络输入</b><img id="network"></div></div><p id="evidence"></p><p id="role" class="warn"></p><div id="scores"></div><p><a id="source_link" target="_blank">原资料出处</a></p>
<h2>4. 留出形态结果：不能只看对应数</h2><div class="scroll"><table><thead><tr><th>属性</th><th>已标参考</th><th>对应数</th><th>恒猜多数类</th><th>误报 / 漏报</th><th>限制</th></tr></thead><tbody id="metrics"></tbody></table></div><p>这些是所选助手参考的一致情况，非专家准确率。细长线7/7与恒猜有线相同，不能当作已学会；另两项样本也很少。模型分数大多靠近0.5，未做概率校准，不能作为可靠置信度。未知目标未加入评分。</p>
<h2>5. 属性能否帮助辨识皴法</h2><div class="scroll"><table><thead><tr><th>具名示例</th><th>来源标签</th><th>直接图像分类</th><th>模型属性 → 类别</th><th>参考属性辅助诊断</th></tr></thead><tbody id="styles"></tbody></table></div><p class="warn">只有2个具名示例，来自同一张HKMA照片，不能与上一轮63例训练、13例回归直接比较。参考属性辅助列使用了人工参考，是诊断而非自主推理。正常属性路径只读取模型分数；它未达到直接分类结果。概念头分数集中在0.5附近，而下游头用0/1参考均值训练，也暴露了中间表示尺度不一致；本轮保留失败结果，未在这2例上调阈值补救。</p>
<h2>6. 全部留出错例</h2><div id="failures"></div><p>宽角块漏检1处，弯曲笔组漏检1处、误检1处。需补同一来源中共存的线形、角块和弧组参考，特别是细长线反例，并在训练内部处理属性分数的校准；之后再换完整新来源验收。当前不把这7个区域改作训练数据。</p>
<h2>7. 夏圭原生细节复查</h2><p>本次资产核查找回了2026-11-12已保存的四处原生放大局部，尺寸约376–418×355–460；上一轮新URL探查没有新增高清图，不能据此说项目没有原生细节。以下复用旧资源，非AI增强，也不是新增独立样本。可见结构边、点状墨迹、淡墨和规则底纹混杂，仍不能仅凭作品说明给每处局部贴单一皴法或完整分笔标签。</p><div id="native"></div>
<h2>8. 对象、角色与绘画能力的边界</h2><p>接口已检查：树叶、空景、未知对象不进入山石属性监督；只有确认的山石皴笔才允许输出皴法标签。此为参考约束测试，不代表本轮学会树石自动区分。轮廓／皴笔联合模型、完整曲线、墨带、笔组先后和物理毛笔控制仍待训练验证。树木画法训练保持关闭。本轮未替换绘画版本，未连接机械臂。</p>
<details><summary>查看全部25处助手参考</summary><img src="reference_board.jpg"></details><p><a href="reference_manifest.json">参考及未知目标</a> · <a href="definitions.json">属性定义</a> · <a href="protocol.json">冻结协议</a> · <a href="results.json">全部模型输出</a> · <a href="verification.json">验证记录</a> · <a href="contract_checks.json">监督与对象约束测试</a> · <a href="native_asset_reconciliation.json">原生资产核对</a></p></main>
<script>let D;const $=id=>document.getElementById(id),names=['细长墨迹','宽角墨块','弯曲笔组'],keys=['elongated_mark','angular_broad_mark','curved_parallel_group'],label=v=>v===null?'待定':v===1?'可见':'未见',style=v=>v===0?'披麻皴':'斧劈皴';function show(){let r=D.cases[$('case').selectedIndex];$('source').src=r.image;$('network').src=r.network_input;$('context').src=r.context;$('scope').textContent=r.id+' / '+(r.split==='train'?'训练视图':'留出视图')+' / 同源单元：'+r.source_unit;$('evidence').textContent='助手可见依据：'+r.visual_evidence;$('role').textContent='角色：未知；窗口不是完整笔画或墨迹掩膜。';$('source_link').href=r.source_url;$('scores').innerHTML=keys.map((k,j)=>'<p>'+names[j]+'：参考 '+label(r.targets[j])+'；预测 '+label(r.predicted[j])+'；分数 '+r.scores[j].toFixed(3)+'</p><div class="bar"><div class="fill" style="width:'+r.scores[j]*100+'%"></div><span class="cut"></span></div>').join('');}fetch('report_data.json').then(r=>r.json()).then(d=>{D=d;d.cases.forEach(r=>{let o=document.createElement('option');o.textContent=r.id+' / '+r.split;$('case').append(o)});$('case').onchange=show;show();keys.forEach((k,j)=>{let r=d.concept_metrics[k],tr=document.createElement('tr');tr.innerHTML='<td>'+names[j]+'</td><td>'+r.positive_test+'正 / '+r.negative_test+'负</td><td>'+r.agreement_count+'/'+r.known_test+'</td><td>'+r.majority_agreement_count+'/'+r.known_test+'</td><td>'+r.fp+' / '+r.fn+'</td><td>'+(r.both_polarities_tested?'仅小样参考':'未检查负例')+'</td>';$('metrics').append(tr)});d.style_probe.forEach(r=>{let tr=document.createElement('tr');tr.innerHTML='<td>'+r.id+'</td><td>'+r.source_label+'</td><td>'+style(r.direct)+'</td><td>'+style(r.concept_bottleneck)+'</td><td>'+style(r.reference_oracle)+'（参考辅助）</td>';$('styles').append(tr)});d.cases.filter(r=>r.split==='test'&&r.errors.length).forEach(r=>{let p=document.createElement('p');p.textContent=r.id+'：'+r.errors.map(k=>names[keys.indexOf(k)]).join('、')+'；'+r.visual_evidence;$('failures').append(p)});d.native.forEach(r=>{let e=document.createElement('details');e.innerHTML='<summary>'+r.id+' / 旧预览 '+r.preview_size.join('×')+' → 已有原生 '+r.native_size.join('×')+'</summary><div class="scores"><div class="card">旧预览<img src="'+r.preview+'"></div><div class="card">已有原生细节<img src="'+r.native+'"></div></div>';$('native').append(e)});window.reportReady=true;});</script></html>'''
 page=page.replace('原始山石示例 →','山石／教学笔形示例 →')
 page=page.replace('<h2>3. 逐区域检查', '<p class="note">元数据复核：HKMA三处独立练笔区域初版误记为山石，现已改为“教学笔形”。它们可用于形态参考，但不能作为山石实例、皴笔角色或绘画动作；所有形态目标、窗口、分组和模型预测均未改。初版与修正记录同时保留。</p><h2>3. 逐区域检查')
 page=page.replace("+' / 同源单元：'+r.source_unit", "+' / 同源单元：'+r.source_unit+' / 对象参考：'+(r.object_type==='teaching_mark'?'教学笔形':'山石区域')")
 page=page.replace('<a href="reference_manifest.json">参考及未知目标</a>', '<a href="reference_manifest_corrected.json">修正后的参考及未知目标</a> · <a href="reference_metadata_correction.json">元数据纠错记录</a> · <a href="reference_manifest.json">初版参考（保留）</a>')
 (ROOT/'index.html').write_text(page,encoding='utf8')
 server=(PROJECT/'scripts/serve_guohua_objects_v2_20261113.py').read_text('utf8').replace('guohua_objects_v2_20261113','cunfa_concept_supervision_20261116')
 a=server.index('ALLOWED={');b=server.index('\nALLOWED.update',a)
 allowed=['index.html','concept_comparison.jpg','reference_board.jpg','report_data.json','reference_manifest.json','reference_manifest_corrected.json','reference_metadata_correction.json','definitions.json','protocol.json','results.json','verification.json','contract_checks.json','native_asset_reconciliation.json','postexperiment_checks.json']
 server=server[:a]+'ALLOWED='+repr(set(allowed))+server[b:]
 server=server.replace('str(Path(__file__).resolve().parents[1])',repr(str(PROJECT)))
 (ROOT/'serve_report.py').write_text(server,encoding='utf8')
 print('REPORT_READY',len(cases),flush=True)

if __name__=='__main__':main()
