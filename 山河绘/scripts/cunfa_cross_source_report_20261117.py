# -*- coding: utf-8 -*-
import sys,json,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from PIL import Image,ImageDraw,ImageFont
from gongbi_training.cunfa_input_scale import input_canvas
ROOT=LAB/'cunfa_cross_source_20261117';PROJECT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text('utf8'))
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 ds=read(ROOT/'reference_manifest.json');results=read(ROOT/'results.json');pred={r['id']:r for r in results['records']};cases=[]
 font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
 byunit={}
 for r in ds['records']:byunit.setdefault(r['reference_unit'],[]).append(r)
 for unit,rows in byunit.items():
  im=Image.open(rows[0]['reference_image']).convert('RGB');d=ImageDraw.Draw(im)
  for j,r in enumerate(rows):d.rectangle(r['box'],outline='#cc7132',width=max(2,im.width//450));d.text((r['box'][0]+2,r['box'][1]+2),str(j+1),font=font(max(14,im.width//60)),fill='#cc7132')
  im.save(ROOT/'views'/(unit+'_windows.jpg'),quality=94)
 for r in ds['records']:
  raw=ROOT/'views'/(r['id']+'_raw.png');shutil.copyfile(r['image'],raw)
  canvas,_=input_canvas(Image.open(r['image']),fill_extent=True);canvas.save(ROOT/'views'/(r['id']+'_network.png'))
  cases.append(dict(id=r['id'],source='views/'+raw.name,context='views/'+r['id']+'_context.png',
   reference='views/'+r['reference_unit']+'_windows.jpg',network='views/'+r['id']+'_network.png',
   split=r['split'],scope=r['evaluation_scope'],object_type=r['object_type'],
   visual_evidence=r['visual_evidence'],source_url=r['source_url'],source_unit=r['reference_unit'],
   source_group=r['source_group'],**{k:pred[r['id']][k] for k in ['targets','outputs','role_reference','role_predictions']}))
 sources=[dict(id=r['id'],url=r['url'],sha256=r.get('sha256'),pages=len(r.get('pages',[])),
  role='development only' if r['id']=='qingpu' else 'new source holdout',
  individual_image_author_independence_verified=False) for r in read(ROOT/'source_inventory.json')]
 write('source_summary.json',sources)
 write('report_data.json',dict(cases=cases,attribute_metrics=results['attribute_metrics'],role_metrics=results['role_metrics'],
  style_probe=results['style_probe'],style_summary=results['style_summary'],sources=sources))
 selected=['ntm_huang_region_0','ntm_mi_region_0','ntm_huang_region_5','ntm_huang_region_6','ntm_xia_region_0']
 labels={'outline_context':'结构边上下文','interior_mark_context':'内部墨迹上下文','diffuse_tone':'连续淡墨面','unmarked_surface':'未着墨面'}
 board=Image.new('RGB',(1500,590),'#f3f0e7');d=ImageDraw.Draw(board)
 d.text((16,12),'新来源直接输出：给对类别名称，不等于分清局部笔法',font=font(28),fill='#243f35')
 for i,ident in enumerate(selected):
  r=next(z for z in cases if z['id']==ident);im=Image.open(ROOT/r['source']);native=im.size;scale=min(270/im.width,180/im.height)
  im=im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))),Image.Resampling.NEAREST);x=i*300+12;board.paste(im,(x,65))
  d.text((x,259),'原生 '+str(native[0])+'×'+str(native[1])+'，仅展示缩放',font=font(15),fill='#805333')
  d.text((x,293),ident,font=font(15),fill='#243f35')
  d.text((x,327),'参考：'+labels.get(r['role_reference'],'待定'),font=font(18),fill='#805333')
  d.text((x,362),'局部：'+labels[r['role_predictions']['local_only']['predicted']],font=font(18),fill='#286e7d')
  d.text((x,397),'邻域：'+labels[r['role_predictions']['local_plus_context']['predicted']],font=font(18),fill='#286e7d')
 d.text((16,469),'角色统计：仅局部6/12；局部＋邻域5/12。淡墨面、未着墨面在本批所选参考中均未找对。',font=font(21),fill='#984d38')
 d.text((16,515),'参考是助手区域判断，非专家逐像素或真实分笔；模型候选未通过，未接入机械臂。',font=font(21),fill='#243f35')
 board.save(ROOT/'comparison.jpg',quality=94)
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>皴法跨资料训练 · 局部角色与形态检查</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f0e7;color:#243f35;font:17px/1.65 "Microsoft YaHei",sans-serif}main{max-width:1500px;margin:auto;padding:24px}h1{font-size:29px}.note{padding:16px;border-left:5px solid #b27235;background:#fff8e7}.warn{color:#994c37}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px}.card{min-width:0;background:#fffdf8;border:1px solid #d0d7c9;padding:10px}.scroll{overflow:auto}table{width:100%;border-collapse:collapse}th,td{padding:9px;border-bottom:1px solid #cbd4c4;text-align:left}select{padding:8px;font:inherit;max-width:100%}#case{width:760px}img{display:block;max-width:100%;height:auto}a{color:#266c69}small{color:#7d654d}@media(max-width:750px){main{padding:14px}.grid{grid-template-columns:1fr}h1{font-size:24px}}</style><main>
<small>2026-11-17 · 新资料开发／来源留出 · 离线研究候选</small><h1>增加资料来源，并把区域角色与形态分开检查</h1>
<p class="note"><b>本轮完成新来源训练与检查，但候选整体未通过。</b>新资料的长墨带参考8/10→9/10，宽角墨块负例4/4→2/4；区域角色仅局部6/12、加入邻域5/12，连续淡墨面和未着墨面均未找对。两个具名披麻／斧劈示例旧模型本来就猜对，新增候选没有额外收益；米点皴仍被二分类头判成斧劈。</p>
<h2>1. 这次增加的资料与分组</h2><p>青浦社区教材下载63页，核查画石步骤、皴法示例、淡墨坡面；新增30处开发区域。美术馆资料中第6页（印刷36页）的黄公望、米芾、夏圭三个作品示例，固定15处新来源检查。共新增45个区域视图，不是45件作品。原48处参考和模型全部保留，新旧共93处。</p>
<p>45处中35处有助手区域角色参考：23处训练、12处检查；其余角色待定。形态、角色、对象分开：树丛不给山石属性或皴笔目标，石内未着墨面仍属于石头，连续淡墨也不是空白背景。教材编写者有署名，具体每幅图例的作者独立性未逐一证实；资料来源分离不等于专家真值或作者独立验收。</p><div class="scroll"><table><thead><tr><th>资料</th><th>页数</th><th>本轮用途</th><th>原出处</th></tr></thead><tbody id="sources"></tbody></table></div>
<h2>2. 实际训练了什么</h2><p>属性基模型从37处扩至67处训练视图，继续使用冻结图像编码器、部分属性目标、C=1分类头与训练内分数校正。未知目标忽略，不从图名自动产生每条线的标签。角色探针单独训练两组：只看局部，以及同时看2.5倍邻域；二者都仅使用青浦的23处已检查角色参考，未使用美术馆测试标签。</p><p>皴法分类另保留三路对照：旧模型冻结；新增属性、仍用4个原具名示例；新增属性并加入2个青浦具名示例。米点皴不加入二分类训练，只用于暴露范围外误判。所有窗口仍是助手给定，尚非全图自主找笔画或对象。</p>
<h2>3. 同图看区域角色的真实失败</h2><img src="comparison.jpg" alt="新来源原图与局部、邻域角色预测对照"><div class="scroll"><table><thead><tr><th>角色模型</th><th>所选角色参考对应数</th><th>说明</th></tr></thead><tbody id="rolemetrics"></tbody></table></div><div class="scroll"><table><thead><tr><th>角色参考</th><th>参考数</th><th>只看局部对应</th><th>加邻域对应</th></tr></thead><tbody id="roles_by_class"></tbody></table></div><p class="warn">这里不是逐像素准确率，也没有恢复画家的真实分笔。部分空景、淡墨被判为内部墨迹；只扩大邻域没有解决跨来源问题。纸色、扫描纹理、区域参考不足等是待检验因素，本轮未分离其因果，不能把错误全部归咎于某一个因素。</p>
<h2>4. 逐张查看93处输入、参考和模型输出</h2><select id="case"></select><p id="scope"></p><div class="grid"><div class="card"><b>原始局部</b><img id="source"></div><div class="card"><b>真正进入角色模型的邻域</b><img id="context"></div><div class="card"><b>属性模型输入</b><img id="network"></div><div class="card"><b>原图窗口位置（参考，不入模型）</b><img id="reference"></div></div><p id="evidence"></p><p id="refnote" class="warn">无完整对象边界或逐笔真值；角色参考与模型输出分开。</p><p id="role_result"></p><p><a id="source_link" target="_blank">原资料出处</a></p><div class="scroll"><table><thead><tr><th>形态属性</th><th>助手参考</th><th>旧冻结模型</th><th>新增资料候选</th></tr></thead><tbody id="scores"></tbody></table></div>
<h2>5. 新资料与旧回归分别统计</h2><select id="metric_scope"><option value="new_NT_source_probe">新美术馆资料局部</option><option value="known_HKMA_regression">旧HKMA七区域回归</option><option value="known_work_new_shape_probe">旧夏圭四区域回归</option></select><div class="scroll"><table><thead><tr><th>形态</th><th>正／负参考</th><th>旧模型对应</th><th>新增候选对应</th><th>旧误报／漏报</th><th>新增误报／漏报</th></tr></thead><tbody id="metrics"></tbody></table></div><p>新资料中角块只有4个明确负例，没有可靠正例，因此不能用它验收斧劈的正向依据。旧HKMA角块5/6→4/6，旧夏圭圆点／墨带检查2/4→1/4，新增资料没有带来稳定兼顾。不同批次结果不直接当作进步百分比。</p>
<h2>6. 皴法名称是否识别，以及范围外怎么办</h2><div class="scroll"><table><thead><tr><th>具名示例</th><th>来源名称</th><th>旧冻结模型</th><th>新增属性＋原4个具名例</th><th>新增属性＋6个具名例</th></tr></thead><tbody id="styles"></tbody></table></div><p class="warn">黄公望示例披麻、夏圭示例斧劈三路均对应，旧模型已经2/2，不能算新增收益。米点示例三路均被强制猜成斧劈：当前没有学会米点类别，也没有通过未知拒识。名称猜对但局部角色、形态出错，不能据此声称理解了皴法或能够画出它。</p>
<h2>7. 全部新来源角色错例与新增参考</h2><div id="role_failures"></div><details><summary>查看45处新增参考（助手标记，非模型输出）</summary><img src="new_reference_board.jpg"></details><p>下一轮优先补跨资料、同场共存的墨带边界、纸纹／未着墨面、淡染与内部短墨迹，核查其角色依据，并明确范围外皴法的训练与拒识方案。当前整幅认可绘画版本保留，分区与对象能力仍未通过；树木画法训练保持关闭。本轮未连接机械臂。</p>
<p><a href="source_summary.json">来源与哈希摘要</a> · <a href="protocol.json">冻结协议</a> · <a href="reference_manifest.json">全部部分参考</a> · <a href="results.json">全部模型直接输出</a> · <a href="verification.json">验证记录</a> · <a href="contract_checks.json">监督约束测试</a></p></main>
<script>let D;const $=id=>document.getElementById(id),keys=['elongated_mark','angular_broad_mark','curved_parallel_group'],names=['细长墨迹','宽角墨块','弯曲笔组'],roles=['outline_context','interior_mark_context','diffuse_tone','unmarked_surface'],rnames=['结构边上下文','内部墨迹上下文','连续淡墨面','未着墨面'],rn=r=>r===null?'待定':rnames[roles.indexOf(r)],label=v=>v===null?'待定':v===1?'可见':'未见',style=v=>v===0?'披麻皴':'斧劈皴';function show(){let r=D.cases[$('case').selectedIndex];for(let id of ['source','context','network','reference'])$(id).src=r[id];$('scope').textContent=r.id+' / '+r.split+' / '+r.scope+' / 对象参考：'+r.object_type;$('evidence').textContent='助手依据：'+r.visual_evidence;$('source_link').href=r.source_url;$('role_result').textContent='角色参考：'+rn(r.role_reference)+'；仅局部：'+rn(r.role_predictions.local_only.predicted)+'；局部＋邻域：'+rn(r.role_predictions.local_plus_context.predicted);$('scores').innerHTML=keys.map((k,j)=>'<tr><td>'+names[j]+'</td><td>'+label(r.targets[j])+'</td>'+['frozen_previous','new_source_candidate'].map(a=>'<td>'+r.outputs[a].scores[j].toFixed(3)+' / '+label(r.outputs[a].predicted[j])+'</td>').join('')+'</tr>').join('');}function metrics(){let scope=$('metric_scope').value;$('metrics').innerHTML='';keys.forEach((k,j)=>{let a=D.attribute_metrics.frozen_previous[scope][k],b=D.attribute_metrics.new_source_candidate[scope][k],tr=document.createElement('tr');tr.innerHTML='<td>'+names[j]+'</td><td>'+a.positive+' / '+a.negative+'</td><td>'+a.agreement+'/'+a.known+'</td><td>'+b.agreement+'/'+b.known+'</td><td>'+a.fp+' / '+a.fn+'</td><td>'+b.fp+' / '+b.fn+'</td>';$('metrics').append(tr)});}fetch('report_data.json').then(r=>r.json()).then(d=>{D=d;d.cases.forEach(r=>{let o=document.createElement('option');o.textContent=r.id+' / '+r.split;$('case').append(o)});$('case').onchange=show;$('metric_scope').onchange=metrics;show();metrics();d.sources.forEach(r=>{let tr=document.createElement('tr');tr.innerHTML='<td>'+r.id+'</td><td>'+r.pages+'</td><td>'+r.role+'</td><td><a href="'+r.url+'">公开来源</a></td>';$('sources').append(tr)});for(let [a,title] of [['local_only','只看局部'],['local_plus_context','局部＋邻域']]){let r=d.role_metrics[a],tr=document.createElement('tr');tr.innerHTML='<td>'+title+'</td><td>'+r.agreement+'/'+r.regions+'</td><td>给定区域，助手上下文参考</td>';$('rolemetrics').append(tr)}roles.forEach((k,j)=>{let a=d.role_metrics.local_only.per_role[k],b=d.role_metrics.local_plus_context.per_role[k],tr=document.createElement('tr');tr.innerHTML='<td>'+rnames[j]+'</td><td>'+a.reference_count+'</td><td>'+a.agreement+'</td><td>'+b.agreement+'</td>';$('roles_by_class').append(tr)});d.style_probe.forEach(r=>{let tr=document.createElement('tr');tr.innerHTML='<td>'+r.id+'</td><td>'+(r.target===null?'米点皴（范围外）':style(r.target))+'</td>'+['frozen_previous','new_attributes_4_named','new_attributes_6_named'].map(a=>'<td>'+style(r.predictions[a])+'</td>').join('')+'</td>';$('styles').append(tr)});d.cases.filter(r=>r.scope==='new_NT_source_probe'&&r.role_reference!==null&&r.role_predictions.local_plus_context.predicted!==r.role_reference).forEach(r=>{let p=document.createElement('p');p.textContent=r.id+'：参考 '+rn(r.role_reference)+' → 邻域模型 '+rn(r.role_predictions.local_plus_context.predicted)+'；'+r.visual_evidence;$('role_failures').append(p)});window.reportReady=true;});</script></html>'''
 page=page.replace(".join('')+'</td>';", ".join('');")
 (ROOT/'index.html').write_text(page,encoding='utf8')
 source=(PROJECT/'scripts/serve_guohua_objects_v2_20261113.py').read_text('utf8').replace('guohua_objects_v2_20261113','cunfa_cross_source_20261117')
 a=source.index('ALLOWED={');b=source.index('\nALLOWED.update',a)
 allowed={'index.html','comparison.jpg','new_reference_board.jpg','report_data.json','source_summary.json','protocol.json','reference_manifest.json','results.json','verification.json','contract_checks.json','postexperiment_checks.json'}
 source=source[:a]+'ALLOWED='+repr(allowed)+source[b:];source=source.replace('str(Path(__file__).resolve().parents[1])',repr(str(PROJECT)))
 (ROOT/'serve_report.py').write_text(source,encoding='utf8');print('REPORT_READY',len(cases),flush=True)
if __name__=='__main__':main()
