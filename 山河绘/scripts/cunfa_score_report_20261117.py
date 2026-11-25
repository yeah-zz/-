# -*- coding: utf-8 -*-
import sys,json,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from PIL import Image,ImageDraw,ImageFont
from gongbi_training.cunfa_input_scale import input_canvas
ROOT=LAB/'cunfa_score_alignment_20261117';PROJECT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text('utf8'))
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 ds=read(ROOT/'reference_manifest.json');result=read(ROOT/'results.json');pred={r['id']:r for r in result['records']}
 font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n);cases=[]
 # Each whole reference is displayed with its predefined windows. The model
 # never receives these outline colours or the reference attribute labels.
 units={}
 for r in ds['records']:
  units.setdefault(r['reference_unit'],[]).append(r)
 for unit,rows in units.items():
  im=Image.open(rows[0]['reference_image']).convert('RGB');d=ImageDraw.Draw(im)
  for j,r in enumerate(rows):d.rectangle(r['box'],outline='#cb7031',width=2);d.text((r['box'][0]+2,r['box'][1]+2),str(j+1),font=font(14),fill='#cb7031')
  im.save(ROOT/'views'/(unit+'_windows.jpg'),quality=94)
 for r in ds['records']:
  dest=ROOT/'views'/(r['id']+'_raw.png');shutil.copyfile(r['image'],dest)
  canvas,_=input_canvas(Image.open(r['image']),fill_extent=True);canvas.save(ROOT/'views'/(r['id']+'_network.png'))
  cases.append(dict(id=r['id'],source='views/'+dest.name,network='views/'+r['id']+'_network.png',
   context='views/'+r['reference_unit']+'_windows.jpg',split=r['split'],scope=r['evaluation_scope'],
   object_type=r['object_type'],visual_evidence=r['visual_evidence'],source_url=r['source_url'],
   source_unit=r['reference_unit'],source_group=r['source_group'],targets=pred[r['id']]['targets'],outputs=pred[r['id']]['outputs']))
 a='legacy_groundtruth_to_soft';b='oof_expanded_sigmoid'
 testcases=[r for r in cases if r['split']=='test'];changes=[]
 for r in testcases:
  for j,k in enumerate(['elongated_mark','angular_broad_mark','curved_parallel_group']):
   if r['targets'][j] is None:continue
   old=r['outputs'][a]['predicted'][j];new=r['outputs'][b]['predicted'][j]
   if old!=new:changes.append(dict(id=r['id'],concept=k,old=old,new=new,target=r['targets'][j],change='recovered' if new==r['targets'][j] else 'regressed'))
 write('report_data.json',dict(cases=cases,shape_metrics=result['shape_metrics'],style_probe=result['style_probe'],style_summary=result['style_summary'],changes=changes))
 # Model-output figure uses source pixels; enlargement is explicitly labelled.
 selected=['hkma_hemp_1','hkma_axe_2','hkma_hemp_2','xia_native_0_shape_1','xia_native_3_shape_0']
 board=Image.new('RGB',(1500,610),'#f3f0e7');d=ImageDraw.Draw(board)
 d.text((16,12),'修正、退步与剩余错例：模型输出与助手参考分开',font=font(28),fill='#243f35')
 for i,ident in enumerate(selected):
  r=next(z for z in cases if z['id']==ident);x=i*300+12;im=Image.open(ROOT/r['source']);native=im.size
  scale=min(270/im.width,190/im.height);im=im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))),Image.Resampling.NEAREST);board.paste(im,(x,65))
  d.text((x,269),'原生 '+str(native[0])+'×'+str(native[1])+'，仅展示缩放',font=font(15),fill='#805333')
  d.text((x,300),ident,font=font(16),fill='#243f35')
  d.text((x,334),'参考：'+' / '.join('?' if v is None else str(v) for v in r['targets']),font=font(19),fill='#805333')
  for arm,y,title in [(a,369,'旧版'),(b,404,'本轮')]:
   d.text((x,y),title+'：'+' / '.join(str(v) for v in r['outputs'][arm]['predicted']),font=font(19),fill='#2a7180')
  d.text((x,448),['弧组漏检找回','角块新增误报','弧组仍误报','圆点仍判长线','长线误报减少'][i],font=font(19),fill='#974d38')
 d.text((16,505),'属性顺序：细长墨迹 / 宽角墨块 / 弯曲笔组；1=可见，0=未见，?=待定。',font=font(21),fill='#243f35')
 d.text((16,547),'所有示例均为已知来源的小样参考，不是新作品独立验收，也不是完整笔画或实机效果。',font=font(20),fill='#805333')
 board.save(ROOT/'comparison.jpg',quality=94)
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>皴法属性输入一致性 · 2026-11-17</title><style>
*{box-sizing:border-box}body{margin:0;color:#243f35;background:#f3f0e7;font:17px/1.65 "Microsoft YaHei",sans-serif}main{max-width:1500px;margin:auto;padding:24px}h1{font-size:29px}img{max-width:100%;height:auto;display:block}.note{padding:16px;border-left:5px solid #b47635;background:#fff9e9}.warn{color:#994c37}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.card{min-width:0;background:#fffdf8;padding:12px;border:1px solid #d2d7cb}.scroll{overflow:auto}table{width:100%;border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #cbd3c4;text-align:left}select{font:inherit;padding:8px;max-width:100%}#case{width:720px}a{color:#256d6a}small{color:#7d634b}.pill{display:inline-block;padding:4px 10px;border:1px solid #9cad9d;border-radius:20px;margin-right:8px}@media(max-width:720px){main{padding:14px}.grid{grid-template-columns:1fr}h1{font-size:24px}}</style><main>
<small>2026-11-17 · 属性反例与训练内分数对齐 · 离线实验</small><h1>修正属性输入差异，保留真实退步与剩余误报</h1>
<p class="note"><b>同一张HKMA照片的2个具名示例，属性路径由1/2变为2/2；旧弧组参考由5/7变为6/7。</b>只改下游输入、不新增资料也达到2/2，因此不能把这个变化全归功于新增反例。新增3处圆钝墨形中仍有2处被判成长线；宽角墨块的对应数未升，且出现了新的误报。未通过独立验收，未替换绘画版本。</p>
<p><span class="pill">9个训练图例，仍为同一位插图作者</span><span class="pill">37处训练视图</span><span class="pill">7处旧回归＋4处已知作品新参考</span></p>
<h2>1. 两项修改分别做对照</h2><p>旧路径让分类头用人工0/1属性均值训练，推理时却给它靠近0.5的模型分数。本轮先按完整训练图例留出，生成基属性模型未见过该图例的预测，用这些训练内预测拟合下游标准化与分类头；推理使用同类模型分数。不是用测试标签调整分类边界。</p>
<p>然后补入5个已有教材图例的19处形态参考，包含圆钝墨形、折向边线和弯曲笔组；它们仅监督形态，不被强行贴成披麻／斧劈。属性分数校正采用训练内留出预测拟合的sigmoid，参数和0.5判断阈值在查看新增预测前固定。校正器本身使用全部训练内预测和参考拟合，这些内部结果不当作独立准确率。</p>
<div class="scroll"><table><thead><tr><th>路径</th><th>属性训练</th><th>分类头输入</th><th>同2个具名示例对应数</th></tr></thead><tbody id="styles_summary"></tbody></table></div><p class="warn">2个示例来自同一张旧照片，结果上限非常有限。它们不是新增作者、不是新作品独立验收，也不能与此前13处作品级暂标测试直接比较。自动完整山石发现、轮廓／皴笔分笔和绘画技能没有新增训练。</p>
<h2>2. 修正与错例同图展示</h2><img src="comparison.jpg" alt="原始墨迹、助手参考、旧模型与本轮预测的对照"><p>本轮找回1处弧组漏检和1处宽角块漏检；同时把原先正确的石内弧线局部误报为宽角墨块。顶部折向墨迹仍误报为弯曲笔组，分数还从约0.506升至0.876。说明分数拉开并不等于可靠，不能拿它直接当绘画置信度。</p>
<h2>3. 逐区域查看48处训练／检查视图</h2><select id="case"></select><p id="scope"></p><div class="grid"><div class="card"><b>原始局部</b><img id="source"></div><div class="card"><b>原图中的助手窗口，非自动检测</b><img id="context"></div><div class="card"><b>真实网络输入</b><img id="network"></div></div><p id="evidence"></p><p id="refnote" class="warn">角色未知；无完整对象边界或逐笔真值。</p><p><a id="source_link" target="_blank">原资料出处</a></p><div class="scroll"><table><thead><tr><th>可观察属性</th><th>助手参考</th><th>旧模型分数／判断</th><th>扩充后原始分数／判断</th><th>训练内校正后分数／判断</th></tr></thead><tbody id="scores"></tbody></table></div>
<h2>4. 分开统计旧回归和新增反例</h2><select id="metric_scope"><option value="known_HKMA_regression">旧HKMA七区域回归</option><option value="known_work_new_shape_probe">已知夏圭作品的四处新增形态参考</option><option value="all_known_shape_probes">合并观察，仅作诊断</option></select><div class="scroll"><table><thead><tr><th>属性</th><th>正／负参考</th><th>旧版对应</th><th>扩充未校正</th><th>扩充并校正</th><th>旧误报／漏报</th><th>本轮误报／漏报</th></tr></thead><tbody id="metrics"></tbody></table></div><p>所有参考由助手检查，未经专家认证。新增区域是已知《雪堂客话图》的原生局部，不是新独立作品。形态负例仍是墨迹，不能改成空白或从画面删掉。未知属性不计分。</p>
<details><summary>分数误差（Brier）对照及解释</summary><div class="scroll"><table><thead><tr><th>属性</th><th>旧版</th><th>扩充未校正</th><th>训练内校正后</th></tr></thead><tbody id="brier"></tbody></table></div><p>Brier是模型分数与助手0/1参考的平方误差，越小越接近这些参考。它在这批所选区域上降低，不能证明在新作品上概率可靠；剩余弧组误报反而更高分，是必须保留的反例。校正采用平衡类权重，也不等同于真实作品中的类别概率。</p></details>
<h2>5. 具名示例的逐张分类</h2><div class="scroll"><table><thead><tr><th>示例</th><th>来源具名</th><th>旧属性路径</th><th>原数据输入对齐</th><th>扩充未校正</th><th>扩充并校正</th><th>旧直接图像分类</th></tr></thead><tbody id="styles"></tbody></table></div><p>下游只有4个原AAM具名示例的类别监督；5个新增教材图例不加入披麻／斧劈类别监督。测试对象和窗口都已给定，尚未实现整幅画自主辨识皴法。对象与皴法尚未通过，树木画法训练保持关闭；本轮未连接机械臂。</p>
<h2>6. 新增形态参考与全部检查错例</h2><details><summary>查看23处新增参考：19训练、4检查</summary><img src="new_references.jpg"></details><div id="failures"></div><p>下一步重点是补不同独立来源中共存的圆点／短笔／长线、轮廓／皴笔与墨带边界，并检查纸纹与染墨干扰；当前参考只能支持区域形态，不能当完整分笔答案。已给定窗口与角色未知的限制需要后续专门解决。</p>
<p><a href="protocol.json">冻结对照协议</a> · <a href="reference_manifest.json">全部参考与未知目标</a> · <a href="results.json">各路径全部原始预测</a> · <a href="verification.json">验证记录</a> · <a href="contract_checks.json">训练内留出与输入约束检查</a></p></main>
<script>let D;const $=id=>document.getElementById(id),keys=['elongated_mark','angular_broad_mark','curved_parallel_group'],names=['细长墨迹','宽角墨块','弯曲笔组'],arms=['legacy_groundtruth_to_soft','oof_original_raw','oof_expanded_raw','oof_expanded_sigmoid'],titles=['旧人工属性 → 模型分数','原18视图＋训练内预测输入','扩至37视图＋同类分数输入','扩至37视图＋训练内sigmoid'],label=v=>v===null?'待定':v===1?'可见':'未见',style=v=>v===0?'披麻皴':'斧劈皴';function show(){let r=D.cases[$('case').selectedIndex];for(let id of ['source','context','network'])$(id).src=r[id];$('scope').textContent=r.id+' / '+r.split+' / '+r.scope+' / '+(r.object_type==='teaching_mark'?'教学笔形':'山石区域')+' / 同源：'+r.source_unit;$('evidence').textContent='助手观察：'+r.visual_evidence;$('source_link').href=r.source_url;$('scores').innerHTML=keys.map((k,j)=>'<tr><td>'+names[j]+'</td><td>'+label(r.targets[j])+'</td>'+[arms[0],arms[2],arms[3]].map(a=>'<td>'+r.outputs[a].scores[j].toFixed(3)+' / '+label(r.outputs[a].predicted[j])+'</td>').join('')+'</tr>').join('');}function metrics(){let scope=$('metric_scope').value;$('metrics').innerHTML='';$('brier').innerHTML='';keys.forEach((k,j)=>{let a=D.shape_metrics[arms[0]][scope][k],b=D.shape_metrics[arms[2]][scope][k],c=D.shape_metrics[arms[3]][scope][k],tr=document.createElement('tr');tr.innerHTML='<td>'+names[j]+'</td><td>'+a.positive+' / '+a.negative+'</td><td>'+a.agreement+'/'+a.known+'</td><td>'+b.agreement+'/'+b.known+'</td><td>'+c.agreement+'/'+c.known+'</td><td>'+a.fp+' / '+a.fn+'</td><td>'+c.fp+' / '+c.fn+'</td>';$('metrics').append(tr);let e=document.createElement('tr');e.innerHTML='<td>'+names[j]+'</td><td>'+a.brier.toFixed(3)+'</td><td>'+b.brier.toFixed(3)+'</td><td>'+c.brier.toFixed(3)+'</td>';$('brier').append(e)});}fetch('report_data.json').then(r=>r.json()).then(d=>{D=d;d.cases.forEach(r=>{let o=document.createElement('option');o.textContent=r.id+' / '+r.split;$('case').append(o)});$('case').onchange=show;$('metric_scope').onchange=metrics;show();metrics();arms.forEach((a,i)=>{let tr=document.createElement('tr');tr.innerHTML='<td>'+titles[i]+'</td><td>'+(i<2?'原18视图':'37视图')+'</td><td>'+(i===0?'人工参考属性均值':'训练内模型预测分数均值')+'</td><td>'+d.style_summary[a]+'/2</td>';$('styles_summary').append(tr)});d.style_probe.forEach(r=>{let tr=document.createElement('tr');tr.innerHTML='<td>'+r.id+'</td><td>'+r.source_label+'</td>'+arms.map(a=>'<td>'+style(r.predictions[a])+'</td>').join('')+'<td>'+style(r.direct_previous)+'</td>';$('styles').append(tr)});d.cases.filter(r=>r.split==='test'&&r.outputs[arms[3]].errors.length).forEach(r=>{let p=document.createElement('p');p.textContent=r.id+'：'+r.outputs[arms[3]].errors.map(k=>names[keys.indexOf(k)]).join('、')+'；'+r.visual_evidence;$('failures').append(p)});window.reportReady=true;});</script></html>'''
 (ROOT/'index.html').write_text(page,encoding='utf8')
 server=(PROJECT/'scripts/serve_guohua_objects_v2_20261113.py').read_text('utf8').replace('guohua_objects_v2_20261113','cunfa_score_alignment_20261117')
 a=server.index('ALLOWED={');b=server.index('\nALLOWED.update',a)
 allowed={'index.html','comparison.jpg','new_references.jpg','report_data.json','protocol.json','reference_manifest.json','results.json','verification.json','contract_checks.json','postexperiment_checks.json'}
 server=server[:a]+'ALLOWED='+repr(allowed)+server[b:];server=server.replace('str(Path(__file__).resolve().parents[1])',repr(str(PROJECT)))
 (ROOT/'serve_report.py').write_text(server,encoding='utf8');print('REPORT',len(cases),'CHANGES',json.dumps(changes),flush=True)

if __name__=='__main__':main()
