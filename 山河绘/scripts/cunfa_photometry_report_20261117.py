# -*- coding: utf-8 -*-
import sys,json,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'cunfa_photometry_20261117';OLD=LAB/'cunfa_cross_source_20261117';PROJECT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text('utf8'))
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 ds=read(OLD/'reference_manifest.json');result=read(ROOT/'results.json');reject=read(ROOT/'rejection_scope_results.json')
 pred={r['id']:r for r in result['records']};audit={r['id']:r for r in read(ROOT/'contrast_audit.json')['records']};cases=[]
 for r in ds['records']:
  dest=ROOT/'views'/(r['id']+'_source.png');shutil.copyfile(r['image'],dest)
  cases.append(dict(id=r['id'],source='views/'+dest.name,legacy='views/'+r['id']+'_legacy.png',
   raw_gray='views/'+r['id']+'_raw_gray.png',linear='views/'+r['id']+'_reference_linear.png',
   split=r['split'],scope=r['evaluation_scope'],source_url=r['source_url'],object_type=r['object_type'],
   visual_evidence=r['visual_evidence'],role_reference=pred[r['id']]['role_reference'],predictions=pred[r['id']]['predictions'],
   selected_prediction=pred[r['id']]['selected_prediction'],audit=audit[r['id']]))
 write('report_data.json',dict(cases=cases,selected=result['selected_arm'],metrics=result['known_role_regression_metrics'],
  oof=result['training_oof_metrics'],rejection=reject,legacy_rejection_metrics_not_admitted=result['rejection_metrics']))
 font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
 board=Image.new('RGB',(1500,580),'#f3f0e7');d=ImageDraw.Draw(board)
 d.text((16,12),'同一未着墨区域：保留灰度变化，检查纸纹是否被过度增强',font=font(28),fill='#243f35')
 r=next(x for x in cases if x['id']=='ntm_huang_region_5')
 for i,(key,title) in enumerate([('source','原始像素'),('legacy','旧：局部直方图均衡'),('raw_gray','原始灰度输入'),('linear','整图统一线性基准')]):
  im=Image.open(ROOT/r[key]);scale=min(340/im.width,275/im.height);im=im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))),Image.Resampling.NEAREST)
  x=i*375+12;board.paste(im,(x,75));d.text((x,385),title,font=font(21),fill='#243f35')
 for x,title,value in [(12,'原生灰度5–95%范围',r['audit']['raw_q95_q05']),(390,'均衡后的范围',r['audit']['equalized_q95_q05']),(770,'线性基准后的范围',r['audit']['linear_q95_q05'])]:
  d.text((x,441),title+'：'+format(value,'.1f'),font=font(22),fill='#805333')
 d.text((16,502),'93处已知区域回归；图像仅为网络输入对照，不是墨迹掩膜或新生成的国画。',font=font(21),fill='#994d38')
 d.text((16,542),'选中方案找回2处未着墨面，但结构边退步、淡墨面未解决，本轮不替换绘画版本。',font=font(20),fill='#243f35')
 board.save(ROOT/'comparison.jpg',quality=94)
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>墨迹／纸纹输入修正与范围外拒识探针</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f0e7;color:#243f35;font:17px/1.65 "Microsoft YaHei",sans-serif}main{max-width:1500px;margin:auto;padding:24px}h1{font-size:29px}.note{padding:16px;border-left:5px solid #b77b3d;background:#fff8e7}.warn{color:#994b37}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}.card{min-width:0;padding:12px;background:#fffdf8;border:1px solid #d0d7c8}img{max-width:100%;height:auto;display:block}.scroll{overflow:auto}table{width:100%;border-collapse:collapse}th,td{padding:9px;border-bottom:1px solid #cbd4c4;text-align:left}select{padding:8px;font:inherit;max-width:100%}#case{width:760px}a{color:#286e69}small{color:#7e644b}@media(max-width:740px){main{padding:14px}.grid{grid-template-columns:1fr}h1{font-size:24px}}</style><main>
<small>2026-11-17 · 已知案例开发回归 · 光度输入与有限拒识探针</small><h1>减少纸纹对比拉伸，保留结构边退步和拒识失败</h1>
<p class="note"><b>训练内选出的方案在同一12处角色参考上由5/12到7/12，未着墨面由0/3到2/3。</b>同时结构边由1/2退至0/2，连续淡墨面仍0/2，尚未兼顾主轮廓和染墨。范围外探针保留4个已知例中的3个，却误拒1个披麻例，也没有拒掉唯一米点例。总体未通过，不替换绘画版本。</p>
<h2>1. 找到了什么输入问题</h2><p>旧流程对每个局部单独做直方图均衡。三个未着墨面的原始5–95%灰度范围为23、18、35，均衡后变为230、229、228，纸纹的对比被明显拉开。整张参考图统一线性亮度基准后，同样范围约27、21、34。这个现象可复现，但不等于已证明所有误判都由均衡引起。</p><img src="comparison.jpg" alt="同一未着墨面在原图、旧均衡、原灰度与统一线性基准下的真实输入">
<p>新方案不在每个局部上重新拉满灰度。线性基准只用整张参考图的90%灰度分位数，把参考亮度映射到240；没有读取纸面掩膜、角色或测试标签，也没有生成新墨迹。它只是图像亮度基准，不是物理墨密度或毛笔参数标定。</p>
<h2>2. 先按训练内留出选择，再检查旧案例</h2><p>角色原始训练仍为青浦23处参考，检查仍为此前已打开的美术馆12处，没有新增独立作品。5个候选固定参数比较：原灰度、整图线性基准、线性基准加多尺度对比统计、仅对比统计、以及训练时加入一份轻微纸色／噪声变化。派生训练视图不增加作者、作品或原始参考数量。</p><p>按完整训练图例留出计算平均类别召回，选中“线性基准＋对比统计”，选择记录先于检查集统计保存。表中保留所有候选，不能事后改选检查集上最有利的一项。内部留出仍来自同一份教材，不是作者独立验收。</p><div class="scroll"><table><thead><tr><th>方案</th><th>训练内平均类别召回</th><th>已知角色参考对应</th><th>检查平均类别召回</th><th>状态</th></tr></thead><tbody id="metrics"></tbody></table></div>
<h2>3. 找回空白不等于保住轮廓</h2><div class="scroll"><table><thead><tr><th>角色参考</th><th>参考数</th><th>旧邻域模型</th><th>训练内选中方案</th><th>仅统计诊断</th></tr></thead><tbody id="roles"></tbody></table></div><p class="warn">仅对比统计能找对两处淡墨面，却损失多处内部墨迹；选中方案保留内部墨迹，但没有恢复结构边或淡墨。光度统计有帮助，不能替代轮廓／皴笔的结构判断。这些仍是助手区域参考的一致情况，不是专家逐像素或真实分笔准确率。</p>
<h2>4. 逐区域比较93处原图与网络输入</h2><select id="case"></select><p id="scope"></p><div class="grid"><div class="card"><b>原始像素</b><img id="source"></div><div class="card"><b>旧均衡输入</b><img id="legacy"></div><div class="card"><b>原灰度输入</b><img id="raw_gray"></div><div class="card"><b>整图线性基准输入</b><img id="linear"></div></div><p id="evidence"></p><p id="refnote" class="warn">无完整对象边界或逐笔真值；模型没有在本轮学习完整曲线、笔顺或物理动作。</p><p id="role_result"></p><p id="contrast"></p><p><a id="source_link" target="_blank">原资料出处</a></p>
<h2>5. 范围外拒识：先修正标签口径，再公开失败</h2><p>初版把7个不同名称示例都当作范围外，其中解索、拖泥带水、钉头与当前两类的家族边界未核实。仅因名称不同就判为负例不充分，因此在保守口径拟合前，将这3个单列待定，不纳入拒识损失；初版口径和结果归档，主表使用保守口径。</p><p>拒识探针只用6个明确披麻／斧劈具名示例，以及4个米点／云头具名示例训练。固定C=1、阈值0.5。它学习的是这套示例口径下的两类成员判断，不是识别所有皴法，也不是通用未知／混合识别。测试为3个已知作品示例加2个旧HKMA示例，不是新独立测试。</p><div class="scroll"><table><thead><tr><th>已知检查示例</th><th>参考口径</th><th>成员分数</th><th>探针动作</th><th>是否对应</th></tr></thead><tbody id="rejection"></tbody></table></div><p class="warn">黄公望披麻示例被误拒，米点示例分数约0.519而被接受。未用这5例把阈值改成0.55等更有利的数值；当前拒识未通过，不接入绘画决策。拒掉某个例子也不等于识别出它是哪一种皴法。</p>
<h2>6. 仍需处理的错误</h2><div id="failures"></div><p>下一步需补可核查的结构边／内部短笔／淡墨面和未着墨面的成对参考，增加真实跨来源纸色、扫描与光照变化；同时先明确皴法类别边界，再补拒识资料。当前主轮廓有回退，所有新头保持研究用途。原有属性与对象模型未替换，树木画法训练保持关闭，本轮未连接机械臂。</p>
<p><a href="protocol.json">冻结光度实验协议</a> · <a href="training_selection.json">训练内选择记录</a> · <a href="contrast_audit.json">逐区域对比范围</a> · <a href="results.json">角色原始结果与停用的初版拒识</a> · <a href="rejection_scope_protocol.json">保守拒识口径</a> · <a href="rejection_scope_results.json">主拒识结果</a> · <a href="verification.json">验证记录</a> · <a href="contract_checks.json">输入不变与弱对比保留测试</a></p></main>
<script>let D;const $=id=>document.getElementById(id),roles=['outline_context','interior_mark_context','diffuse_tone','unmarked_surface'],rnames=['结构边上下文','内部墨迹上下文','连续淡墨面','未着墨面'],rn=r=>r===null?'待定':rnames[roles.indexOf(r)],arms=['legacy_hist_context','raw_gray_context','reference_linear_context','linear_plus_contrast','contrast_only','linear_augmented'],titles=['旧局部均衡邻域模型','原灰度＋邻域','整图线性基准＋邻域','线性基准＋对比统计','仅对比统计诊断','线性＋统计＋训练扰动'];function show(){let r=D.cases[$('case').selectedIndex];for(let id of ['source','legacy','raw_gray','linear'])$(id).src=r[id];$('scope').textContent=r.id+' / '+r.split+' / '+r.scope+' / 对象参考：'+r.object_type;$('source_link').href=r.source_url;$('evidence').textContent='助手依据：'+r.visual_evidence;$('role_result').textContent='角色参考：'+rn(r.role_reference)+'；旧邻域模型：'+rn(r.predictions.legacy_hist_context)+'；选中方案：'+rn(r.selected_prediction);let a=r.audit;$('contrast').textContent='5–95%灰度范围：原生 '+a.raw_q95_q05.toFixed(1)+'；局部均衡 '+a.equalized_q95_q05.toFixed(1)+'；整图线性基准 '+a.linear_q95_q05.toFixed(1);}fetch('report_data.json').then(r=>r.json()).then(d=>{D=d;d.cases.forEach(r=>{let o=document.createElement('option');o.textContent=r.id+' / '+r.split;$('case').append(o)});$('case').onchange=show;show();arms.forEach((a,i)=>{let m=d.metrics[a],o=d.oof[a],tr=document.createElement('tr');tr.innerHTML='<td>'+titles[i]+'</td><td>'+(o?(o.macro_recall*100).toFixed(1)+'%':'冻结对照')+'</td><td>'+m.agreement+'/'+m.regions+'</td><td>'+(m.macro_recall*100).toFixed(1)+'%</td><td>'+(a===d.selected?'训练内选中':'保留对照')+'</td>';$('metrics').append(tr)});roles.forEach((k,j)=>{let a=d.metrics.legacy_hist_context.per_role[k],b=d.metrics[d.selected].per_role[k],c=d.metrics.contrast_only.per_role[k],tr=document.createElement('tr');tr.innerHTML='<td>'+rnames[j]+'</td><td>'+a.count+'</td><td>'+a.agreement+'</td><td>'+b.agreement+'</td><td>'+c.agreement+'</td>';$('roles').append(tr)});d.rejection.records.filter(r=>r.split==='test').forEach(r=>{let tr=document.createElement('tr');tr.innerHTML='<td>'+r.id+'</td><td>'+(r.reference_known?'属于当前两类':'范围外')+'</td><td>'+r.known_score.toFixed(3)+'</td><td>'+(r.accept_known?'保留两类候选':'范围外／不确定')+'</td><td>'+(Number(r.accept_known)===r.reference_known?'对应':'失败')+'</td>';$('rejection').append(tr)});d.cases.filter(r=>r.role_reference!==null&&r.split==='test'&&r.selected_prediction!==r.role_reference).forEach(r=>{let p=document.createElement('p');p.textContent=r.id+'：参考 '+rn(r.role_reference)+' → 选中方案 '+rn(r.selected_prediction)+'；'+r.visual_evidence;$('failures').append(p)});window.reportReady=true;});</script></html>'''
 (ROOT/'index.html').write_text(page,encoding='utf8')
 server=(PROJECT/'scripts/serve_guohua_objects_v2_20261113.py').read_text('utf8').replace('guohua_objects_v2_20261113','cunfa_photometry_20261117')
 a=server.index('ALLOWED={');b=server.index('\nALLOWED.update',a)
 allowed={'index.html','comparison.jpg','report_data.json','protocol.json','training_selection.json','contrast_audit.json','results.json','rejection_scope_protocol.json','rejection_scope_results.json','verification.json','contract_checks.json','postexperiment_checks.json'}
 server=server[:a]+'ALLOWED='+repr(allowed)+server[b:];server=server.replace('str(Path(__file__).resolve().parents[1])',repr(str(PROJECT)))
 (ROOT/'serve_report.py').write_text(server,encoding='utf8');print('REPORT_READY',len(cases),flush=True)
if __name__=='__main__':main()
