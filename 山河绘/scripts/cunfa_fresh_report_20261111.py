# -*- coding: utf-8 -*-
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'cunfa_fresh_validation_20261111';OLD=LAB/'cunfa_context_reference_20261111'
def read(n):return json.loads((ROOT/n).read_text('utf8'))
def save(n,d):(ROOT/n).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 s=read('summary.json');d=read('test_manifest.json');pr=read('predictions.json');freeze=read('freeze_manifest.json');checks=read('verification.json');byid={r['id']:r for r in d['records']};views={r['id']:r for r in pr['views']};font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
 methods={'local_plus_context':'冻结模型：局部＋上下文','local_only':'冻结模型：仅局部','context_only':'冻结模型：仅上下文'}
 # Feature-space neighbours are post-hoc illustrations, not causal explanations
 # of the SVM decision, and do not alter labels or the checkpoint.
 oldrows=json.loads((OLD/'input_manifest.json').read_text('utf8'))['rows'];train=np.load(OLD/'features.npz')['local_plus_context'];test=np.load(ROOT/'features.npz')['local_plus_context'];neighbours=[]
 (ROOT/'neighbours').mkdir(exist_ok=True)
 for i,r in enumerate(pr['records']):
  similarity=train@test[i];order=np.argsort(similarity)[::-1][:3];top=[]
  for j in order:
   old=oldrows[j];dest=ROOT/'neighbours'/(old['id']+'.png')
   if not dest.exists():Image.open(old['image']).save(dest)
   top.append(dict(id=old['id'],label=old['label'],artist=old['artist_group'],similarity=float(similarity[j]),image='neighbours/'+old['id']+'.png',source_url=old['source_url']))
  neighbours.append(dict(id=r['id'],nearest=top,not_a_causal_explanation=True))
 save('posthoc_neighbours.json',neighbours)
 payload=[];(ROOT/'contexts').mkdir(exist_ok=True)
 for r in pr['records']:
  source=byid[r['id']];image=Image.open(source['source_image']).convert('RGB');dd=ImageDraw.Draw(image);dd.rectangle(source['source_box'],outline='#078099',width=3);image.thumbnail((900,700));image.save(ROOT/'contexts'/(r['id']+'.jpg'),quality=92)
  item=dict(**r,local='inputs/'+r['id']+'_local.png',context='inputs/'+r['id']+'_context.png',position='contexts/'+r['id']+'.jpg',source_url=source['source_url'],label_scope=source['label_scope'],source_box=source['source_box'],context_box=views[r['id']]['context_box'],attribution=source.get('attribution'),nearest=next(n['nearest'] for n in neighbours if n['id']==r['id']));payload.append(item)
 exclusions=[dict(id=r['id'],image='crops/'+Path(r['image']).name,reason=r['exclusion_reason']) for r in d['records'] if r['admission']!='source_label_test']
 majority='披麻皴';majority_accuracy=sum(r['label']==majority for r in pr['records'])/len(pr['records'])
 save('decision.json',dict(candidate='frozen local+context',new_author_source_label_gate='failed',formal_expert_gate='not_available',adopted=False,permitted_use='research comparison and annotation assistance only',stroke_generation_enabled=False,robot_execution_enabled=False,
  reasons=['only7/13 source labels matched','axe_cut_recall1/5','Xia_Gui0/4','context underperformed frozen local-only and training-majority control'],no_test_label_used_for_training=True))
 queue=[dict(id=r['id'],image=r['image'],source_image=r['source_image'],source_url=r['source_url'],source_box=r['source_box'],candidate_label=r['label'],admission=r['admission'],independent_reviewed_label=None,reviewer=None,expert_certified=False,local_cun_group_members=None,uncertainty_note=None) for r in d['records']];save('independent_review_queue.json',queue)
 out=dict(summary=s,methods=methods,records=payload,results=pr['results'],excluded_regions=exclusions,excluded_sources=d['exclusions'],verification=checks,freeze=freeze,decision=read('decision.json'),training_majority_control=dict(label=majority,accuracy=majority_accuracy,posthoc_control=True,class_chosen_from_old_training32_vs31=True));save('report_data.json',out)
 im=Image.new('RGB',(1440,930),'#f4f1e8');dr=ImageDraw.Draw(im);dr.text((32,22),'冻结模型的新作者验证：本轮未通过',font=font(34),fill='#243d32');dr.text((32,84),'新13个候选区域 · 4位馆藏署名/归属作者 · 4件作品 · 2个发布机构',font=font(23),fill='#435b4c');dr.text((32,126),'先冻结模型和参数，再收集新图与确定裁剪；13例全部报告，3例在预测前排除。',font=font(21),fill='#435b4c')
 for i,(label,v) in enumerate([('冻结局部＋上下文',s['metrics']['local_plus_context']['label_agreement']),('冻结仅局部对照',s['metrics']['local_only']['label_agreement']),('始终猜训练集多数类（披麻）',majority_accuracy)]):
  y=220+i*105;dr.text((32,y),label,font=font(24),fill='#243d32');dr.rectangle((580,y+2,580+v*650,y+36),fill='#b36b4d' if i==0 else '#7b8f79');dr.text((1190,y-1),f'{v:.1%}',font=font(27),fill='#243d32')
 dr.text((32,565),'上下文模型：披麻6/8，斧劈1/5；夏圭4例全部判为披麻。',font=font(25),fill='#9a3c29');dr.text((32,616),'王时敏3/4　文徵明3/4　马远1/1　夏圭0/4',font=font(24),fill='#243d32');dr.text((32,687),'结论：上一轮85.7%的探索表现，未在这批新作者作品上复现。',font=font(25),fill='#9a3c29');dr.text((32,738),'不能据此说已懂皴法；不进入皴笔生成，不替换当前绘画版本。',font=font(24),fill='#9a3c29');dr.text((32,798),'馆藏作品标签＋助手局部筛选，仍非专家真值；13个裁剪仅来自4件作品。',font=font(21),fill='#435b4c');dr.text((32,845),'新图未用于训练或调参；机械臂未连接。待复核区域及错例均已保留。',font=font(21),fill='#435b4c');im.save(ROOT/'comparison.png')
 board=Image.new('RGB',(1400,4*270+60),'#f4f1e8');b=ImageDraw.Draw(board);b.text((15,10),'模型未见作者：全部13例的来源标签 → 冻结模型预测',font=font(24),fill='#243d32')
 for i,r in enumerate(payload):
  x=i%4*350;y=i//4*270+60;image=Image.open(ROOT/r['local']);image.thumbnail((325,160));board.paste(image,(x+10,y));p=r['predictions']['local_plus_context']['predicted'];b.text((x+10,y+169),r['id'],font=font(17),fill='black');b.text((x+10,y+195),r['label']+' → '+p,font=font(20),fill='#26654a' if r['label']==p else '#a23e2a');b.text((x+10,y+227),r['artist'],font=font(18),fill='#435b4c')
 board.save(ROOT/'all_predictions.jpg',quality=93)
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>皴法冻结验证 · 新作者与全部错例</title><style>body{margin:0;background:#f4f1e8;color:#243d32;font:16px/1.7 system-ui}main{max-width:1260px;margin:auto;padding:26px}h1{font-size:29px}h2{font-size:23px;margin-top:34px}.notice{background:#f1ded0;padding:18px;border-left:5px solid #ac6040}.overview{width:100%;height:auto}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(285px,1fr));gap:16px}.card{background:#fffdf7;padding:15px;border:1px solid #d7d6c7;border-radius:7px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:8px}.pair img{width:100%;height:170px;object-fit:contain;background:white}.image{width:100%;height:auto}.good{color:#25684b}.bad{color:#a03e2b}small{display:block;color:#5b6b5b;overflow-wrap:anywhere}select{max-width:100%;padding:10px;margin:6px 8px 6px 0;background:#fffdf7;border:1px solid #b5b8a4}a{color:#235c41}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #c9cdbc}.tablewrap{overflow-x:auto}details{margin-top:12px}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere}.thumb{width:90px;height:90px;object-fit:contain;vertical-align:middle}</style><main><h1>新作者验证：冻结模型没有通过</h1><div class="notice"><b>结果：局部＋上下文仅7/13符合来源标签（53.8%），仅局部为8/13（61.5%）。</b><br>上下文模型对斧劈的来源标签召回为1/5，夏圭4例全部误判。上一轮旧资料上的85.7%没有在本批新作者作品上复现。该候选不能进入皴笔生成或替换当前绘画版本。</div><p>模型在收集新图前已冻结，保留分类头、编码器、预处理代码的SHA-256。测试来自王时敏、文徵明、马远、夏圭4位未进入该分类模型训练/调参的作者，共4件作品、2个发布机构。先核对来源及裁剪，再做预测；主候选沿用既有分类头，对照头仅在冻结前使用旧63例拟合；新图未参与训练或调参。</p><p><b>标签范围：</b>仍是馆藏作品技法说明/目录标签＋助手局部筛选，专家逐区域认证为0。13个区域来自4件作品，并非13件独立作品，不能把这次一致率称为专家准确率或全面验收。</p><img class="overview" src="comparison.png" alt="冻结模型新作者结果"><h2>同一13例的冻结对照</h2><div class="tablewrap"><table><thead><tr><th>方法</th><th>来源一致率</th><th>macro-F1</th><th>披麻召回</th><th>斧劈召回</th></tr></thead><tbody id="metrics"></tbody></table></div><p>训练集多数类是披麻（旧63例中32比31）；始终猜披麻在新13例上为8/13，此项是事后补充的简单对照。上下文相较仅局部有1例改对、2例改错。所有数据均保留，不因预测结果重新裁剪、换标签或删除错例。</p><h2>按作品/作者看结果</h2><div class="tablewrap"><table><thead><tr><th>馆藏署名/归属作者</th><th>符合来源标签 / 该作者区域数</th></tr></thead><tbody id="artists"></tbody></table></div><p>马远只保留了1例，1/1不能说明该作者已经稳定识别。4位作者各只有一件作品，本轮也没有证明对每位作者的新作品有效。</p><h2>模型直接输出：全部13例分类</h2><select id="method"></select><select id="artist"><option value="">全部作者</option></select><select id="status"><option value="">全部结果</option><option value="wrong">与来源标签不符</option><option value="correct">与来源标签一致</option></select><span id="count"></span><div id="grid" class="grid"></div><h2>预测前排除的3个局部</h2><p>梅枝占主导、印章污染、内部皴纹不足的区域未参与预测。它们是在模型输出前被排除，原图和排除原因保留如下。</p><div id="excluded" class="grid"></div><h2>来源核查</h2><p><a href="https://www.dpm.org.cn/collection/paint/231531.html" target="_blank" rel="noopener">王时敏山水图</a>有披麻皴说明；<a href="https://www.dpm.org.cn/collection/paint/234350.html" target="_blank" rel="noopener">马远梅石溪凫</a>有斧劈说明；<a href="https://www.dpm.org.cn/collection/paint/230757.html" target="_blank" rel="noopener">夏圭雪堂客话</a>有小斧劈及短线秃笔说明；<a href="https://digitalarchive.npm.gov.tw/Collection/Detail/5621?dep=P" target="_blank" rel="noopener">文徵明山水卷</a>的技法目录列披麻皴。后者使用页面直接展示的照片预览，去掉色卡与装裱边缘后再选区域，没有操作下载验证码表单。</p><p>明文徵明山水 卷。國立故宮博物院，台北，CC BY 4.0 @ www.npm.gov.tw。</p><details><summary>其他候选来源为什么没有硬贴标签</summary><pre id="sourceExclusions"></pre></details><h2>本轮能得出的结论</h2><p>冻结上下文模型目前缺乏稳定的新作品辨识能力。夏圭这件作品上的全部失败需要保留，不能先改标签来解释掉。来源差异、少量作品与局部标签、细线型小斧劈的覆盖不足等都需进一步检验，本次实验不能单独确定失败原因。</p><p>已为16个候选建立独立复核清单，确认标签仍为空；自动预测没有变成训练真值。这13例已经揭示结果，后续如用于开发，必须另找新的测试作品。完整笔组、真实笔顺、墨带生成和实机参数仍未通过。</p><details><summary>冻结与完整性检查</summary><pre id="checks"></pre></details><p><a href="all_predictions.jpg" target="_blank">全部13例预测长图</a> · <a href="report_data.json">完整对照数据</a> · <a href="independent_review_queue.json">未填标签复核清单</a></p></main><script>
const el=id=>document.getElementById(id),esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let DATA;const pct=v=>(v*100).toFixed(1)+'%';
fetch('report_data.json').then(r=>r.json()).then(d=>{DATA=d;el('method').innerHTML=Object.entries(d.methods).map(([k,v])=>`<option value="${k}">${v}</option>`).join('');el('artist').innerHTML+=[...new Set(d.records.map(r=>r.artist))].map(a=>`<option>${a}</option>`).join('');el('metrics').innerHTML=Object.entries(d.summary.metrics).map(([k,m])=>`<tr><td>${esc(d.methods[k])}</td><td>${pct(m.label_agreement)}</td><td>${m.macro_f1.toFixed(3)}</td><td>${pct(m.class_recall[0])}</td><td>${pct(m.class_recall[1])}</td></tr>`).join('');el('artists').innerHTML=Object.entries(d.results.local_plus_context.per_artist).map(([a,m])=>`<tr><td>${a}</td><td>${Math.round(m.label_agreement*m.samples)} / ${m.samples}</td></tr>`).join('');el('excluded').innerHTML=d.excluded_regions.map(r=>`<article class="card"><img class="image" src="${r.image}" alt="预测前排除的局部"><small>${r.id}</small><p>${esc(r.reason)}</p></article>`).join('');el('sourceExclusions').textContent=JSON.stringify(d.excluded_sources,null,2);el('checks').textContent=JSON.stringify({freeze:d.freeze,verification:d.verification,decision:d.decision},null,2);for(const id of ['method','artist','status'])el(id).onchange=draw;draw();window.reportReady=true;});
function draw(){const k=el('method').value,a=el('artist').value,s=el('status').value;const rows=DATA.records.filter(r=>(!a||a===r.artist)&&(!s||s==='wrong'&&r.label!==r.predictions[k].predicted||s==='correct'&&r.label===r.predictions[k].predicted));el('count').textContent=`显示 ${rows.length}/13`;el('grid').innerHTML=rows.map(r=>`<article class="card"><div class="pair"><div><small>局部输入</small><img src="${r.local}" alt="${r.id}局部"></div><div><small>固定2.5倍上下文</small><img src="${r.context}" alt="${r.id}上下文"></div></div><small>${r.id} · ${r.artist} · ${r.publisher}</small><b>来源候选：${r.label}</b><br><b class="${r.label===r.predictions[k].predicted?'good':'bad'}">冻结模型：${r.predictions[k].predicted}</b><details><summary>所有模型与原图位置</summary>${Object.entries(r.predictions).map(([key,p])=>`<small class="${p.predicted===r.label?'good':'bad'}">${esc(DATA.methods[key])}：${p.predicted}；分类间隔${p.signed_margin.toFixed(3)}（不是概率）</small>`).join('')}<img class="image" src="${r.position}" alt="蓝框仅为裁剪位置"><small>${esc(r.label_scope)}<br>蓝框是裁剪位置，不是提取效果。${esc(r.attribution||'')}</small><a href="${r.source_url}" target="_blank" rel="noopener">核对原始来源</a></details><details><summary>特征空间近邻（事后诊断）</summary><small>这不是SVM决策的因果解释，也没有用于修正标签。</small>${r.nearest.map(n=>`<p><img class="thumb" src="${n.image}" alt="旧训练样本${n.id}"><small>${n.label} · ${n.artist} · 相似度${n.similarity.toFixed(3)}</small></p>`).join('')}</details></article>`).join('');}
</script></html>'''
 (ROOT/'index.html').write_text(page,encoding='utf8')
 note=f'''# 冻结上下文模型新作者验证（2026-11-11）

未通过，拒绝采用，不进入生成或实机。

先冻结旧63例上训练好的SVM头（C=1）、MobileNetV3编码器和预处理快照，再采集新图。新作者：王时敏、文徵明、马远、夏圭，四件作品、两个发布机构。16个候选中3个马远局部在预测前因梅枝主导、印章污染、皴纹不足排除；13例全部报告，披麻8/斧劈5。馆藏作品技法或目录标签＋助手局部筛选，不是独立专家真值，认证为0。

冻结双路7/13=53.8%，macroF1=.4583，披麻6/8，斧劈1/5；仅局部8/13=61.5%，macroF1=.6061；仅上下文7/13。按作者：王时敏3/4、文徵明3/4、马远1/1、夏圭0/4。旧训练集多数类披麻32/31，总猜披麻本批也8/13（事后补充对照）。双路相对局部1例改善2例退步。新数据未做fit、调参、选择模型或修改阈值，所有初始冻结文件哈希不变。

新数据只在这一轮揭示结果，不能解释为独立专家验收。下一轮若用13例开发，须另设新的封存作品。四件作品不能支持全面泛化或统计显著性结论。上一轮85.7%旧资料探索结果本批未复现，不把不同测试集变化说成训练进步或精确泛化下降幅度。

来源：DPM231531/234350/230757；NPM5621（明文徵明山水卷，技法目录披麻皴）。NPM仅下载HTML实际展示的公开预览，未操作验证码下载；原始预览和内容裁剪框[0,46,595,374]保留，去除色卡与装裱。王原祁233555、戴进231419、王鉴231773、唐寅229559没有直接作品级皴法标签，未据一般画风硬贴；NPM11259同时列披麻和斧劈，未硬归纯类。

事后近邻只用于诊断，不是SVM因果解释；16例独立复核清单确认栏均空。主候选未重训；对照头冻结前仅用旧63例拟合。新图未参与训练或调参，未安装依赖、下载权重、替换生产版本或连接机械臂。

输出：{ROOT}
冻结清单SHA256：{hashlib.sha256((ROOT/'freeze_manifest.json').read_bytes()).hexdigest()}
测试清单SHA256：{s['test_manifest_sha256']}
'''
 (Path(__file__).resolve().parents[1]/'research'/'cunfa_fresh_validation_status_20261111.md').write_text(note,encoding='utf8');print('REPORT_READY',flush=True)
if __name__=='__main__':main()
