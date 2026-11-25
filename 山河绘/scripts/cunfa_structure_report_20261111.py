# -*- coding: utf-8 -*-
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,cv2,joblib
from PIL import Image,ImageDraw,ImageFont
from cunfa_structure_train_20261111 import structure,ROOT,OLD
def read(n):return json.loads((ROOT/n).read_text('utf8'))
def save(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 s=read('summary.json');d=read('dataset_manifest.json');p=read('protocol.json');checks=read('verification.json');rows=d['external_holdout']
 names={'frozen_gabor':'旧Gabor模型','structure_statistics':'新结构统计模型（主试验）','structure_plus_gabor':'结构＋Gabor（辅助试验）','colour_control':'明暗/颜色对照'}
 results={k:read(k+'_external.json') for k in names};pred={k:{r['id']:r for r in v['predictions']} for k,v in results.items()}
 (ROOT/'contexts').mkdir(exist_ok=True);(ROOT/'diagnostics').mkdir(exist_ok=True)
 font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
 for r in rows:
  # Context is for checking the ROI, never input to this classifier.
  im=Image.open(r['source_image']).convert('RGB');w,h=im.size;dr=ImageDraw.Draw(im);dr.rectangle(r['source_box'],outline='#007b90',width=max(2,int(max(w,h)/300)));im.thumbnail((850,650));im.save(ROOT/'contexts'/(r['id']+'.jpg'),quality=90)
  feat,(ink,theta,coh)=structure(Image.open(r['image']),True)
  assert np.isfinite(feat).all()
  a=Image.fromarray(np.uint8(255*(1-ink))).convert('RGB');b=np.zeros((128,128,3),'uint8');b[:,:,0]=np.uint8(theta/np.pi*179);b[:,:,1]=np.uint8(coh*255);b[:,:,2]=np.uint8(ink*255);b=Image.fromarray(cv2.cvtColor(b,cv2.COLOR_HSV2RGB))
  v=Image.new('RGB',(420,215),'#fffdf7');draw=ImageDraw.Draw(v);v.paste(a.resize((160,160)),(15,30));v.paste(b.resize((160,160)),(220,30));draw.text((15,3),'局部对比墨迹响应',font=font(16),fill='black');draw.text((220,3),'方向/一致性响应',font=font(16),fill='black');draw.text((15,192),'仅为特征示意，不是提取线稿或绘画效果',font=font(16),fill='#8f432d');v.save(ROOT/'diagnostics'/(r['id']+'.png'))
 payload=[]
 for r in rows:
  v={k:r[k] for k in ['id','label','artist_group','work_group','source_url','label_scope','source_box']};v.update(image='crops/'+Path(r['image']).name,context='contexts/'+r['id']+'.jpg',diagnostic='diagnostics/'+r['id']+'.png',predictions={k:pred[k][r['id']] for k in names});payload.append(v)
 # An annotation queue is deliberately unanswered, with no prediction-as-gold.
 review=[dict(id=r['id'],image=r['image'],source_image=r['source_image'],source_box=r['source_box'],candidate_source_label=r['label'],source_url=r['source_url'],scope=r['label_scope'],
  reviewed_local_label=None,reviewer=None,review_date=None,evidence_region=None,uncertain_reason=None,allowed_labels=['披麻皴','斧劈皴','混合','无法确认','非山石/裁剪不合格'],eligible_for_formal_test=False) for r in d['development_records']+rows]
 save('expert_review_queue.json',review)
 view=dict(summary=s,methods=names,results=results,records=payload,verification=checks,protocol=p,exclusions=d['exclusions'],old_development_cv={k:read(k+'_development_cv.json') for k in ['structure_statistics','structure_plus_gabor','colour_control']})
 save('report_data.json',view)
 # Static overview intentionally puts primary failed result alongside fixed controls.
 im=Image.new('RGB',(1440,900),'#f4f1e8');dr=ImageDraw.Draw(im)
 dr.text((35,22),'皴法辨识第二轮：新增作者测试，结构模型未通过',font=font(32),fill='#243b32')
 dr.text((35,80),'训练：原40个区域 / 9位作者　　外部测试：新23个区域 / 4位作者',font=font(23),fill='#40574c')
 dr.text((35,122),'累计候选63个：披麻32、斧劈31；仍缺专家逐区域真值，裁剪数量不等于独立作品数量。',font=font(21),fill='#40574c')
 dr.text((40,190),'同一组新23例的来源标签一致率',font=font(25),fill='#243b32')
 for i,key in enumerate(names):
  y=250+i*100;val=s['metrics'][key]['label_agreement'];dr.text((40,y),names[key],font=font(23),fill='#243b32');dr.rectangle((565,y+4,565+val*650,y+34),fill='#b46a4c' if key=='structure_statistics' else '#527c69');dr.text((1150,y-3),f'{val:.1%} ({round(val*23)}/23)',font=font(24),fill='#243b32')
 dr.text((40,675),'主试验：披麻仅3/12符合来源标签，斧劈10/11；类别表现明显失衡。',font=font(24),fill='#993c2b')
 dr.text((40,720),'结构＋Gabor与旧模型同为16/23，不能声称稳定提升；本轮不替换绘画模型。',font=font(23),fill='#993c2b')
 dr.text((40,770),'昨天的42.5%是旧40例按作者留出；今天的69.6%是新23例，不能直接比较为进步。',font=font(21),fill='#40574c')
 dr.text((40,818),'本轮仅研究局部分类，未训练完整笔组生成、整幅皴法定位或机械臂复现。',font=font(21),fill='#40574c');im.save(ROOT/'comparison.png')
 canvas=Image.new('RGB',(1440,6*280+75),'#f4f1e8');dd=ImageDraw.Draw(canvas);dd.text((15,12),'新23例全部展示：来源标签 / 旧模型 / 结构模型 / 融合模型',font=font(24),fill='#243b32')
 for i,r in enumerate(rows):
  x=i%4*360;y=i//4*280+65;a=Image.open(r['image']);a.thumbnail((335,135));canvas.paste(a,(x+12,y));dd.text((x+12,y+140),r['id'],font=font(17),fill='black');dd.text((x+12,y+165),'来源：'+r['label']+' '+r['artist_group'],font=font(17),fill='#40574c')
  for j,(key,label) in enumerate([('frozen_gabor','旧'),('structure_statistics','结构'),('structure_plus_gabor','融合')]):
   v=pred[key][r['id']];dd.text((x+12,y+190+24*j),label+'：'+v['predicted'],font=font(18),fill='#25674d' if v['predicted']==r['label'] else '#a33b2b')
 canvas.save(ROOT/'all_external_predictions.jpg',quality=93)
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>皴法辨识第二轮 · 新作者对照</title><style>body{background:#f4f1e8;color:#233b31;font:16px/1.7 system-ui;margin:0}main{max-width:1250px;margin:auto;padding:26px}h1{font-size:29px}h2{font-size:23px;margin-top:34px}.notice{padding:18px;background:#f1dbcd;border-left:5px solid #a25539}.overview{width:100%;height:auto}select{max-width:100%;padding:10px;margin:6px 8px 6px 0;background:#fffdf7;border:1px solid #9ea99d;border-radius:6px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(265px,1fr));gap:16px}.card{background:#fffdf7;border:1px solid #dedbcd;border-radius:8px;padding:16px}.crop{width:100%;height:170px;object-fit:contain;background:white}.context{width:100%;height:auto}.good{color:#21664a}.bad{color:#a03d2a}small{display:block;color:#596b5e;overflow-wrap:anywhere}a{color:#235d46}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #c8c9bb}.tablewrap{overflow-x:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{margin-top:12px}summary{cursor:pointer}</style><main><h1>皴法辨识第二轮：新作者对照与标签复核</h1><div class="notice"><b>结论：结构统计主试验未通过，没有可采用的新皴法模型。</b><br>本轮新增23个有馆藏说明依据的局部，来自董其昌、刘珏、王履、刘松年4位新作者。与原40个区域合计63个候选，披麻32、斧劈31，涉及13位署名/归属作者。训练仍只使用原40个，新23例不参与训练、调参或选择主方法。<br>这些是来源支持的暂定标签，专家逐区域金标准为0。不能把63个裁剪理解为63件独立作品，也不能把来源一致率当成真实专家准确率。</div><p>结果与方案在输出清单中保留；本轮未连接机械臂、未替换认可的绘画版本。新结构信息是方向、局部一致性、宽窄、连通和间距统计，尚未建立画家真实笔组、起落笔和皴法语义的监督。</p><img class="overview" src="comparison.png" alt="同一外部测试集对照"><h2>同一外部测试集上的比较</h2><div class="tablewrap"><table><thead><tr><th>方法</th><th>来源一致率</th><th>macro-F1</th><th>披麻召回</th><th>斧劈召回</th><th>按作者平均一致率</th></tr></thead><tbody id="metrics"></tbody></table></div><p>主试验相较旧Gabor有2例改对、5例改错。融合候选与旧模型均为16/23，微小F1差别不构成稳定改善。按作者平均值也一并显示，避免某位作者裁剪较多掩盖失败。新测试集只涉及4位作者，不能作显著性或广泛泛化结论。</p><h2>查看每一张：原图局部、来源与预测</h2><select id="method"></select><select id="artist"><option value="">全部新作者</option></select><select id="status"><option value="">全部结果</option><option value="wrong">仅与来源不符</option><option value="correct">仅与来源一致</option></select><span id="count"></span><div class="grid" id="grid"></div><h2>为什么没有判为进步</h2><p>纯结构统计把12例披麻中的9例判成斧劈，说明这些统计量没有稳定区分两类。加入纹理虽平衡了两类表现，但没有超过固定旧模型。局部方向和宽窄并不能替代笔触归属、山石结构与笔组关系；本轮未训练这些真实语义关系。部分裁剪原始宽度仅65–100像素、含树木或设色，来源标签仍需复核，这些是需要逐项检验的限制，不能据此认定唯一失败原因。</p><h2>来源与标签范围</h2><p><a href="https://www.dpm.org.cn/collection/paint/230692.html" target="_blank" rel="noopener">董其昌山水册</a>只取明确提及披麻的第七开；<a href="https://www.dpm.org.cn/collection/paint/230678.html" target="_blank" rel="noopener">董其昌山水卷</a>只取说明为披麻的远山；<a href="https://www.dpm.org.cn/collection/paint/232500.html" target="_blank" rel="noopener">刘珏夏云欲雨</a>以长披麻说明为依据；<a href="https://www.dpm.org.cn/collection/paint/228140.html" target="_blank" rel="noopener">王履华山图册</a>与<a href="https://www.dpm.org.cn/collection/paint/234462.html" target="_blank" rel="noopener">刘松年四景山水</a>有小斧劈说明。明确混用技法的王翚作品留作待定，没有硬塞入纯斧劈类。局部选择由助手复核，不冒充专家标签。</p><h2>下一步已有的准备</h2><p>已建立63例未填答案的专家复核清单，分别记录来源标签、像素框、可确认区域及不确定原因；模型预测没有写成训练真值。应先获得清晰完整山石与笔组级参考，并封存新的独立作者测试，再检验可学习的上下文模型。本轮23例已经查看结果，后续不能再称为未见测试。混合/未知类、整幅定位、皴笔生成和实机参数尚未验证。</p><details><summary>查看冻结方案与完整性检查</summary><pre id="checks"></pre></details><p><a href="all_external_predictions.jpg" target="_blank">全部23例预测长图</a> · <a href="report_data.json">实验审计数据</a> · <a href="expert_review_queue.json">未填标签复核清单</a></p></main><script>
const el=id=>document.getElementById(id),esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let DATA;const pct=n=>(n*100).toFixed(1)+'%';
fetch('report_data.json').then(r=>r.json()).then(d=>{DATA=d;el('method').innerHTML=Object.entries(d.methods).map(([k,v])=>`<option value="${k}">${esc(v)}</option>`).join('');el('method').value='structure_statistics';el('artist').innerHTML+=[...new Set(d.records.map(r=>r.artist_group))].map(a=>`<option>${a}</option>`).join('');el('metrics').innerHTML=Object.entries(d.summary.metrics).map(([k,m])=>`<tr><td>${esc(d.methods[k])}</td><td>${pct(m.label_agreement)}</td><td>${m.macro_f1.toFixed(3)}</td><td>${pct(m.class_recall[0])}</td><td>${pct(m.class_recall[1])}</td><td>${pct(m.artist_macro_label_agreement)}</td></tr>`).join('');el('checks').textContent=JSON.stringify({verification:d.verification,protocol:d.protocol,exclusions:d.exclusions},null,2);for(const k of ['method','artist','status'])el(k).onchange=draw;draw();window.reportReady=true;});
function draw(){const key=el('method').value,a=el('artist').value,status=el('status').value;const rows=DATA.records.filter(r=>(!a||a===r.artist_group)&&(!status||status==='wrong'&&r.predictions[key].predicted!==r.label||status==='correct'&&r.predictions[key].predicted===r.label));el('count').textContent=`显示 ${rows.length}/23`;el('grid').innerHTML=rows.map(r=>{const p=r.predictions[key];return `<article class="card"><img class="crop" src="${r.image}" alt="${esc(r.id)}"><small>${r.id} · ${r.artist_group}</small><b>来源候选：${r.label}</b><br><b class="${p.predicted===r.label?'good':'bad'}">${esc(DATA.methods[key])}：${p.predicted}</b><small>没有独立专家真值；符合来源不等于确认正确。</small><details><summary>全部方法对照</summary>${Object.entries(r.predictions).map(([k,v])=>`<small class="${v.predicted===r.label?'good':'bad'}">${esc(DATA.methods[k])}：${v.predicted}</small>`).join('')}</details><details><summary>原作位置与标签范围</summary><p>${esc(r.label_scope)}</p><img class="context" src="${r.context}" alt="蓝框仅标记裁剪位置，不是模型提取"><small>蓝框仅是裁剪位置；像素框 ${r.source_box.join(', ')}</small><a href="${r.source_url}" target="_blank" rel="noopener">核对馆藏说明</a></details><details><summary>算法实际看了什么（非绘画效果）</summary><img class="context" src="${r.diagnostic}" alt="计算特征响应，不是线稿"><small>颜色仅表示局部方向与响应强度，不是识别的笔触类别。未把响应当作画家真实分笔。</small></details></article>`}).join('');}
</script></html>'''
 (ROOT/'index.html').write_text(page,encoding='utf8')
 note=f'''# 皴法辨识外部作者对照（2026-11-11）

未通过，不采用新模型，未连接机械臂。

新增23个候选区域，4位作者；合计63个已知类候选（披麻32、斧劈31）/13位作者。来源为DPM实际作品说明，新增作者董其昌（仅第七开和卷中远山）、刘珏、王履、刘松年。王翚混用技法排除。独立专家局部真值仍为0。63个裁剪不等于63幅独立作品；每类区域数量到达初始目标，可靠标签目标尚未达到。

训练仅旧40例/9位作者，新增23例只作外部测试。冻结代码/方案/清单后执行；未使用新标签调参、训练或选择主方法。主试验为128方形归一化后的多尺度HOG、方向一致性、墨带宽窄、连通和间距统计＋RBF SVM；这些是手工结构代理，不是学得的真实笔组语义，也不是新的深度完整笔触模型。融合统计Gabor为预先指定辅助试验。C仅由旧训练作者内层GroupKFold选择。模型在E盘保存，现有Python环境无新增依赖。

同一23例来源标签一致率：冻结旧Gabor16/23（69.6%，macroF1=.6933）；主结构模型13/23（56.5%，macroF1=.5208）；融合16/23（69.6%，macroF1=.6957）；颜色12/23（52.2%，macroF1=.4559）。主模型披麻3/12，斧劈10/11，明显偏斧劈；相对旧模型2例改善、5例退步。按作者平均值亦在summary中，区域数加权与作者平均不可混用。旧42.5%是不同旧数据上的作者交叉验证，不能与新69.6%直接声称进步。

本轮结论：加入这些结构统计没有证明稳定改善，不能说已经懂得皴法。尚缺经复核的整块山石、完整笔触与笔组关系监督。待专家确认的清单已建立，全部reviewed_local_label为空，不以自动预测替代真值。本轮测试已暴露，后续模型需新增未见作者测试。未知/混合类、整幅定位、条件笔组生成、实机复现均未通过。

输出：{ROOT}
方案SHA256：{s['protocol_sha256']}
数据清单SHA256：{s['manifest_sha256']}
复现：设置项目要求的E盘缓存环境与OPENBLAS_NUM_THREADS/MKL_NUM_THREADS/OMP_NUM_THREADS=4后，E:/Anaconda/python.exe -B运行scripts/cunfa_collect_sources_20261111.py、cunfa_prepare_holdout_20261111.py、cunfa_structure_train_20261111.py、cunfa_structure_report_20261111.py。新训练脚本校验既有protocol，不允许静默覆盖改动后的方案。图片源以source_manifest记录，所有下载E盘，无实机连接。
'''
 (Path(__file__).resolve().parents[1]/'research'/'cunfa_structure_status_20261111.md').write_text(note,encoding='utf8');print('REPORT_READY',flush=True)
if __name__=='__main__':main()
