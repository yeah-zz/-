# -*- coding: utf-8 -*-
"""Audit the frozen experiment and publish a read-only comparison, no robot IO."""
import sys,json,hashlib,html
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from sklearn.decomposition import PCA
from train_cunfa_gabor_20261110 import fit_projection,project
ROOT=LAB/'cunfa_gabor_baseline_20261110'
def read(n):return json.loads((ROOT/n).read_text('utf8'))
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 s=read('summary.json');manifest=read('dataset_manifest.json');rows=manifest['records'];byid={r['id']:r for r in rows};ev=read('full_gabor_evaluation.json');pred={p['id']:p for p in ev['predictions']}
 assert hashlib.sha256((ROOT/'dataset_manifest.json').read_bytes()).hexdigest()==s['manifest_sha256']
 assert len(pred)==40 and len(set(pred))==40
 for r in rows:
  assert hashlib.sha256(Path(r['image']).read_bytes()).hexdigest()==r['crop_sha256']
  assert hashlib.sha256(Path(r['source_image']).read_bytes()).hexdigest()==r['source_sha256']
 for f in ev['folds']:
  tr=[byid[i] for i in f['train_ids']];te=[byid[i] for i in f['test_ids']]
  for k in ['artist_group','work_group','crop_sha256']:
   assert {r[k] for r in tr}.isdisjoint({r[k] for r in te})
  assert all(r['admission']=='pilot_only' for r in tr+te)
 assert ev['unknown_probe']['probe_artist_absent_from_training']
 # Compare the custom dual PCA with sklearn's independent full-SVD PCA.
 # Pairwise distances avoid sign ambiguities in equivalent eigenvectors.
 rng=np.random.RandomState(17);x=rng.normal(size=(18,30)).astype('float64');q=fit_projection(x,'full_gabor');ours=(x-q['mean'])@q['axes'].T
 ref=PCA(n_components=8,svd_solver='full').fit_transform(x)
 a=np.linalg.norm(ours[:,None]-ours[None,:],axis=-1);b=np.linalg.norm(ref[:,None]-ref[None,:],axis=-1)
 err=float(np.max(abs(a-b)));assert err<1e-4
 checks=dict(manifest_hash_matches=True,all_69_crop_and_source_hashes_match=True,outer_artist_work_exact_crop_overlap=0,
  each_known_region_predicted_once=True,unknown_probes_excluded_from_fitting=True,dual_pca_matches_independent_svd_max_error=err,
  serialised_model_reload_matches=s['model_reload_equivalent'],formal_expert_test_samples=0,robot_connected=False)
 write('verification.json',checks)
 font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
 im=Image.new('RGB',(1400,860),'#f4f1e8');d=ImageDraw.Draw(im)
 d.text((38,24),'皴法辨识基线：本轮未通过',font=font(35),fill='#202e2b')
 d.text((38,82),'40个来源依据候选区域 · 两类各20 · 9位署名/归属作者 · 12个作品/示范',font=font(21),fill='#354a45')
 d.text((38,118),'按作者留出测试；同一作品的裁剪不会同时进入训练和测试。非专家金标准。',font=font(20),fill='#354a45')
 d.text((42,185),'来源标签 ↓  /  模型预测 →',font=font(22),fill='#202e2b')
 for j,cl in enumerate(s['classes']):d.text((280+j*190,230),cl,font=font(25),fill='#202e2b')
 for i,cl in enumerate(s['classes']):
  d.text((55,315+i*150),cl,font=font(25),fill='#202e2b')
  for j in range(2):
   x0=240+j*190;y0=275+i*150;d.rectangle((x0,y0,x0+175,y0+135),fill='#b5d2c2' if i==j else '#ecd2bf');d.text((x0+65,y0+40),str(ev['confusion'][i][j]),font=font(35),fill='#202e2b')
 d.text((710,205),'留出测试：与来源标签的一致率',font=font(25),fill='#202e2b')
 for k,(name,label) in enumerate([('full_gabor','40组Gabor图 + PCA + SVM'),('pooled_gabor','Gabor均值/标准差 + SVM'),('colour_control','明暗/颜色对照 + SVM')]):
  y0=270+k*100;v=s['metrics'][name]['label_agreement'];d.text((710,y0),label,font=font(20),fill='#202e2b');d.rectangle((710,y0+37,710+v*480,y0+62),fill='#547b70');d.text((1190,y0+27),f'{v:.1%}',font=font(24),fill='#202e2b')
 d.text((710,585),'全量Gabor：17/40符合来源标签',font=font(23),fill='#983e2f')
 d.text((38,660),'结论：未超过简单明暗/颜色对照，不能据此声称已具备可靠皴法辨识能力。',font=font(24),fill='#983e2f')
 for k,t in enumerate(['只输入人工选定的山石局部；尚未实现整幅自动找皴法。','资料仍有作品级弱标签，正式专家测试为0；每类30–50个目标尚未达到。','“需复核”仅由分类间隔触发，不等于已训练好“混合/未知”类别。']):d.text((38,710+k*38),t,font=font(21),fill='#354a45')
 im.save(ROOT/'results.png')
 # Show every known sample, in frozen manifest order, with OOF predictions.
 known=[r for r in rows if r['id'] in pred];canvas=Image.new('RGB',(1400,10*245+75),'#f4f1e8');dd=ImageDraw.Draw(canvas)
 dd.text((25,14),'全部40个留出预测：来源标签 → 模型预测（非训练集重预测）',font=font(24),fill='#202e2b')
 for i,r in enumerate(known):
  p=pred[r['id']];x=i%4*350;y=i//4*245+65;a=Image.open(r['image']).convert('RGB');a.thumbnail((320,140));canvas.paste(a,(x+15,y+5));col='#2b6551' if p['label']==p['predicted'] else '#a33f32'
  dd.text((x+12,y+150),r['id'],font=font(17),fill='#202e2b');dd.text((x+12,y+177),p['label']+' → '+p['predicted'],font=font(20),fill=col);dd.text((x+12,y+205),r['artist_group'],font=font(17),fill='#354a45')
 canvas.save(ROOT/'all_predictions.jpg',quality=91)
 payload=[]
 for r in known:
  p=pred[r['id']];payload.append(dict(**{k:r[k] for k in ['id','label','artist_group','work_group','publisher_group','label_scope','source_url','source_box']},predicted=p['predicted'],candidate=p['candidate'],correct=p['predicted']==r['label'],image='crops/'+Path(r['image']).name,source_image=r['source_image'],absolute_margin=p['absolute_margin'],review_threshold=p['review_threshold']))
 probes=[r for r in rows if r['admission']=='unknown_probe'];probe=ev['unknown_probe'];probeview=[]
 for i,r in enumerate(probes):probeview.append(dict(id=r['id'],image='crops/'+Path(r['image']).name,source_url=r['source_url'],predicted=s['classes'][probe['predicted'][i]],review=probe['flagged_for_review'][i],evidence=r['evidence']))
 data=dict(summary=s,records=payload,probes=probeview,verification=checks,excluded=read('excluded_sources.json'),excluded_regions=[dict(id=r['id'],reason=r['exclusion_reason']) for r in rows if r['admission']=='excluded'])
 write('report_data.json',data)
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>皴法辨识 · 来源与留出测试</title><style>body{margin:0;background:#f4f1e8;color:#21342e;font:16px/1.7 system-ui}main{max-width:1280px;margin:auto;padding:28px}h1{font-size:30px}h2{font-size:23px;margin-top:34px}.notice{background:#f1d9c9;padding:18px;border-left:5px solid #a04b36}a{color:#235d4d}select,button{padding:10px;margin:5px;border:1px solid #9ea9a1;background:white;border-radius:6px}#grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}.card{background:#fffdf7;padding:16px;border-radius:8px;border:1px solid #dedbcf}.crop{display:block;width:100%;height:180px;object-fit:contain;background:white}.bad{color:#9f3d2c}.good{color:#23664a}small{display:block;color:#57665f;overflow-wrap:anywhere}.overview{width:100%;height:auto}details{margin-top:12px}table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #ccc;text-align:left;padding:10px}#probes{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}</style><main><h1>皴法辨识：先核对来源，再检验模型</h1><p>本轮是披麻皴 / 斧劈皴分类研究，结果未通过。没有替换已认可的绘画版本，未连接机械臂。</p><div class="notice"><b>不应解读成“已经懂得皴法”</b><br>40个有来源依据的候选区域，两类各20个，来自9位署名/归属作者、12个作品/示范、3个发布机构。专家逐区域复核与正式专家测试均为0；每类30–50个目标尚未达到。博物馆的作品级技法说明不等于每个裁剪的独立真值。</div><p>所有裁剪在训练前冻结；按作者留出9轮，调参只在各轮训练作者中进行。表中每张预测均来自未见该作者的模型。PCA、标准化只拟合训练数据。其他皴法示例不参与训练。</p><img class="overview" src="results.png" alt="分类结果与对照"><h2>方法与结论</h2><p>基线借鉴2014年Gabor＋PCA＋SVM方法族：128×128灰度均衡、5尺度×8方向的Gabor幅值图，最多8个主成分，RBF SVM。原论文研究点/线/面三类，本轮改为两种具体皴法，不是原文结果的精确复现。</p><p>全量Gabor的来源标签一致率42.5%，macro-F1为0.422；均值/标准差Gabor为45.0%；明暗/颜色对照为47.5%。这些结果没有支持可靠的跨作者辨识。请同时检查来源标签、裁剪分辨率和模型的失败案例。</p><h2>逐张来源与留出预测</h2><select id="cl"><option value="">全部来源标签</option><option>披麻皴</option><option>斧劈皴</option></select><select id="status"><option value="">全部结果</option><option value="wrong">仅错例</option><option value="correct">仅符合来源标签</option><option value="review">仅触发需复核</option></select><select id="artist"><option value="">全部作者</option></select><span id="count"></span><div id="grid"></div><h2>其他皴法：拒识压力检查</h2><p>以下5例来自同一署名示范者，使用完全未见该作者的折模型。二分类器仍会强制输出两类之一；“需复核”阈值只按训练折分类间隔设定，不能当成混合/未知分类已通过。</p><div id="probes"></div><h2>来源与未完成项</h2><p>图例依据：<a href="https://education.asianart.org/wp-content/uploads/sites/6/2019/09/Chinese-Brushstrokes-Vocabulary-AAM_0.pdf" target="_blank" rel="noopener">亚洲艺术博物馆用笔图例（Lampo Leong）</a>；<a href="https://www.npm.gov.tw/NewChineseArtDownload.ashx?bid=4360" target="_blank" rel="noopener">台北故宫文章：同幅作品不同皴法局部</a>；各样本内链接至故宫博物院对应藏品说明。需补清晰且独立的作品/作者、逐区域专家复核，预先封存测试，再比较结合山石结构和笔组的模型。不能把本轮自动预测当作新标签。</p><details><summary>来源排除与完整性检查</summary><pre id="checks" style="white-space:pre-wrap;overflow-wrap:anywhere"></pre></details><p><a href="all_predictions.jpg" target="_blank">查看全部预测长图</a> · <a href="report_data.json">下载审计数据</a> · <a href="verification.json">完整性检查</a></p></main><script>
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let DATA;
fetch('report_data.json').then(r=>r.json()).then(d=>{DATA=d;artist.innerHTML+=[...new Set(d.records.map(r=>r.artist_group))].sort().map(a=>`<option>${esc(a)}</option>`).join('');for(const x of [cl,status,artist])x.onchange=draw;draw();probes.innerHTML=d.probes.map(r=>`<article class="card"><img class="crop" src="${r.image}"><b>${esc(r.id)}</b><small>${esc(r.evidence)}</small><p>强制二类：${esc(r.predicted)}<br>${r.review?'触发需复核':'未触发需复核'}</p></article>`).join('');checks.textContent=JSON.stringify({checks:d.verification,excluded_sources:d.excluded,excluded_regions:d.excluded_regions},null,2);window.reportReady=true;});
function draw(){const selected=DATA.records.filter(r=>(!cl.value||r.label===cl.value)&&(!artist.value||r.artist_group===artist.value)&&(!status.value||status.value==='wrong'&&!r.correct||status.value==='correct'&&r.correct||status.value==='review'&&r.candidate==='需复核'));count.textContent=`显示 ${selected.length} / 40`;grid.innerHTML=selected.map(r=>`<article class="card"><img class="crop" src="${r.image}" alt="${esc(r.id)}"><small>${esc(r.id)} · ${esc(r.artist_group)}</small><b class="${r.correct?'good':'bad'}">${esc(r.label)} → ${esc(r.predicted)}</b><small>${esc(r.candidate==='需复核'?'分类间隔触发需复核':'未触发复核；不代表预测可靠')}</small><details><summary>核对来源与标签范围</summary><p>作品/示范：${esc(r.work_group)}<br>标签范围：${esc(r.label_scope)}<br>原图像素框：${r.source_box.join(', ')}</p><a href="${esc(r.source_url)}" target="_blank" rel="noopener">查看原始来源</a><small>分类间隔 ${r.absolute_margin.toFixed(3)} / 阈值 ${r.review_threshold.toFixed(3)}；不是置信概率。</small></details></article>`).join('');}
</script></html>'''
 # `status` is a built-in window property, so use explicit element references.
 page=page.replace('let DATA;','let DATA;const cl=document.getElementById("cl"),status=document.getElementById("status"),artist=document.getElementById("artist"),grid=document.getElementById("grid"),count=document.getElementById("count"),probes=document.getElementById("probes"),checks=document.getElementById("checks");')
 (ROOT/'index.html').write_text(page,encoding='utf8')
 note=f'''# 皴法辨识初始基线（2026-11-10）

本轮未通过，不替换绘画系统，不连接机械臂。

数据：40个来源依据候选区域（披麻皴20、斧劈皴20），9位署名/归属作者，12个作品/示范，3个发布机构。5例其他类型仅作压力检查。69个裁剪中24个在训练前因空白、植被或边框等排除。旧30个弱来源样本未用。正式专家局部标签/测试均为0，未达每类30–50个目标。

来源：AAM用笔图例、NPM万壑松风局部说明、DPM七件藏品描述。AAM全为Lampo Leong署名；李唐同幅画包含两种局部标签，不能按整幅画统一标签。DPM属于作品级技法依据＋人工裁剪，不是专家局部真值。源文件与裁剪SHA-256写入清单。

方法：40个Gabor幅值图、128×128直方图均衡灰度、最多8维PCA、训练投影标准化、RBF SVM；C∈[0.1,1,10]内层作者GroupKFold选择。外层LeaveOneGroupOut按作者9轮。与原2014论文三类任务不同，属于方法族适配。主模型17/40符合来源标签（42.5%），macro-F1=0.4217，混淆矩阵行真列预测[[10,10],[13,7]]。Gabor统计特征45.0%；颜色明暗对照47.5%。未显示结构辨识优势。

完整性：所有69个源/裁剪哈希、作者/作品/精确裁剪不跨折、40个各预测一次、未知探针不进训练、模型存取等价通过。双PCA与独立SVD距离误差{err:.3g}。没有将正式未知类别训练完成，“需复核”仅为分类间隔阈值。没有整幅检测、笔触生成或实机动作。

复现：先显式设置E盘TEMP/TMP/PIP_CACHE_DIR/TORCH_HOME/HF_HOME/XDG_CACHE_HOME/CUDA_CACHE_PATH、PYTHONDONTWRITEBYTECODE=1及OPENBLAS_NUM_THREADS/MKL_NUM_THREADS/OMP_NUM_THREADS=4；在项目目录用E:/Anaconda/python.exe -B运行scripts/prepare_cunfa_two_class_20261110.py、scripts/train_cunfa_gabor_20261110.py、scripts/report_cunfa_gabor_20261110.py。既有threadpoolctl在枚举OpenBLAS时不兼容，本轮用启动环境限制线程，没有升级或修改现有环境。

下一步：先复核错误区域标签，补独立作者和清晰皴笔资料；按作者预先封存专家测试。冻结新数据后再比较图像上下文/笔组结构方法，不以本次预测自动扩充标签。原认可版本维持基线。

输出目录：{ROOT}
清单SHA-256：{s['manifest_sha256']}
'''
 out=Path(__file__).resolve().parents[1]/'research';out.mkdir(exist_ok=True);(out/'cunfa_gabor_baseline_status_20261110.md').write_text(note,encoding='utf8')
 print(json.dumps(checks,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
