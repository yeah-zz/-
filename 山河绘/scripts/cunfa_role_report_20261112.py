# -*- coding: utf-8 -*-
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'cunfa_role_reference_20261112';PREV=LAB/'cunfa_fresh_validation_20261111'
def read(n):return json.loads((ROOT/n).read_text('utf8'))
def save(n,d):(ROOT/n).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 native=read('native_manifest.json');roles=read('partial_roles.json');refs=read('source_references.json');probe=read('source_probe_results.json');policy=read('legacy_policy_audit.json');checks=read('supervision_checks.json');pred={r['id']:r for r in probe['predictions']};font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
 old=json.loads((PREV/'summary.json').read_text('utf8'));assert old['test_regions']==13 and old['metrics']['local_plus_context']['confusion']==[[6,2],[4,1]]
 assert sha(PREV/'test_manifest.json')==policy['legacy_manifest_sha256'];assert len(refs['records'])==8 and all(r['style_target']>=0 for r in refs['records']);assert roles['known_pixel_masks']==0
 for r in refs['records']:assert sha(r['image'])==r['crop_sha256'] and sha(r['source_image'])==r['source_sha256']
 for r in native['records']:assert sha(r['native_image'])==r['native_sha256'] and sha(r['old_image'])==r['old_sha256']
 cases=[]
 translations={'structure_candidate':'轮廓/山面转折候选','cun_candidate':'皴笔作用候选','wash_or_paper_unknown':'染墨/底纹待定','wash_or_merged_ink_unknown':'染墨/叠墨待定'}
 for r in native['records']:
  role=next(x for x in roles['records'] if x['id']==r['id']);regions=[dict(**z,role_display=translations[z['role_candidate']]) for z in role['regions']]
  cases.append(dict(id=r['id'],comparison='native/'+r['id']+'_comparison.jpg',native='native/'+r['id']+'.png',overlay='roles/'+r['id']+'.png',old_box=r['old_box'],native_box=r['native_box'],old_size=r['old_size'],native_size=r['native_size'],gray_registration_correlation=r['thumbnail_native_gray_correlation'],regions=regions,
   old_source_label='斧劈皴（作品级）',old_model_prediction='披麻皴',confirmed_local_named_style=None,source_url=r['source_url']))
 refview=[]
 for r in refs['records']:refview.append(dict(id=r['id'],image='references/'+r['id']+'.png',label=r['local_style'],publisher=r['publisher'],scope=r['label_scope'],source_url=r['source_url'],source_evidence=r['source_evidence'],predicted=pred[r['id']]['predicted']))
 verification=dict(old_13_test_manifest_unchanged=True,old_13_confusion_unchanged=True,native_four_same_regions_not_new_samples=True,all_source_and_crop_hashes_match=True,source_caption_examples=8,role_candidate_regions=14,confirmed_role_masks=0,automatic_predictions_as_labels=0,all16_legacy_work_labels_blocked_for_new_training=True,
  unknown_gradient_checks_passed=all(checks.values()),new_role_model_trained=False,source_example_classification_head_trained=True,source_probe_C_fixed_at1=True,no_hyperparameter_search=True,production_replaced=False,robot_connected=False)
 # checks contains robot_connected=False, so aggregate only the assertion fields.
 verification['unknown_gradient_checks_passed']=all(v for k,v in checks.items() if k!='robot_connected')
 save('verification.json',verification);save('report_data.json',dict(cases=cases,references=refview,probe=probe,policy=policy,supervision_checks=checks,verification=verification))
 cover=Image.new('RGB',(1440,820),'#f4f1e8');dd=ImageDraw.Draw(cover);dd.text((24,20),'这一轮修正监督入口：笔迹作用与皴法名称分开',font=font(33),fill='#243e32');dd.text((24,82),'高分不能替代可靠的局部标签；旧13例结果原样保留，没有追溯重算。',font=font(23),fill='#455d4d')
 for i,(head,body) in enumerate([('原生细节复查','同4个区域取得约4倍原生像素，不是AI增强，也不是新增独立测试。'),('具名示例参考','新增2个香港艺术馆教学例，复用6个已有具名参考，合计8个原始单位。'),('训练入口修复','整幅画标签不自动传给局部；未确认的角色框、待定像素不参加对应监督。'),('训练与限制','8例固定参数检查6/8；14个角色框仍待定，未训练角色或完整笔组模型。')]):
  y=175+i*130;dd.text((24,y),head,font=font(27),fill='#2d6850');dd.text((24,y+45),body,font=font(22),fill='#243e32')
 dd.text((24,735),'本轮没有通过验收的新绘画模型，没有替换当前版本，也没有连接机械臂。',font=font(23),fill='#98452f');cover.save(ROOT/'overview.png')
 # A single useful visual: source ink remains unchanged, coloured boxes are candidates.
 board=Image.new('RGB',(1500,780),'#f4f1e8');d=ImageDraw.Draw(board);d.text((18,10),'同一山石局部：原生墨迹 → 角色候选（助手标记，非模型分割）',font=font(29),fill='#243e32')
 for i,filename in enumerate(['native/fresh_230757_0.png','roles/fresh_230757_0.png']):
  im=Image.open(ROOT/filename);im.thumbnail((630,600));xx=20+i*750;board.paste(im,(xx,105));d.text((xx,64),['原生像素：保留墨带与底纹','候选框：并未确认墨迹归属'][i],font=font(23),fill='#243e32')
 d.text((20,655),'棕色：结构候选；蓝色：皴笔候选；灰棕：染墨/底纹待定。框内并非全部属于同类。',font=font(22),fill='#5c614a');d.text((20,703),'未确认的像素、完整曲线、笔顺和下压量仍为空，不把候选框灌成训练真值。',font=font(22),fill='#98452f');d.text((20,749),'来源：故宫博物院《雪堂客话图》公开放大图；本图不是提取线稿或绘画复现效果。',font=font(19),fill='#455d4d');board.save(ROOT/'role_comparison.jpg',quality=94)
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>皴法监督修正 · 原生细节与角色参考</title><style>body{margin:0;background:#f4f1e8;color:#243e32;font:16px/1.7 system-ui}main{max-width:1280px;margin:auto;padding:26px}h1{font-size:29px}h2{font-size:23px;margin-top:34px}.notice{padding:18px;background:#e3eadd;border-left:5px solid #507a65}.warn{background:#f1dfcf;padding:16px}.overview{width:100%;height:auto}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}.card{background:#fffdf7;border:1px solid #d9d8c6;border-radius:7px;padding:15px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}.pair img{width:100%;height:350px;object-fit:contain;background:#ddd7c8}.card img{width:100%;height:220px;object-fit:contain;background:white}small{display:block;overflow-wrap:anywhere;color:#5b6b5d}a{color:#235d42}button,select{max-width:100%;padding:10px;margin:6px;background:#fffdf7;border:1px solid #b1b69e}.good{color:#26694e}.bad{color:#a03f2b}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{margin-top:12px}summary{cursor:pointer}table{border-collapse:collapse;width:100%}th,td{padding:10px;border-bottom:1px solid #c9cbbb;text-align:left}.tablewrap{overflow-x:auto}</style><main><h1>先修监督范围：不再把整幅画标签当成局部答案</h1><div class="notice"><b>本轮做了数据与训练入口修正，没有产生通过验收的新绘画模型。</b><br>原来的13例7/13、夏圭4例与来源标签不符等结果均保留。回查确认，这些仍是作品级技法说明＋助手裁剪，并非专家逐局部真值。因此不能据此把每一条墨线都强行标作小斧劈，也不能把“判为披麻”反过来当新标签。</div><img class="overview" src="overview.png" alt="监督修正工作说明"><h2>原生像素：回查同4个区域</h2><p>从故宫公开放大查看器取得4268×4095图像中对应区域的原生像素。旧预览为1024×982。只读取同图所需区域，保留查看器元数据、瓦片来源和SHA-256；这不是AI增强，也不是4张新作品。缩回旧尺寸后的灰度相关性约0.94–0.97，核对的是同一位置。</p><select id="case"></select><div class="pair"><div><h3>原生墨迹（未改画）</h3><img id="native" alt="原生墨迹"></div><div><h3>助手角色候选框</h3><img id="overlay" alt="候选框，非模型输出"></div></div><p id="sizes"></p><div class="warn">棕色框表示轮廓/山面转折候选，蓝色表示皴笔作用候选，灰棕表示染墨/底纹或叠墨待定。框只是复核范围，不能假设框内全部像素都属于同类；确认的角色掩膜仍为0，局部具名皴法确认仍为0。没有画出已验证的完整单笔。</div><div class="tablewrap"><table><thead><tr><th>区域</th><th>可见作用候选</th><th>具名皴法</th><th>参与训练</th></tr></thead><tbody id="regions"></tbody></table></div><details><summary>查看旧预览放大与原生像素对照</summary><img class="overview" id="nativeComparison" alt="同图分辨率对照"></details><h2>直接具名的来源参考</h2><p>新增香港艺术馆教材中2个明确标注的斧劈/披麻练习图，复用AAM的4个具名示例、NPM文章2个具名局部，共8个原始参考单位，3个来源组。没有把旋转裁剪当新作品，也没有把HKMA教材标题中的王无邪当成未核实示范图的作者。具名示例能提供该图示的类别依据，不能自动提供逐笔起止、墨迹归属或实机压力。</p><div class="grid" id="references"></div><h2>本轮训练检查：仅具名示例分类头</h2><p>沿用已有ImageNet编码器并冻结它，只训练线性SVM示例分类头，C固定为1。按AAM/HKMA/NPM来源组留出，8例中6例符合来源标签；错例是AAM短披麻、NPM具名斧劈。这不是独立新作者验收，不能和原13例53.8%比较，也不能用75%声称整幅识别、笔组或实画能力已提升。14个角色候选框完全未进入训练。</p><h2>算法入口的实际修改</h2><ul><li>皴法名称与笔迹作用分别保存，作品级名称保留作来源信息，不自动传播到局部监督。</li><li>只有直接具名的教学示例、来源明确标注的局部，或有证据的专家局部复核，才能为对应的局部分类任务提供标签。</li><li>角色框必须有检查过的掩膜才能参加角色监督。待定像素不当背景，待定样式不当负类，模型预测不当答案。</li><li>原生像素和既有放大图都保留可追溯关系；不把插值/增强当作新信息。</li></ul><p>本轮入口审计把旧16个“作品级说明＋裁剪”的记录挡在新的严格局部监督之外，但不修改旧实验清单或分数。8个具名示例通过新入口；14个角色框仍被挡住，防止候选框变成伪真值。</p><details><summary>监督回归检查与完整性</summary><pre id="checks"></pre></details><h2>还缺什么</h2><p>需要能核对局部勾、皴、擦染关系的清晰步骤或逐笔示范，再补角色掩膜、笔组成员和不确定区。本轮没有可靠染墨掩膜、完整分笔或真实笔顺，因此没有启动角色/笔组模型训练。现有绘画版本维持原状，机械臂未连接。</p><p><a href="https://www.dpm.org.cn/collection/paint/230757.html" target="_blank" rel="noopener">故宫原作说明</a> · <a href="https://hk.art.museum/Archive/resources/S_Wucius_WONG_Suggested_Class_Activities_CHIN.pdf" target="_blank" rel="noopener">香港艺术馆教学活动建议</a> · <a href="report_data.json">复核数据</a> · <a href="source_references.jpg" target="_blank">8个参考长图</a></p></main><script>
const el=id=>document.getElementById(id),esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let DATA;
fetch('report_data.json').then(r=>r.json()).then(d=>{DATA=d;el('case').innerHTML=d.cases.map(r=>`<option value="${r.id}">${r.id}</option>`).join('');el('case').onchange=draw;draw();el('references').innerHTML=d.references.map(r=>`<article class="card"><img src="${r.image}" alt="${r.id}"><small>${r.id} · ${r.publisher}</small><b>来源具名：${r.label}</b><p class="${r.label===r.predicted?'good':'bad'}">来源组留出预测：${r.predicted}</p><small>作用掩膜、完整分笔：未标注</small><details><summary>核对来源范围</summary><small>${esc(r.scope)}<br>${esc(r.source_evidence)}</small><a href="${r.source_url}" target="_blank" rel="noopener">原始来源</a></details></article>`).join('');el('checks').textContent=JSON.stringify({supervision:d.supervision_checks,verification:d.verification},null,2);window.reportReady=true;});
function draw(){const r=DATA.cases.find(r=>r.id===el('case').value);el('native').src=r.native;el('overlay').src=r.overlay;el('nativeComparison').src=r.comparison;el('sizes').textContent=`旧裁剪 ${r.old_size.join('×')} → 对应原生区域 ${r.native_size.join('×')}；灰度位置核对相关性 ${r.gray_registration_correlation.toFixed(3)}。同图复核，未重新计分。`;el('regions').innerHTML=r.regions.map(g=>`<tr><td>${g.id.split('_region')[1]}</td><td>${esc(g.role_display)}</td><td>待定</td><td>否</td></tr>`).join('');}
</script></html>'''
 (ROOT/'index.html').write_text(page,encoding='utf8')
 note=f'''# 皴法角色参考与监督入口修正（2026-11-12）

旧13例测试保持7/13及混淆[[6,2],[4,1]]，未修改、未重计分。回查指出整幅画具名技法不能作为每个局部/像素的真值，部分裁剪可见结构转折、点状笔迹与规则底纹；弱墨作用仍待定。不能据此追溯改标签或自动认定哪一种纹理。

取得同4ROI的原生像素，公开查看器4268x4095，对应旧1024x982，原生区域397x439、418x460、397x355、376x418；降回旧尺寸灰度相关性.946/.944/.974/.964。读取公开查看器实际使用的DeepZoom路径，保留XML/查看器代码、{len(native['downloaded_tiles'])}个区域瓦片及哈希，无受限下载、无超分辨率、不是新测试图。

新增HKMA具名教学2例，复用AAM4和NPM具名局部2，总8原始示例。HKMA示范图作者身份不明，不能把课程主题王无邪认作图作者；所有HKMA图放同一来源组。角色与完整分笔真值仍为0。利用8个明确具名单位，仅训练冻结ImageNet特征上的C=1线性SVM头，来源组留出6/8；不能与旧13例比较，不是专家认证或新作者验收，不采用。

新模块gongbi_training/cunfa_supervision.py分离角色和具名标签：作品级标签不传播到局部；直接具名示例/来源局部或有证据专家复核才给样式目标；角色候选必须检查掩膜后才可用；unknown像素忽略而非背景，unknown样式忽略而非负类。测试核验未标像素梯度为0、改变未标logit不改变loss、空掩膜安全、自动预测/混用标签被拒绝。旧16记录全部被新严格入口挡住，历史复现实验保留。

4处共14个助手候选框，仅复核范围，不是填满同类像素的框监督。具名皴法、角色确认、完整曲线、笔顺、压力均空，不用于这轮训练。角色模型不就绪，当前已认可绘画版本不替换，未连接机械臂。

输出：{ROOT}
来源HKMA教材PDF SHA256：{read('source_manifest.json')['sha256']}
'''
 (Path(__file__).resolve().parents[1]/'research'/'cunfa_role_reference_status_20261112.md').write_text(note,encoding='utf8');print('REPORT_READY',flush=True)
if __name__=='__main__':main()
