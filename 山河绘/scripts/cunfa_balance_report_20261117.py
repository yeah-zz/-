# -*- coding: utf-8 -*-
import sys,json,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from gongbi_training.cunfa_boundary_relations import canvas_points
from gongbi_training.cunfa_context_roles import context_box
ROOT=LAB/'cunfa_boundary_balance_20261117';OLD=LAB/'cunfa_boundary_relations_20261117';PROJECT=Path(__file__).resolve().parents[1]
ARMS=['frozen_previous','old_refs_balanced','negative_refs_unbalanced','negative_refs_balanced'];TITLES=['上轮冻结联合模型','只改平衡损失','只补非轮廓点','补点＋平衡损失']
def read(p):return json.loads(Path(p).read_text('utf8'))
def main():
 r=read(ROOT/'results.json');d=read(ROOT/'reference_manifest.json');old=read(OLD/'report_data.json');z=np.load(ROOT/'dense_outputs.npz');rows={a['id']:a for a in d['records']};neg={a['id']:a for a in d['negative_references']};cases=[];(ROOT/'views').mkdir(exist_ok=True);font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
 for case,pred in zip(old['cases'],r['records']):
  assert case['id']==pred['id'];ident=case['id'];case.update(pred)
  if case['split']!='train':case['split']='known_regression'
  for k in ['input','local','near','reference','source_image']:
   if not (ROOT/case[k]).exists():shutil.copyfile(OLD/case[k],ROOT/case[k])
  if ident in neg:
   row=rows[ident];size=Image.open(row['reference_image']).size;nb=context_box(row['box'],size,2.5);im=Image.open(ROOT/case['input']).convert('RGB');pd=ImageDraw.Draw(im)
   for x,y in canvas_points(neg[ident]['negative_points'],nb,224):pd.ellipse((x-3,y-3,x+3,y+3),fill='#dd4136')
   im.save(ROOT/case['reference'])
  case['maps']={}
  for arm in ARMS:
   edge=z[arm+'_edge'][len(cases)];direction=z[arm+'_direction'][len(cases)];side=z[arm+'_side'][len(cases),0];base='views/'+ident+'_'+arm
   Image.fromarray(np.clip(edge*255,0,255).astype('uint8')).save(ROOT/(base+'_edge.png'));rgb=side[:,:,None]*np.array([40,95,215])+(1-side[:,:,None])*np.array([25,165,95]);Image.fromarray(rgb.astype('uint8')).save(ROOT/(base+'_side.png'))
   im=Image.open(ROOT/case['input']).convert('RGB');pd=ImageDraw.Draw(im)
   for yy in range(4,56,7):
    for xx in range(4,56,7):
     if edge[yy,xx]<.5:continue
     theta=.5*np.arctan2(direction[1,yy,xx],direction[0,yy,xx]);dx=7*np.cos(theta);dy=7*np.sin(theta);cx=xx*4;cy=yy*4;pd.line((cx-dx,cy-dy,cx+dx,cy+dy),fill='#d74f32',width=1)
   im.save(ROOT/(base+'_direction.png'));case['maps'][arm]={k:base+'_'+k+'.png' for k in ['edge','side','direction']}
  cases.append(case)
 data=dict(cases=cases,arms=ARMS,titles=TITLES,selected=r['selected_arm'],metrics=r['check_metrics'],aux=r['auxiliary_metrics']['seen_nizan_sparse2'],oof=r['training_oof_metrics'],oof_aux=r['training_oof_auxiliary'],selection=read(ROOT/'training_selection.json'))
 (ROOT/'report_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
 board=Image.new('RGB',(1500,900),'#f3f0e7');pd=ImageDraw.Draw(board);pd.text((12,10),'正负监督对照：原始输入及四组轮廓分数（未校准）',font=font(25),fill='#243f35')
 for j,title in enumerate(['实际输入']+TITLES):pd.text((12+j*298,55),title,font=font(18),fill='#805333')
 for i,ident in enumerate(['nizan_0','nizan_1','wanghui_2']):
  case=next(c for c in cases if c['id']==ident);y=93+i*260
  for j,path in enumerate([case['input']]+[case['maps'][a]['edge'] for a in ARMS]):
   im=Image.open(ROOT/path).convert('RGB').resize((224,224),Image.Resampling.NEAREST);board.paste(im,(12+j*298,y))
  pd.text((12,y+225),ident+' / '+('已知回归' if i<2 else '训练非轮廓参考'),font=font(17),fill='#243f35')
 board.save(ROOT/'comparison.jpg',quality=94)
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>轮廓误报修正 · 非轮廓参考与平衡损失</title><style>*{box-sizing:border-box}body{margin:0;background:#f3f0e7;color:#243f35;font:17px/1.7 "Microsoft YaHei",sans-serif}main{max-width:1500px;margin:auto;padding:24px}h1{font-size:29px}h2{font-size:23px}img{display:block;max-width:100%;height:auto}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.card{min-width:0;background:#fffdf8;padding:12px}.note{padding:15px;background:#fff8e7;border-left:5px solid #b27539}.warn{color:#99523c}.scroll{overflow:auto}td,th{padding:9px;border-bottom:1px solid #cbd4c4;text-align:left}table{width:100%;border-collapse:collapse}select{max-width:100%;font:inherit;padding:8px;margin-bottom:10px}a{color:#286d69}small{color:#7b654d}@media(max-width:750px){main{padding:14px}.grid{grid-template-columns:1fr}h1{font-size:24px}}</style><main>
<small>2026-11-17 · 离线研究对照 · 无新增独立测试作品</small><h1>补明确非轮廓点，并平衡轮廓正负监督</h1><p id="summary" class="note"></p>
<h2>这次修改</h2><p>保留49处角色训练、6个原轮廓训练窗口及全部旧目标，新增6个训练局部、18个非轮廓点，覆盖内部墨迹、连续淡墨、未着墨面各2处。红点只表示“此处不是轮廓”；有墨迹的内部点不会被标成空白，未知像素也不变背景。参考为助手稀疏核查，不是专家逐像素真值。</p><details><summary>查看新增非轮廓点（原图与红点）</summary><img src="negative_references.jpg"></details><p>三组使用相同缓存图像特征、卷积结构、初值、100轮训练与0.5固定阈值：只改损失为正负两类等权、只补非轮廓点、以及两者结合。平衡损失分别平均正点与负点的损失，再等权合并，避免标得较多的一类压过另一类。走向、两侧及角色损失不变。按整本教材／整段视频／整件作品分4组留出，训练内综合角色平均召回与轮廓平衡对应率选择候选。</p><p class="warn">倪瓒和传沈周作品均已在前轮打开，本轮全部是已知回归，没有新来源独立验收。训练参考没有加入这些检查作品，预测仍不给模型参考线、点或标签。还没有自动识别完整石块、提取真实完整笔触或学会皴法复刻。</p>
<h2>误报与漏检同时检查</h2><div class="scroll"><table><thead><tr><th>方法</th><th>已标轮廓覆盖</th><th>已标非轮廓排除</th><th>非轮廓误报</th><th>两类平均</th><th>走向误差</th></tr></thead><tbody id="aux"></tbody></table></div><p>本表只统计上一轮倪瓒2个稀疏窗口的同一批点，不是全图精度。覆盖下降就是新增漏检，不能仅以误报变少判定成功；没有在这些点上调阈值。</p><h2>角色是否受损</h2><div class="scroll"><table><thead><tr><th>方法</th><th>训练内角色平均召回</th><th>旧12处</th><th>已看传沈周6处</th><th>已看倪瓒7处</th><th>选择</th></tr></thead><tbody id="metrics"></tbody></table></div>
<h2>实际分数图对照</h2><img src="comparison.jpg" alt="原始输入与全部四组轮廓分数"><p class="warn">灰图是模型未经校准的轮廓分数，亮区不等于完整曲线；训练局部和已知回归已明确区分。没有用人工参考修饰预测，也没有将分数图转换成机械臂落笔路径。</p>
<h2>146处逐项核查</h2><select id="case"></select> <select id="model"></select><p id="scope"></p><div class="grid"><div class="card">实际图像输入<img id="input"></div><div class="card">助手稀疏参考<img id="reference"></div><div class="card">所选模型轮廓分数<img id="edge"></div><div class="card">所选模型两侧分数<img id="side"></div></div><details><summary>查看预测轴向走向；短线不是笔触路径</summary><img id="direction"></details><p id="evidence"></p><p id="refnote" class="warn">无完整对象边界或真实笔触答案；没有稀疏参考的窗口显示原图。</p><p><a id="source_link" target="_blank">原资料</a></p><table><thead><tr><th>参考与方法</th><th>区域角色</th></tr></thead><tbody id="predictions"></tbody></table><h2>训练内选中候选的全部角色错例</h2><div id="failures"></div><p>树木画法训练保持关闭。未替换认可绘画版本，未训练新皴法类别或未知拒识，未连接机械臂。</p><p><a href="reference_manifest.json">参考及分组</a> · <a href="protocol.json">冻结协议与正负数量</a> · <a href="training_selection.json">训练内选择</a> · <a href="results.json">全部输出</a> · <a href="verification.json">核验</a> · <a href="contract_checks.json">损失与标签检查</a></p></main>
<script>let D;const $=id=>document.getElementById(id),names={outline_context:'结构边',interior_mark_context:'内部墨迹',diffuse_tone:'连续淡墨面',unmarked_surface:'未着墨面'},rn=r=>names[r]||'待定／排除';function show(){let r=D.cases[$('case').selectedIndex],a=D.arms[$('model').selectedIndex];$('input').src=r.input;$('reference').src=r.reference;for(let k of ['edge','side','direction'])$(k).src=r.maps[a][k];$('scope').textContent=r.id+' / '+r.split+' / '+D.titles[D.arms.indexOf(a)];$('evidence').textContent='助手角色参考：'+r.evidence;$('source_link').href=r.source_url;$('predictions').innerHTML='<tr><td>参考</td><td>'+rn(r.role_reference)+'</td></tr>'+D.arms.map((a,i)=>'<tr><td>'+D.titles[i]+'</td><td>'+rn(r.predictions[a])+'</td></tr>').join('')}fetch('report_data.json').then(r=>r.json()).then(d=>{D=d;let a=d.selected,m=d.aux[a],old=d.aux.frozen_previous;$('summary').textContent='三组训练已完成。训练内选中：'+d.titles[d.arms.indexOf(a)]+'。已知倪瓒稀疏点：轮廓覆盖 '+(100*old.positive_recall).toFixed(1)+'% → '+(100*m.positive_recall).toFixed(1)+'%，非轮廓误报 '+(100*(1-old.negative_specificity)).toFixed(1)+'% → '+(100*(1-m.negative_specificity)).toFixed(1)+'%。仍需同时看角色与漏检；本轮没有替换绘画版本。';d.arms.forEach((a,i)=>{let m=d.aux[a],tr=document.createElement('tr');tr.innerHTML='<td>'+d.titles[i]+'</td>'+[m.positive_recall,m.negative_specificity,1-m.negative_specificity,m.balanced_boundary_accuracy].map(v=>'<td>'+(100*v).toFixed(1)+'%</td>').join('')+'<td>'+m.axial_angle_mean_degrees.toFixed(1)+'°</td>';$('aux').append(tr);tr=document.createElement('tr');tr.innerHTML='<td>'+d.titles[i]+'</td><td>'+(d.oof[a]?(100*d.oof[a].macro_recall).toFixed(1)+'%':'冻结对照')+'</td>'+['known12','seen_shen6','seen_nizan7'].map(k=>'<td>'+d.metrics[k][a].agreement+'/'+d.metrics[k][a].regions+'</td>').join('')+'<td>'+(a===d.selected?'训练内选中':'对照')+'</td>';$('metrics').append(tr);let o=document.createElement('option');o.textContent=d.titles[i];$('model').append(o)});d.cases.forEach(r=>{let o=document.createElement('option');o.textContent=r.id+' / '+r.split;$('case').append(o)});$('model').selectedIndex=d.arms.indexOf(d.selected);$('case').selectedIndex=d.cases.findIndex(r=>r.id==='nizan_0');$('case').onchange=show;$('model').onchange=show;show();d.cases.filter(r=>(r.id.startsWith('ntm_')||r.id.startsWith('shen_attributed_')||r.id.startsWith('nizan_'))&&r.role_reference!==null&&r.selected_prediction!==r.role_reference).forEach(r=>{let p=document.createElement('p');p.textContent=r.id+'：参考 '+rn(r.role_reference)+' → '+rn(r.selected_prediction)+'；'+r.evidence;$('failures').append(p)});window.reportReady=true});</script></html>'''
 (ROOT/'index.html').write_text(page,encoding='utf8');server=(PROJECT/'scripts/serve_guohua_objects_v2_20261113.py').read_text('utf8').replace('guohua_objects_v2_20261113','cunfa_boundary_balance_20261117');a=server.index('ALLOWED={');b=server.index('\nALLOWED.update',a);allowed={'index.html','comparison.jpg','negative_references.jpg','report_data.json','reference_manifest.json','protocol.json','training_selection.json','results.json','verification.json','contract_checks.json','postexperiment_checks.json'};server=server[:a]+'ALLOWED='+repr(allowed)+server[b:];server=server.replace('str(Path(__file__).resolve().parents[1])',repr(str(PROJECT)));(ROOT/'serve_report.py').write_text(server,encoding='utf8');print('REPORT_READY',len(cases))
if __name__=='__main__':main()
