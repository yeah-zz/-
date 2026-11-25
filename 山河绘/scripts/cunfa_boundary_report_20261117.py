# -*- coding: utf-8 -*-
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from gongbi_training.cunfa_photometry import appearance_canvas
from gongbi_training.cunfa_context_roles import context_box
from gongbi_training.cunfa_boundary_relations import canvas_points
ROOT=LAB/'cunfa_boundary_relations_20261117';PROJECT=Path(__file__).resolve().parents[1]
ARMS=['frozen_expanded_global','frozen_expanded_spatial_pyramid','role_only','boundary_relations'];TITLES=['冻结全局候选','冻结空间候选','同结构仅角色','加入边界／走向／两侧监督']
LABEL={'outline_context':'结构边','interior_mark_context':'内部墨迹','diffuse_tone':'连续淡墨面','unmarked_surface':'未着墨面',None:'待定／排除'}
def read(p):return json.loads(Path(p).read_text('utf8'))
def main():
 d=read(ROOT/'reference_manifest.json');r=read(ROOT/'results.json');z=np.load(ROOT/'dense_outputs.npz');curves={c['id']:c for c in d['curves']};sources={};cases=[];refs={};font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
 for i,(row,pred) in enumerate(zip(d['records'],r['records'])):
  assert row['id']==pred['id'];ident=row['id'];key=row['reference_image']
  if key not in sources:
   ref=Image.open(key).convert('RGB');refs[key]=ref;native=ref.size;ref1=ref.copy();ref1.thumbnail((1000,1000));name='views/source_'+str(len(sources))+'.jpg';ref1.save(ROOT/name,quality=92);sources[key]=(name,native)
  ref=refs[key];nb=context_box(row['box'],ref.size,2.5);inp=appearance_canvas(ref.crop(nb),ref,'reference_linear');inp.save(ROOT/'views'/(ident+'_input.png'))
  edge=Image.fromarray(np.clip(z['boundary_relations_edge'][i]*255,0,255).astype('uint8'));edge.save(ROOT/'views'/(ident+'_edge.png'))
  directions=inp.copy();dd=ImageDraw.Draw(directions);ang=z['boundary_relations_direction'][i]
  for yy in range(4,56,7):
   for xx in range(4,56,7):
    if z['boundary_relations_edge'][i,yy,xx]<.5:continue
    theta=.5*np.arctan2(ang[1,yy,xx],ang[0,yy,xx]);dx=7*np.cos(theta);dy=7*np.sin(theta);cx=xx*4;cy=yy*4;dd.line((cx-dx,cy-dy,cx+dx,cy+dy),fill='#d74f32',width=1)
  directions.save(ROOT/'views'/(ident+'_direction.png'))
  side=z['boundary_relations_side'][i,0];rgb=(side[:,:,None]*np.array([40,95,215])+(1-side[:,:,None])*np.array([25,165,95])).astype('uint8');Image.fromarray(rgb).save(ROOT/'views'/(ident+'_side.png'))
  annotated=inp.copy();pd=ImageDraw.Draw(annotated)
  if ident in curves:
   c=curves[ident];pd.line([tuple(x) for x in canvas_points(c['line'],nb,224)],fill='#ed5736',width=2)
   for k,col in [('rock','#2860df'),('open','#19a55f')]:
    for x,y in canvas_points(c[k],nb,224):pd.ellipse((x-3,y-3,x+3,y+3),fill=col)
  annotated.save(ROOT/'views'/(ident+'_reference.png'))
  name,native=sources[key];cases.append(dict(**pred,source_image=name,native_size=native,box=row['box'],input='views/'+ident+'_input.png',local='views/'+ident+'_local.png',near='views/'+ident+'_near.png',reference='views/'+ident+'_reference.png',edge='views/'+ident+'_edge.png',side='views/'+ident+'_side.png',direction='views/'+ident+'_direction.png',evidence=row['visual_evidence'],source_url=row['source_url'],object_type=row['object_type'],group=row['role_fold_group'],has_sparse_reference=ident in curves))
 data=dict(cases=cases,selected=r['selected_arm'],metrics=r['check_metrics'],oof=r['training_oof_metrics'],aux=r['auxiliary_metrics'],arms=ARMS,titles=TITLES);(ROOT/'report_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
 board=Image.new('RGB',(1500,790),'#f3f0e7');pd=ImageDraw.Draw(board);pd.text((15,10),'轮廓与两侧关系：稀疏参考和模型直接输出分开显示',font=font(27),fill='#243f35')
 for i,ident in enumerate(['nizan_0','nizan_1','nizan_5']):
  case=next(x for x in cases if x['id']==ident);y=65+i*225
  for j,k in enumerate(['input','reference','edge','side']):
   im=Image.open(ROOT/case[k]).convert('RGB').resize((210,210),Image.Resampling.NEAREST);board.paste(im,(12+j*230,y))
  pd.text((945,y+5),ident+' / 新作品检查',font=font(20),fill='#243f35');pd.text((945,y+40),'参考：'+LABEL[case['role_reference']],font=font(20),fill='#805333')
  for j,a in enumerate(ARMS):pd.text((945,y+78+j*29),TITLES[j]+'：'+LABEL[case['predictions'][a]],font=font(17),fill='#286e7c')
 for j,title in enumerate(['实际图像输入','助手参考；非模型输出','预测轮廓概率；未校准','预测两侧；蓝山面／绿空景']):pd.text((12+j*230,750),title,font=font(15),fill='#805333')
 board.save(ROOT/'comparison.jpg',quality=94)
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>轮廓走向与两侧关系 · 训练对照</title><style>*{box-sizing:border-box}body{margin:0;background:#f3f0e7;color:#243f35;font:17px/1.7 "Microsoft YaHei",sans-serif}main{max-width:1500px;margin:auto;padding:24px}h1{font-size:29px}h2{font-size:23px}img,canvas{display:block;max-width:100%;height:auto}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.card{padding:12px;background:#fffdf8;min-width:0}.note{padding:15px;border-left:5px solid #b27539;background:#fff8e7}.warn{color:#98513b}.scroll{overflow:auto}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #cbd4c4;text-align:left}select{font:inherit;padding:8px;max-width:100%;width:800px}a{color:#286d69}small{color:#7c654c}@media(max-width:750px){main{padding:14px}.grid{grid-template-columns:1fr}h1{font-size:24px}}</style><main>
<small>2026-11-17 · 离线研究 · 本轮没有替换实画版本</small><h1>从给局部分类，推进到稀疏轮廓、走向和两侧关系监督</h1><p class="note" id="summary"></p>
<h2>这次训练了什么</h2><p>保留49处角色训练参考，新增6个训练窗口的粗轮廓走向参考，其中5个同时有山面与空景两侧点。参考来自既有教材及王翬作品；同一本青浦教材全部并为一组，视频所有帧同组，两件博物馆作品各自成组。训练内依次留出这4个完整组选择候选，未用旧错例或新作品挑参数。</p><p>冻结图像编码器前两段，保留56×56特征图，接入可训练卷积头。两组结构、初值、100轮训练、角色资料完全一致；对照组只学角色，实验组另学轮廓附近范围、轴向走向与两侧点。未知像素不当背景。输入仍是给定2.5倍邻域及窗口位置，并非自动发现整块石头；预测时不提供参考线、点、角色标签或阶段差分。</p><p class="warn">橙线只是助手核查的粗定位与走向，蓝点为山面、绿点为空景；不是精确墨带边界或画家真实分笔。一处左侧仍是淡山面的区域取消了空景点，避免把山体误教成纸面。表格参考均非专家真值。</p>
<div class="scroll"><table><thead><tr><th>模型</th><th>训练内类别平均召回</th><th>旧12处</th><th>已看沈周候选6处</th><th>新倪瓒7处</th><th>选择</th></tr></thead><tbody id="metrics"></tbody></table></div>
<h2>新作品检查</h2><p>倪瓒《虞山林壑图》整件在下载前固定为本轮新检查作品，身份及原图来自大都会博物馆。选择9处候选，1处灰面近看有细线而保持待定，树叶排除山石监督；1处纸面候选边缘含印章，预测前调整位置。留下7处角色参考，含1处连续灰调。新作品的2条粗轮廓与两侧点仅作评分，不参与训练。仍来自与部分开发资料相同的博物馆，未完成全项目作品近重复核查，也不是专家或已认证作者独立验收。</p><div class="scroll"><table><thead><tr><th>新作品角色</th><th>数量</th><th>冻结全局</th><th>冻结空间</th><th>仅角色</th><th>加关系监督</th></tr></thead><tbody id="roles"></tbody></table></div>
<h2>原图、参考与直接输出</h2><img src="comparison.jpg" alt="新作品原始输入、助手粗线点参考与模型直接概率输出"><p class="warn">灰图为预测轮廓分数，蓝绿图为两侧分数，均未概率校准；全图着色包含无监督区域的外推，不是已验证分区，不是可执行轨迹，也不表示恢复了完整笔触。</p><div class="scroll"><table><thead><tr><th>稀疏辅助检查</th><th>窗口数</th><th>轮廓参考覆盖</th><th>非轮廓点排除</th><th>两侧点平均对应率</th><th>走向平均误差</th></tr></thead><tbody id="aux"></tbody></table></div><p>辅助指标只在橙线附近与已标点上计算，无法代表完整边界精度、整图IoU或真实分笔。仅角色组没有训练辅助头，不对其随机辅助输出作比较。</p>
<h2>146处逐项核查</h2><select id="case"></select><p id="scope"></p><div class="grid"><div class="card">实际图像输入<img id="input"></div><div class="card">助手稀疏参考<img id="reference"></div><div class="card">模型轮廓分数<img id="edge"></div><div class="card">模型两侧分数<img id="side"></div></div><p id="evidence"></p><p id="refnote" class="warn">无完整对象边界或真实笔触答案；没有稀疏参考的窗口第二栏显示原输入。</p><p><a id="source_link" target="_blank">原始资料</a></p><table><thead><tr><th>参考及方法</th><th>角色</th></tr></thead><tbody id="predictions"></tbody></table>
<h2>选中候选的全部角色错例</h2><div id="failures"></div><p>树木画法训练保持关闭。皴法类别及未知拒识本轮没有重训；这些区域角色与稀疏边界输出不证明系统理解了皴法或具备笔组生成能力。原始轨迹与认可整幅版本保留，本轮未连接机械臂。</p><p><a href="reference_manifest.json">冻结参考与修改记录</a> · <a href="protocol.json">协议及训练代码哈希</a> · <a href="training_selection.json">训练内选择</a> · <a href="results.json">全部预测</a> · <a href="verification.json">核验</a> · <a href="source_inventory.json">新作品馆藏来源</a></p></main>
<script>let D;const $=id=>document.getElementById(id),names={outline_context:'结构边',interior_mark_context:'内部墨迹',diffuse_tone:'连续淡墨面',unmarked_surface:'未着墨面'},rn=r=>names[r]||'待定／排除';function show(){let r=D.cases[$('case').selectedIndex];for(let k of ['input','reference','edge','side'])$(k).src=r[k];$('scope').textContent=r.id+' / '+r.split+' / 作品组：'+r.group+' / 稀疏参考：'+r.has_sparse_reference;$('evidence').textContent='助手依据：'+r.evidence;$('source_link').href=r.source_url;$('predictions').innerHTML='<tr><td>参考</td><td>'+rn(r.role_reference)+'</td></tr>'+D.arms.map((a,i)=>'<tr><td>'+D.titles[i]+'</td><td>'+rn(r.predictions[a])+'</td></tr>').join('')}fetch('report_data.json').then(r=>r.json()).then(d=>{D=d;let a=d.selected,k=d.metrics.known12[a],s=d.metrics.seen_shen6[a],n=d.metrics.fresh_nizan7[a];$('summary').textContent='训练内选中：'+d.titles[d.arms.indexOf(a)]+'；旧12处 '+k.agreement+'/'+k.regions+'，已看作品 '+s.agreement+'/'+s.regions+'，新作品 '+n.agreement+'/'+n.regions+'。两组训练已完成；这些仍是助手参考上的探索结果，没有通过整体能力验收。';d.arms.forEach((a,i)=>{let tr=document.createElement('tr'),o=d.oof[a];tr.innerHTML='<td>'+d.titles[i]+'</td><td>'+(o?(100*o.macro_recall).toFixed(1)+'%':'冻结对照')+'</td>'+['known12','seen_shen6','fresh_nizan7'].map(s=>'<td>'+d.metrics[s][a].agreement+'/'+d.metrics[s][a].regions+'</td>').join('')+'<td>'+(a===d.selected?'训练内选中':'对照')+'</td>';$('metrics').append(tr)});Object.keys(names).forEach(role=>{let tr=document.createElement('tr'),n=d.metrics.fresh_nizan7[d.arms[0]].per_role[role].count;tr.innerHTML='<td>'+rn(role)+'</td><td>'+n+'</td>'+d.arms.map(a=>'<td>'+d.metrics.fresh_nizan7[a].per_role[role].agreement+'</td>').join('');$('roles').append(tr)});for(let s of ['train','fresh_check']){let m=d.aux[s],tr=document.createElement('tr');tr.innerHTML='<td>'+(s==='train'?'训练窗口，非泛化':'新作品稀疏参考')+'</td><td>'+m.windows+'</td>'+['boundary_positive_recall','boundary_negative_specificity','side_balanced_accuracy'].map(k=>'<td>'+(100*m[k]).toFixed(1)+'%</td>').join('')+'<td>'+m.axial_angle_mean_degrees.toFixed(1)+'°</td>';$('aux').append(tr)}d.cases.forEach(r=>{let o=document.createElement('option');o.textContent=r.id+' / '+r.split;$('case').append(o)});$('case').onchange=show;$('case').selectedIndex=d.cases.findIndex(r=>r.id==='nizan_0');show();d.cases.filter(r=>(r.id.startsWith('ntm_')||r.id.startsWith('shen_attributed_')||r.id.startsWith('nizan_'))&&r.role_reference!==null&&r.selected_prediction!==r.role_reference).forEach(r=>{let p=document.createElement('p');p.textContent=r.id+'：参考 '+rn(r.role_reference)+' → '+rn(r.selected_prediction)+'；'+r.evidence;$('failures').append(p)});window.reportReady=true});</script></html>'''
 page=page.replace('<h2>原图、参考与直接输出</h2>','<p class="warn"><b>本轮关系监督仍未通过。</b>新作品轮廓参考附近覆盖约96%，但非轮廓点排除仅约5%，说明大量非轮廓也被报成轮廓。已知沈周候选角色4/6→5/6，新作品整体未优于冻结空间模型。</p><h2>原图、参考与直接输出</h2>')
 page=page.replace('<p id="evidence"></p>','<details><summary>查看模型轴向走向场：橙色短线是预测方向，非笔触轨迹</summary><img id="direction"></details><p id="evidence"></p>')
 page=page.replace("['input','reference','edge','side']","['input','reference','edge','side','direction']")
 (ROOT/'index.html').write_text(page,encoding='utf8')
 server=(PROJECT/'scripts/serve_guohua_objects_v2_20261113.py').read_text('utf8').replace('guohua_objects_v2_20261113','cunfa_boundary_relations_20261117');a=server.index('ALLOWED={');b=server.index('\nALLOWED.update',a);allowed={'index.html','comparison.jpg','report_data.json','reference_manifest.json','protocol.json','training_selection.json','results.json','verification.json','source_inventory.json','contract_checks.json','postexperiment_checks.json'};server=server[:a]+'ALLOWED='+repr(allowed)+server[b:];server=server.replace('str(Path(__file__).resolve().parents[1])',repr(str(PROJECT)));(ROOT/'serve_report.py').write_text(server,encoding='utf8');print('REPORT_READY',len(cases))
if __name__=='__main__':main()
