# -*- coding: utf-8 -*-
import sys,json,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'cunfa_relational_roles_20261117';PROJECT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text('utf8'))
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 ds=read(ROOT/'reference_manifest.json');result=read(ROOT/'results.json');pred={r['id']:r for r in result['records']};cases=[]
 for r in ds['records']:
  dest=ROOT/'views'/(r['id']+'_source.png');shutil.copyfile(r['image'],dest)
  cases.append(dict(id=r['id'],source='views/'+dest.name,local='views/'+r['id']+'_local.png',near='views/'+r['id']+'_near.png',far='views/'+r['id']+'_far.png',
   split=r['split'],scope=r['evaluation_scope'],added=r.get('added_this_round',False),source_url=r['source_url'],
   object_type=r['object_type'],visual_evidence=r['visual_evidence'],role_reference=pred[r['id']]['role_reference'],
   predictions=pred[r['id']]['predictions'],selected_prediction=pred[r['id']]['selected_prediction'],fold_group=r['role_fold_group']))
 write('report_data.json',dict(cases=cases,selected=result['selected_arm'],metrics=result['known_check_metrics'],oof=result['training_oof_metrics'],
  selection=read(ROOT/'training_selection.json')))
 font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
 labels={'outline_context':'结构边','interior_mark_context':'内部墨迹','diffuse_tone':'连续淡墨面','unmarked_surface':'未着墨面'}
 selected=['ntm_xia_region_2','ntm_huang_region_6','ntm_mi_region_0','ntm_xia_region_3','ntm_xia_region_0']
 board=Image.new('RGB',(1500,620),'#f3f0e7');d=ImageDraw.Draw(board)
 d.text((15,12),'结构边与淡墨：找回的位置、代价和未通过的候选一并保留',font=font(27),fill='#243f35')
 for i,ident in enumerate(selected):
  r=next(x for x in cases if x['id']==ident);im=Image.open(ROOT/r['source']);native=im.size;s=min(270/im.width,175/im.height);im=im.resize((max(1,round(im.width*s)),max(1,round(im.height*s))),Image.Resampling.NEAREST);x=i*300+12;board.paste(im,(x,65))
  d.text((x,252),'原生 '+str(native[0])+'×'+str(native[1])+'，仅展示缩放',font=font(15),fill='#805333')
  d.text((x,285),ident,font=font(15),fill='#243f35');d.text((x,318),'参考：'+labels[r['role_reference']],font=font(19),fill='#805333')
  for y,key,title in [(352,'frozen_previous','上轮'),(386,'expanded_flat_spatial','选中候选'),(420,'expanded_hierarchical','两阶段'),(454,'expanded_hierarchical_rotated','旋转训练')]:
   d.text((x,y),title+'：'+labels[r['predictions'][key]],font=font(18),fill='#286e7c')
 d.text((15,515),'原12处已知参考：上轮7/12；训练内选中候选6/12。找回一处结构边，也出现新的误判。',font=font(21),fill='#994d38')
 d.text((15,560),'这些是给定区域的角色判断，非完整笔触提取、自动石块识别或机械臂实画。',font=font(21),fill='#243f35')
 board.save(ROOT/'comparison.jpg',quality=94)
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>结构边与淡墨 · 补参考、空间关系与两阶段对照</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f0e7;color:#243f35;font:17px/1.65 "Microsoft YaHei",sans-serif}main{max-width:1500px;margin:auto;padding:24px}h1{font-size:29px}.note{padding:16px;border-left:5px solid #b77535;background:#fff8e7}.warn{color:#994c37}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px}.card{min-width:0;padding:11px;background:#fffdf8;border:1px solid #ced6c6}.scroll{overflow:auto}table{width:100%;border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #cbd4c4;text-align:left}img{max-width:100%;height:auto;display:block}select{font:inherit;padding:8px;max-width:100%;width:760px}a{color:#286d69}small{color:#7c654c}@media(max-width:750px){main{padding:14px}.grid{grid-template-columns:1fr}h1{font-size:24px}}</style><main>
<small>2026-11-17 · 给定区域的角色研究 · 已知案例开发回归</small><h1>补长结构边参考，比较空间关系与两阶段判断</h1>
<p class="note"><b>本轮新增15处可用角色参考，完成4组训练；整体仍未通过。</b>38处训练参考内部选中“大窗口＋空间位置”方案，在原12处检查上6/12，低于上轮7/12。结构边找回1处（0/2→1/2），但内部墨迹和留白退步。两阶段方案找对1处淡墨，却损失其他角色，没有替换绘画版本。</p>
<h2>1. 先补数据，而不是把多训练次数当进展</h2><p>上轮6处结构边训练参考多为小石块边线，淡墨参考也集中在同一张坡面图。本轮复用已有视频中勾结构／加皴阶段的静态画面，再补同一教材的高低石坡图。19个候选中15处接入角色训练，4处因夹有细线或归属不明保留待定。没有新增独立作品、作者或专家标签。</p><p>视频的先后画面只用来核查参考，模型不读取前帧、时间戳、差分或参考掩膜；同一视频的全部阶段和裁剪归在同一个训练留出组。石坡图仍与原青浦教材同源。角色训练23→38处；检查仍是原美术馆12处已打开参考，并非新来源独立验收。</p><details><summary>查看19个补充候选：15处用于角色监督，4处待定</summary><img src="added_references.jpg"></details>
<h2>2. 四个候选具体改变什么</h2><p>一组只增加参考、保持原近邻输入；一组增加6倍窗口，并保留灰度变化在3×3区域里的位置；一组先判断有明显墨迹／连续淡墨／未着墨，再对有墨迹分结构边和内部墨迹；最后一组加入90、180、270度旋转训练。旋转视图与原图共享来源组，每个原参考的总损失权重仍为1，不把旋转当新增作品。</p><p class="warn">6倍窗口只是更大的给定画面，不等于自动识别到完整石头。结构边参考包含外缘和已核查的山面转折，目前还没有恢复完整笔画归属、曲线或笔顺；两阶段分数也未经可靠概率校准。</p>
<div class="scroll"><table><thead><tr><th>方案</th><th>训练内平均类别召回</th><th>原12处角色对应</th><th>检查平均类别召回</th><th>状态</th></tr></thead><tbody id="metrics"></tbody></table></div><p>训练内选择要求每类召回至少0.5，再按平均类别召回选择；选择记录先于检查指标保存。选中候选并未在旧检查上取得整体改善，不能事后改选最有利的一列。内部留出仍非作者／来源独立测试。</p>
<h2>3. 主轮廓找回与其他退步一起看</h2><img src="comparison.jpg" alt="原始局部与上轮、选中候选、两阶段和旋转训练的真实预测"><div class="scroll"><table><thead><tr><th>角色</th><th>参考数</th><th>上轮</th><th>补参考近邻</th><th>训练内选中</th><th>两阶段</th><th>两阶段旋转</th></tr></thead><tbody id="roles"></tbody></table></div><p>近景石头顶边被找回，大崖右侧边仍漏判；旋转训练虽找回大崖边，却又漏掉近景顶边。两阶段找对一处连续淡墨，但把若干内部点状墨迹误判成淡墨。不同方案各有取舍，不能把每列最好的一项拼成一个不存在的成功模型。</p>
<h2>4. 112处视图逐张检查</h2><select id="case"></select><p id="scope"></p><div class="grid"><div class="card"><b>原始局部</b><img id="source"></div><div class="card"><b>局部实际输入</b><img id="local"></div><div class="card"><b>2.5倍邻域</b><img id="near"></div><div class="card"><b>6倍窗口</b><img id="far"></div></div><p id="evidence"></p><p id="refnote" class="warn">无完整对象边界或逐笔真值；给定窗口与对象参考不当作模型已经学会的能力。</p><p><a id="source_link" target="_blank">原资料出处</a></p><div class="scroll"><table><thead><tr><th>参考与模型</th><th>区域角色</th></tr></thead><tbody id="predictions"></tbody></table></div>
<h2>5. 选中候选的全部检查错例</h2><div id="failures"></div><p>原始轨迹和已认可整幅版本继续保留。区域角色探针没有进入线条删除或机械臂落笔决策，不能因某处被判为淡墨／留白就自动删掉原线。皴法类别和范围外拒识头本轮没有重训；上一轮拒识失败仍成立。树木画法训练保持关闭，本轮未连接机械臂。</p><p>后续应补结构边与内部皴笔在同一石块中共存的可靠参考，并区分边界、内部线带与连续淡墨的墨带范围；这些证据需要支持实际分笔，而不只给局部名称。当前少量助手参考和跨来源不稳定性仍是限制。</p>
<p><a href="reference_manifest.json">参考与分组</a> · <a href="protocol.json">冻结实验协议</a> · <a href="training_selection.json">训练内选择</a> · <a href="results.json">全部直接输出</a> · <a href="verification.json">验证记录</a> · <a href="contract_checks.json">来源、旋转位置与待定目标检查</a></p></main>
<script>let D;const $=id=>document.getElementById(id),roles=['outline_context','interior_mark_context','diffuse_tone','unmarked_surface'],rnames=['结构边上下文','内部墨迹上下文','连续淡墨面','未着墨面'],rn=r=>r===null?'待定':rnames[roles.indexOf(r)],arms=['frozen_previous','expanded_flat_near','expanded_flat_spatial','expanded_hierarchical','expanded_hierarchical_rotated'],titles=['上轮冻结模型','补参考＋原近邻输入','大窗口＋空间位置','两阶段判断','两阶段＋旋转训练'];function show(){let r=D.cases[$('case').selectedIndex];for(let k of ['source','local','near','far'])$(k).src=r[k];$('scope').textContent=r.id+' / '+r.split+' / '+r.scope+' / 留出组：'+r.fold_group;$('evidence').textContent='助手依据：'+r.visual_evidence;$('source_link').href=r.source_url;$('predictions').innerHTML='<tr><td>参考</td><td>'+rn(r.role_reference)+'</td></tr>'+arms.map((a,i)=>'<tr><td>'+titles[i]+'</td><td>'+rn(r.predictions[a])+'</td></tr>').join('');}fetch('report_data.json').then(r=>r.json()).then(d=>{D=d;d.cases.forEach(r=>{let o=document.createElement('option');o.textContent=r.id+' / '+r.split;$('case').append(o)});$('case').onchange=show;show();arms.forEach((a,i)=>{let m=d.metrics[a],o=d.oof[a],tr=document.createElement('tr');tr.innerHTML='<td>'+titles[i]+'</td><td>'+(o?(100*o.macro_recall).toFixed(1)+'%':'不同训练集，不比较')+'</td><td>'+m.agreement+'/'+m.regions+'</td><td>'+(100*m.macro_recall).toFixed(1)+'%</td><td>'+(a===d.selected?'训练内选中':'保留对照')+'</td>';$('metrics').append(tr)});roles.forEach((k,j)=>{let tr=document.createElement('tr');tr.innerHTML='<td>'+rnames[j]+'</td><td>'+d.metrics.frozen_previous.per_role[k].count+'</td>'+arms.map(a=>'<td>'+d.metrics[a].per_role[k].agreement+'</td>').join('');$('roles').append(tr)});d.cases.filter(r=>r.split==='test'&&r.role_reference!==null&&r.selected_prediction!==r.role_reference).forEach(r=>{let p=document.createElement('p');p.textContent=r.id+'：参考 '+rn(r.role_reference)+' → 选中候选 '+rn(r.selected_prediction)+'；'+r.visual_evidence;$('failures').append(p)});window.reportReady=true;});</script></html>'''
 (ROOT/'index.html').write_text(page,encoding='utf8')
 server=(PROJECT/'scripts/serve_guohua_objects_v2_20261113.py').read_text('utf8').replace('guohua_objects_v2_20261113','cunfa_relational_roles_20261117')
 a=server.index('ALLOWED={');b=server.index('\nALLOWED.update',a)
 allowed={'index.html','comparison.jpg','added_references.jpg','report_data.json','reference_manifest.json','protocol.json','training_selection.json','results.json','verification.json','contract_checks.json','postexperiment_checks.json'}
 server=server[:a]+'ALLOWED='+repr(allowed)+server[b:];server=server.replace('str(Path(__file__).resolve().parents[1])',repr(str(PROJECT)))
 (ROOT/'serve_report.py').write_text(server,encoding='utf8');print('REPORT_READY',len(cases),flush=True)
if __name__=='__main__':main()
