# -*- coding: utf-8 -*-
import sys,json,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'cunfa_branch_isolation_20261117';OLD=LAB/'cunfa_boundary_balance_20261117';PROJECT=Path(__file__).resolve().parents[1]
ARMS=['frozen_selected','shared','detached_heads','separate_trunks'];TITLES=['上轮训练内选中','共享训练，同资料对照','阻断辅助梯度','独立辅助分支']
def read(p):return json.loads(Path(p).read_text('utf8'))
def main():
 r=read(ROOT/'results.json');d=read(ROOT/'reference_manifest.json');old=read(OLD/'report_data.json');z=np.load(ROOT/'dense_outputs.npz');cases=[];(ROOT/'views').mkdir(exist_ok=True);font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
 for case,pred in zip(old['cases'],r['records']):
  assert case['id']==pred['id'];ident=case['id'];case.update(pred)
  if case['split']!='train':case['split']='known_regression'
  for k in ['input','local','near','reference','source_image']:
   if not (ROOT/case[k]).exists():shutil.copyfile(OLD/case[k],ROOT/case[k])
  case['maps']={}
  for arm in ARMS:
   edge=z[arm+'_edge'][len(cases)];direction=z[arm+'_direction'][len(cases)];side=z[arm+'_side'][len(cases),0];base='views/'+ident+'_'+arm
   Image.fromarray(np.clip(edge*255,0,255).astype('uint8')).save(ROOT/(base+'_edge.png'));rgb=side[:,:,None]*np.array([40,95,215])+(1-side[:,:,None])*np.array([25,165,95]);Image.fromarray(rgb.astype('uint8')).save(ROOT/(base+'_side.png'));im=Image.open(ROOT/case['input']).convert('RGB');pd=ImageDraw.Draw(im)
   for yy in range(4,56,7):
    for xx in range(4,56,7):
     if edge[yy,xx]<.5:continue
     theta=.5*np.arctan2(direction[1,yy,xx],direction[0,yy,xx]);dx=7*np.cos(theta);dy=7*np.sin(theta);cx=xx*4;cy=yy*4;pd.line((cx-dx,cy-dy,cx+dx,cy+dy),fill='#d74f32',width=1)
   im.save(ROOT/(base+'_direction.png'));case['maps'][arm]={k:base+'_'+k+'.png' for k in ['edge','side','direction']}
  cases.append(case)
 gradients=read(ROOT/'gradient_diagnostics.json');data=dict(cases=cases,arms=ARMS,titles=TITLES,selected=r['selected_arm'],metrics=r['check_metrics'],aux=r['auxiliary_metrics']['seen_nizan_sparse2'],oof=r['training_oof_metrics'],oof_aux=r['training_oof_auxiliary'],selection=read(ROOT/'training_selection.json'),gradients=gradients);(ROOT/'report_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
 board=Image.new('RGB',(1500,900),'#f3f0e7');pd=ImageDraw.Draw(board);pd.text((12,10),'共享、梯度隔离和独立分支：原图与轮廓分数直接对照',font=font(25),fill='#243f35')
 for j,title in enumerate(['实际输入']+TITLES):pd.text((12+j*298,55),title,font=font(18),fill='#805333')
 for i,ident in enumerate(['nizan_0','nizan_1','wanghui_2']):
  c=next(x for x in cases if x['id']==ident);y=93+i*260
  for j,path in enumerate([c['input']]+[c['maps'][a]['edge'] for a in ARMS]):board.paste(Image.open(ROOT/path).convert('RGB').resize((224,224),Image.Resampling.NEAREST),(12+j*298,y))
  pd.text((12,y+225),ident+' / '+('已知回归' if i<2 else '训练局部'),font=font(17),fill='#243f35')
 board.save(ROOT/'comparison.jpg',quality=94)
 # Reuse the verified local report shell while replacing all experiment prose.
 page=(OLD/'index.html').read_text('utf8');start=page.index('<small>');end=page.index('<h2>误报与漏检同时检查</h2>')
 introduction='''<small>2026-11-17 · 梯度隔离对照 · 全部为已知回归</small><h1>把轮廓与角色的训练干扰单独检验</h1><p id="summary" class="note"></p><h2>本轮改了什么</h2><p>保留完全相同的146处输入、49处角色训练、6个原轮廓窗口与6个非轮廓窗口18点。三组的损失、初值、100轮训练及0.5阈值相同：继续共享；辅助头读取共享特征但阻断辅助损失向其回传；辅助任务使用独立的卷积分支。角色分支容量不变，独立分支方案额外增加参数。</p><p>共享方案精确复现上一轮“补点＋平衡损失”，用于隔离比较；“上轮训练内选中”使用原参考，单列作历史参照，不能混作同资料对照。阻断和独立分支在全部146处的角色预测均精确等于原只训练角色的模型，这是防止辅助损失改写角色分支的验证，不是新增自主辨识能力。</p><p>按整本教材、整段视频、整件作品分4组留出，仍用训练内角色平均召回与轮廓平衡对应率选择。没有用检查作品调阈值、选模型或补标签。参考仍是助手稀疏核查，全部检查图都已查看，<b>没有新来源独立验收</b>。</p><h2>邻近墨迹复核</h2><img src="pair_review.jpg" alt="同一局部的淡轮廓及邻近内部线候选核查"><p class="warn">定位框为助手观察，非模型输出；内侧线可能是结构转折，内部短墨迹可能含点苔或植被，不能仅按几何位置强制标成皴笔或删线。本轮只记录待定，未把这些事后观察接入训练。</p>'''
 page=page[:start]+introduction+page[end:]
 a=page.index('<h2>实际分数图对照</h2>');b=page.index('<h2>146处逐项核查</h2>')
 section='''<h2>梯度诊断</h2><p>下表是已拟合共享方案的来源批次中，角色损失与辅助损失对共享层梯度的余弦。负值表示该批次方向相反；全训练集合却为正，不能据此断言所有更新冲突或已找到全部失败原因。阻断方案辅助梯度为零，软件检查通过。</p><table><thead><tr><th>来源批次</th><th>角色梯度范数</th><th>辅助梯度范数</th><th>余弦</th></tr></thead><tbody id="gradients"></tbody></table><h2>实际分数图对照</h2><img src="comparison.jpg" alt="原始输入、历史候选及三种训练路由输出"><p class="warn">亮区只是未校准轮廓分数，不等于完整曲线。独立分支误报更少时也可能漏掉大多数轮廓，不能只看误报下降。本轮没有超过现有方案的整体能力，未替换绘画版本。</p>'''
 page=page[:a]+section+page[b:]
 page=page.replace('<table><thead><tr><th>来源批次','<div class="scroll"><table><thead><tr><th>来源批次').replace('<tbody id="gradients"></tbody></table>','<tbody id="gradients"></tbody></table></div>')
 # Replace the old summary expression, retaining the checked controls/tables.
 oldsummary="'三组训练已完成。训练内选中：'+d.titles[d.arms.indexOf(a)]+'。已知倪瓒稀疏点：轮廓覆盖 '+(100*old.positive_recall).toFixed(1)+'% → '+(100*m.positive_recall).toFixed(1)+'%，非轮廓误报 '+(100*(1-old.negative_specificity)).toFixed(1)+'% → '+(100*(1-m.negative_specificity)).toFixed(1)+'%。仍需同时看角色与漏检；本轮没有替换绘画版本。'"
 newsummary="'三组隔离对照已完成，训练内选中：'+d.titles[d.arms.indexOf(a)]+'。已知倪瓒角色 '+d.metrics.seen_nizan7[a].agreement+'/7；稀疏轮廓覆盖 '+(100*m.positive_recall).toFixed(1)+'%，非轮廓误报 '+(100*(1-m.negative_specificity)).toFixed(1)+'%。恢复角色基线仍不代表轮廓通过；本轮没有替换绘画版本。'"
 assert oldsummary in page;page=page.replace(oldsummary,newsummary).replace('d.aux.frozen_previous','d.aux.frozen_selected')
 page=page.replace("window.reportReady=true", "Object.entries(d.gradients.shared_fitted).forEach(([g,m])=>{let tr=document.createElement('tr');tr.innerHTML='<td>'+g+'</td><td>'+m.role_gradient_norm.toFixed(3)+'</td><td>'+m.auxiliary_gradient_norm.toFixed(3)+'</td><td>'+(m.cosine===null?'无辅助监督':m.cosine.toFixed(3))+'</td>';$('gradients').append(tr)});window.reportReady=true")
 page=page.replace('轮廓误报修正 · 非轮廓参考与平衡损失','角色与轮廓 · 共享梯度隔离对照');page=page.replace('<a href="contract_checks.json">损失与标签检查</a>','<a href="contract_checks.json">梯度路由检查</a> · <a href="gradient_diagnostics.json">梯度诊断</a> · <a href="pair_review.json">邻近线复核</a>');(ROOT/'index.html').write_text(page,encoding='utf8')
 server=(PROJECT/'scripts/serve_guohua_objects_v2_20261113.py').read_text('utf8').replace('guohua_objects_v2_20261113','cunfa_branch_isolation_20261117');a=server.index('ALLOWED={');b=server.index('\nALLOWED.update',a);allowed={'index.html','comparison.jpg','pair_review.jpg','pair_review.json','gradient_diagnostics.json','report_data.json','reference_manifest.json','protocol.json','training_selection.json','results.json','verification.json','contract_checks.json','postexperiment_checks.json'};server=server[:a]+'ALLOWED='+repr(allowed)+server[b:];server=server.replace('str(Path(__file__).resolve().parents[1])',repr(str(PROJECT)));(ROOT/'serve_report.py').write_text(server,encoding='utf8');print('REPORT_READY',len(cases))
if __name__=='__main__':main()
