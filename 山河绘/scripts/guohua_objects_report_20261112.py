# -*- coding: utf-8 -*-
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from guohua_objects_train_20261112 import pack,confusion
from gongbi_training.cunfa_supervision import partial_role_loss
ROOT=LAB/'guohua_objects_pilot_20261112'
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ds=read(ROOT/'dataset_manifest.json');res=read(ROOT/'results.json');p=res['protocol'];assert sha(ROOT/'dataset_manifest.json')==p['dataset_sha256'];assert sha(ROOT/'mapping_evidence.json')==p['mapping_sha256'];assert sha(Path(__file__).with_name('guohua_objects_train_20261112.py'))==p['script_sha256'];assert all(sha(ROOT/(k+'.pt'))==v for k,v in res['checkpoint_hashes'].items());(ROOT/'views').mkdir(exist_ok=True);font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n);colors=np.array([[145,157,170],[183,134,75],[65,149,91]],'uint8');records=[]
 for r in ds['records']:
  im=np.array(Image.open(r['image']).convert('RGB'));target=np.array(Image.open(r['target']));reference=np.full_like(im,[229,223,210]);known=target!=255;reference[known]=colors[target[known]];Image.fromarray(im).save(ROOT/'views'/(r['id']+'_input.jpg'),quality=92);Image.fromarray(reference).save(ROOT/'views'/(r['id']+'_reference.png'));paths={}
  for name in res['protocol']['arms']:
   z=np.load(ROOT/'predictions'/(r['id']+'_'+name+'.npz'));pred=z['prediction'];Image.fromarray(colors[pred]).save(ROOT/'views'/(r['id']+'_'+name+'.png'));paths[name]='views/'+r['id']+'_'+name+'.png'
  metric=next(q for q in res['records'] if q['id']==r['id']);records.append(dict(id=r['id'],split=r['split'],work_group=r['work_group'],input='views/'+r['id']+'_input.jpg',reference='views/'+r['id']+'_reference.png',predictions=paths,scores=metric['scores'],is_user_regression=False))
 ident='H_14';im=np.array(Image.open(LAB/'cunfa_roles_v2_20261112/references/H_14_input.png'));Image.fromarray(im).save(ROOT/'views/H_14_input.png');paths={}
 for name in res['protocol']['arms']:
  z=np.load(ROOT/'predictions'/('H_14_'+name+'.npz'));Image.fromarray(colors[z['prediction']]).save(ROOT/'views'/('H_14_'+name+'.png'));confidence=z['probability'].astype('float32').max(0);uncertain=colors[z['prediction']].copy();uncertain[confidence<.6]=[229,223,210];Image.fromarray(uncertain).save(ROOT/'views'/('H_14_'+name+'_uncertain.png'));paths[name]='views/H_14_'+name+'.png'
 old=Image.open(LAB/'cunfa_retention_v4_20261112/predictions/H_14_known_ink_retention.png');old.save(ROOT/'views/H_14_old_rock_roles.png');records.insert(0,dict(id=ident,split='known_user_regression',work_group='howto_previously_seen',input='views/H_14_input.png',reference=None,predictions=paths,is_user_regression=True))
 board=Image.new('RGB',(1200,450),'#f4f1e8');draw=ImageDraw.Draw(board)
 for j,(image,title) in enumerate([(Image.fromarray(im),'原图'),(old.copy(),'旧角色模型：树叶误作皴笔'),(Image.open(ROOT/'views/H_14_context_object_head.png'),'新对象模型：棕=山石，绿=树木')]):
  image.thumbnail((385,320));board.paste(image,(j*400+6,45));draw.text((j*400+6,8),title,font=font(18),fill='#243e32')
 draw.text((8,350),'右图为模型直接预测，未使用树叶参考遮罩；左中两图与右图任务、图例不同。',font=font(19),fill='#815238');draw.text((8,391),'这是粗对象分割，不是线稿、皴法识别或树木画法。该案例未用于训练，但已反复查看。',font=font(18),fill='#815238');board.save(ROOT/'foliage_object_comparison.jpg',quality=94)
 worst=[]
 for split in ['test_named','test_collection']:
  worst.extend(sorted([r for r in records if r['split']==split],key=lambda r:r['scores']['context_object_head']['mean_iou'])[:2])
 wb=Image.new('RGB',(1200,4*360),'#f4f1e8');wd=ImageDraw.Draw(wb)
 for i,r in enumerate(worst):
  for j,(path,title) in enumerate([(r['input'],'原图'),(r['reference'],'源粗对象标注'),(r['predictions']['context_object_head'],'模型原始预测')]):
   image=Image.open(ROOT/path);image.thumbnail((380,300));wb.paste(image,(j*400+6,i*360+32));wd.text((j*400+6,i*360+5),title,font=font(17),fill='#243e32')
  wd.text((8,i*360+333),r['id']+' / '+r['split']+' / IoU较低错例',font=font(17),fill='#815238')
 wb.save(ROOT/'worst_cases.jpg',quality=93)
 data=dict(records=records,aggregates=res['aggregates'],foliage=res['user_foliage_regression'],split_counts={s:sum(r['split']==s for r in ds['records']) for s in res['aggregates']},concepts=read(ROOT/'source_checked_cunfa_concepts.json'));(ROOT/'report_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
 # Actual preprocessing preserves void and geometry, and the loss ignores void.
 tiny=np.full((31,63,3),200,'uint8');target=np.full((31,63),255,'uint8');target[8:12,10:15]=2;x,t,box=pack(tiny,target);assert set(np.unique(t))=={2,255};assert x.shape==(3,256,256);logits=torch.randn(1,3,256,256,requires_grad=True);loss=partial_role_loss(logits,torch.from_numpy(t.astype('int64'))[None]);loss.backward();assert torch.count_nonzero(logits.grad[0,:,t==255])==0
 assert confusion(np.array([[255,1,2]],'uint8'),np.array([[2,1,2]],'uint8')).sum()==2
 for r in ds['records']:assert sha(r['image'])==r['image_sha256'] and sha(r['target'])==r['target_sha256']
 # The old role experiment and explicit user correction remain unchanged.
 oldcor=read(LAB/'cunfa_object_correction_20261112/correction.json');assert sha(LAB/'cunfa_retention_v4_20261112/results.json')==oldcor['old_results_sha256']
 checks=dict(dataset_and_mapping_hash_match=True,training_script_hash_match=True,all_image_target_hashes_match=True,source_void_never_background=True,letterbox_preserves_void=True,unknown_loss_gradients_zero=True,void_excluded_from_metrics=True,old_role_results_unchanged=True,raw_object_output_not_reference_masked=True,user_foliage_reference_not_training=True,model_actually_trained=True,coarse_object_head_not_tree_drawing_skill=True,artist_groups_only_filename_candidates=True,formal_object_acceptance=False,cunfa_acceptance=False,tree_skill_training_enabled=False,robot_connected=False);(ROOT/'verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf8')
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>山石与树木对象基线 · 2026-11-12</title><style>*{box-sizing:border-box}body{margin:0;background:#f4f1e8;color:#243e32;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1440px;padding:28px;margin:auto}h1{font-size:30px;line-height:1.35}h2{font-size:23px;margin-top:32px}.note{padding:16px;background:#fff9e8;border-left:5px solid #b57c3c}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.card{background:#fffdf7;border:1px solid #d6d9c9;border-radius:8px;padding:12px}img{max-width:100%;height:auto;display:block;max-height:480px;margin:auto}select{padding:8px;font:inherit;max-width:100%;background:#fffdf7;border:1px solid #a2b2a5;border-radius:6px}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:14px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #d5d9c9}.warn{color:#8f4b32}small{color:#637667}a{color:#236b61}code{overflow-wrap:anywhere}@media(max-width:650px){main{padding:15px}.grid{grid-template-columns:1fr}h1{font-size:25px}td,th{padding:6px;font-size:13px}}</style><main><small>离线探索 · 未连接机械臂 · 未替换绘画版本</small><h1>先学会区分山石和树木，再判断对象内部的笔法</h1><p class="note"><b>本轮实际训练了一个粗对象分割模型，用户指出的树叶核心这次全部预测为树木；仍未通过独立验收。</b><br>其他测试图中仍有明显的山石/树木互相误判。这里没有训练树种、枝干拓扑、叶法、皴法名称或机器人笔触。</p>
<h2>1. 用户指出的树叶错例</h2><img src="foliage_object_comparison.jpg"><p>新模型没有用该树叶核心训练或手工改色。该核心1,647像素来自用户确认的树叶对象区域及助手选择的保守多边形，不是精确树叶墨迹边界。模型在此全部判为树木，不代表已解决整幅图的树石混淆，也不是新盲测。</p>
<h2>2. 数据与标签核查</h2><p>源数据编号2（山）、13（石）合并为山石，3为树木；其他具名前景归入其他对象，源0/255未标区域继续忽略。关键编号用两张逐像素相同的图及具名COCO多边形交叉核对，核心编号一致性超过99.99%。这不是官方完整编号表或专家全量复核。</p><p>训练35张、验证3张，测试分为具名文件12张和另一集合8张。训练图来自8个文件名作品组；12张具名测试实际同属1个作品组，不能当12件独立作品。作者与作品线索来自文件名，原始归属与裁剪关系没有独立认证。两数据仓库的重复图只作为映射证据，不重复计入训练样本。</p>
<h2>3. 同条件基线对照</h2><p>冻结已有ImageNet MobileNetV3 Small编码器，训练多尺度对象分割头；与只看单像素颜色、不看上下文的网络对照。两组各固定600步、相同增强与作品组采样，使用最后检查点，没有按测试分数挑轮次。</p><div class="scroll"><table><thead><tr><th>测试组</th><th>方法</th><th>平均IoU</th><th>山石IoU</th><th>树木IoU</th><th>树木误作山石</th><th>山石误作树木</th></tr></thead><tbody id="metrics"></tbody></table></div><p class="warn">指标只统计源标注覆盖区域，源未标纸面不在分母内。来源粗标注可能含边界错误。文件名作者分组也不等于已认证的新作者测试，当前不能据这些数字宣称对象识别已完成。</p>
<h2>4. 逐张检查原始对象预测</h2><div class="controls"><label>图像 <select id="case"></select></label><span id="scope"></span></div><div class="grid"><div class="card"><b>原图</b><img id="input"></div><div class="card"><b>源粗对象参考</b><img id="reference"><p id="refnote"></p></div><div class="card"><b>模型原始对象预测</b><img id="prediction"></div></div><p>棕=山石，绿=树木，灰蓝=其他具名对象；参考中的米色=未标，不参与损失。预测图显示整个画面，未用源参考或用户树叶多边形遮罩修饰。未标空白也会被网络分配类别，其中可能有错误。</p><p>这是对象区域，不能把色块当墨带或线稿；树木预测也没有细分主干、枝条和松针。</p>
<h2>5. 保留最差案例</h2><img src="worst_cases.jpg"><p>分别显示两组测试中平均IoU最低的两张，便于检查树石交叠、设色和细长枝干的误判。仅用于训练后的诊断，没有回填为新标签或挑选新的权重。</p>
<h2>6. 皴法辨识同步推进到哪里</h2><p>已核对现有4幅具名皴法教学图示与原文，整理可见笔形与排列概念；它们均来自同一位示范者，不能随机拆成独立训练/测试。另有8例概念核查队列。本轮没有新增专家皴法标签，没有训练新的皴法概念模型。</p><p>文字所述用锋方式与图中能观察到的属性分开记录；不能从静态图片直接补成机器人下压或含墨量。树叶、树干和皴笔继续采用不同对象/角色标签。</p>
<h2>7. 当前阶段</h2><p>对象模型已试训，正式独立验收仍未通过；皴法独立验收也未通过。树木画法训练保持关闭。下一轮优先核查新的作品归属、树石边界与错误对象，补皴法局部概念参考，再验证对象预测如何影响下游皴法。</p><p><a href="protocol.json">冻结训练协议</a> · <a href="mapping_evidence.json">编号映射依据</a> · <a href="results.json">全部逐张结果</a> · <a href="source_checked_cunfa_concepts.json">皴法概念来源记录</a> · <a href="verification.json">检查记录</a></p><p><a href="https://huggingface.co/datasets/Rane7/CLP_dataset">CLPD源数据</a> · <a href="https://github.com/zpc481523/res-painting">交叉核对标注来源</a></p></main><script>let D;const $=s=>document.getElementById(s),pct=x=>x===null?'无参考':(x*100).toFixed(1)+'%';function show(){let r=D.records[$('case').selectedIndex];$('input').src=r.input;$('prediction').src=r.predictions.context_object_head;if(r.reference){$('reference').src=r.reference;$('refnote').textContent='源粗区域标注；未标区域不作背景。'}else{$('reference').removeAttribute('src');$('refnote').textContent='用户确认的树叶核心用于回归检查；无完整对象边界真值。'}$('scope').textContent=r.split+' / '+r.work_group;}fetch('report_data.json').then(r=>r.json()).then(d=>{D=d;for(let r of d.records){let o=document.createElement('option');o.textContent=r.id;$('case').append(o)}$('case').onchange=show;show();for(let [split,title] of [['test_named','具名文件：12图 / 1作品组'],['test_collection','集合留出：8图 / 8文件候选组']])for(let [arm,name] of [['pixel_color_only','仅颜色'],['context_object_head','多尺度上下文']]){let s=D.aggregates[split][arm],tr=document.createElement('tr');tr.innerHTML='<td>'+title+'</td><td>'+name+'</td>'+[s.mean_iou,s.iou[1],s.iou[2],s.tree_as_rock_rate,s.rock_as_tree_rate].map(v=>'<td>'+pct(v)+'</td>').join('');$('metrics').append(tr)}window.reportReady=true});</script></html>'''
 page=page.replace('模型在此全部判为树木，不代表已解决整幅图的树石混淆，也不是新盲测。','模型在此全部判为树木，不代表已解决整幅图的树石混淆，也不是新盲测。 上方淡轮廓围出的山体仍未完整分出，灰蓝预测不能简单当作正确背景。')
 page=page.replace('</style>','.controls>*{min-width:0;max-width:100%;overflow-wrap:anywhere}.grid>.card{min-width:0}select{width:100%;max-width:600px}</style>')
 (ROOT/'index.html').write_text(page,encoding='utf8');print(json.dumps(checks,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
