# -*- coding: utf-8 -*-
"""Audit static inference and distinguish real sources, synthetic stress, and guards."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
from pima_static_model_20261109 import ROOT,StaticTracer,variants,old_records,trace_many,seed_for
from pima_static_guard_20261109 import guarded,sample
import json,hashlib,shutil
import numpy as np,cv2,torch
from PIL import Image,ImageDraw,ImageFont
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18);SMALL=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
def read(p):return json.loads(p.read_text('utf8'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def path_image(images,curve,color):
 im=Image.fromarray(np.uint8(np.clip(1-images[1],0,1)*255)).convert('RGB');d=ImageDraw.Draw(im);d.line([tuple(p) for p in curve],fill=color,width=1);return im
def stress_gallery(training):
 array=np.load(ROOT/'static_tracer_predictions.npz');r=next(x for x in old_records() if x['meta']['split']!='train');cases=variants(r);board=Image.new('RGB',(1200,340*len(cases)),'#f1eee6');d=ImageDraw.Draw(board)
 names={'base':'真实候选的静态局部','dry_gap':'人工断续压力测试','known_crossing':'人工交叉压力测试','bend_left':'人工弯折压力测试·左','bend_right':'人工弯折压力测试·右'}
 for row,c in enumerate(cases):
  pred=array[c['id']+'_prediction'];q=array[c['id']+'_reference'];panels=[Image.fromarray(np.uint8((1-c['images'][0])*255)).convert('RGB'),path_image(c['images'],q,'#168bb0'),path_image(c['images'],pred,'#111')]
  for col,im in enumerate(panels):
   im.thumbnail((370,270));board.paste(im,(col*400+40,row*340+35));d.text((col*400+8,row*340+5),[names[c['mode']],'已知目标路径','新模型自由运行·原始输出'][col],font=SMALL,fill='#243e37')
  result=next(x for x in training['results'] if x['id']==c['id']);d.text((8,row*340+307),f'中心偏差 {result["metrics"]["symmetric_distance_px"]:.2f}px；收笔偏差 {result["metrics"]["endpoint_error_px"]:.1f}px；不代表真实新画作表现',font=SMALL,fill='#555')
 board.save(ROOT/'stress_preview.jpg',quality=95)
def band_observation(gray,q):
 widths=[];tones=[];valid=[];ribbon=Image.new('RGBA',(gray.shape[1],gray.shape[0]),(255,255,255,255));d=ImageDraw.Draw(ribbon)
 normals=[]
 for i,p in enumerate(q):
  tangent=q[min(i+2,len(q)-1)]-q[max(i-2,0)];tangent=tangent/max(float(np.linalg.norm(tangent)),1e-6);normal=np.array([-tangent[1],tangent[0]]);normals.append(normal);locations=p+np.linspace(-8,8,33)[:,None]*normal;profile=sample(gray,locations);paper=float(np.median(np.r_[profile[:4],profile[-4:]]));alpha=np.clip((paper-profile)/max(paper,.05),0,1);peak=float(alpha[14:19].max());threshold=max(.035,peak*.35)
  left=right=16
  while left>0 and alpha[left-1]>threshold:left-=1
  while right<32 and alpha[right+1]>threshold:right+=1
  good=peak>.06 and left>0 and right<32;width=float((right-left+1)*.5)
  widths.append(width);tones.append(peak);valid.append(good)
 for i in range(len(q)-1):
  if not valid[i] or not valid[i+1]:continue
  poly=[q[i]+normals[i]*widths[i]/2,q[i+1]+normals[i+1]*widths[i+1]/2,q[i+1]-normals[i+1]*widths[i+1]/2,q[i]-normals[i]*widths[i]/2];shade=int(255*(1-min(1,(tones[i]+tones[i+1])/2)));d.polygon([tuple(p) for p in poly],fill=(shade,shade,shade,255))
 return dict(width_image_pixels=widths,relative_image_contrast=tones,valid=valid,physical_pressure=None,ink_volume=None,robot_speed=None,width_to_robot_mapping=None),ribbon.convert('RGB')
def external_gallery(raw,guard):
 board=Image.new('RGB',(1200,420*2),'#f1eee6');d=ImageDraw.Draw(board);bands=[]
 for row,refid in enumerate(['H01','H02']):
  gi=guard['images'][row];ri=raw['images'][row];rgb=np.array(Image.open(ROOT/'external'/gi['image']).convert('RGB'));orig=Image.fromarray(rgb);rm=ri['reference_matches'][0];gm=gi['reference_matches'][0];rc=next(c for c in ri['candidates'] if c['id']==rm['candidate_id']);gc=next(c for c in gi['candidates'] if c['id']==gm['candidate_id'])
  views=[]
  for curve,color in [(None,None),(rc['curve'],'#98527d'),(gc['curve'],'#087b89')]:
   im=orig.copy();dr=ImageDraw.Draw(im)
   if curve:dr.line([tuple(p) for p in curve],fill=color,width=2)
   box=(340,190,435,315) if row==0 else (340,30,640,210);im=im.crop(box);scale=min(370/im.width,300/im.height);im=im.resize((round(im.width*scale),round(im.height*scale)));views.append(im)
  for col,im in enumerate(views):board.paste(im,(col*400+15,row*420+36));d.text((col*400+8,row*420+5),[refid+' 原始局部（展示裁切）','原始模型最接近候选','墨迹约束后最接近候选'][col],font=SMALL,fill='#243e37')
  d.text((8,row*420+365),f'单条参考覆盖：原始 {rm["visible_reference_coverage_within_3px"]:.0%} → 约束后 {gm["visible_reference_coverage_within_3px"]:.0%}；参考是近似静态曲线，不是专家逐笔真值',font=SMALL,fill='#555')
  gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255;observation,ribbon=band_observation(gray,np.array(gc['curve'],np.float32));observation.update(reference=refid,source_candidate=gc['id'],method='measured image cross-sections, not a learned physical brush model');bands.append(observation)
  if row==0:
   b=Image.new('RGB',(800,360),'#f1eee6');bd=ImageDraw.Draw(b)
   for col,im in enumerate([orig.crop((340,190,435,315)),ribbon.crop((340,190,435,315))]):
    im=im.resize((210,276));b.paste(im,(col*400+30,50));bd.text((col*400+8,8),['原图墨带','测量笔宽/明暗的几何预览·非实画'][col],font=SMALL,fill='#243e37')
   b.save(ROOT/'band_observation_preview.jpg',quality=95)
 board.save(ROOT/'external_local_preview.jpg',quality=95);save(ROOT/'image_band_observations.json',dict(records=bands,physical_calibration_complete=False,physical_robot_parameters_generated=False))
def main():
 torch.set_num_threads(4);training=read(ROOT/'static_training_report.json');raw=read(ROOT/'static_discovery_report.json');guard=read(ROOT/'guarded_discovery_report.json');protocol=read(ROOT/'protocol.json');source=read(ROOT/'external_sources.json')
 ck=torch.load(ROOT/'static_tracer.pt',map_location='cpu',weights_only=False);net=StaticTracer().cuda();net.load_state_dict(ck['model']);net.eval();z=np.load(ROOT/'static_tracer_predictions.npz')
 r=next(x for x in old_records() if x['meta']['split']!='train');case=variants(r,False)[0];out=trace_many(net,case['images'],[seed_for(case['q'])])[0];assert np.allclose(out['curve'],z[case['id']+'_prediction'],atol=1e-4)
 assert ck['real_train_ids']==['S01','S02'];assert training['external_sources_read_by_training'] is False
 assert guard['external_is_now_development_not_blind_test'];assert protocol['external_used_for_algorithm_revision'] is True
 blank=guarded(net,np.zeros((2,100,100),np.float32),[dict(point=[50,50],direction=[1,0])])[0];assert len(blank['curve'])==1
 for image in guard['images']:
  assert image['manual_seed_given'] is False and image['reference_curve_given'] is False
  for c in image['candidates']:assert np.isfinite(c['curve']).all()
 old=read(LAB/'pima_single_stroke_20261109/verification.json')
 for item in old['approved_plans_fingerprints']:assert sha(Path(item['path']))==item['sha256']
 stress_gallery(training);external_gallery(raw,guard)
 decision=dict(adopted=False,full_image_extraction_passed=False,style_recognition_passed=False,real_cunfa_reproduction_passed=False,
  reasons=['Only two real candidate strokes train the new head','External raw model drifts away from ink','Image guard improves one visible curve but another bending contour stays incomplete','Original painter identity and true single-stroke labels for new sources remain unverified','Physical calibration not performed'],
  actual_new_expert_verified_training_strokes=0,new_public_work_sources=2,guard_is_hybrid_not_pure_model=True,robot_connected=False)
 save(ROOT/'decision.json',decision)
 save(ROOT/'verification.json',dict(passed=True,checkpoint_raw_rollout_reproduced=True,no_external_training_or_checkpoint_selection=True,
  external_algorithm_revision_disclosed=True,blank_source_does_not_create_curve=True,no_manual_seed_on_external=True,
  synthetic_stress_not_counted_as_real_demonstrations=True,approved_plans_unchanged=True,robot_connected=False,
  checkpoint_sha256=sha(ROOT/'static_tracer.pt'),new_source_artist_disjointness_verified=False))
 snap=ROOT/'source_snapshot';snap.mkdir(exist_ok=True)
 for n in ['pima_static_protocol_20261109.py','pima_static_model_20261109.py','pima_static_discover_20261109.py','pima_static_guard_20261109.py','pima_static_report_20261109.py']:shutil.copy2(Path(__file__).with_name(n),snap/n)
 save(ROOT/'next_data_requirements.json',dict(priority='verified independent real demonstrations with observable contact/lift, crossing ownership, dry-brush continuation, and bends',current_new_work_sources=2,current_verified_new_artist_stroke_labels=0,style_labels='source-title weak labels only',physical_robot_parameters=None))
 rows=''.join(f'<tr><td>{x["mode"]}</td><td>{x["metrics"]["symmetric_distance_px"]:.2f}</td><td>{x["metrics"]["endpoint_error_px"]:.1f}</td></tr>' for x in training['results'] if x['split']!='train')
 source_links=''.join(f'<li><a href="{s["page"]}">{s.get("title",s.get("artist",s["id"]))}</a>：'+('作者可确认；当前取流需登录，未使用。' if s['id']=='tong_museum_index' else '不同作品已观察；原画者身份未确认，未加入训练。')+'</li>' for s in source)
 page='''<!doctype html>
<html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>静态单笔发现与外部作品开发对照</title>
<style>body{max-width:1200px;margin:24px auto;padding:0 16px;background:#f3eee5;color:#253f38;font:17px/1.65 system-ui}h1{font-size:29px}section{padding:20px;background:#fffdf8;border-radius:12px;margin:18px 0}img{max-width:100%;height:auto}table{width:100%;border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}a{color:#326889}.status{border-left:6px solid #a37543}.small{font-size:14px;color:#555}</style>
<h1>从静态画面提出单笔候选：本轮离线结果</h1>
<section class="status"><b>自动候选流程已跑通，完整画面提取和皴法复刻仍未通过。</b><p>模型原始输出会偏离墨迹。补入视频遮挡处理与墨迹约束后，一条简单可见轮廓能连续跟上；较大弯折轮廓仍有遗漏、分段。约束后的结果是混合算法，不表示模型自己学会了修正。本轮不替换认可的绘画方案，未连接机械臂。</p></section>
<section><h2>真实数据与压力测试分开记录</h2><ul><li>新增 2 幅不同作品的公开示范画面，均未加入训练；原画者身份尚未核实，不能声称完成了作者独立性验证。</li><li>已确认作者的博物馆示范取流需要登录，当前只保存来源索引，未绕过限制。</li><li>新模型实际仍用旧的 2 条候选参考训练，另 1 条同山石样本用于开发和选权重；没有新增专家逐笔真值。</li><li>交叉、干笔断续与弯折本轮采用已知路径的人工增强，共 10 个训练变体；它们不是 10 条真实新笔触，也不证明真实皴法效果。</li></ul><ul>SOURCES</ul></section>
<section><h2>新模型与自动起点</h2><p>输入改为单张静态图，使用局部墨迹与纸面归一化特征，不再取得前后差分。模型预测延续方向、停止概率、剩余距离和收笔位置。训练参考的预平滑移动限制为 0.6 像素，下面的原始模型轨迹没有事后平滑。</p><p>完整画面的起点由图像端点检测提出，不提供人工笔尖或参考路径。三幅画面分别有 300、376、582 个可用端点，按空间分布各试跑 80 个；这只是计算预算，不能把 80 个候选叫作 80 笔，更不代表全图已检查完。</p></section>
<section><h2>局部对照：保留成功和失败</h2><p>左：原图局部；中：原始模型最接近候选；右：图像约束后最接近候选。这个裁切仅用于展示，推理发生在完整输入帧中。近似参考只有两条可见轮廓，未标区域保持未知。</p><img src="external_local_preview.jpg"><p>简单轮廓 H01 的单条候选能覆盖参考。弯折轮廓 H02 仍不能由一条候选完整恢复，不能用多个碎段的合计覆盖率掩盖这个问题。参考有约 3 像素的人工描绘误差，覆盖率不是专家准确率。</p><p>首次外部检查后，这两幅作品参与了算法问题诊断，因此最终结果属于外部开发对照，不再称独立盲测。原始冻结权重检查已单独归档。</p></section>
<section><h2>完整画面：图像约束后全部候选</h2><img src="new_howto_frame_00_guarded.jpg"><img src="new_howto_frame_01_guarded.jpg"><img src="new_fuchun_frame_01_guarded.jpg"><p>交叉不确定、短候选和重复候选保留在 JSON 中；这里的覆盖图不等于可下发绘画轨迹。文字、残余笔杆和复杂树木仍会干扰候选。</p><details><summary>查看未加约束时的原始模型失败结果</summary><img src="new_howto_frame_00_proposals.jpg"><img src="new_howto_frame_01_proposals.jpg"></details></section>
<section><h2>人工压力测试：交叉与弯折仍不足</h2><img src="stress_preview.jpg"><table><tr><th>同山石开发条件</th><th>曲线距离 px</th><th>终点偏差 px</th></tr>ROWS</table><p>输入合同已从逐笔差分改为静态图像，不能直接把这些指标与上一轮数字当作同条件提升。交叉和弯折的终点仍有明显偏差，未通过采用条件。</p></section>
<section><h2>墨带形状与实机参数</h2><p>沿可用候选测量原图横截面的相对宽度和明暗，并生成几何墨带预览。下面的宽度来自输入图像测量，不是模型学会的真实毛笔接触，也不是实际皴法重绘。</p><img src="band_observation_preview.jpg"><p>图像笔宽尚未映射到实际毫米、毛笔下压量、速度或含墨量。相关实机参数保持为空，未生成新的上机动作。</p></section>
<section><h2>目前最需要补的数据</h2><p>需要可确认作者、逐笔落起清楚的真实交叉、干笔断续和弯折示范；当前新增来源尚不足以提供这些可靠标签。先补真实笔迹归属与收笔参考，再验证完整笔组和皴法类别。不能把这轮自动候选整批当作训练答案。</p><p>笔组组织、皴法辨识和真实墨带复现尚未完成，继续保留现有认可版本作为基线。</p></section>
<section><h2>报告文件</h2><p><a href="protocol.json">实验与数据使用约定</a> · <a href="external_sources.json">来源核查</a> · <a href="static_training_report.json">GPU 训练及压力测试</a> · <a href="static_discovery_report.json">首次原始模型检查</a> · <a href="guarded_discovery_report.json">混合约束开发对照</a> · <a href="image_band_observations.json">图像墨带测量</a> · <a href="verification.json">核验</a> · <a href="decision.json">采用结论</a></p></section></html>'''.replace('SOURCES',source_links).replace('ROWS',rows)
 (ROOT/'index.html').write_text(page,encoding='utf8');print('STATIC_REPORT_VERIFIED',ROOT,flush=True)
if __name__=='__main__':main()
