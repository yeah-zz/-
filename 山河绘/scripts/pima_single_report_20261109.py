# -*- coding: utf-8 -*-
"""Reproduce saved neural traces and publish a precise local research report."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
from pima_single_train_20261109 import ROOT,CurveTracer,records,rollout,unit,assess
import json,hashlib,shutil
import numpy as np,torch
from PIL import Image
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(4);report=json.loads((ROOT/'single_curve_training_report.json').read_text('utf8'));manifest=json.loads((ROOT/'single_reference_manifest.json').read_text('utf8'));data=records()
 ck=torch.load(ROOT/'single_curve_tracer.pt',map_location='cpu',weights_only=False);model=CurveTracer().cuda();model.load_state_dict(ck['model']);model.eval();z=np.load(ROOT/'single_curve_predictions.npz')
 for r in data:
  q=r['q'];pred,logs,reason=rollout(model,r['images'],q[0],unit(q[2]-q[0]));saved=z[r['meta']['id']+'_prediction'];assert pred.shape==saved.shape and np.allclose(pred,saved,atol=1e-4)
  assert np.isfinite(pred).all();assert all(0<=a['stop_probability']<=1 for a in logs)
 assert len(data)==3;assert [r['meta']['id'] for r in data if r['meta']['split']=='train']==['S01','S02']
 assert [r['meta']['id'] for r in data if r['meta']['split']!='train']==['S03']
 assert all(r['expert_verified'] is False and r['physical_pressure'] is None for r in manifest['records'])
 assert all(r['target_endpoint_or_length_supplied_to_model'] is False and r['guide_curve_supplied_to_model'] is False for r in report['results'])
 prior=LAB/'pima_demonstration_20261109/verification.json';old=json.loads(prior.read_text('utf8'));plans=old['approved_plans_fingerprints']
 for p in plans:assert sha(Path(p['path']))==p['sha256'],'Approved plan changed'
 original=json.loads((ROOT/'teacher_forcing_only/single_curve_training_report.json').read_text('utf8'))
 dev=next(x for x in report['results'] if x['split']!='train');olddev=next(x for x in original['results'] if x['split']!='train')
 decision=dict(adopted=False,limited_same_rock_trace_improvement=True,complete_cunfa_reproduction_passed=False,
  reason='Only two assistant-curated training events and one development event from the same rock; residual endpoint errors and curve jitter remain. Event imagery and initial tip/tangent are supplied.',
  learned_style_recognition=False,autonomous_full_image_stroke_discovery=False,physical_brush_reproduction=False,robot_connected=False,
  development_selected_during_training=True,independent_test_works=0)
 save(ROOT/'decision.json',decision)
 verify=dict(passed=True,checkpoint_full_rollout_replayed=True,all_curves_finite=True,excluded_cases_not_trained=True,
  training_ids=['S01','S02'],same_rock_development_ids=['S03'],independent_test_works=0,development_used_for_checkpoint_selection=True,
  reference_guide_not_model_input=True,target_endpoint_and_length_not_rollout_inputs=True,expert_truth_not_claimed=True,
  approved_plans_unchanged=True,approved_plans_fingerprints=plans,robot_connected=False)
 save(ROOT/'verification.json',verify)
 for item in report['results']:item['initial_state_source']='candidate reference first point and initial tangent; oracle seed, not autonomous detection'
 report['model_sha256']=sha(ROOT/'single_curve_tracer.pt');report['reference_manifest_sha256']=sha(ROOT/'single_reference_manifest.json');save(ROOT/'single_curve_training_report.json',report)
 # Compact first training example and held-out DEVELOPMENT example; no retouching.
 src=Image.open(ROOT/'single_curve_preview.jpg');board=Image.new('RGB',(1200,740),'#f1eee6');board.paste(src.crop((0,0,1200,370)),(0,0));board.paste(src.crop((0,740,1200,1110)),(0,370));board.save(ROOT/'compact_preview.jpg',quality=95)
 snapshot=ROOT/'source_snapshot';snapshot.mkdir(exist_ok=True)
 for name in ['pima_stroke_inspect_20261109.py','pima_tip_candidates_20261109.py','pima_single_prepare_20261109.py','pima_single_reference_20261109.py','pima_single_train_20261109.py','pima_single_report_20261109.py']:
  shutil.copy2(Path(__file__).with_name(name),snapshot/name)
 series=[]
 for r in data:
  result=next(x for x in report['results'] if x['id']==r['meta']['id']);series.append(dict(id=result['id'],split=result['split'],box=r['meta']['event_crop'],reference=r['q'].tolist(),prediction=result['predicted_curve']))
 rows=''.join('<tr><td>'+x['id']+'</td><td>'+('训练' if x['split']=='train' else '同山石开发，参与选权重')+'</td><td>'+f"{x['metrics']['symmetric_distance_px']:.2f}"+'</td><td>'+f"{x['metrics']['endpoint_error_px']:.1f}"+'</td><td>'+f"{x['straight_endpoint_oracle']['symmetric_distance_px']:.2f}"+'</td></tr>' for x in report['results'])
 page='''<!doctype html>
<html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>披麻皴逐笔参考与连续曲线试训</title>
<style>body{max-width:1200px;margin:24px auto;padding:0 16px;background:#f3eee5;color:#253f38;font:17px/1.65 system-ui}h1{font-size:29px}section{padding:20px;background:#fffdf8;border-radius:12px;margin:18px 0}img{max-width:100%;height:auto}table{width:100%;border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}a{color:#326889}.status{border-left:6px solid #a37543}button,select,input{font:inherit;margin:8px}canvas{max-width:100%;background:white;border:1px solid #ddd}.small{font-size:14px;color:#555}</style>
<h1>披麻皴：逐笔起止与连续曲线</h1>
<section class="status"><b>已完成逐笔候选整理与两轮 GPU 试训；完整皴法复刻仍未通过。</b><p>本轮在给定单笔前后图像、起点与初始方向时，可以输出一条连续曲线并预测停止。还不能仅凭整幅成品自主完成分笔、认皴或实机复刻。原来认可的绘画方案未替换，未连接机械臂。</p></section>
<section><h2>这次补了什么监督</h2><p>细看 11 个原始事件窗口，整理 5 条较清晰的长皴笔候选，保留其中 3 条参考。两条用于训练，一条用于同一山石上的开发与权重选择。另两条因起落、交汇或旧线归属不清而排除。</p><p>参考由视频可观察运动、笔前/笔后新增墨迹及人工复核共同整理，仍是候选监督。没有专家逐笔真值、接触传感器、压力或含墨量实测。0.05 秒取帧只用于细查，不能解释为接触时刻的同等精度。</p><img src="reference_curves.jpg"></section>
<section><h2>模型直接输出</h2><p>第四列为保存权重自由运行后的原始中心轨迹。推理没有取得完整参考曲线、目标终点或目标长度；起点和初始方向取自候选参考，是已知起始提示，并非模型自主检测。单笔前后事件图像及范围也由外部提供。</p><img src="single_curve_preview.jpg"><p class="small">这是中心轨迹验证。黑色细线不代表真实毛笔笔宽，也不是完整皴法墨迹重绘。</p></section>
<section><h2>逐步观察模型如何形成完整曲线</h2><p><select id="case"></select><button id="play">播放轨迹</button><input id="step" type="range" min="1" value="1"><span id="state"></span></p><canvas id="trace" width="900" height="560"></canvas><p class="small">蓝色：候选参考；黑色：模型逐步输出。这里是离线轨迹回放，不是机械臂实时画面。</p></section>
<section><h2>完整曲线检查</h2><table><tr><th>事件</th><th>用途</th><th>曲线平均距离 px</th><th>终点偏差 px</th><th>已知终点直线基线距离 px</th></tr>ROWS</table><p>平均距离是预测与候选参考之间的双向最近点距离，不能解释为专家准确率。开发事件参与了权重选择，不是独立盲测；同一山石上表现较好也不能证明新山石泛化。</p><p>最初仅逐步监督的模型在开发事件中越过收笔位置约 24.6px，并撞到局部边界。补充偏离恢复与收笔后的终止增强后，保留第 300 步权重：开发事件平均距离约 0.57px、终点偏差约 2.6px。两条训练事件仍有约 7.7px、5.8px 的终点偏差，曲线也有抖动。</p><p>清空图像输入后结果明显变差，说明输出依赖图像；这种对照不证明模型已理解皴法语义。</p></section>
<section><h2>仍需解决</h2><p>继续补清晰、独立来源的单笔示范；处理交叉、干笔断续和弯折笔触；改善收笔定位及曲线抖动。下一阶段需要把单笔从完整画面中找出来，再验证笔组组织、皴法辨识、墨带形状与实机参数。</p><p>当前不采用这批权重替换完整绘画模型。模型没有学习真实机械臂下压量、含墨量，也没有进行实画。</p></section>
<section><h2>可追溯文件</h2><p><a href="single_reference_manifest.json">逐笔参考与排除原因</a> · <a href="single_curve_training_report.json">训练、原始输出和对照指标</a> · <a href="verification.json">核验结果</a> · <a href="decision.json">采用结论</a></p><p><a href="https://www.ganjingworld.com/video/1h5vi07sks9KFNZ72hjMX9VrP1331c">教师示范来源</a> · <a href="inspection/37_212.6_213.15.jpg">第三笔起点细查</a> · <a href="single_observations.jpg">原始前后墨迹观察</a></p></section>
<script>
const cases=SERIES;let timer=null;const sel=document.getElementById('case'),slider=document.getElementById('step'),ctx=document.getElementById('trace').getContext('2d');cases.forEach((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=c.id+' '+(c.split==='train'?'训练':'同山石开发');sel.add(o)});
function path(points,color,count,scale,ox,oy){if(!points.length)return;ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();points.slice(0,count).forEach((p,i)=>{const x=ox+p[0]*scale,y=oy+p[1]*scale;i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke()}
function draw(){const c=cases[Number(sel.value)],w=c.box[2]-c.box[0],h=c.box[3]-c.box[1],scale=Math.min(750/w,490/h),ox=(900-w*scale)/2,oy=25;ctx.clearRect(0,0,900,560);path(c.reference,'#1394c3',c.reference.length,scale,ox,oy);path(c.prediction,'#111',Number(slider.value),scale,ox,oy);document.getElementById('state').textContent=slider.value+' / '+c.prediction.length+' 点'}
function change(){slider.max=cases[Number(sel.value)].prediction.length;slider.value=1;draw()}sel.onchange=change;slider.oninput=draw;document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;return}timer=setInterval(()=>{slider.value=Number(slider.value)>=Number(slider.max)?1:Number(slider.value)+2;draw()},60)};change();
</script></html>'''.replace('ROWS',rows).replace('SERIES',json.dumps(series,ensure_ascii=False))
 (ROOT/'index.html').write_text(page,encoding='utf8');print('SINGLE_VERIFIED',dev['metrics'],'old',olddev['metrics'],flush=True)
if __name__=='__main__':main()
