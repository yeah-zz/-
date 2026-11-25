# -*- coding: utf-8 -*-
"""Publish raw failure alongside reviewed references, without changing baselines."""
import sys,json,hashlib,html
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pima_rock_reference_20261109 import ROOT,LAB
from pima_static_model_20261109 import StaticTracer,trace_many
from pima_static_discover_20261109 import proposals,evaluate_reference
import torch,numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 torch.set_num_threads(4);reg=json.loads((ROOT/'final_state_registration.json').read_text('utf8'));manifest=json.loads((ROOT/'reference_manifest.json').read_text('utf8'));rgb=np.array(Image.open(ROOT/'finished_frame.jpg').convert('RGB'))
 ims,sk,jnear,seeds,stats=proposals(rgb,max_seeds=80);results=[]
 refs=[dict(id=r['id'],points=r['curve'],annotation_uncertainty_px=5) for r in reg['records'] if r['status']=='reviewed_candidate']
 for name,ckpath in [('old_model',LAB/'pima_static_discovery_20261109'/'static_tracer.pt'),('new_model',ROOT/'new_work_static_tracer.pt')]:
  ck=torch.load(ckpath,map_location='cpu',weights_only=False);net=StaticTracer().cuda();net.load_state_dict(ck['model']);net.eval();raw=trace_many(net,ims,seeds,max_steps=250);curves=[dict(id=f'{name}_{i}',curve=r['curve'].tolist(),stop_reason=r['stop_reason']) for i,r in enumerate(raw)]
  matches=[evaluate_reference(curves,r) for r in refs];nearest=[dict(id=r['id'],nearest_selected_seed_distance_px=float(np.linalg.norm(np.array([s['point'] for s in seeds])-np.array(r['points'][0]),axis=1).min())) for r in refs]
  result=dict(model=name,stats=stats,curves=curves,matches=matches,seed_diagnostics_after_inference=nearest,
   input='completed static frame at 306.439s; no reference endpoints, directions, masks, video delta or rock ROI',
   final_frame_seen_by_training=False,source_work_seen_by_new_weights=True,scene_is_not_independent_work_test=True,
   individual_final_stroke_ownership_known=False,shape_overlap_is_not_artist_stroke_accuracy=True)
  results.append(result);im=Image.fromarray(rgb);d=ImageDraw.Draw(im);palette=['#1b7da0','#a87231','#1c855f','#815a99']
  for i,r in enumerate(curves):
   if len(r['curve'])>=5:d.line([tuple(p) for p in r['curve']],fill=palette[i%4],width=2)
  im.save(ROOT/f'finished_{name}_raw.jpg',quality=95);print('FINAL',name,matches,nearest,flush=True)
  if name=='new_model':
   replay=trace_many(net,ims,seeds[:8],max_steps=250);assert all(np.allclose(np.array(curves[i]['curve']),r['curve'],atol=1e-4) for i,r in enumerate(replay))
 write(ROOT/'final_discovery_report.json',dict(results=results,adopted=False,complete_rock_closed_loop_passed=False,robot_connected=False))
 old=json.loads((LAB/'pima_single_stroke_20261109'/'verification.json').read_text('utf8'));finger=[]
 for r in old['approved_plans_fingerprints']:
  sha=hashlib.sha256(Path(r['path']).read_bytes()).hexdigest();assert sha==r['sha256'];finger.append(dict(**r,unchanged=True))
 train=torch.load(ROOT/'new_work_static_tracer.pt',map_location='cpu',weights_only=False);assert train['training_events']==['R01','R05','R06','R07'];assert set(train['training_events']).isdisjoint({'R02','R03','R04','S01','S02','S03'})
 write(ROOT/'verification.json',dict(passed=True,checkpoint_raw_rollout_reproduced=True,excluded_candidates_not_trained=True,
  new_weights_training_works=1,source_disjoint_from_new_weights_development_works=1,verified_author_disjoint_test_works=0,
  final_static_image_not_used_for_training=True,training_stage_time_seconds=54.25,final_evaluation_time_seconds=reg['final_time_seconds'],
  autonomous_discovery_not_given_reference_seed=True,unknown_overlap_not_claimed_as_instance_gold=True,
  approved_plans_unchanged=True,approved_plans_fingerprints=finger,robot_connected=False,
  new_checkpoint_sha256=hashlib.sha256((ROOT/'new_work_static_tracer.pt').read_bytes()).hexdigest()))
 write(ROOT/'decision.json',dict(adopted=False,complete_rock_closed_loop_passed=False,actual_new_expert_verified_strokes=0,
  new_assistant_reviewed_motion_curve_candidates=4,uncertain_candidates_excluded=3,learned_instance_ownership_passed=False,
  named_cunfa_recognition_passed=False,physical_brush_mapping_validated=False,
  reason='Four new motion-curve candidates are insufficient for autonomous discovery. The new model fails automatic starts and termination. Shared group delta cannot supply trustworthy individual ownership in overlaps.'))
 font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
 board=Image.new('RGB',(1320,400),'#f2efe6');d=ImageDraw.Draw(board)
 panels=[Image.fromarray(rgb),Image.open(ROOT/'final_transferred_reference.jpg'),Image.open(ROOT/'finished_new_model_raw.jpg')]
 titles=['教程完成画面·局部','视频核查曲线参考·非模型输出','新模型自动提取·未通过']
 for i,im in enumerate(panels):
  board.paste(im.crop((190,260,510,460)).resize((430,269)),(i*440,45));d.text((i*440+8,10),titles[i],font=font,fill='#234238')
 d.text((10,330),'蓝：4条复核候选；棕：3条待定。参考只转移几何，最终单笔墨迹归属仍未确认。',font=small,fill='#333')
 d.text((10,360),'本轮未替换认可版本；无机械臂连接。模型失败结果原样保留。',font=small,fill='#333');board.save(ROOT/'cycle_preview.jpg',quality=96)
 data=dict(records=manifest['records'],matrix=reg['matrix']);payload=json.dumps(data,ensure_ascii=False).replace('</','<\\/')
 # The source fMP4 decodes in OpenCV but is rejected by the browser demuxer.
 # A timed, sampled GIF provides an honest local excerpt without new codecs.
 # Original PNGs, original fMP4 and source timestamps remain the evidence.
 samples=json.loads((ROOT/'fine.json').read_text('utf8'));frames=[]
 for s in samples:
  if 50.15<=s['t']<=55.30:
   im=Image.open(ROOT/'fine'/s['file']).crop((235,235,520,465)).resize((570,460));frames.append(im.convert('P',palette=Image.Palette.ADAPTIVE,colors=192))
 frames[0].save(ROOT/'tutorial_excerpt.gif',save_all=True,append_images=frames[1:],duration=100,loop=0,optimize=False)
 page='''<!doctype html><meta charset="utf-8"><title>披麻皴：新来源逐笔核查</title><style>body{margin:0;background:#f2efe6;color:#243e37;font:17px/1.65 system-ui}main{max-width:1320px;margin:auto;padding:28px;box-sizing:border-box}h1{font-size:28px}h2{font-size:23px}.notice{background:#fff7e4;border-left:5px solid #a76b32;padding:15px}img{max-width:100%;display:block}section{background:white;padding:20px;margin:22px 0}a{color:#176e91}canvas{max-width:100%;background:white;display:block}button,input{font:inherit}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:9px;text-align:left}.muted{color:#666}video{width:min(856px,100%)}</style><main>
 <h1>先核对真实笔触，再验证模型能否自己找到</h1>
 <p class="notice"><b>整块山石闭环尚未通过，新模型不采用。</b>这轮新增4条经视频核查的曲线候选，3条待定。它们不是专家逐笔真值；尚不能声称已经训练出披麻皴辨识或忠实复刻。</p>
 <img src="cycle_preview.jpg" alt="真实完成画、参考曲线与失败模型对照">
 <section><h2>这轮实际做了什么</h2><p>顺序解码已保存的短披麻皴示范，对关键窗口按原始约30帧/秒复核。7个接触候选中，R01、R05、R06、R07形成4条运动与新增笔迹支持的曲线；R02–R04只有2–3帧能观察，不作确定训练答案。</p><p>只用这幅新作品的4条候选，从头训练600步静态曲线追踪器。训练输入是54秒时的静态阶段图，不给差分或曲线引导。最终306秒的完成画只做提取验证，未放进本轮训练。故本轮还未达到“从最终画面学习单笔归属”的完整目标。</p><p>最终画面与参考配准使用30个RANSAC内点，内点RMS约0.51像素。这只验证位置能对齐，不能证明后续叠皴后仍能唯一分清早期每笔的墨迹。</p><p>源作品来自公开账号“米小鹿画画”，原画者身份及播放速度未独立确认。另一幅刘鹏凯作品与本轮权重的训练作品分开，但此前参与过方法开发；不称为独立盲测，也未完成按作者分开的验收。</p></section>
 <section><h2>教程与核查参考</h2><p><a href="https://www.ganjingworld.com/video/1gc2lo9n27j5ZShwAja69iQ3N12d1c">《画几块石头，示范短披麻皴》来源</a>。视频约50秒处切换为近景，跨镜头不做逐笔差分。</p><img src="tutorial_excerpt.gif" alt="教程50.15至55.30秒局部抽帧回放，约10帧每秒"><p class="muted">教程真实画面的抽帧回放：50.15–55.30秒，约10帧/秒，保持约0.1秒的采样时间间隔；不是原始帧率，也不是模型生成。逐笔核查另使用原始约30帧/秒的PNG。</p><p>下面是笔组前后帧与逐笔候选；底图是若干相邻前帧／后帧的亮度合成，并非单张原帧；遮挡合成残影不当笔迹。这里的差分包含整个笔组，不是单笔墨迹真值。交叠处在数据中保留未知。</p><img src="reference_review.jpg" loading="lazy"></section>
 <section><h2>笔组参考回放</h2><p>仅回放视频核查参考，<b>不是模型自主笔顺</b>。原轮廓已存在，蓝色按视频中可观察的相对次序加入。未确认的3笔保持待定；本轮未给整块山石编造完整笔顺。</p><canvas id="replay" width="856" height="480"></canvas><button id="play">播放参考</button> <input id="step" type="range" min="0" max="4" value="0"> <span id="caption"></span></section>
 <section><h2>模型直接输出与限制</h2><p>以下保留全部原始自动候选，没有把参考起点、参考方向、墨迹差分或山石裁剪框送给提取过程。端点候选仍由现有图像算法产生；最多追踪80个起点，其余未追踪，不能当作全图已经处理完毕。</p><p>新模型在已知起点的训练诊断中能跟住一条候选，但在自动起点下仍漏检、漂移、错误终止。600步权重按预先固定预算保存，没有挑最好看的检查点。这一失败说明新增4条参考仍不足以支撑自主提取。</p><h3>完成画：旧模型原始输出</h3><img src="finished_old_model_raw.jpg" loading="lazy"><h3>完成画：新模型原始输出（未通过）</h3><img src="finished_new_model_raw.jpg" loading="lazy"><h3>不同作品：新模型输出（开发对照）</h3><img src="liupengkai_other_work_new_model.jpg" loading="lazy"><p>局部参考未覆盖整幅，不能计算全局精确率。图像宽度、浓淡尚未映射成毛笔下压量、含墨量或速度；本页不提供机器人控制接口。</p></section>
 <section><h2>继续补哪些数据</h2><p>需要身份和作品来源可核对、近景中起落可见、最终画面里单笔仍可分辨的示范。优先补完整轮廓、交汇与提笔前后帧，逐条建立墨迹归属、曲线、端点和角色参考。下一轮先补足自动起点与单笔归属监督，再验收完整山石；不继续把整幅提取结果整批当答案。</p><p><a href="reference_manifest.json">参考与排除记录</a> · <a href="training_report.json">训练诊断</a> · <a href="final_discovery_report.json">最终静态画提取记录</a> · <a href="verification.json">完整性检查</a> · <a href="decision.json">未采用说明</a></p></section>
 <script>const DATA=__DATA__,c=document.querySelector('#replay'),ctx=c.getContext('2d'),slider=document.querySelector('#step'),caption=document.querySelector('#caption'),bg=new Image();const rows=DATA.records.filter(r=>r.status==='reviewed_candidate');function paint(){ctx.clearRect(0,0,c.width,c.height);ctx.drawImage(bg,0,0);let count=+slider.value;for(let r of rows.slice(0,count)){ctx.beginPath();r.reference_curve.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.strokeStyle='#187fa5';ctx.lineWidth=2;ctx.stroke()}caption.textContent='已回放 '+count+' / 4 条复核候选；其余待定'}bg.onload=paint;bg.src='references/group_before.png';slider.oninput=paint;let timer;document.querySelector('#play').onclick=()=>{clearInterval(timer);slider.value=0;paint();timer=setInterval(()=>{slider.value=+slider.value+1;paint();if(+slider.value>=4)clearInterval(timer)},900)};</script></main>'''.replace('__DATA__',payload)
 (ROOT/'index.html').write_text(page,encoding='utf8');print('REPORT',ROOT,flush=True)
if __name__=='__main__':main()
