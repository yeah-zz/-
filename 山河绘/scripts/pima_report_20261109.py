# -*- coding: utf-8 -*-
"""Audit the pilot, build an honest local report, and preserve approved plans."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
import json,hashlib,shutil,html
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from pima_next_group_20261109 import ROOT,NextGroupNet,load_data
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(4)
 m=json.loads((ROOT/'sequence_manifest.json').read_text('utf8'));r=json.loads((ROOT/'next_group_report.json').read_text('utf8'));z=np.load(ROOT/'next_group_predictions.npz')
 ck=torch.load(ROOT/'next_group_pilot.pt',map_location='cpu',weights_only=False);net=NextGroupNet().cuda();net.load_state_dict(ck['model']);net.eval()
 with torch.no_grad():replay=net(torch.tensor(z['inputs'],device='cuda')).sigmoid().cpu().numpy()
 assert np.allclose(replay,z['prediction'],atol=1e-5)
 assert np.isfinite(replay).all() and replay.min()>=0 and replay.max()<=1
 assert r['independent_test_groups']==0 and r['adopted'] is False
 assert all(x['expert_paths'] is None and x['individual_stroke_count'] is None and x['pressure'] is None for x in m['records'])
 assert {x['source_group'] for x in m['records']}=={'liupengkai_qiushan_same_painting'}
 trainfiles={x[0]['file'] for x in load_data() if x[0]['split']=='train'}
 devfiles={x[0]['file'] for x in load_data() if x[0]['split']!='train'}
 assert not trainfiles&devfiles
 sources=[]
 for lesson in [29,30,31,35,36,37,40]:
  if lesson==31:meta=json.loads((ROOT/'raw/ganjing_metadata.json').read_text('utf8'))
  else:meta=json.loads((ROOT/'sources'/str(lesson)/'metadata.json').read_text('utf8'))
  selected=lesson in [36,37]
  sources.append(dict(lesson=lesson,title=meta['name'],author=meta['author']['name'],page=meta['url'],
    selected=selected,source_family='liupengkai_qiushan_same_painting',
    reason='Selected rock outline/cun windows only; other times excluded' if selected else 'Not admitted: inspected content mainly foliage/trunks, or mixed stages outside pilot scope'))
 save(ROOT/'source_curation.json',sources)
 plans=[]
 for rel in ['approved_execution_refine_20261107/G03_01_500/plan.json','object_hierarchy_20261107/G03_01/plan.json','structural_roles_20261107/G03_01/plan.json']:
  p=LAB/rel
  if p.exists():plans.append(dict(path=str(p),sha256=sha(p)))
 verify=dict(passed=True,checkpoint_replay=True,finite_probabilities=True,split_files_disjoint=True,
  independent_sources_disjoint=False,reason='Same painter and same painting: no independent test exists',
  no_expert_path_or_pressure_claims=True,raw_group_observations=len(m['records']),effective_train_groups=r['train_groups'],
  same_work_development_groups=r['same_work_dev_groups'],approved_plans_fingerprints=plans,robot_connected=False)
 save(ROOT/'verification.json',verify)
 # A compact *real observation* illustration, explicitly separated from predictions.
 font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
 board=Image.new('RGB',(1100,410),'#f1eee6');d=ImageDraw.Draw(board)
 d.text((15,8),'真实教程观察：轮廓 → 加入皴笔 → 叠加笔组（不是模型重绘）',font=font,fill='#243e37')
 for i,t in enumerate([440,480,560,760]):
  im=Image.open(ROOT/'sequence'/f'36_{t:04d}.png').crop((390,65,625,300)).convert('RGB').resize((260,260))
  board.paste(im,(15+i*275,65));d.text((15+i*275,340),f'第36课 {t}–{t+9}秒',font=small,fill='#222')
 d.text((15,375),'视频前后变化可观察；单笔身份、真实下压和含墨量仍未知。',font=small,fill='#555');board.save(ROOT/'real_demonstration_preview.jpg',quality=95)
 # Show raw output AND the validity domain; do not silently clean model mistakes.
 p=z['prediction'][0,0];v=z['valid'][0,0];raw=Image.fromarray(np.uint8((1-p)*255)).resize((320,320));bounded=Image.fromarray(np.uint8((1-p*v)*255)).resize((320,320))
 cmp=Image.new('RGB',(680,380),'#f1eee6');cmp.paste(raw,(10,40));cmp.paste(bounded,(350,40));dd=ImageDraw.Draw(cmp);dd.text((10,10),'模型原始输出',font=small,fill='#222');dd.text((350,10),'仅显示有效监督区域（非原始输出）',font=small,fill='#222');cmp.save(ROOT/'prediction_validity.jpg',quality=94)
 (ROOT/'decision.json').write_text(json.dumps(dict(adopted=False,real_reproduction_passed=False,learned_next_stroke_identity=False,
  reason='Raw predictions include large untrained/background areas and do not reliably select the next complete stroke; weak-mask score does not establish real cunfa reproduction.',robot_connected=False),indent=2),encoding='utf8')
 snapshot=ROOT/'source_snapshot';snapshot.mkdir(exist_ok=True)
 for n in ['pima_collect_20261109.py','pima_sources_20261109.py','pima_timeline_20261109.py','pima_sequence_20261109.py','pima_next_group_20261109.py','pima_report_20261109.py']:
  shutil.copy2(Path(__file__).with_name(n),snapshot/n)
 manifest_sha=sha(ROOT/'training_manifest.json');r['training_manifest_sha256']=manifest_sha;r['source_family_count']=1;r['phase_input_is_assistant_coarse_hint_not_model_recognition']=True;r['weak_mask_score_is_not_reproduction_accuracy']=True;save(ROOT/'next_group_report.json',r)
 table=''.join(f'<tr><td>{s["lesson"]}</td><td><a href="{html.escape(s["page"])}">{html.escape(s["title"])}</a></td><td>{"仅选山石时段" if s["selected"] else "本轮排除"}</td></tr>' for s in sources)
 frames=[]
 for lesson in [36,37]:
  for f in sorted((ROOT/'sequence').glob(f'{lesson}_*.png')):
   if lesson==36 and not 440<=int(f.stem.split('_')[1])<=760:continue
   frames.append(dict(src='sequence/'+f.name,label=f'真实第{lesson}课 {f.stem.split("_")[1]}–{int(f.stem.split("_")[1])+9}秒'))
 page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>披麻皴真实过程试验</title><style>body{margin:24px auto;max-width:1200px;background:#f4f0e7;color:#243e37;font:17px/1.6 system-ui}h1{font-size:29px}img{max-width:100%;height:auto;background:white}section{background:#fffdf8;padding:20px;margin:18px 0;border-radius:12px}.status{border-left:6px solid #a56d31}table{width:100%;border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}button,input{font:inherit;margin:8px}#step{width:78%}#frame{max-height:580px}a{color:#346a8d}.small{font-size:14px;color:#555}</style><h1>披麻皴：从真实绘画过程开始</h1>
<section class="status"><b>本轮完成了真实过程数据整理和 GPU 小模型试训，完整皴法复刻仍未通过。</b><p>模型原始输出仍有大片背景响应，且没有稳定选出下一组完整笔画。本轮权重未采用，原来认可的 489 笔方案未替换。没有连接机械臂。</p></section>
<section><h2>真实教程中的山石变化</h2><p>以下是教师示范的时间观察，经过配准与短时分位数去遮挡；不是模型生成，也不是专家逐笔真值。</p><img src="/real_demonstration_preview.jpg"><p><button id="play">播放观察序列</button><input id="step" type="range" min="0" value="0"><span id="label"></span></p><img id="frame"><p class="small">这是观察窗口回放，不是机械臂实时画面。相邻窗口可能新增多笔，不能把窗口数当成笔画数。</p></section>
<section><h2>数据与模型实际做到哪里</h2><table><tr><th>项目</th><th>本轮情况</th></tr><tr><td>公开教师示范检查</td><td>7 段，选出 2 段中的山石时段</td></tr><tr><td>原始前后增量记录</td><td>23 组；过滤弱增量后 11 组训练、7 组同画开发观察</td></tr><tr><td>独立作品测试</td><td>0；两段来自同一画家、同一幅画</td></tr><tr><td>监督来源</td><td>真实视频前后墨迹差分，自动弱监督；无专家分笔、下压或含墨量真值</td></tr><tr><td>新模型</td><td>整幅局部参考＋当前墨迹＋山石区域＋粗阶段提示 → 下一组墨迹概率；无固定笔数和竖向模板</td></tr><tr><td>GPU 试训</td><td>400 次更新，开发损失选择第 120 步；新权重独立保存</td></tr></table><p>同画弱增量 Dice 为 0.251；“全部剩余墨迹”基线为 0.227。这个小幅差异不代表皴法复刻正确，更不代表新作品准确率。</p></section>
<section><h2>模型直接输出及失败位置</h2><p>第四列是原始概率输出。大面积灰底、重复响应旧墨迹，说明“下一组该画什么”尚未学稳；这里保留失败结果。</p><img src="/next_group_preview.jpg"><details><summary>原始输出与有效监督范围的区别</summary><img src="/prediction_validity.jpg"><p>有效区域之外没有得到同等监督，不能靠显示时裁白来宣称模型成功。</p></details></section>
<section><h2>下一阶段的训练准入条件</h2><p>从已找到的真实长皴笔视频继续细化可观察的起止窗口，区分单笔和笔组；接触不清或交叉无法分辨的位置保持未知。补入独立教师或独立作品后封存测试。完整曲线、真实分笔与终止策略还未完成，不能将本轮概率图直接当作机器人轨迹。</p></section><section><h2>来源与复核文件</h2><table>TABLE</table><p><a href="/next_group_report.json">训练报告</a> · <a href="/source_curation.json">来源筛选</a> · <a href="/verification.json">核验结果</a> · <a href="/sequence_manifest.json">时间与弱参考记录</a></p><p class="small">文件和媒体保存在 E 盘；本页仅供本地观察，不提供运动指令。</p></section>
<script>const frames=FRAMES;let timer=null;const slider=document.getElementById('step');slider.max=frames.length-1;function show(){const f=frames[Number(slider.value)];document.getElementById('frame').src=f.src;document.getElementById('label').textContent=f.label}slider.oninput=show;document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;return}timer=setInterval(()=>{slider.value=(Number(slider.value)+1)%frames.length;show()},1000)};show();</script></html>'''.replace('TABLE',table).replace('FRAMES',json.dumps(frames,ensure_ascii=False))
 page=page.replace('src="/','src="').replace('href="/','href="')
 (ROOT/'index.html').write_text(page,encoding='utf8')
 print('VERIFIED',len(m['records']),'replay and unchanged plan fingerprints saved',flush=True)
if __name__=='__main__':main()
