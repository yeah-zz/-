# -*- coding: utf-8 -*-
"""Show real tradeoffs; coverage is not full stroke accuracy."""
from source_constrained_strokes_20261110 import *
def main():
 ev=json.loads((OUT/'evaluation.json').read_text('utf8'));r=ev['results'][0];cells=[]
 for role,title in [('contour','轮廓'),('structure','结构转折'),('cun','皴笔候选')]:
  values=[]
  for key in ['raw_model_coverage','source_constrained_coverage']:
   a=[z for z in r[key] if z['role']==role];values.append([float(np.mean([z[k] for z in a])) for k in ['union_visible_coverage_3px','best_single_path_visible_coverage_3px']])
  cells.append(dict(role=role,name=title,old_union=values[0][0],new_union=values[1][0],old_single=values[0][1],new_single=values[1][1]))
 decision=dict(adopted=False,complete_strokes_passed=False,new_neural_weights=False,
  reason='More source-aligned visible coverage, but contour/structural single-path continuity regresses; many small pending paths remain.',
  partial_reference_count=27,reference_kind='assistant static visible guides, incomplete and uncertain; not artist stroke truth',metrics=cells,
  source_parameters_selected_on_procedural_data=True,method_developed_with_real_outputs_visible=True,independent_artist_test=False,
  candidate_paths=r['result_paths'],pending_short=r['pending_short'],robot_connected=False)
 save('decision.json',decision)
 table=''.join(f"<tr><td>{c['name']}</td><td>{c['old_union']:.1%} → {c['new_union']:.1%}</td><td>{c['old_single']:.1%} → {c['new_single']:.1%}</td></tr>" for c in cells)
 scenes=''.join(f"<section><h2>{z['id']}</h2><div class='pair'><img src='{z['id']}_source.png'><img src='{z['id']}_paths.png'><img src='{z['id']}_overlay.png'></div><p>候选路径 {z['result_paths']}，其中短段待定 {z['pending_short']}；角色、对象与真实分笔尚未确认。</p></section>" for z in ev['results'][1:])
 html='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原图约束提取 · 同图验证</title><style>body{background:#f4f1e9;color:#263e35;font:16px/1.7 system-ui;margin:0}main{max-width:1480px;margin:30px auto;padding:0 20px}h1{font-size:30px}.notice{padding:16px;background:#fff0d4;border-left:4px solid #b28b42}img{max-width:100%;background:white}.pair{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.pair img{width:100%}table{border-collapse:collapse;width:100%;background:white;margin:24px 0}td,th{padding:12px;text-align:left;border-bottom:1px solid #ddd}section{margin:30px 0}a{color:#287b68}p{max-width:1100px}@media(max-width:700px){.pair{grid-template-columns:1fr}td,th{font-size:13px;padding:6px}}</style><main><h1>原图约束提取：少了乱飘，完整笔触仍未通过</h1><p>2026-11-10 · 离线实验 · 没有新训练权重 · 没有机械臂控制接口</p><div class="notice"><strong>本轮不替换认可版本。</strong>原图上的可见线条覆盖有所提高，但单条连续路径仍不足，结构转折存在退步；不能把整体覆盖增加当成完整笔触恢复成功。</div><h2>同一区域、自动读取、无人工起点或参考路径输入</h2><img src="comparison.png" alt="原图、旧模型、图像约束候选和原图叠加"><p>算法先读取原图暗线带，再用多尺度线带中心抑制粗墨网状分叉。续接仅限原图已有墨带内部、端点互相唯一优选的位置；白缝与竞争分支不强接。浅灰短段仍保存为待定，没有靠删除它们制造数量改善。</p><h2>同时检查找到多少与是否连贯</h2><p>以下仅针对27段人工可见结构参考，容差为原图3像素。不是完整真值、全局准确率或画家真实分笔。结构参考可能经过模糊墨迹；局部扩大覆盖也可能包含多余路径。</p><table><tr><th>参考类别</th><th>全部候选共同覆盖：旧 → 新</th><th>最佳单条路径覆盖：旧 → 新</th></tr>'''+table+f'''</table><p>右侧山石组输出 {r['result_paths']} 条候选路径，其中 {r['pending_short']} 条短于4像素、保留待定。这是图像路径分解数量，不能视为应当实画的笔数。</p><h2>另外三个局部采用相同参数</h2><p>下面都是既有开发资料，未作为独立盲测；没有全量真实参考，不能报告它们的提取准确率。</p>'''+scenes+'''<p><a href="evaluation.json">全部诊断数据</a> · <a href="decision.json">未采用原因</a> · <a href="counterexample_checks.json">白缝、并行线和竞争分支检查</a></p></main>'''
 (OUT/'index.html').write_text(html,encoding='utf-8')
 print(json.dumps(decision,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
