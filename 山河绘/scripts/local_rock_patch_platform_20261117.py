# -*- coding: utf-8 -*-
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
ROOT=LAB/'local_rock_patch_20261117'
def main():
 page=Path(__file__).with_name('hierarchical_paint_platform_20261105.html').read_text('utf8')
 page=page.replace('按块完成 · 块内分层 · 连续笔画候选','单块主石局部对照 · 三套旧新计划并列')
 page=page.replace('输入细段 → 连续候选','所选计划原有笔数／候选笔数')
 page=page.replace('原图支持的断段连接','本轮新增连接（本轮为0）')
 page=page.replace("o.textContent=({G03_01:","o.textContent=s.label||({G03_01:")
 page=page.replace('await load(c.default);',"$('scene').value=c.default;await load(c.default);focusRock();")
 page=page.replace('<div class="buttons"><button id="play"','<div class="buttons"><button id="focusRock">定位本块主石</button><label><input type="checkbox" id="localZoom" checked>局部放大</label><label><input type="checkbox" id="localOnly" checked>本块结束即停回放</label><button id="play"')
 page=page.replace("c.scale(1.5,1.5);return c", "c.scale(1.5,1.5);if($('localZoom').checked){c.scale(496/120,496/120);c.translate(-102,-339)}return c")
 page=page.replace("$('phases').replaceChildren(...labels.map", "$('phases').replaceChildren(...(m&&m.local_phase_labels||labels).map")
 page=page.replace('${labels[next.phase]}','${(plan.modules[next.module].local_phase_labels||labels)[next.phase]}')
 page=page.replace("m.sam_key?'自动对象候选 · 未验证为独立山石':'仅空间分组 · 对象归属待定'", "m.local_phase_labels?'本块阶段由助手核查；非模型自主识别':plan.modules.length===13?'已有对象归属方案；未证明泛化':'原500笔对象候选'")
 page=page.replace("function tick(t)", "function localRange(){if(!plan)return [0,0];if(plan.modules.length===13){let v=plan.actions.filter(a=>a.module===4);return [v[0].sequence-1,v[v.length-1].sequence]}let a=plan.actions.findIndex(a=>{let p=plan.curves[a.curve].points;return p.some(q=>q[0]>132&&q[0]<205&&q[1]>370&&q[1]<411)});return [Math.max(0,a),Math.min(plan.actions.length,Math.max(0,a)+21)]}function focusRock(){if(!plan)return;playing=false;cursor=localRange()[0];$('play').textContent='播放笔序';render()}$('focusRock').onclick=focusRock;$('localZoom').onchange=render;function tick(t)")
 page=page.replace("if(cursor>=plan.actions.length){playing=false;$('play').textContent='播放笔序'}render()", "if($('localOnly').checked&&cursor>=localRange()[1]){cursor=localRange()[1];playing=false;$('play').textContent='播放笔序'}if(cursor>=plan.actions.length){playing=false;$('play').textContent='播放笔序'}render()")
 page=page.replace("$('play').onclick=()=>{if(cursor>=plan.actions.length)cursor=0;", "$('play').onclick=()=>{if($('localOnly').checked&&(cursor<localRange()[0]||cursor>=localRange()[1]))cursor=localRange()[0];if(cursor>=plan.actions.length)cursor=0;")
 page=page.replace("$('scene').onchange=()=>load($('scene').value).catch(fail)", "$('scene').onchange=()=>load($('scene').value).then(focusRock).catch(fail)")
 oldnote='区域是自动分割候选，尚未验证为独立山石；“内部长线”不等于已识别具体皴法。归属待定区保留标识。回放速度只影响动画，不是机械臂速度。'
 page=page.replace(oldnote,'原500笔认可版本完整保留；本轮只在既有489笔对象计划中处理前景主石21条，其余468条路径和笔序不变。阶段安排由助手核查；图像拟合不是新识别模型，纹理皴法仍待定。原500笔的对象尚未合并，局部定位只作比较。回放非实机、非相机。')
 page=page.replace('<footer>','<footer><p><a href="/report.html">查看原图、500笔基线、489笔基线、候选和风险说明</a></p>')
 (ROOT/'index.html').write_text(page,encoding='utf8')
 m=json.loads((ROOT/'metrics.json').read_text('utf8'));v=json.loads((ROOT/'verification.json').read_text('utf8'))
 report='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>前景中部主石 · 局部替换对照</title><style>*{box-sizing:border-box}body{font:17px/1.75 "Microsoft YaHei",sans-serif;color:#243f35;background:#f3f0e7;margin:0}main{max-width:1400px;margin:auto;padding:24px}img{max-width:100%;height:auto;display:block}a{color:#286d69}.note{background:#fff8e7;padding:15px;border-left:5px solid #b37639}.warn{color:#99523c}.scroll{overflow:auto}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ccd4c7;text-align:left}@media(max-width:700px){main{padding:14px}h1{font-size:25px}}</style><main><h1>从现有整幅中，只处理一块前景山石</h1><p class="note">本轮保留你认可的原500笔整幅，并复用已有489笔对象计划；选出主石21条路径，另外468条路径与对应笔序逐项不变。已接入同一个离线回放平台，可对照三套计划。没有替换正式版本。</p><p><a href="/">打开离线笔序回放</a> · 选择“单块主石局部候选”后点“定位本块主石”；关闭“局部放大”可检查整幅。回放速度不是机械臂速度。</p><h2>原图与旧新局部</h2><img src="/G03_01_local_candidate/rock.jpg" alt="原图、认可500笔同局部、已有489笔、拟合候选、有限笔宽风险和助手阶段安排"><p class="warn">左下相对墨带图是图像截面估计的风险预览：交汇与小闭环仍会粘连，未采用为笔宽方案，更没有换算成下压、速度或含墨量。整体图像只有496×496像素，放大不能恢复原图没有的笔形信息。</p><h2>这轮的实际改动</h2><p>局部保留21条，保护完整外缘和共享边4条原几何；17条尝试沿原图局部明暗作有界法向拟合，端点固定，最大约0.65像素。没有跨空白接线，没有新增或删除笔画。变化较小，不能称为提取突破。</p><p>阶段安排从旧“轮廓4条＋内部17条”，改为“轮廓／共享边待定4条→结构转折4条→内部纹理候选13条”。其中前侧边界先画；包含底边与内部转折的连续路径整条保留，不拆成短笔。共享边、具体皴法和真实画家分笔仍未确认。这是助手对本幅的核查，不是新模型自动语义识别。</p><div class="scroll"><table><thead><tr><th>检查</th><th>结果</th></tr></thead><tbody><tr><td>认可500笔原文件</td><td>哈希不变，副本逐字节一致</td></tr><tr><td>489笔工作基线原文件</td><td>哈希不变，另建候选</td></tr><tr><td>其他468条路径与动作记录</td><td>逐项不变</td></tr><tr><td>局部全部端点、外缘与共享边4条</td><td>保持</td></tr><tr><td>每条曲线绘制一次、对象无回访</td><td>技术检查通过</td></tr><tr><td>新神经模型训练、皴法辨识、实机验证</td><td>本轮均未完成或未进行</td></tr></tbody></table></div><h2>候选是否采用</h2><p>暂不采用。局部细碎纹理和小闭环仍在，图像拟合只带来小幅位置变化。现在已有明确的局部对照与接入接口；后续只替换这一块的曲线或笔触，不用重建整幅系统，也不把旧500→489的笔数变化算成本轮成果。</p><p>下一步应在这块石头上建立可靠的完整笔触与墨带参考，先区分可见结构转折、内部墨迹及设色边界，再比较真正重建的对象候选。当前旧路径不当作训练真值。树木与亭子的提取、曲线及回放记录没有因本轮修改而改变。</p><p><a href="/">回到对照回放</a> · <a href="/G03_01_approved500/plan.json">原500笔计划副本</a> · <a href="/G03_01_baseline489/plan.json">489笔工作基线</a> · <a href="/G03_01_local_candidate/plan.json">局部候选</a></p><p>本轮离线；没有连接、使能或移动机械臂。</p></main></html>'''
 (ROOT/'report.html').write_text(report,encoding='utf8');print('PLATFORM_READY',m['fitted_paths'],v['max_baseline_sample_distance_px'])
if __name__=='__main__':main()
