# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import copy,json,shutil
import numpy as np,cv2
from PIL import Image,ImageDraw
from object_hierarchy_20261107 import object_masks,schedule,save,read,sha,FONT,SMALL,color
from structural_roles_20261107 import classify
from hierarchical_paint_core_20261105 import order_metrics
ROOT=LAB/'structural_roles_20261107';PRIOR=LAB/'object_hierarchy_20261107/G03_01'

def draw(rows,ids=None,phase=None):
    im=np.full((992,992,3),255,np.uint8)
    for i,r in enumerate(rows):
        if ids is not None and i not in ids:continue
        if phase is not None and r['phase']>phase:continue
        c=[(45,100,180),(160,95,35),(70,125,75)][r['phase']]
        cv2.polylines(im,[np.rint(np.asarray(r['points'])*2).astype(np.int32)],False,c,2,cv2.LINE_AA)
    return Image.fromarray(im)

def main():
    ROOT.mkdir(exist_ok=True);folder=ROOT/'G03_01';folder.mkdir(exist_ok=True)
    old=read(PRIOR/'plan.json');rows=copy.deepcopy(old['curves']);modules=copy.deepcopy(old['modules']);rgb=np.array(Image.open(PRIOR/'source.png').convert('RGB'))
    profiles=read(LAB/'approved_execution_refine_20261107/G03_01_500/width_targets.json')['profiles'];masks=object_masks()
    decisions=classify(rows,modules,masks,rgb,profiles)
    labels=dict(rock=['主轮廓／山体结构线','皴线勾勒候选','点景／待定细节'],tree=['主枝干','次级枝条','枝叶／松针候选'],architecture=['屋檐立柱等骨架','内部构件','建筑细节'])
    for r,d in zip(rows,decisions):
        r.update(previous_phase=r['phase'],phase=d['phase'],stroke_role=d['role'],role_uncertain=d['uncertain'],role_verified=False,
            role_inference='source/object-context rules; not trained semantic recognition')
    for m in modules:m['phase_labels']=labels.get(m['kind'],['主要结构','次要线条','细节'])
    for r in rows:r['phase_label']=modules[r['module']]['phase_labels'][r['phase']]
    route,order=schedule(rows,modules)
    plan=copy.deepcopy(old);plan.update(parent=str(PRIOR/'plan.json'),parent_sha256=sha(PRIOR/'plan.json'),curves=rows,modules=modules,actions=route,module_order=order,
        role_inference_method='source/object-context rules',traditional_cun_subtype_recognition=False,new_training=False,ready_for_robot=False)
    plan['stats']={**old['stats'],'baseline_order':order_metrics(old['actions']),'hierarchy_order':order_metrics(route)}
    save(folder/'plan.json',plan);save(folder/'role_decisions.json',decisions)
    for n in ('source.png','modules.png'):shutil.copy2(PRIOR/n,folder/n)
    boards=[]
    for mid,name in [(2,'中央主山石'),(8,'松树整体')]:
        ids=[i for i,r in enumerate(rows) if r['module']==mid];board=Image.new('RGB',(1560,595),'#f4f1e8');dr=ImageDraw.Draw(board)
        for phase in range(3):
            count=sum(rows[i]['phase']<=phase for i in ids);board.paste(draw(rows,ids,phase).resize((496,496)),(phase*520+10,45))
            dr.text((phase*520+10,10),f'{name} · {modules[mid]["phase_labels"][phase]} · 累计{count}笔',font=FONT,fill='#263e34')
        dr.text((10,552),'按实际阶段累积显示：蓝为第一阶段，棕为第二阶段，绿为第三阶段；角色是规则候选，未做专家准确率验证。',font=SMALL,fill='#263e34')
        board.save(folder/('tree.jpg' if mid==8 else 'rock.jpg'),quality=95);boards.append(board)
    combined=Image.new('RGB',(1560,1190),'#f4f1e8')
    for j,b in enumerate(boards):combined.paste(b,(0,j*595))
    combined.save(ROOT/'role_stages.jpg',quality=95);combined.save(folder/'comparison.jpg',quality=95);shutil.copy2(folder/'comparison.jpg',folder/'sequence_comparison.jpg')
    save(ROOT/'catalog.json',dict(scenes=[dict(id='G03_01',label='G03_01 · 轮廓/皴线与主干/枝叶分层')],default='G03_01',robot_connected=False))
    save(ROOT/'execution_binding.json',dict(scene=None,journal=None,plan_sha256=None))
    page=(LAB/'object_hierarchy_20261107/index.html').read_text('utf8')
    page=page.replace("labels.map((t,i)=>","(m?.phase_labels||labels).map((t,i)=>")
    page=page.replace('${labels[next.phase]}','${(plan.modules[next.module].phase_labels||labels)[next.phase]}')
    page=page.replace('<label><input type="checkbox" id="ghost" checked>', '<label><input type="checkbox" id="byrole" checked> 按阶段着色</label><label><input type="checkbox" id="ghost" checked>')
    page=page.replace('c.strokeStyle=color(act.module);',"c.strokeStyle=$('byrole').checked?['#2d64b4','#a05f23','#467d4b'][act.phase]:color(act.module);")
    page=page.replace("$('ghost').onchange=render;","$('byrole').onchange=render;$('ghost').onchange=render;")
    page=page.replace('已画：按块着色','已画默认按阶段着色：蓝=结构，棕=内部/皴线，绿=枝叶细节；取消勾选则按对象着色')
    page=page.replace('${a.length_px.toFixed(1)} px','${a.length_px.toFixed(1)} px · ${plan.curves[a.curve].phase_label}${plan.curves[a.curve].role_uncertain?"（待核对）":""}')
    page=page.replace('本幅对象归属由助手校正：完成当前对象全部笔画后才切换对象。轮廓/内部/细节仍为启发式分层；','本幅对象归属由助手校正；角色依据原图外边界、木质颜色邻域、笔宽与结构关系推断。完成主轮廓后才进入皴线；树木先主干，再次级枝条，最后枝叶。角色有待核对，非训练后语义识别证明；')
    (ROOT/'index.html').write_text(page,encoding='utf8')
    counts={m['name']:{m['phase_labels'][phase]:sum(r['module']==m['id'] and r['phase']==phase for r in rows) for phase in range(3)} for m in modules}
    save(ROOT/'role_summary.json',dict(counts=counts,changed_roles=sum(r['phase']!=r['previous_phase'] for r in rows),uncertain=sum(d['uncertain'] for d in decisions),
        no_new_training=True,source_specific_object_hierarchy_inherited=True,role_accuracy_not_established=True,geometry_unchanged=True))
    html='''<!doctype html><meta charset="utf-8"><title>对象内笔触分层</title><style>body{font:16px/1.7 Microsoft YaHei,system-ui;background:#f4f1e8;color:#243e33;max-width:1560px;padding:24px;margin:auto}img{width:100%}pre{white-space:pre-wrap}</style><h1>先轮廓后皴线，先主枝干后枝叶</h1><p>489笔坐标、对象归属和数量均未变化，只修改笔触角色与顺序。本轮用原图和对象上下文规则：山石外边界与贯通结构线先画，内部皴线后画；树木用木质颜色邻域、所在结构位置及可观察笔宽区分主干/枝条/枝叶。没有训练新模型，尚不能证明对其他画风可靠识别，也没有识别披麻皴等具体类别。</p><p>外边界来自上一轮本幅校正对象掩码，其误差会传递到角色；原图颜色、淡线和交汇也会影响判断。待定笔画保留且带标记，不以规则标签充当专家真值。排序通过不等于角色识别正确。</p><p><a href="/">打开分阶段笔序回放</a></p><h2>山石阶段</h2><img src="/G03_01/rock.jpg"><h2>松树阶段</h2><img src="/G03_01/tree.jpg"><pre>'''+json.dumps(counts,ensure_ascii=False,indent=2)+'</pre>'
    (ROOT/'report.html').write_text(html,encoding='utf8')
    print(json.dumps(read(ROOT/'role_summary.json'),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
