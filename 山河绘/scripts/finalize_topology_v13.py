# -*- coding: utf-8 -*-
"""Record negative transfer and optional review readiness without hiding it."""
import sys,json,shutil,threading,urllib.request
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure
configure()
import numpy as np,torch
from PIL import Image
from stroke_topology_v13 import ROOT,PARENT,jsave,digest,targets,loss_fn
from topology_trace_v13 import extract

def run():
    result=json.loads((ROOT/'RESULTS_SUMMARY.json').read_text());real=json.loads((ROOT/'evaluation'/'real_metrics.json').read_text())['records'];fixed=json.loads((ROOT/'frozen_center_ablation'/'metrics.json').read_text())['records'];by={r['id']:r for r in fixed}
    comparison=[]
    for r in real:
        comparison.append(dict(id=r['id'],old_best_single_path_coverage=r['before']['mean_best_single_path_coverage'],new_joint_model=r['new_final']['mean_best_single_path_coverage'],fixed_old_center_with_new_topology=by[r['id']]['fixed_pixels_final']['mean_best_single_path_coverage'],old_fragments=r['before']['mean_reference_fragments'],new_joint_fragments=r['new_final']['mean_reference_fragments'],frozen_fragments=by[r['id']]['fixed_pixels_final']['mean_reference_fragments']))
    result.update(real_comparison=comparison,full_model_real_regression_detected=any(r['new_joint_model']<r['old_best_single_path_coverage']-.03 for r in comparison),frozen_center_ablation_done=True,independent_real_test_available=False,new_review_annotations_used=False)
    jsave(ROOT/'RESULTS_SUMMARY.json',result)
    # Check the added weak-prior channel also ignores purple regions.
    d=dict(width=48,height=48,sample_id='purple',strokes=[dict(role='contour',points=[[5,20],[42,20]])],regions=[dict(kind='uncertain',points=[[20,0],[30,0],[30,40],[20,40]])]);t,m=targets(d,False);t=np.concatenate((t,np.full_like(t[:1],.5)));m=np.concatenate((m,((1-m[0])*(1-m[1])*m[7])[None]));o=torch.randn(1,8,48,48,requires_grad=True);loss,_=loss_fn(o,torch.from_numpy(t[None]),torch.from_numpy(m[None]));loss.backward();assert o.grad[:,:,0:40,21:30].abs().sum()==0
    u=np.linspace(0,2*np.pi,360);p=np.stack((64+40*np.cos(u),64+40*np.sin(u)),1);d=dict(width=128,height=128,sample_id='closed',strokes=[dict(role='contour',points=p.tolist())],regions=[]);t,m=targets(d,True);paths,_,_=extract(t);assert len(paths)==1 and np.linalg.norm(paths[0][0]-paths[0][-1])<3
    uniform=np.zeros((8,128,128),np.float32);uniform[0]=.7;uniform[1]=uniform[3]=1;paths,_,_=extract(uniform);assert not paths
    # Reuse the previously tested editor code. Validate all four draft labels
    # and read their API without submitting or modifying user annotations.
    review=ROOT/'user_review';sys.path.insert(0,str(review));from review_app import make_server,validate_document
    package=json.loads((review/'package_manifest.json').read_text());states=[]
    for r in package['samples']:
        d=json.loads((review/'labels'/(r['id']+'.json')).read_text());validate_document(d,r,package['package_id']);assert digest(review/r['image'])==r['image_sha256'];states.append(dict(id=r['id'],state=d['review_state'],provenance=d.get('label_provenance')))
    server=make_server(review,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();address='http://127.0.0.1:'+str(server.server_port)
    try:
        assert len(json.load(urllib.request.urlopen(address+'/api/manifest'))['samples'])==4
        for r in package['samples']:
            assert json.load(urllib.request.urlopen(address+'/api/labels/'+r['id']))['document']['sample_id']==r['id']
    finally:server.shutdown();server.server_close();thread.join(timeout=5)
    jsave(ROOT/'review_tool_verification.json',dict(api_and_schema_checked=True,initial_label_states=states,no_user_labels_modified_by_smoke_test=True))
    for file in [ROOT/'small_fit'/'final.pt',ROOT/'main'/'final.pt',ROOT/'pairs'/'final.pt']:
        data=torch.load(file,map_location='cpu',weights_only=False)['model'];assert all(bool(torch.isfinite(x).all()) for x in data.values())
    for f in ROOT.rglob('*.jpg'):
        with Image.open(f) as im:im.verify()
    verification=json.loads((ROOT/'verification.json').read_text());verification.update(purple_weak_prior_gradient_zero=True,legitimate_closed_contour_preserved=True,uniform_region_does_not_create_fake_loop=True,frozen_center_ablation_preserved=True,review_tool_api_checked=True,parent_unchanged=digest(PARENT)==json.loads((ROOT/'small_fit'/'diagnostic.json').read_text())['baseline_sha256']);jsave(ROOT/'verification.json',verification)
    report=(ROOT/'REPORT.md').read_text();report+='''
## 最终采用判断与固定主线对照

完整 v13 联合模型在多张真实图片上退步，因此不替换 v12。程序小样和程序留出场景虽有明显改善，不能据此宣称真实山石提取更好。

补做固定 v12 主线概率及旧版脊线骨架的模块对照，只换分支图处理、交叉配对及端点连接。结果可避免部分联合模型退步，但改善仍不一致。这个对照同时冻结了概率与脊线提取，因此只能定位为“新定位/追踪组合需要拆开校准”，不能单凭它断言退步只来自某一个神经输出。

| 局部 | v12 | 完整 v13 | 固定 v12 主线后的新连接 |
|---|---:|---:|---:|
'''
    for r in comparison:report+='| '+r['id']+' | '+f"{r['old_best_single_path_coverage']:.1%} | {r['new_joint_model']:.1%} | {r['fixed_old_center_with_new_topology']:.1%}"+' |\n'
    report+='''
以上均为部分参考线的最佳单路径覆盖，非真实整图精确率。错误连接可能增加单路径覆盖，因此还应结合碎片和疑似混笔提示、实际叠加图检查；不挑每张最高分拼成一个“最佳模型”。

固定主线图在 frozen_center_ablation/comparison.jpg，逐图数据在其 metrics.json。

## 尚未完成的真实数据环节

本轮尚没有新收到的、完整复核的真实清晰山石线稿。小样拟合通过只针对三个已知程序场景；尚不能替代“3～5张真实清晰图的完整逐笔拟合”检查。

已准备 user_review/start_review.cmd，含 R01–R04 四个清晰局部，初始线均为助手草稿。用户可逐笔修正、补全并确认想让机械臂采用的分笔；工具保持未标区域未知，不会自动宣称完整标注。它们尚未用于本轮训练，后续仅在用户完成复核、明确标注范围后另行纳入。

下一步优先固定现有较稳的主线模块，完成这批真实小样的完整复核，再分别校准关键点和连接。先验证真实小样，暂不扩大到几百几千张自动草标图。
'''
    (ROOT/'REPORT.md').write_text(report,encoding='utf-8')
    for name in ('frozen_center_ablation_v13.py','finalize_topology_v13.py','prepare_review_topology_v13.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/('source_'+name))
    print('FINAL_V13_VERIFIED',json.dumps(dict(regression_detected=result['full_model_real_regression_detected'],review_ready=str(review),status='experimental_not_promoted')),flush=True)
if __name__=='__main__':run()
