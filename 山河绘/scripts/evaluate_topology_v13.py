# -*- coding: utf-8 -*-
"""Separate overfit check, procedural holdout and partial real diagnostics."""
import sys,json,shutil
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure
configure()
import numpy as np,torch,cv2
from PIL import Image,ImageDraw
from stroke_topology_v13 import ROOT,OLD,PARENT,model_new,infer,procedural,targets,loss_fn,jsave,digest
from topology_trace_v13 import extract
from stroke_pairs_v13 import load
from stroke_link_v12 import connect,dense
from evaluate_clean_strokes_v12 import path_metrics,line_view
from human_mountain_v11 import fields,FONT,SMALL,paste

def direct(model,rgb):
    with torch.no_grad():
        p=model(torch.from_numpy(rgb.transpose(2,0,1).copy())[None].cuda().float()/255).float();p[:,[0,5,6,7]]=p[:,[0,5,6,7]].sigmoid();p[:,1:3]=torch.nn.functional.normalize(p[:,1:3],dim=1);p[:,3:5]=torch.nn.functional.normalize(p[:,3:5],dim=1);return p[0].cpu().numpy()

def score(reference,paths,mask):return path_metrics(reference,paths,mask)

def real_checks(model,link_model,threshold):
    folder=ROOT/'evaluation';folder.mkdir(exist_ok=True);records=json.loads((OLD/'manifest.json').read_text(encoding='utf-8'))['records'];rows=[]
    for r in records:
        out=folder/r['id'];out.mkdir(exist_ok=True);rgb=np.asarray(Image.open(r['image']).convert('RGB'));doc=json.loads(Path(r['labels']).read_text(encoding='utf-8'));_,_,mask,_=fields(doc)
        if r['id']=='U_A06':mask[:,0:86,0:96]=0
        pred=infer(model,rgb);left,top=r.get('padding',[0,0]);b=r.get('source_box',[0,0,rgb.shape[1],rgb.shape[0]]);w,h=b[2]-b[0],b[3]-b[1];content=pred[:,top:top+h,left:left+w];raw,_,_=extract(content,False);joined,info,_=extract(content,True);offset=np.array([left,top]);raw=[p+offset for p in raw];joined=[p+offset for p in joined];linked,events=connect(joined,pred[0],rgb,link_model,threshold)
        previous=json.loads((OLD/'evaluation'/r['id']/'trajectory.json').read_text());before=[np.asarray(p,float) for p in previous['paths']];reference=[s['points'] for s in doc['strokes']]
        results=dict(id=r['id'],split='reused_development' if r['split']!='train' else 'train',provenance=r['provenance'],before=score(reference,before,mask[3]),new_fields_no_junction_pairing=score(reference,[dense(p) for p in raw],mask[3]),new_fields_junction_pairing=score(reference,[dense(p) for p in joined],mask[3]),new_final=score(reference,linked,mask[3]),junction_pairings=len(info['junction_pairings']),endpoint_joins=len(events),independent_expert_accuracy=False)
        rows.append(results);jsave(out/'metrics.json',results);np.savez_compressed(out/'probabilities.npz',prediction=pred);jsave(out/'trajectory.json',dict(paths=[p.round(2).tolist() for p in linked],junction_pairings=info['junction_pairings'],gap_joins=events,physical_z=None,physical_speed=None,ready_for_robot=False,actually_drawn=False,stroke_ids_not_artist_verified=True))
        key=rgb.copy();key[pred[5]>.35]=[235,40,25];key[pred[6]>.35]=[25,80,235]
        before_im=line_view(rgb.shape[:2],before);raw_im=line_view(rgb.shape[:2],raw);after_im=line_view(rgb.shape[:2],linked,True);reference_im=line_view(rgb.shape[:2],reference)
        for name,a in [('original',rgb),('reference',reference_im),('before',before_im),('new_fields_unpaired',raw_im),('after',after_im),('predicted_keypoints',key)]:Image.fromarray(a).save(out/(name+'.png'))
        board=Image.new('RGB',(2000,460),'#f5f3ec');d=ImageDraw.Draw(board);d.text((8,5),r['id']+' | '+('用户部分标注' if r['group_kind']=='user_reviewed' else '助手草标，非专家真值')+' | '+('训练图' if r['split']=='train' else '未参与本轮训练的重复开发图'),font=FONT,fill='black')
        for j,(label,a) in enumerate([('原图',rgb),('参考线（不是预测）',reference_im),('v12 最终轨迹',before_im),('v13 未配对分支',raw_im),('v13 最终候选笔画',after_im)]):d.text((j*400+8,42),label,font=SMALL,fill='black');paste(board,a,j*400+8,78,384,370)
        board.save(out/'comparison.jpg',quality=95);print('REAL_TOPOLOGY',r['id'],round(results['before']['mean_best_single_path_coverage'],3),round(results['new_final']['mean_best_single_path_coverage'],3),len(events),flush=True)
    jsave(folder/'real_metrics.json',dict(records=rows,all_references_partial=True,new_clean_references_not_expert=True,all_nontraining_comparisons_reused_development=True,full_precision_unavailable=True))
    for name,ids in [('clean_comparison',['C03','C05','T01','T02']),('user_comparison',['U_A01','U_A03','U_A06'])]:
        board=Image.new('RGB',(2000,460*len(ids)+35),'#f5f3ec');d=ImageDraw.Draw(board);d.text((8,4),'固定展示：参考与预测分栏；真实图片仍是部分标注，不是独立画作准确率',font=FONT,fill='black')
        for i,identity in enumerate(ids):board.paste(Image.open(folder/identity/'comparison.jpg'),(0,35+i*460))
        board.save(folder/(name+'.jpg'),quality=95)
    return rows

def procedural_checks(model,link_model,threshold):
    folder=ROOT/'diagnostics';folder.mkdir(exist_ok=True);rows=[];known=[];examples=[]
    for i in range(20):
        rgb,t,m,reference=procedural(740000+i);pred=direct(model,rgb);raw,_,_=extract(pred,False);joined,_,_=extract(pred,True);linked,events=connect(joined,pred[0],rgb,link_model,threshold);valid=np.ones(rgb.shape[:2],np.uint8)
        rows.append(dict(seed=740000+i,neural_predicted_input=True,unpaired=score(reference,[dense(p) for p in raw],valid),paired=score(reference,linked,valid)))
        control,_,_=extract(t[:8],False);oracle,_,_=extract(t[:8],True);known.append(dict(seed=740000+i,input='known reference fields, NOT neural prediction',unpaired=score(reference,control,valid),paired=score(reference,oracle,valid)))
        if i in (1,3,6):examples.append((rgb,t,pred,reference,control,oracle,raw,linked))
    jsave(folder/'unseen_procedural_neural.json',dict(records=rows,real_artwork_accuracy=False));jsave(folder/'known_reference_tracer.json',dict(records=known,input_not_neural_prediction=True))
    for label,oracle_only in [('known_reference',True),('unseen_procedural_neural',False)]:
        board=Image.new('RGB',(1600,400*len(examples)+40),'#f5f3ec');d=ImageDraw.Draw(board);d.text((8,5),'已知参考字段追踪诊断，不是模型提取成果' if oracle_only else '未训练程序场景上的神经模型输出，不是真实画作准确率',font=FONT,fill='black')
        for i,(rgb,t,p,reference,control,oracle,raw,linked) in enumerate(examples):
            before,after=(control,oracle) if oracle_only else (raw,linked)
            for j,(name,a) in enumerate([('程序输入图',rgb),('正确参考轨迹',line_view(rgb.shape[:2],reference)),('配对前分支',line_view(rgb.shape[:2],before,True)),('配对后候选',line_view(rgb.shape[:2],after,True))]):d.text((j*400+8,45+i*400),name,font=SMALL,fill='black');paste(board,a,j*400+8,75+i*400,384,355)
        board.save(folder/(label+'.jpg'),quality=95)
    return rows,known

def verification(model,records):
    # The assignment-invariant loss cannot prefer axis slot ordering.
    _,t,m,_=procedural(750001,64);out=torch.randn(1,8,64,64,requires_grad=True);target=torch.from_numpy(t[None]);mask=torch.from_numpy(m[None]);a,_=loss_fn(out,target,mask);swapped=out[:,[0,3,4,1,2,5,6,7]];b,_=loss_fn(swapped,target,mask);assert torch.allclose(a,b,atol=1e-6)
    # Explicit purple regions contribute no direct loss gradient in any head.
    doc=dict(width=64,height=64,sample_id='test',strokes=[dict(role='contour',points=[[5,20],[58,20]])],regions=[dict(kind='uncertain',points=[[25,0],[38,0],[38,50],[25,50]])]);t,m=targets(doc,False);p=torch.randn(1,8,64,64,requires_grad=True);loss,_=loss_fn(p,torch.from_numpy(t[None]),torch.from_numpy(m[None]));loss.backward();assert p.grad[:,:,0:50,26:38].abs().sum()==0
    assert all(bool(torch.isfinite(x).all()) for x in model.state_dict().values());train_manifest=json.loads((ROOT/'main'/'manifest.json').read_text());assert all(digest(r['image'])==r['image_sha256'] and digest(r['labels'])==r['label_sha256'] for r in train_manifest['real']);assert not any(r['id'] in ('T01','T02','U_A06') for r in train_manifest['real'])
    images=0
    for path in ROOT.rglob('*.jpg'):
        with Image.open(path) as im:im.verify()
        images+=1
    for path in (ROOT/'evaluation').rglob('trajectory.json'):
        d=json.loads(path.read_text());assert not d['ready_for_robot'] and d['physical_z'] is None
        for points in d['paths']:assert np.isfinite(np.asarray(points)).all()
    jsave(ROOT/'verification.json',dict(axis_slot_permutation_invariant=True,purple_all_head_loss_gradients_zero=True,weights_finite=True,source_snapshots_unchanged=True,test_labels_not_trained=True,verified_boards=images,robot_connected=False,experimental_not_promoted=True))

def finish(rows,synthetic,known):
    mean=lambda rr,side,key:float(np.mean([r[side][key] for r in rr]));pairs=json.loads((ROOT/'pairs'/'training_summary.json').read_text());small=json.loads((ROOT/'small_fit'/'diagnostic.json').read_text());training=json.loads((ROOT/'main'/'training_summary.json').read_text());result=dict(status='experimental_not_promoted',small_fit_updates=600,main_updates=1000,endpoint_pair_classifier_updates=800,new_real_expert_labels=0,partial_real_training_crops=11,procedural_main_training_scenes=120,
      known_small_fit_gate_passed=small['localization_gate_passed'],calibration_admissible_negative_pairs=pairs['calibration_admissible_negative_pairs'],known_reference_unpaired_vs_paired_best_path=[mean(known,'unpaired','mean_best_single_path_coverage'),mean(known,'paired','mean_best_single_path_coverage')],unseen_procedural_unpaired_vs_paired_best_path=[mean(synthetic,'unpaired','mean_best_single_path_coverage'),mean(synthetic,'paired','mean_best_single_path_coverage')],real_independent_artwork_accuracy=False,semantic_cunfa_classification=False,pressure_or_ink_darkness_not_trained=True,actually_drawn=False)
    jsave(ROOT/'RESULTS_SUMMARY.json',result)
    report='''# v13 起止点、交叉位置与双方向字段实验

本轮增加了 8 通道预测：主线中心、两组无向切线、起止点、交叉位置和双方向有效性。没有训练具体皴法名称、笔宽浓淡或机械臂物理动作。

先用 3 组已知程序轨迹做 600 次小样拟合，验证网络能学会这些目标；它们不是真实画作，也不是泛化分数。主训练重新从 v12 权重初始化，没有使用小样过拟合权重；混合原有 11 个真实局部与 120 个程序场景，训练 1000 次。

真实图仍是部分标注：6 个清晰局部由助手草描，5 个为用户此前复核的山体主线。没有把它们当成完整真值；没有描出的部分保持未知，紫色区排除所有直接训练损失。少量旧模型软正则只用于其余未知区，不是人工标签。

交叉追踪采用骨架分支图，再用局部方向和双方向字段进行保守配对。起止点与交叉热图已经训练；当前起止点图仍主要用于诊断，尚未单独控制抬笔。交叉图参与保留关键区域，连接选择仍包含几何规则，不能把所有收益归为神经训练。

另训练了 800 次三分类端点评分：同笔、不同笔、无法确定。5,400 个训练配对、1,800 个校准配对、1,800 个未训练程序测试配对。校准中过旧候选筛选条件的负例为 589 对，较 v12 的 5 对更充分；但它们来自有限的程序构造族，不能当作真实国画的可靠错接率。对于不可由图像唯一确定的干笔/抬笔关系，采用待复核标签。

T01、T02、A06 不进入本轮梯度训练，但已在前轮反复查看，均作为重复开发对照，不称新的独立盲测。A06 排除已知同扫描重叠和紫色区域评分。预测时不使用参考线、绿色/紫色区域去擦除或修饰输出。

## 真实图片逐笔对照

以下是每条参考线由最佳单条预测路径覆盖的比例，容差 3 像素。全部是真实图片的部分标注指标，不能代表全图精确率。

| 图片 | v12 最终轨迹 | v13 未配对分支 | v13 交叉配对 | v13 含保守缺口连接 |
|---|---:|---:|---:|---:|
'''
    for r in rows:
        report+='| '+r['id']+' | '+' | '.join(f"{r[s]['mean_best_single_path_coverage']:.1%}" for s in ['before','new_fields_no_junction_pairing','new_fields_junction_pairing','new_final'])+' |\n'
    report+=f'''
已知正确字段的独立追踪诊断（20 个程序场景，逐场景均值）：最佳单路径覆盖 {result['known_reference_unpaired_vs_paired_best_path'][0]:.1%} → {result['known_reference_unpaired_vs_paired_best_path'][1]:.1%}。主模型在另外 20 个未训练程序场景上的结果：{result['unseen_procedural_unpaired_vs_paired_best_path'][0]:.1%} → {result['unseen_procedural_unpaired_vs_paired_best_path'][1]:.1%}。二者明确分开，不作为真实山石准确率。

## 查看

- evaluation/clean_comparison.jpg：预先固定的 C03、C05、T01、T02。
- evaluation/user_comparison.jpg：预先固定的 A01、A03、A06。
- small_fit/comparison.jpg：已知程序小样训练拟合，非泛化。
- diagnostics/known_reference.jpg：追踪器已知输入诊断。
- diagnostics/unseen_procedural_neural.jpg：未训练程序场景中的神经模型输出。
- evaluation/real_metrics.json：逐图覆盖、碎片数与部分参考混笔检查提示。
- user_review/start_review.cmd：四个清晰局部的逐笔复核工具。初始标签仍是助手草稿，尚未用于本轮新复核训练。

当前结果仅为实验候选。没有替换既有绘画模型，没有连接、使能或移动机械臂。输入、缓存、权重和输出均在 E 盘。
'''
    (ROOT/'REPORT.md').write_text(report,encoding='utf-8')
    for name in ('stroke_topology_v13.py','topology_trace_v13.py','stroke_pairs_v13.py','evaluate_topology_v13.py','check_small_topology_v13.py'):
        shutil.copy2(Path(__file__).with_name(name),ROOT/('source_'+name))
    print('V13_COMPLETE',json.dumps(result),flush=True)

def run():
    torch.set_num_threads(4);cv2.setNumThreads(2);ROOT.mkdir(exist_ok=True);model=model_new().cuda();model.load_state_dict(torch.load(ROOT/'main'/'final.pt',map_location='cuda',weights_only=False)['model']);model.eval();link_model,threshold=load();rows=real_checks(model,link_model,threshold);synthetic,known=procedural_checks(model,link_model,threshold);verification(model,rows);finish(rows,synthetic,known)
if __name__=='__main__':run()
