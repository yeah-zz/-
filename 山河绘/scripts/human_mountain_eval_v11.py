# -*- coding: utf-8 -*-
"""Shared postprocessing before/after, partial-label fit metrics only."""
import sys,json,math,time,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,cv2,torch
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from gongbi_training.stroke_train_v08 import model_new,infer
from original_d_tracer import LandscapeBrushGenerator as Trace
from human_mountain_v11 import ROOT,PARENT,IDS,FONT,SMALL,jsave,digest,paste,fields

def extract(prediction):
    # Locate ridges along the learned normal before thinning. Skeletonizing a
    # filled mountain response creates its medial axis, not the painted lines.
    p=prediction[0];smooth=cv2.GaussianBlur(p.astype(np.float32),(0,0),.65);theta=.5*np.arctan2(prediction[2],prediction[1]);ny=np.cos(theta);nx=-np.sin(theta);yy,xx=np.mgrid[:p.shape[0],:p.shape[1]].astype(np.float32)
    side1=cv2.remap(smooth,xx+nx*1.5,yy+ny*1.5,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101);side2=cv2.remap(smooth,xx-nx*1.5,yy-ny*1.5,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    ridge=(smooth>=side1)&(smooth>=side2)&(smooth-(side1+side2)*.5>.001)
    n,labels,stats,_=cv2.connectedComponentsWithStats(((p>=.30)&ridge).astype(np.uint8),8);keep=np.zeros(p.shape,np.uint8)
    strong=np.unique(labels[p>=.50]);good=np.zeros(n,bool);good[strong]=True;good[0]=False;good[stats[:,cv2.CC_STAT_AREA]<12]=False;keep=good[labels]
    binary=skeletonize(keep).astype(np.uint8)*255;binary[[0,-1],:]=0;binary[:,[0,-1]]=0
    paths=[];sk=np.zeros_like(binary)
    for path in Trace._trace_skeleton_paths(binary):
        points=np.asarray(path,float)
        if len(points)<2 or np.linalg.norm(np.diff(points,axis=0),axis=1).sum()<12:continue
        paths.append(points);cv2.polylines(sk,[np.rint(points).astype(np.int32)],False,1,1)
    return sk,paths

def metrics(doc,pred,sk,mask):
    center,_,_,_=fields(doc);center=center*(mask[3]>0);distance=distance_transform_edt(sk==0) if sk.any() else np.full(sk.shape,10000.)
    tolerance3=float(np.mean(distance[center>0]<=3));tolerance6=float(np.mean(distance[center>0]<=6));per=[]
    for stroke in doc['strokes']:
        sampled=[];points=np.asarray(stroke['points'],float)
        for a,b in zip(points[:-1],points[1:]):
            t=np.linspace(0,1,max(2,math.ceil(np.linalg.norm(b-a)*2)));sampled.extend(a[None]+(b-a)[None]*t[:,None])
        xy=np.rint(sampled).astype(int);xy[:,0]=np.clip(xy[:,0],0,sk.shape[1]-1);xy[:,1]=np.clip(xy[:,1],0,sk.shape[0]-1)
        valid=mask[3,xy[:,1],xy[:,0]]>0
        if valid.any():per.append(dict(id=stroke['id'],within3=float(np.mean(distance[xy[valid,1],xy[valid,0]]<=3))))
    bg=mask[1]>0
    return dict(partial_reference_coverage_3px=tolerance3,partial_reference_coverage_6px=tolerance6,reference_strokes_with_90pct_points_covered=sum(x['within3']>=.9 for x in per)/max(1,len(per)),
      explicit_background_pixels=int(bg.sum()),explicit_background_high_probability_fraction=float(np.mean(pred[0,bg]>=.5)) if bg.any() else None,
      explicit_background_candidate_pixels=int(sk[bg].sum()) if bg.any() else None,candidate_pixels=int(sk.sum()),mean_partial_reference_distance_px=float(distance[center>0].mean()),
      per_user_stroke=per,real_precision_unavailable=True,independent_test=False)

def draw_result(rgb,sk):
    line=np.full_like(rgb,255);line[sk>0]=[25,25,25];overlay=rgb.copy();mask=cv2.dilate(sk,np.ones((2,2),np.uint8))>0;overlay[mask]=np.uint8(rgb[mask]*.35+np.array([235,35,25])*.65);return line,overlay

def run(root=None):
    global ROOT
    if root is not None:ROOT=Path(root)
    torch.set_num_threads(4);cv2.setNumThreads(2);manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'));records=manifest['records'];folder=ROOT/'evaluation';folder.mkdir(exist_ok=True)
    model=model_new().cuda();model.load_state_dict(torch.load(ROOT/'final.pt',map_location='cuda',weights_only=False)['model']);model.eval();rows=[];board=Image.new('RGB',(2000,455*len(records)+45),'#f5f3ec');d=ImageDraw.Draw(board)
    d.text((10,5),'A01–A05：人工标注训练拟合对照（同一画作，非独立测试）；两版使用相同后处理',font=FONT,fill='black')
    for i,r in enumerate(records):
        identity=r['id'];out=folder/identity;out.mkdir(exist_ok=True);rgb=np.asarray(Image.open(r['snapshot_image']).convert('RGB'));doc=json.loads(Path(r['snapshot_labels']).read_text(encoding='utf-8'));z=np.load(ROOT/'snapshot'/(identity+'_targets.npz'));old=z['parent'];new=infer(model,rgb);oldsk,oldpaths=extract(old);newsk,newpaths=extract(new)
        oldline,_=draw_result(rgb,oldsk);newline,overlay=draw_result(rgb,newsk);reference=np.full_like(rgb,255);center,_,mask,_=fields(doc);center=center*(mask[3]>0);reference[cv2.dilate(center,np.ones((2,2),np.uint8))>0]=[235,75,20]
        for name,a in [('original',rgb),('user_reference',reference),('before',oldline),('after',newline),('after_overlay',overlay)]:Image.fromarray(a).save(out/(name+'.png'))
        np.savez_compressed(out/'probabilities.npz',before=old,after=new)
        jsave(out/'candidate_paths.json',dict(coordinate_units='pixels',actually_drawn=False,physical_z=None,physical_speed=None,ready_for_robot=False,paths=[p.round(2).tolist() for p in newpaths]))
        comparison=Image.new('RGB',(2000,455),'#f5f3ec');cd=ImageDraw.Draw(comparison)
        for j,(label,a) in enumerate([('原图',rgb),('你的山体主线（参考）',reference),('训练前候选线',oldline),('训练后候选线',newline),('训练后叠加（红色）',overlay)]):
            cd.text((j*400+8,5),identity+' '+label,font=SMALL,fill='black');paste(comparison,a,j*400+8,38,384,405)
        comparison.save(out/'comparison.jpg',quality=95);board.paste(comparison,(0,i*455+40));rows.append(dict(id=identity,before=metrics(doc,old,oldsk,mask),after=metrics(doc,new,newsk,mask),before_paths=len(oldpaths),after_paths=len(newpaths),label_sha256=r['label_sha256']))
        print('EVALUATED',identity,len(oldpaths),len(newpaths),flush=True)
    board.save(folder/'all_reviewed.jpg',quality=95)
    # Use a predeclared three-row subset for a more legible first preview.
    small=Image.new('RGB',(2000,455*3+45),'#f5f3ec');sd=ImageDraw.Draw(small);sd.text((10,5),'固定展示 A01 / A04 / A05；全部五张另存。人工标注图上的拟合，非独立测试。',font=FONT,fill='black')
    for i,identity in enumerate(['A01','A04','A05']):small.paste(Image.open(folder/identity/'comparison.jpg'),(0,45+i*455))
    small.save(folder/'first_preview.jpg',quality=95)
    source=LAB/'data'/'processed'/'user_qianli_07'/'image.png';rgb=np.asarray(Image.open(source).convert('RGB'));old=np.load(LAB/'full_landscape_probe_20261024'/'user_qianli_07'/'raw_probabilities.npz')['geometry'];new=infer(model,rgb)
    if digest(PARENT)!=json.loads((ROOT/'config.json').read_text())['parent_sha256']:raise RuntimeError('parent changed')
    oldsk,oldpaths=extract(old);newsk,newpaths=extract(new);oldline,_=draw_result(rgb,oldsk);newline,overlay=draw_result(rgb,newsk);whole=folder/'same_work_full_scene';whole.mkdir(exist_ok=True)
    for name,a in [('original',rgb),('before',oldline),('after',newline),('overlay',overlay)]:Image.fromarray(a).save(whole/(name+'.png'))
    np.savez_compressed(whole/'probabilities.npz',before=old,after=new)
    scene=Image.new('RGB',(2000,500),'#f5f3ec');dd=ImageDraw.Draw(scene);dd.text((10,5),'同一画作全图推理：含已训练局部及未标注内容，不代表其他画作上的泛化；细线候选非实画',font=FONT,fill='black')
    for j,(label,a) in enumerate([('原图',rgb),('训练前候选线',oldline),('训练后候选线',newline),('训练后叠加（红色）',overlay)]):dd.text((j*500+8,45),label,font=FONT,fill='black');paste(scene,a,j*500+8,85,484,400)
    scene.save(whole/'comparison.jpg',quality=95)
    def weighted(key,side):return sum(r[side][key]*records[i]['strokes'] for i,r in enumerate(rows))/sum(r['strokes'] for r in records)
    summary=dict(reviewed_images=len(records),reviewed_strokes=sum(r['strokes'] for r in records),source_groups=1,independent_test=False,score_type='training fit to partial user mountain-mainline traces, not full-picture precision',
      stroke_weighted_coverage_3px_before=weighted('partial_reference_coverage_3px','before'),stroke_weighted_coverage_3px_after=weighted('partial_reference_coverage_3px','after'),
      stroke_weighted_coverage_6px_before=weighted('partial_reference_coverage_6px','before'),stroke_weighted_coverage_6px_after=weighted('partial_reference_coverage_6px','after'),
      postprocessing=dict(method='normal-direction nonmaximum suppression before common hysteresis and thinning',normal_offset_px=1.5,min_ridge_contrast=.001,low=.30,high=.50,minimum_component_pixels=12,minimum_path_length_pixels=12,same_for_both=True,manual_prediction_cleanup=False),
      unmarked_houses_trees_falls='unknown, not evaluated as false positives',cunfa_classification_trained=False,width_darkness_trained=False,actual_drawing=False,robot_ready=False,rows=rows,whole_scene=dict(before_paths=len(oldpaths),after_paths=len(newpaths),independent_test=False))
    jsave(folder/'metrics.json',summary)
    report=f'''# 人工山体主线校准实验

仅使用你确认的 A01–A05，共 {summary['reviewed_strokes']} 条描线。统一解释为山体主线；没有据此训练具体皴法类别。未描的房屋、树木、瀑布及其他内容保持未知。

本阶段训练 {json.loads((ROOT/'training_summary.json').read_text())['successful_updates']} 次有效更新；保留旧模型，未连接机械臂。只训练 RGB 到主线中心及无向切线。粗细、浓淡、物理下压和笔顺均未在本轮训练。

这是同一画作上局部人工标注的拟合实验，没有独立画作验证，不能据此报告全图准确率。对照中两版使用相同阈值和后处理，没有手工删改预测线。橙色人工参考与黑色模型输出分栏，人工线没有混入输出。

按每张标线数量加权的部分参考线点覆盖（3px容差）：{summary['stroke_weighted_coverage_3px_before']:.3f} → {summary['stroke_weighted_coverage_3px_after']:.3f}；6px容差：{summary['stroke_weighted_coverage_6px_before']:.3f} → {summary['stroke_weighted_coverage_6px_after']:.3f}。覆盖高也可能伴随重复、粗宽响应、额外杂线，不能当作精确率。

逐图背景区域误检和标线覆盖见 evaluation/metrics.json。只在用户明确圈出的背景内统计误检；未标出的细节不作为负例。

查看 evaluation/first_preview.jpg、evaluation/all_reviewed.jpg，以及 evaluation/same_work_full_scene/comparison.jpg。同画全图展示包括已训练区域，并非独立测试。
'''
    (ROOT/'REPORT.md').write_text(report,encoding='utf-8')
    for f in folder.rglob('*.png'):
        with Image.open(f) as im:im.verify()
    for p in model.parameters():
        if not torch.isfinite(p).all():raise RuntimeError('nonfinite model parameter')
    jsave(ROOT/'verification.json',dict(weights_finite=True,png_verified=True,parent_unchanged=True,snapshot_hashes_match=all(digest(r['snapshot_labels'])==r['label_sha256'] for r in records),robot_connected=False))
    print('EVALUATION_COMPLETE',json.dumps({k:v for k,v in summary.items() if k not in ['rows','postprocessing']}),flush=True)
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path);args=parser.parse_args();run(args.root)
