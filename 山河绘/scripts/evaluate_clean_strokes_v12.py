# -*- coding: utf-8 -*-
"""Separate pixel prediction, path linking and known-reference diagnostics."""
import sys,json,math,shutil,time
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,torch,cv2
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
from clean_strokes_v12 import ROOT,jsave,digest,FONT
from human_mountain_v11 import fields,paste
from human_mountain_eval_v11 import extract
from gongbi_training.stroke_train_v08 import model_new,infer
from stroke_link_v12 import dense,synthetic,train as train_link,connect

def path_metrics(reference,paths,valid):
    pp=[dense(p) for p in paths if len(p)>=2];refs=[]
    for p in reference:
        q=dense(p);xy=np.rint(q).astype(int);xy[:,0]=np.clip(xy[:,0],0,valid.shape[1]-1);xy[:,1]=np.clip(xy[:,1],0,valid.shape[0]-1);q=q[valid[xy[:,1],xy[:,0]]>0]
        if len(q)>1:refs.append(q)
    if not pp:return dict(point_coverage_3px=0.,mean_best_single_path_coverage=0.,single_path_90pct_fraction=0.,mean_reference_fragments=0.,path_count=0,mixed_reference_paths_flagged=0)
    flat=np.concatenate(pp);labels=np.concatenate([np.full(len(p),i) for i,p in enumerate(pp)]);tree=cKDTree(flat);points=[];bests=[];fragment=[];single=[]
    for ref in refs:
        d,index=tree.query(ref);near=labels[index[d<=3]];points.extend((d<=3).tolist());candidates,counts=np.unique(near,return_counts=True);fragment.append(int((counts>=max(4,.05*len(ref))).sum()));order=np.argsort(-counts);best=0.
        for k in candidates[order[:8]]:best=max(best,float(np.mean(cKDTree(pp[int(k)]).query(ref)[0]<=3)))
        bests.append(best);single.append(best>=.9)
    # A partial-reference mixture is a review flag, not a real false-merge
    # accuracy claim. Distinct artist strokes cannot always be recovered.
    rf=np.concatenate(refs);ri=np.concatenate([np.full(len(p),i) for i,p in enumerate(refs)]);rtree=cKDTree(rf);mixed=0
    for p in pp:
        d,index=rtree.query(p);matched=ri[index[d<=3]]
        if len(matched)<20:continue
        _,counts=np.unique(matched,return_counts=True);mixed+=int((counts>=max(8,.12*len(matched))).sum()>1)
    return dict(point_coverage_3px=float(np.mean(points)),mean_best_single_path_coverage=float(np.mean(bests)),single_path_90pct_fraction=float(np.mean(single)),mean_reference_fragments=float(np.mean(fragment)),path_count=len(pp),mixed_reference_paths_flagged=mixed)

def line_view(shape,paths,color=False):
    canvas=np.full((*shape,3),255,np.uint8)
    for i,p in enumerate(paths):
        c=tuple(int(v) for v in cv2.cvtColor(np.uint8([[[i*37%180,190,175]]]),cv2.COLOR_HSV2RGB)[0,0]) if color else (30,30,30)
        cv2.polylines(canvas,[np.rint(p).astype(np.int32)],False,c,1,cv2.LINE_AA)
    return canvas

def predict_all():
    torch.set_num_threads(4);cv2.setNumThreads(2);manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'));model=model_new().cuda();model.load_state_dict(torch.load(ROOT/'final.pt',map_location='cuda',weights_only=False)['model']);model.eval();records=manifest['records']
    for r in records:
        out=ROOT/'evaluation'/r['id'];out.mkdir(exist_ok=True);rgb=np.asarray(Image.open(r['image']).convert('RGB'));old=np.load(r['arrays'])['parent'];new=infer(model,rgb);np.savez_compressed(out/'probabilities.npz',before=old,after=new);r['prediction']=str(out/'probabilities.npz')
        print('PREDICTED',r['id'],flush=True)
    del model;torch.cuda.empty_cache();return records

def evaluate_records(records,model,threshold):
    summaries=[]
    for r in records:
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));doc=json.loads(Path(r['labels']).read_text(encoding='utf-8'));_,_,mask,_=fields(doc);z=np.load(r['prediction']);old=z['before'];new=z['after']
        # Some narrow native tutorial crops were padded for batching. Trace
        # only the native content rectangle, identically for both models;
        # artificial padding borders are not part of the source artwork.
        left,top=r.get('padding',[0,0]);box=r.get('source_box',[0,0,rgb.shape[1],rgb.shape[0]]);width=box[2]-box[0];height=box[3]-box[1]
        def content_paths(pred):
            _,paths=extract(pred[:,top:top+height,left:left+width]);return [p+np.array([left,top]) for p in paths]
        before=content_paths(old);raw=content_paths(new);linked,events=connect(raw,new[0],rgb,model,threshold);out=ROOT/'evaluation'/r['id'];reference=[s['points'] for s in doc['strokes']]
        # A06 known same-scan overlap is removed from scoring only. Raw
        # predictions and plans do not use annotation masks or paths.
        if r['id']=='U_A06':mask[:,0:86,0:96]=0
        # connect() resamples input paths once even when it makes no joins.
        # Give the unlinked controls the same resampling for fair attribution.
        summary=dict(id=r['id'],split=r['split'],provenance=r['provenance'],before=path_metrics(reference,[dense(p) for p in before],mask[3]),new_pixels_old_tracer=path_metrics(reference,[dense(p) for p in raw],mask[3]),new_pixels_learned_links=path_metrics(reference,linked,mask[3]),hypothesized_joins=len(events),independent_expert_accuracy=False,native_content_only=True,matching_initial_path_resampling=True)
        jsave(out/'metrics.json',summary);jsave(out/'trajectory.json',dict(paths=[p.round(2).tolist() for p in linked],joins=events,coordinates='pixels',ready_for_robot=False,physical_z=None,actually_drawn=False,pen_order='not calibrated or artist-verified'));summaries.append(summary)
        before_image=line_view(rgb.shape[:2],before);raw_image=line_view(rgb.shape[:2],raw);plan=line_view(rgb.shape[:2],linked,True);ref_image=line_view(rgb.shape[:2],reference)
        for name,a in [('original',rgb),('before',before_image),('new_centerline',raw_image),('planned_paths',plan),('partial_reference',ref_image)]:Image.fromarray(a).save(out/(name+'.png'))
        board=Image.new('RGB',(1800,425),'#f5f3ec');d=ImageDraw.Draw(board);d.text((8,4),r['id']+' | '+r['split']+' | '+('用户局部标注' if r['group_kind']=='user_reviewed' else '助手草标，非专家准确率'),font=FONT,fill='black')
        for j,(label,a) in enumerate([('原图',rgb),('参考线（不是预测）',ref_image),('上一版中心线',before_image),('本轮中心线',raw_image),('本轮逐笔轨迹（颜色分笔）',plan)]):d.text((j*360+5,31),label,font=FONT,fill='black');paste(board,a,j*360+5,62,348,352)
        board.save(out/'comparison.jpg',quality=95);print('REAL_CHECK',r['id'],round(summary['before']['mean_best_single_path_coverage'],3),round(summary['new_pixels_learned_links']['mean_best_single_path_coverage'],3),len(events),flush=True)
    jsave(ROOT/'evaluation'/'summary.json',dict(records=summaries,provenance='new clean references are assistant drafts; user references partial and reused development',full_precision_not_available=True))
    ids=['C03','C05','T01','T02'];board=Image.new('RGB',(1800,425*len(ids)+35),'#f5f3ec');d=ImageDraw.Draw(board);d.text((8,5),'固定展示：两张清晰训练图＋全部两张不同来源对照；参考线与模型输出分栏',font=FONT,fill='black')
    for i,identity in enumerate(ids):board.paste(Image.open(ROOT/'evaluation'/identity/'comparison.jpg'),(0,35+425*i))
    board.save(ROOT/'evaluation'/'clean_comparison.jpg',quality=95)
    return summaries

def oracle_checks(model,threshold):
    summaries=[];chosen=[]
    for i in range(16):
        rgb,paths,prob=synthetic(530000+i);doc=dict(width=320,height=320,sample_id='known',strokes=[dict(role='contour',points=p.tolist()) for p in paths],regions=[]);center,angle,mask,dist=fields(doc);pred=np.concatenate((np.exp(-dist[None]**2/(2*1.1**2)),angle,np.zeros((3,320,320),np.float32))).astype(np.float32)
        if i%3==0:
            point=paths[0][110];xx,yy=np.rint(point).astype(int);patch=np.zeros(center.shape,np.uint8);cv2.circle(patch,(xx,yy),3,1,-1);pred[0,(patch>0)&(pred[0]>.3)]=.28
        _,before=extract(pred);after,events=connect(before,pred[0],rgb,model,threshold);valid=np.ones(center.shape,np.uint8);summary=dict(seed=530000+i,input='known reference probability and tangent, NOT model output',before=path_metrics(paths,[dense(p) for p in before],valid),after=path_metrics(paths,after,valid),joins=len(events));summaries.append(summary)
        if i in [0,1,3]:chosen.append((i,rgb,pred[0],before,after))
    jsave(ROOT/'diagnostics'/'known_reference_metrics.json',dict(records=summaries,procedural_not_real_extraction=True))
    board=Image.new('RGB',(1600,370*len(chosen)+45),'#f5f3ec');d=ImageDraw.Draw(board);d.text((8,5),'追踪器诊断：输入已知参考线，含人为低置信缺口；不是神经模型的提取成果',font=FONT,fill='black')
    for row,(identity,rgb,p,before,after) in enumerate(chosen):
        for j,(label,a) in enumerate([('程序化输入图',rgb),('已知参考线概率',np.uint8(255*(1-p))),('原追踪片段',line_view(p.shape,before,True)),('学习连接后轨迹',line_view(p.shape,after,True))]):d.text((j*400+5,45+row*370),str(identity)+' '+label,font=FONT,fill='black');paste(board,Image.fromarray(a),j*400+5,75+row*370,385,325)
    board.save(ROOT/'diagnostics'/'known_reference_comparison.jpg',quality=95);return summaries

def run():
    records=predict_all();training=[r for r in records if r['split']=='train'];model,threshold=train_link(training);summaries=evaluate_records(records,model,threshold);oracle=oracle_checks(model,threshold)
    jsave(ROOT/'final_summary.json',dict(neural_updates=1000,link_classifier_updates=500,clean_real_train_images=6,user_reviewed_train_images=5,clean_test_crops=2,clean_test_source_illustrations=1,new_test_source_group='wang_steps',clean_references_not_expert=True,known_procedural_neural_samples=80,known_procedural_link_train_scenes=160,known_procedural_link_validation_scenes=35,known_reference_diagnostic_scenes=16,selected_source_illustrations=7,independent_artwork_generalization_not_verified=True,semantic_cunfa_classification=False,pressure_calibration=False,robot_connected=False,actually_drawn=False))
    for f in (ROOT/'evaluation').rglob('*.png'):
        with Image.open(f) as im:im.verify()
    state=torch.load(ROOT/'final.pt',map_location='cpu',weights_only=False)['model'];assert all(bool(torch.isfinite(v).all()) for v in state.values())
    assert all(digest(r['image'])==r['image_sha256'] and digest(r['labels'])==r['label_sha256'] for r in records)
    for name in ['clean_stroke_data_v12.py','stroke_link_v12.py','evaluate_clean_strokes_v12.py']:shutil.copy2(Path(__file__).with_name(name),ROOT/('final_'+name))
    jsave(ROOT/'verification.json',dict(images_verified=True,weights_finite=True,source_labels_unchanged=True,test_labels_not_trained=True,robot_connected=False));print('ALL_V12_CHECKS_COMPLETE',flush=True)
if __name__=='__main__':run()
