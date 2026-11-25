# -*- coding: utf-8 -*-
"""Evaluate grouped strokes and open-set distance on exactly the archived Gabor folds."""

from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from gongbi_training.runtime import LAB, configure
from recognition_groups_20261123 import label, read, sha, verify_baselines
from recognition_scene_topology_20261124 import write


OUT=LAB/"crosswork_cunfa_20261124"
GABOR=LAB/"cunfa_gabor_baseline_20261110"


def measures(image, cv2, skeletonize, hog, observed_marks):
    rgb=np.asarray(ImageOps.contain(image.convert("RGB"),(160,160)))
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    a=cv2.GaussianBlur(gray,(0,0),.7)
    relief=np.maximum(cv2.GaussianBlur(a,(0,0),6)-a,0)
    relief=np.clip(relief/max(.06,float(np.quantile(relief,.97))),0,1)
    h=hog(cv2.resize(relief,(128,128)),orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1))
    h=h.reshape(8,8,8)
    hog_features=np.r_[h.mean((0,1)),h.std((0,1)),h[:4].mean((0,1)),h[4:].mean((0,1))]
    marks,density=observed_marks(Image.fromarray(rgb),cv2,skeletonize)
    single=np.zeros(10)
    group=np.zeros(13)
    if marks:
        lengths=np.asarray([m[2] for m in marks])
        widths=np.concatenate([m[1] for m in marks])
        angles=np.asarray([np.arctan2(*(m[0][-1]-m[0][0])[::-1]) for m in marks])
        center=np.asarray([m[0].mean(axis=0) for m in marks])
        spacing=np.linalg.norm(center[:,None]-center[None,:],axis=-1)
        np.fill_diagonal(spacing,np.inf)
        closest=np.min(spacing,axis=1) if len(center)>1 else np.array([0.])
        single=np.r_[np.quantile(lengths,[.5,.85]),np.quantile(widths,[.5,.85,.95]),
                     np.mean([m[3]/m[2] for m in marks]),np.mean([m[1].std() for m in marks]),
                     np.mean([m[4].mean() for m in marks]),np.mean([m[4].std() for m in marks]),np.log1p(len(marks))]
        weights=np.histogram(angles%np.pi,bins=8,range=(0,np.pi),weights=lengths)[0]/lengths.sum()
        group=np.r_[weights,density,np.abs(np.average(np.exp(2j*angles),weights=lengths)),
                    np.quantile(closest,[.5,.85]),lengths.max()/lengths.sum()]
    surface=[]
    for sigma in (1.,2.5,5.):
        base=cv2.GaussianBlur(relief,(0,0),sigma)
        dx=cv2.Sobel(base,cv2.CV_32F,1,0,ksize=3)
        dy=cv2.Sobel(base,cv2.CV_32F,0,1,ksize=3)
        dx2,dy2,cross=(cv2.GaussianBlur(v,(0,0),sigma*2) for v in (dx*dx,dy*dy,dx*dy))
        coherence=np.hypot(dx2-dy2,2*cross)/np.maximum(dx2+dy2,1e-7)
        xx=cv2.Sobel(base,cv2.CV_32F,2,0,ksize=3)
        yy=cv2.Sobel(base,cv2.CV_32F,0,2,ksize=3)
        xy=cv2.Sobel(base,cv2.CV_32F,1,1,ksize=3)
        major=np.hypot(xx-yy,2*xy)
        surface.extend([float(np.quantile(coherence,.5)),float(np.quantile(coherence,.9)),
                        float(np.quantile(major,.85)*sigma**2),
                        float(np.corrcoef(coherence.ravel(),relief.ravel())[0,1])
                        if coherence.std()>1e-6 and relief.std()>1e-6 else 0.])
    return single,group,np.r_[surface,hog_features]


def main():
    configure()
    import cv2
    from skimage.feature import hog
    from skimage.morphology import skeletonize
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from digital_specialty_round import observed_marks
    from train_cunfa_gabor_20261110 import metric

    OUT.mkdir(exist_ok=True)
    baseline=read(GABOR/"full_gabor_evaluation.json")
    manifest=read(GABOR/"dataset_manifest.json")
    lookup={r["id"]:r for r in manifest["records"]}
    rows=[lookup[p["id"]] for p in baseline["predictions"]]
    unknown=[r for r in manifest["records"] if r["label"]=="其他／无法确认"]
    names=("披麻皴","斧劈皴")
    groups_by_id={i:f["held_artist_group"] for f in baseline["folds"] for i in f["test_ids"]}
    groups=np.array([groups_by_id[r["id"]] for r in rows])
    y=np.array([names.index(r["label"]) for r in rows])
    assert len(rows)==40 and np.bincount(y).tolist()==[20,20]
    assert all(r["label"]==p["label"] for r,p in zip(rows,baseline["predictions"]))
    assert all(len({groups[i] for i,r in enumerate(rows) if r["work_group"]==work})==1 for work in {r["work_group"] for r in rows})
    audit=[]
    features=[]
    paper_features=[]
    for row in rows+unknown:
        with Image.open(row["image"]) as im:
            values=measures(im,cv2,skeletonize,hog,observed_marks)
            rgb=np.asarray(im.convert("RGB"),dtype=float)/255
            intensity=np.asarray(im.convert("L"),dtype=float)/255
            paper=rgb[intensity>=np.quantile(intensity,.70)]
            paper_features.append(np.r_[paper.mean(axis=0),paper.std(axis=0),np.quantile(paper,[.2,.8])])
            audit.append(dict(id=row["id"],source=row["source_url"],kind=row["label"],
                 artist=row["artist_group"],work=row["work_group"],crop=row["source_box"],
                 shape=list(im.size),image_hash=sha(row["image"]),source_hash=sha(row["source_image"]),
                 label_scope=row["label_scope"],contrast=float(np.asarray(im.convert("L")).std())))
            features.append(values)
    a,b,c=[np.stack([row[k] for row in features]) for k in range(3)]
    matrices=dict(single=a,group=np.c_[a,b],surface=c,fused=np.c_[a,b,c],background_proxy=np.asarray(paper_features))
    outer=list(LeaveOneGroupOut().split(y,y,groups))
    rng=np.random.default_rng(24)
    res={}
    for mode,x in matrices.items():
        predictions=np.zeros(len(y),int)
        considered=np.zeros(len(y),bool)
        train_true=[]
        train_pred=[]
        probes=[]
        for tr,te in outer:
            scaler=StandardScaler().fit(x[tr])
            classify=SVC(C=1,kernel="rbf",class_weight="balanced").fit(scaler.transform(x[tr]),y[tr])
            predict=lambda idx:classify.predict(scaler.transform(x[idx]))
            fit=predict(tr)
            train_true.extend(y[tr])
            train_pred.extend(fit)
            centroid=np.stack([scaler.transform(x[tr[y[tr]==value]]).mean(0) for value in (0,1)])
            calibration=[]
            for train_dev,val_dev in LeaveOneGroupOut().split(tr,y[tr],groups[tr]):
                sub=tr[train_dev]
                val=tr[val_dev]
                if len(np.unique(y[sub]))<2:
                    continue
                s=StandardScaler().fit(x[sub])
                means=np.stack([s.transform(x[sub[y[sub]==v]]).mean(0) for v in (0,1)])
                vals=s.transform(x[val])
                calibration.extend(np.sqrt(np.mean((vals-means[y[val]])**2,axis=1)))
            limit=float(np.quantile(calibration,.85))
            target=predict(te)
            values=scaler.transform(x[te])
            distance=np.sqrt(np.mean((values-centroid[target])**2,axis=1))
            predictions[te]=target
            considered[te]=distance<=limit
            if "Lampo_Leong" in set(groups[te]):
                idx=np.arange(len(y),len(x))
                result=predict(idx)
                distance=np.sqrt(np.mean((scaler.transform(x[idx])-centroid[result])**2,axis=1))
                probes=[dict(id=r["id"],class_candidate=names[k],distance=float(d),threshold=limit,
                             rejected=bool(d>limit)) for r,k,d in zip(unknown,result,distance)]
        res[mode]=dict(validation=metric(y,predictions),training_resubstitution_pooled=metric(np.asarray(train_true),np.asarray(train_pred)),
            correct_and_accepted=int(np.count_nonzero((y==predictions)&considered)),
            wrong_and_accepted=int(np.count_nonzero((y!=predictions)&considered)),
            known_rejected=int(np.count_nonzero(~considered)),unknown_probes=probes,
            rows=[dict(id=r["id"],work=r["work_group"],artist=r["artist_group"],expected=names[t],
                       predicted=names[p],accepted=bool(k)) for r,t,p,k in zip(rows,y,predictions,considered)])
    majority=np.zeros(len(y),int)
    for tr,te in outer:
        majority[te]=int(np.argmax(np.bincount(y[tr])))
    random_scores=[float(np.mean(rng.integers(0,2,len(y))==y)) for _ in range(200)]
    shuffled=[]
    for _ in range(40):
        v=rng.permutation(y)
        trial=[]
        truth=[]
        for tr,te in outer:
            s=StandardScaler().fit(matrices["fused"][tr])
            model=SVC(C=1).fit(s.transform(matrices["fused"][tr]),v[tr])
            trial.extend(model.predict(s.transform(matrices["fused"][te])))
            truth.extend(v[te])
        shuffled.append(float(np.mean(np.asarray(trial)==truth)))
    book=dict(folds=baseline["folds"],rows=audit,artist_counts=dict(Counter(r["artist"] for r in audit[:len(y)])),
              work_count=len({r["work"] for r in audit[:len(y)]}),gabor={k:baseline[k] for k in ("label_agreement","macro_f1","confusion")},
              variants=res,majority=metric(y,majority),random_mean=float(np.mean(random_scores)),
              shuffled_mean=float(np.mean(shuffled)),shuffled_upper_95=float(np.quantile(shuffled,.95)),
              independent_validation=False,manifest_sha256=sha(GABOR/"dataset_manifest.json"),
              checkpoints=verify_baselines(),distance_definition="z-scaled RMS to predicted class centroid; 85th percentile inner author holdout")
    write(OUT/"cunfa.json",book)
    board=Image.new("RGB",(1060,500),"#f4f1e8")
    pen=ImageDraw.Draw(board)
    label(pen,(14,8),"披麻皴与斧劈皴的按作者对照",24)
    label(pen,(14,46),"矩阵行是来源标签，列是模型预测。两个类别的顺序均为披麻、斧劈。",15)
    columns=(14,240,465,595,738,890)
    names_ui=("单笔特征","单笔与笔组","方向与纹理","全特征融合","亮底颜色")
    for x,title in zip(columns,("方法","混淆矩阵","标签一致","已知正确且接受","拒绝已知","拒绝未知")):
        label(pen,(x,80),title,15)
    rows_ui=[("原 Gabor 基线",str(baseline["confusion"]),"17/40","—","—","—")]
    for name,r in zip(names_ui,res.values()):
        rows_ui.append((name,str(r["validation"]["confusion"]),f"{round(r['validation']['label_agreement']*40)}/40",
                        f"{r['correct_and_accepted']}/40", f"{r['known_rejected']}/40",
                        f"{sum(t['rejected'] for t in r['unknown_probes'])}/{len(unknown)}"))
    for i,line in enumerate(rows_ui):
        top=108+i*34
        if i%2==0:
            pen.rectangle((6,top-2,1049,top+24),fill="#e9ede4")
        for x,value in zip(columns,line):
            label(pen,(x,top),value,15)
    label(pen,(14,327),"五个范围外示例",16)
    probe_titles=("解索皴","卷云皴","米点皴","横拖笔示例","钉头皴示例")
    for j,probe in enumerate(unknown):
        with Image.open(probe["image"]) as im:
            tile=ImageOps.contain(im.convert("RGB"),(180,115))
        board.paste(tile,(14+j*205,355))
        label(pen,(14+j*205,475),probe_titles[j],14)
    board.save(OUT/"cunfa_comparison.png")


if __name__=="__main__":
    main()
