# -*- coding: utf-8 -*-
"""Fit a fresh scene head on CLPD and compare with the frozen project baseline."""

from collections import Counter
import json
from pathlib import Path
import random

from professional_clpd import ROOT, MAP, fingerprint, near_distance, work_key, write
from gongbi_training.runtime import LAB, configure


def main():
    configure()
    import cv2
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from gongbi_training.scene_objects import SceneObjects
    from guohua_objects_train_20261112 import pack, WEIGHTS
    from recognition_groups_20261123 import SCENE, verify_baselines, sha

    out=ROOT/"scene_pilot"
    out.mkdir(exist_ok=True)
    manifest=ROOT/"clpd/manifest.json"
    records=json.loads(manifest.read_text("utf-8"))["rows"]
    history=[]
    for folder in ("guohua_objects_pilot_20261112","guohua_objects_v2_20261113",
                   "guohua_objects_v3_20261116","guohua_objects_v4_sparse_20261116",
                   SCENE.name,"object_prospective_20261118"):
        directory=LAB/folder
        for pattern in ("*manifest.json","*split*.json"):
            for path in directory.glob(pattern):
                try: document=json.loads(path.read_text("utf-8"))
                except (ValueError,OSError): continue
                if isinstance(document,dict) and isinstance(document.get("records"),list):
                    history.extend(r for r in document["records"] if isinstance(r,dict))
    prior={}
    for row in history:
        path=row.get("image") or row.get("image_path")
        if isinstance(path,str) and Path(path).is_file():
            prior[str(Path(path))]=row.get("image_sha256")
    checked=[]
    for path,digest in prior.items():
        with Image.open(path) as image:
            checked.append((path,digest or sha(path),work_key(path),fingerprint(image)))
    for row in records:
        matches=[]
        for path,digest,work,hashes in checked:
            exact=digest==row["image_sha256"]
            title=row["work_key"]==work
            near=near_distance(row["phash"],hashes)<=6
            if exact or title or near:
                matches.append({"image":path,"exact":exact,"named_work":title,"near_hash":near})
        row["historical_matches"]=matches
    write(out/"overlap_audit.json",dict(prior_images_examined=len(checked),rows=[
        dict(id=r["id"],split=r["split"],work=r["work_key"],matches=r["historical_matches"]) for r in records]))
    baselines=verify_baselines()
    s0_checkpoint=SCENE/"mapping_fixed.pt"
    expected=json.loads((SCENE/"checkpoint_freeze.json").read_text("utf-8"))["mapping_fixed"]
    assert sha(s0_checkpoint)==expected
    torch.manual_seed(20261125)
    torch.set_num_threads(4)
    device="cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark=False
    lut=np.full(256,255,np.uint8)
    for key,value in MAP.items():
        lut[key]=value
    resolution=256
    train=[r for r in records if r["split"]=="train"]
    dev=[r for r in records if r["split"]=="development"]
    test=[r for r in records if r["split"]=="evaluation"]
    assert set(r["group"] for r in train).isdisjoint(r["group"] for r in dev+test)
    assert set(r["group"] for r in dev).isdisjoint(r["group"] for r in test)
    epochs=12 if device=="cuda" else 6
    config=dict(input_size=resolution,epochs=epochs,batch=6,device=device,
                setup="frozen ImageNet MobileNetV3-Small encoder, fresh multiscale scene head",
                checkpoint_selection="fixed final epoch",source_revision=records[0]["revision"],
                optimizer="AdamW",learning_rate=.002,seed=20261125,
                dataset_sha256=sha(manifest),frozen_project_sha256=expected,encoder_sha256=sha(WEIGHTS),raw_to_scene=MAP)
    write(out/"experiment.json",config)
    for r in records:
        assert sha(r["image"])==r["image_sha256"] and sha(r["mask"])==r["mask_sha256"]
    def load(row):
        with Image.open(row["image"]) as image, Image.open(row["mask"]) as target:
            return np.asarray(image.convert("RGB")),lut[np.asarray(target)]

    def predict(model,rgb):
        x,_,(l,t,w,h,ow,oh)=pack(rgb,size=resolution)
        with torch.inference_mode():
            pred=model(torch.from_numpy(x)[None].to(device)).softmax(1)[0].cpu().numpy()
        return np.stack([cv2.resize(p[t:t+h,l:l+w],(ow,oh)) for p in pred])

    def confusion(target,pred,known=None):
        keep=target!=255
        if known is not None:
            keep &= known
        return np.bincount(4*target[keep].astype(int)+pred[keep],minlength=16).reshape(4,4)

    def measures(matrix):
        def ratio(n,d):
            return float(n/d) if d else None
        result={}
        for k,name in ((1,"mountain_rock"),(2,"trees")):
            tp=int(matrix[k,k]); actual=int(matrix[k].sum()); pred=int(matrix[:,k].sum())
            result[name]=dict(precision=ratio(tp,pred),recall=ratio(tp,actual),f1=ratio(2*tp,pred+actual),
                              iou=ratio(tp,pred+actual-tp),support=actual)
        return result

    def boundary(target):
        rock=target==1
        trees=target==2
        if not rock.any() or not trees.any():
            return np.zeros(target.shape,bool)
        near_rock=cv2.distanceTransform((~rock).astype(np.uint8),cv2.DIST_L2,5)<=8
        near_tree=cv2.distanceTransform((~trees).astype(np.uint8),cv2.DIST_L2,5)<=8
        return (rock & near_tree) | (trees & near_rock)

    def groups(values, pred, valid):
        confidence=values.max(0)[valid]
        errors=pred[valid]
        return np.minimum(np.rint(confidence*1023).astype(int),1023),errors

    def observe(model, cohort):
        total=np.zeros((4,4),int)
        edge=np.zeros((4,4),int)
        correct_hist=np.zeros(1024,int)
        hist=np.zeros(1024,int)
        tree_hist=np.zeros(1024,int)
        rock_hist=np.zeros(1024,int)
        by_work={}
        for row in cohort:
            rgb,t=load(row)
            p=predict(model,rgb)
            prediction=p.argmax(0)
            m=confusion(t,prediction)
            total+=m
            edge+=confusion(t,prediction,boundary(t))
            by_work.setdefault(row["group"],np.zeros((4,4),int))[:] += m
            valid=t!=255
            bins,labels=groups(p,prediction,valid)
            hist+=np.bincount(bins,minlength=1024)
            correct_hist+=np.bincount(bins[labels==t[valid]],minlength=1024)
            tree_bins=np.minimum(np.rint(p[2]*1023).astype(int),1023)
            tree_hist+=np.bincount(tree_bins[t==2],minlength=1024)
            rock_hist+=np.bincount(tree_bins[t==1],minlength=1024)
        cover=np.cumsum(hist[::-1])
        hits=np.cumsum(correct_hist[::-1])
        risk=[]
        for target in (.50,.70,.85,.95,1.):
            i=int(np.searchsorted(cover,target*cover[-1]))
            prev=0 if i==0 else cover[i-1]
            prev_hits=0 if i==0 else hits[i-1]
            fraction=(target*cover[-1]-prev)/max(1,cover[i]-prev)
            risk.append(dict(coverage=target,risk=float(1-(prev_hits+fraction*(hits[i]-prev_hits))/(target*cover[-1])),
                             confidence_boundary=(1023-i)/1023, tie_handling="uniform within score bin"))
        tree_cum=np.cumsum(tree_hist[::-1])
        rock_cum=np.cumsum(rock_hist[::-1])
        operating=[]
        if tree_cum[-1] and rock_cum[-1]:
            for recall in (.60,.75,.90):
                idx=int(np.searchsorted(tree_cum,recall*tree_cum[-1]))
                operating.append(dict(tree_recall=float(tree_cum[idx]/tree_cum[-1]),
                    rock_as_tree=float(rock_cum[idx]/rock_cum[-1]),tree_score_cutoff=(1023-idx)/1023))
        return dict(metrics=measures(total),boundary=measures(edge),confusion=total.tolist(),
                    boundary_confusion=edge.tolist(),coverage_risk=risk,tree_recall_tradeoff=operating,
                    per_work={w:measures(v) for w,v in sorted(by_work.items())})

    s0=SceneObjects().to(device).eval()
    s0.load_state_dict(torch.load(s0_checkpoint,map_location="cpu",weights_only=True)["model"])
    s2=SceneObjects().to(device)
    for layer in list(s2.lateral)+list(s2.head):
        if hasattr(layer,"reset_parameters"):
            layer.reset_parameters()
    tensors=[pack(*load(r),size=resolution)[:2] for r in train]
    count=sum(np.bincount(y[y!=255],minlength=4) for _,y in tensors)
    assert np.all(count>0),count
    weights=torch.tensor(1/np.sqrt(count/count.sum()),dtype=torch.float32,device=device)
    opt=torch.optim.AdamW((p for p in s2.parameters() if p.requires_grad),lr=.002)
    rng=random.Random(20261125)
    curve=[]
    for epoch in range(epochs):
        ids=list(range(len(tensors)))
        rng.shuffle(ids)
        losses=[]
        s2.train()
        for start in range(0,len(ids),6):
            examples=[tensors[i] for i in ids[start:start+6]]
            variants=[(x[:,:,::-1].copy(),y[:,::-1].copy()) if rng.random()<.5 else (x,y) for x,y in examples]
            x,y=(np.stack(items) for items in zip(*variants))
            opt.zero_grad(set_to_none=True)
            logits=s2(torch.from_numpy(x).to(device))
            loss=torch.nn.functional.cross_entropy(logits,torch.from_numpy(y).long().to(device),
                                                    weight=weights,ignore_index=255)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        s2.eval()
        dev_score=observe(s2,dev)
        entry=dict(epoch=epoch+1,weighted_train_loss=sum(losses)/len(losses),
                   dev_iou={k:v["iou"] for k,v in dev_score["metrics"].items()})
        curve.append(entry)
        print(entry,flush=True)
    weights_path=out/"s2_clpd_head.pt"
    torch.save(dict(model=s2.state_dict(),config=config),weights_path)
    cohorts={"development":dev,"work_group_evaluation":test}
    unmatched=[r for r in test if not r["historical_matches"]]
    if unmatched:
        cohorts["evaluation_without_detected_historical_match"]=unmatched
    results={key:dict(n=len(rows),groups=len({r["group"] for r in rows}),S0=observe(s0,rows),S2=observe(s2,rows))
             for key,rows in cohorts.items()}
    write(out/"results.json",dict(config=config,loss_curve=curve,results=results,baseline_hashes=baselines,
              checkpoint_sha256=sha(weights_path),group_counts=dict(Counter(r["split"] for r in records)),
              independent_artwork_acceptance=False))
    font=ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",15)
    palette=np.array([[227,224,214],[178,111,65],[48,112,86],[77,113,158]],np.uint8)
    chart=Image.new("RGB",(780,262),"#f7f3e9")
    pen=ImageDraw.Draw(chart)
    pen.text((14,8),"CLPD S2 训练损失及开发组树石 IoU",font=font,fill="#243d31")
    for index,(name,color,series,limit) in enumerate([
        ("加权交叉熵","#a45e39",[v["weighted_train_loss"] for v in curve],None),
        ("山石 IoU","#a45e39",[v["dev_iou"]["mountain_rock"] or 0 for v in curve],1),
        ("树木 IoU","#246b60",[v["dev_iou"]["trees"] or 0 for v in curve],1)]):
        x0=32+index*255; limit=limit or max(series)
        pen.rectangle((x0,46,x0+210,220),outline="#acabad")
        pen.line([(x0+i*210/max(1,len(series)-1),220-174*v/limit) for i,v in enumerate(series)],fill=color,width=3)
        pen.text((x0,230),f"{name}  {series[0]:.3f} → {series[-1]:.3f}",font=font,fill=color)
    chart.save(out/"training_curve.png")
    display=sorted(test,key=lambda r:r["id"])[:6]
    board=Image.new("RGB",(882,40+220*len(display)),"#f7f3e9")
    pen=ImageDraw.Draw(board)
    pen.text((12,10),"同组留出样本 · 原图 / 发布方标签 / S0 / S2",font=font,fill="#243d31")
    example=None
    for i,r in enumerate(display):
        rgb,truth=load(r)
        p0=predict(s0,rgb).argmax(0).astype(np.uint8)
        p2=predict(s2,rgb).argmax(0).astype(np.uint8)
        pen.text((10,37+i*220),r["id"],font=font,fill="#243d31")
        annotated=palette[truth.clip(0,3)]
        annotated[truth==255]=[151,147,151]
        for j,array in enumerate((rgb,annotated,palette[p0],palette[p2])):
            im=Image.fromarray(array)
            im.thumbnail((206,193))
            board.paste(im,(10+218*j,58+i*220))
        if example is None and all(np.any(m==k) for m in (p0,p2) for k in (1,2)):
            example=(r,rgb,p0,p2)
    board.save(out/"scene_comparison.png")
    if example is not None:
        from professional_scene_drawing import render_pair
        render_pair(out,*example)
    assert verify_baselines()==baselines and sha(s0_checkpoint)==expected
    print("Completed grouped training and evaluation.",flush=True)


if __name__=="__main__":
    main()
