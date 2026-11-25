# -*- coding: utf-8 -*-
"""Exercise the frozen conflict check on textured, coarsely labelled rock windows."""

import json

import numpy as np
from PIL import Image, ImageDraw

from gongbi_training.runtime import LAB, configure
from gongbi_training.source_scene_mapping import source_to_scene
from recognition_full_trunk_20261123 import assess
from recognition_groups_20261123 import SCENE, label, read, sha, verify_baselines
from recognition_trunk_archive_transfer_20261124 import color_overlay
from recognition_trunk_envelope_20261123 import RULES, best_envelope
from recognition_trunk_identity_20261123 import check_candidate
from recognition_trunk_rock_context_20261124 import conflicts, measure, scene_probabilities
from recognition_trunk_source_background_20261123 import source_background, source_rows


OUT = LAB / "trunk_rock_context_holdout_20261124"
RULE_RUN = LAB / "trunk_rock_context_20261124"
SCREEN = dict(window=(44,72),stride=32,minimum_rock_share=.96,windows_per_image=2,minimum_center_spacing=65)


def write(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),"utf-8")


def choose_windows(gray,reference,cv2):
    width,height=SCREEN["window"]
    ys,xs=reference.shape
    contrast=cv2.magnitude(cv2.Sobel(gray,cv2.CV_32F,1,0),cv2.Sobel(gray,cv2.CV_32F,0,1))
    shortlisted=[]
    for y in range(32,ys-height-12,SCREEN["stride"]):
        for x in range(8,xs-width-8,SCREEN["stride"]):
            box=(x,y,x+width,y+height)
            sel=np.s_[y:y+height,x:x+width]
            if np.mean(reference[sel]==1)<SCREEN["minimum_rock_share"]:
                continue
            activity=float(np.mean(contrast[sel]))
            adjacent=reference[max(0,y-23):min(ys,y+height+23),max(0,x-23):min(xs,x+width+23)]
            tree_near=float(np.mean(adjacent==2))
            shortlisted.append((box,activity,tree_near))
    kept=[]
    for key in (lambda r:r[1], lambda r:(r[2],r[1])):
        ordered=sorted(shortlisted,key=key,reverse=True)
        next_window=next((box for box,_,_ in ordered if all(
            np.hypot(box[0]-other[0],box[1]-other[1])>=SCREEN["minimum_center_spacing"] for other in kept)),None)
        if next_window is not None:
            kept.append(next_window)
    return kept[:SCREEN["windows_per_image"]]


def main():
    configure()
    import cv2
    import torch
    from skimage.graph import MCP_Geometric
    from gongbi_training.scene_objects import SceneObjects
    from recognition_role_gap_20261123 import scene_evidence
    import recognition_trunk_rock_context_20261124 as frozen

    torch.set_num_threads(4)
    baseline=verify_baselines()
    proto=read(RULE_RUN/"protocol.json")
    assert proto["script_sha256"]==sha(frozen.__file__) and proto["mixed_scale_rule"]==frozen.MIXED
    excluded={sha(row["image"]) for row in read(RULE_RUN/"results.json")["records"]}
    excluded.update(row["image_sha256"] for row in read(OUT/"reference_freeze.json")["cases"])
    trained=[row for row in read(SCENE/"dataset_manifest.json")["records"] if row["split"]=="train"]
    excluded.update(sha(row["image"]) for row in trained)
    sources=[row for row in read(SCENE/"dataset_manifest.json")["records"] if row["split"]=="new_test"]
    for dir_name,manifest in (("guohua_faint_sources_20261116","dataset_manifest.json"),
                              ("object_prospective_20261118","dataset_manifest.json"),
                              ("guohua_priority_check_20261116","test_manifest.json")):
        sources.extend(read(LAB/dir_name/manifest)["records"])
    images=[]
    for row in sources:
        digest=sha(row["image"])
        mask=row.get("source_mask") or row.get("mask")
        if digest in excluded or mask is None:
            continue
        excluded.add(digest)
        with Image.open(row["image"]) as source,Image.open(mask) as target:
            gray=np.asarray(source.convert("L"))
            classes=source_to_scene(np.asarray(target))
        boxes=choose_windows(gray,classes,cv2)
        if boxes:
            images.append(dict(id=row["id"],work_group=row["work_group"],image=row["image"],sha256=digest,
                               mask=mask,mask_sha256=sha(mask),boxes=boxes))
    write("scan_reference_freeze.json",dict(windows=images,settings=SCREEN,model_predictions_used_for_screening=False))
    ckpt=SCENE/"mapping_fixed.pt"
    assert sha(ckpt)==proto["checkpoint_sha256"]
    write("scan_protocol.json",dict(script_sha256=sha(__file__),reference_sha256=sha(OUT/"scan_reference_freeze.json"),
                                  conflict_rule_sha256=sha(frozen.__file__),checkpoint_sha256=sha(ckpt),
                                  single_entrance_candidates_only=True))
    net=SceneObjects()
    net.load_state_dict(torch.load(ckpt,map_location="cpu",weights_only=False)["model"])
    net.eval()
    results=[]
    hits=[]
    for row in images:
        with Image.open(row["image"]) as source,Image.open(row["mask"]) as target:
            original=source.convert("RGB")
            rgb=np.asarray(original)
            according_to_source=source_to_scene(np.asarray(target))
        low,connected,_,_=scene_evidence(rgb,net,cv2,torch,MCP_Geometric)
        tree=(low==2)&(connected==2)
        lab=cv2.GaussianBlur(cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2LAB),(0,0),.6)
        _,profile=source_background(lab,cv2)
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        probs=None
        for box in row["boxes"]:
            x0,y0,x1,y1=box
            band=best_envelope(list(source_rows(profile,lab,box)),y1-y0,appearance=True)
            estimate=assess(dict(crop=box,sections=[]),band,tree[y0:y1,x0:x1])
            center=[(a+b)/2 for _,a,b,_ in estimate["path"]]
            drift=float(np.ptp(center)/band["span"]) if center else 0.
            allowed=estimate["accepted"] and drift<=RULES["maximum_lateral_drift"]
            checked=check_candidate(estimate["path"],allowed,lab,profile,tree,review_body=True)
            path=checked["retained"]
            withheld=False
            rock_fraction=None
            features={}
            if path:
                if probs is None:
                    probs=scene_probabilities(rgb,net,torch,cv2)
                features=measure(path,probs,gray)
                withheld=conflicts(features)["mixed_same_side"]
                interior=np.concatenate([according_to_source[y,left+1:right] for y,left,right,_ in path])
                rock_fraction=float(np.mean(interior==1))
                hits.append((row["id"],box,path,original,withheld,rock_fraction))
            results.append(dict(id=row["id"],work_group=row["work_group"],crop=box,generated_candidate=bool(path),
                                identity_status=checked["status"],path=path,withheld=withheld,
                                source_rock_fraction=rock_fraction,evidence=features))
    for hit in hits:
        print(hit[0],hit[1],len(hit[2]),hit[4],hit[5],flush=True)
    if hits:
        panel=Image.new("RGB",(380*min(3,len(hits)),48+277*((len(hits)+2)//3)),"#f4f1e8")
        pen=ImageDraw.Draw(panel)
        label(pen,(12,8),"粗标岩面中漏入树干候选的区域",21)
        for i,(ident,box,path,original,withheld,rock_fraction) in enumerate(hits):
            x,y=i%3*380+12,42+i//3*277
            panel.paste(color_overlay(original,box,path,[]),(x,y+22))
            label(pen,(x,y),ident[:42],13)
            status="转待核" if withheld else "仍为树干候选"
            label(pen,(x,y+259),f"{status} · 内部粗标岩面 {rock_fraction:.0%}",14)
        panel.save(OUT/"rock_scan_candidates.png")
    assert baseline==verify_baselines()
    write("scan_results.json",dict(records=results,painting_baselines=baseline,adopted=False,robot_connected=False))
    print("Scanned",len(results),"windows",len(images),"images",len(hits),"baseline candidates",flush=True)


if __name__=="__main__":
    main()
