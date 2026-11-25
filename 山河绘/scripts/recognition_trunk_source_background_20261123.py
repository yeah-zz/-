# -*- coding: utf-8 -*-
"""Estimate paper tone in original coordinates and probe trunk ROI stability."""

import json

import numpy as np
from PIL import Image, ImageDraw

from gongbi_training.runtime import LAB, configure
from recognition_full_trunk_20261123 import assess
from recognition_groups_20261123 import SCENE, label, read, sha, verify_baselines
from recognition_trunk_envelope_20261123 import RULES, best_envelope, envelope_rows, foreground_profile


OUT = LAB / "recognition_trunk_source_background_20261123"
PREV = LAB / "recognition_trunk_envelope_20261123"
METHODS = ("source_lightness", "source_lab")
VIEWS = ("original","wider","left","right")
SETTINGS = dict(paper_kernel=(23,3), bright_tolerance=3, smoothing=(23,7),
                background_scale=35, chroma_weight=.8, match_tolerance=2, min_consensus=3)


def write(name,data):
    (OUT / name).write_text(json.dumps(data,ensure_ascii=False,indent=2),"utf-8")


def source_background(lab,cv2):
    light=lab[:,:,0]
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,SETTINGS["paper_kernel"])
    upper=cv2.morphologyEx(light,cv2.MORPH_CLOSE,kernel)
    upper=cv2.GaussianBlur(upper,(3,3),.7)
    light_difference=np.maximum(upper-light,0)
    weights=np.exp(-light_difference/SETTINGS["bright_tolerance"])
    smooth=cv2.GaussianBlur(weights,SETTINGS["smoothing"],0)
    numerator=cv2.GaussianBlur(lab[:,:,1:]*weights[:,:,None],SETTINGS["smoothing"],0)
    chroma=numerator / np.maximum(smooth[:,:,None],1e-6)
    color_diff=np.linalg.norm(lab[:,:,1:]-chroma,axis=2)
    light_profile=np.minimum(light_difference*1.15/SETTINGS["background_scale"],1.2)
    lab_profile=np.minimum(np.hypot(light_difference*1.15,SETTINGS["chroma_weight"]*color_diff)
                           / SETTINGS["background_scale"],1.2)
    return light_profile,lab_profile


def source_rows(profile,lab,box):
    """Keep flank samples in original coordinates, including outside the request box."""
    x0,y0,x1,y1=box
    width=profile.shape[1]
    left,right=np.triu_indices(x1-x0,RULES["min_width"])
    left,right=left+x0,right+x0
    keep=(right-left<=RULES["max_width"]) & (left>=3) & (right<width-3)
    left,right=left[keep],right[keep]
    extent=right-left
    sample=np.rint(left[:,None]+extent[:,None]*np.linspace(.17,.83,7)).astype(int)
    offset=np.array(RULES["probe"])
    outer_left=np.maximum(left[:,None]-offset,0)
    outer_right=np.minimum(right[:,None]+offset,width-1)
    for y in range(y0,y1):
        row=profile[y]
        body=np.quantile(row[sample],.28,axis=1)
        flank_left=row[outer_left].mean(axis=1)
        flank_right=row[outer_right].mean(axis=1)
        weak=body-np.maximum(flank_left,flank_right)
        strong=body-np.minimum(flank_left,flank_right)
        allowed=(weak>=RULES["minimum_envelope_step"]) | ((weak>=-.055)&(strong>=.23))
        paired=weak>=RULES["supported_flank_step"]
        score=np.clip(weak,-.055,.55)*15 + np.clip(strong,0,.65)*9 + .95*np.minimum(body,.9)-.045*extent
        light=lab[y,:,0]
        interior=np.median(light[sample],axis=1)
        brightness=interior-np.minimum(light[left],light[right])
        both=interior-np.maximum(light[left],light[right])
        allowed &= brightness>=RULES["minimum_interior_lift"]
        score+=.36*np.clip(brightness,0,16)+.15*np.clip(both,-8,12)
        selected=np.flatnonzero(allowed&(score>0))
        selected=selected[np.argsort(-score[selected],kind="stable")[:90]]
        yield dict(y=y-y0,bounds=np.stack((left[selected]-x0,right[selected]-x0),axis=1),
                   score=score[selected],supported=paired[selected],color=lab[y][sample[selected]].mean(axis=1))


def compare_views(view_results,sections):
    tables=[{y:(a,b) for y,a,b,_ in value["path"]} for value in view_results]
    tol=SETTINGS["match_tolerance"]
    rows=[]
    for y,left,right in sections:
        pairs=[table[y] for table in tables if y in table]
        largest=max((sum(max(abs(l-a),abs(r-b))<=tol for a,b in pairs) for l,r in pairs),default=0)
        matches=sum(max(abs(a-left),abs(b-right))<=tol for a,b in pairs)
        rows.append(dict(y=y,views_with_pairs=len(pairs),views_agree=largest,views_match_reference=matches))
    return rows


def consensus(results):
    good=[{y:(a,b) for y,a,b,_ in r["path"]} for r in results if r["accepted"]]
    tol=SETTINGS["match_tolerance"]
    rows=[]
    for y in sorted({y for r in good for y in r}):
        pairs=[r[y] for r in good if y in r]
        groups=[[q for q in pairs if max(abs(v-w) for v,w in zip(p,q))<=tol] for p in pairs]
        strongest=max(groups,key=len,default=[])
        if len(strongest)>=SETTINGS["min_consensus"]:
            left,right=np.rint(np.median(strongest,axis=0)).astype(int)
            rows.append([int(y),int(left),int(right),len(strongest)])
    return rows


def render(cases):
    panel=Image.new("RGB",(1280,80+194*len(cases)),"#f4f1e8")
    draw=ImageDraw.Draw(panel)
    label(draw,(12,12),"同一树干在不同裁剪中的外缘",25)
    label(draw,(12,47),"四列使用同一张原图的底色。青色为各自候选，橙色是原图复核点。",16)
    for i,item in enumerate(cases):
        y=77+i*194
        label(draw,(12,y),item["id"]+"  "+item["role"],15)
        for j,variant in enumerate(item["views"]):
            arm=variant["results"]["source_lab"]
            x0,y0,x1,y1=variant["crop"]
            with Image.open(item["image"]) as f:
                im=f.convert("RGB").crop(variant["crop"])
            scale=min(5,128/im.height,238/im.width)
            tile=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.NEAREST)
            pen=ImageDraw.Draw(tile)
            for yy,l,r,_ in arm["path"]:
                for edge in (l,r):
                    bx=(edge-x0+.5)*scale;by=(yy-y0+.5)*scale
                    pen.ellipse((bx-1,by-1,bx+1,by+1),fill="#0eb69c")
            for yy,l,r in item["sections"]:
                for edge in (l,r):
                    bx=(edge-x0+.5)*scale;by=(yy-y0+.5)*scale
                    pen.ellipse((bx-2,by-2,bx+2,by+2),outline="#e28c42",width=1)
            x=12+320*j
            panel.paste(tile,(x,y+27))
            count=sum(s["both_within_tolerance"] for s in arm["reference_sections"])
            status="树干候选" if arm["accepted"] else "待核"
            names=("原范围","加宽","左移","右移")
            tag=f"{names[j]} · {count}/{len(item['sections'])} 处对齐" if item["sections"] else f"{names[j]} · {status}"
            label(draw,(x,y+160),tag,15)
    panel.save(OUT/"source_background_comparison.png")


def main():
    configure()
    import cv2
    import torch
    from skimage.graph import MCP_Geometric
    from gongbi_training.scene_objects import SceneObjects
    from recognition_role_gap_20261123 import scene_evidence

    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True,parents=True)
    lock=verify_baselines()
    old=read(PREV/"results.json")
    previous_views={(r["id"],r["perturbation"]):r for r in read(PREV/"roi_sensitivity.json")["records"]}
    weight=SCENE/"mapping_fixed.pt"
    assert sha(weight)==read(SCENE/"checkpoint_freeze.json")["mapping_fixed"]
    write("protocol.json",dict(script_sha256=sha(__file__),previous_sha256=sha(PREV/"results.json"),
                               previous_view_sha256=sha(PREV/"roi_sensitivity.json"),
                               checkpoint_sha256=sha(weight),settings=SETTINGS,
                               policies="same scene model and pair tracking per original ROI; different background",
                               references_used_for_scoring_only=True, known_regression=True))
    net=SceneObjects()
    net.load_state_dict(torch.load(weight,map_location="cpu",weights_only=False)["model"])
    net.eval()
    cache={}
    cases=[]
    for row in old["cases"]:
        case={k:v for k,v in row.items() if k!="results"}
        path=case["image"]
        assert sha(path)==case["image_sha256"]
        if path not in cache:
            with Image.open(path) as f:
                rgb=np.asarray(f.convert("RGB"))
            low,connected,_,_=scene_evidence(rgb,net,cv2,torch,MCP_Geometric)
            lab=cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
            lab=cv2.GaussianBlur(lab,(0,0),.6)
            cache[path]=lab,(low==2)&(connected==2),source_background(lab,cv2)
        lab,tree,profiles=cache[path]
        x0,y0,x1,y1=case["crop"]
        boxes=[case["crop"],[x0-4,y0,x1+4,y1],[x0-3,y0,x1-3,y1],[x0+3,y0,x1+3,y1]]
        variants=[]
        for view,box in zip(VIEWS,boxes):
            l,t,r,b=box
            z=np.s_[t:b,l:r]
            lab_crop=lab[z]
            profile_old=foreground_profile(lab_crop,cv2)
            inputs=[("crop_rim",envelope_rows(profile_old,lab_crop,True))]
            inputs.extend((method,source_rows(profile,lab,box)) for method,profile in zip(METHODS,profiles))
            scores={}
            for method,data in inputs:
                band=best_envelope(list(data),b-t,appearance=True)
                result=assess(dict(case,crop=box),band,tree[z])
                center=[(a+c)/2 for _,a,c,_ in result["path"]]
                drift=float(np.ptp(center)/result["span"]) if center else 0.
                result.update(lateral_drift_relative_to_span=drift,
                              accepted=bool(result["accepted"] and drift<=RULES["maximum_lateral_drift"]))
                scores[method]=result
            baseline=scores["crop_rim"]
            original=row["results"]["luminous_body"]
            if view=="original":
                assert baseline["path"]==original["path"] and baseline["accepted"]==original["accepted"]
            existing=previous_views.get((case["id"],view))
            if existing:
                assert existing["accepted"]==baseline["accepted"]
                assert existing["matched"]==sum(r["both_within_tolerance"] for r in baseline["reference_sections"])
            variants.append(dict(view=view,crop=box,results=scores))
        by_arm={m:[v["results"][m] for v in variants] for m in ("crop_rim",*METHODS)}
        details={m:compare_views(r,case["sections"]) for m,r in by_arm.items()}
        voted={}
        tol=SETTINGS["match_tolerance"]
        for m,arm in by_arm.items():
            stable=consensus(arm)
            index={yy:(l,r) for yy,l,r,_ in stable}
            correct=sum(y in index and max(abs(a-index[y][0]),abs(b-index[y][1]))<=tol
                        for y,a,b in case["sections"])
            voted[m]=dict(path=stable,matched=int(correct))
        case.update(views=variants, consistency=details, consensus=voted)
        cases.append(case)
        print(case["id"],{m:[sum(k["both_within_tolerance"] for k in u["reference_sections"]) for u in v]
                          for m,v in by_arm.items()},flush=True)
    render(cases)
    assert lock==verify_baselines()
    write("results.json",dict(cases=cases,painting_baselines=lock,adopted=False,
                               independent_validation=False,robot_connected=False))


if __name__=="__main__":
    main()
