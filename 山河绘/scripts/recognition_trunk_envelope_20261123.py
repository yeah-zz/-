# -*- coding: utf-8 -*-
"""Trace the envelope of one trunk using sustained interior/exterior evidence."""

import json

import numpy as np
from PIL import Image, ImageDraw

from gongbi_training.runtime import LAB, configure
from recognition_full_trunk_20261123 import SETTINGS as PREVIOUS_SETTINGS, assess
from recognition_groups_20261123 import SCENE, label, read, sha, verify_baselines


OUT = LAB / "recognition_trunk_envelope_20261123"
PREV = LAB / "recognition_full_trunk_20261123"
RULES = dict(min_width=3, max_width=24, probe=(2,3,4), lateral_background_width=3,
             minimum_envelope_step=.045, supported_flank_step=.085, max_row_step=3,
             max_skip_rows=2, minimum_span_fraction=.58, minimum_supported_fraction=.40,
             minimum_interior_lift=2.0, maximum_lateral_drift=.45)
ARMS = ("old", "envelope", "envelope_appearance", "luminous_body")
DISPLAY = ("上轮候选", "同一色带外缘", "外缘与沿干色彩一致性", "检查内部淡色与走向")


def write(name, data):
    (OUT / name).write_text(json.dumps(data,ensure_ascii=False,indent=2),"utf-8")


def foreground_profile(lab, cv2):
    """Estimate what differs from the local background on each side of the ROI."""
    width=lab.shape[1]
    rim=RULES["lateral_background_width"]
    left=np.median(lab[:,:rim],axis=1)
    right=np.median(lab[:,-rim:],axis=1)
    sides=[cv2.GaussianBlur(value[:,None,:],(1,0),2.8) for value in (left,right)]
    blend=np.linspace(0,1,width,dtype=np.float32)[None,:,None]
    background=sides[0]*(1-blend)+sides[1]*blend
    difference=(lab-background)*np.array((1.15,1.,1.),dtype=np.float32)
    profile=np.minimum(np.linalg.norm(difference,axis=2)/35,1.2)
    return profile


def envelope_rows(profile,lab,luminous_body=False):
    """Rank pairs whose span contains a distinct band, checking both outer flanks."""
    width=profile.shape[1]
    left,right=np.triu_indices(width,RULES["min_width"])
    keep=(right-left<=RULES["max_width"]) & (left>=2) & (right<width-2)
    left,right=left[keep],right[keep]
    span=right-left
    sample=np.rint(left[:,None]+span[:,None]*np.linspace(.17,.83,7)).astype(int)
    offset=np.array(RULES["probe"])
    outer_left=np.maximum(left[:,None]-offset,0)
    outer_right=np.minimum(right[:,None]+offset,width-1)
    for row,pixels in enumerate(profile):
        body=pixels[sample]
        interior=np.quantile(body,.28,axis=1)
        outside_a=np.mean(pixels[outer_left],axis=1)
        outside_b=np.mean(pixels[outer_right],axis=1)
        left_step=interior-outside_a
        right_step=interior-outside_b
        weaker=np.minimum(left_step,right_step)
        stronger=np.maximum(left_step,right_step)
        permitted=(weaker>=RULES["minimum_envelope_step"]) | ((weaker>=-.055)&(stronger>=.23))
        supported=weaker>=RULES["supported_flank_step"]
        contrast=np.clip(weaker,-.055,.55)*15+np.clip(stronger,0,.65)*9
        score=contrast+.95*np.minimum(interior,.9)-.045*span
        if luminous_body:
            lightness=lab[row,:,0]
            body_light=np.median(lightness[sample],axis=1)
            soft_side=np.maximum(lightness[left],lightness[right])
            dark_side=np.minimum(lightness[left],lightness[right])
            lift=body_light-dark_side
            both_lift=body_light-soft_side
            permitted &= lift>=RULES["minimum_interior_lift"]
            score+=.36*np.clip(lift,0,16)+.15*np.clip(both_lift,-8,12)
        index=np.flatnonzero(permitted & (score>0))
        index=index[np.argsort(-score[index],kind="stable")[:90]]
        yield dict(y=row,bounds=np.stack((left[index],right[index]),axis=1),
                   score=score[index],supported=supported[index],
                   color=lab[row][sample[index]].mean(axis=1))


def best_envelope(rows, total_height, appearance=False):
    """Track consistent band width and flanks without knowing the reviewed edges."""
    best=(float("-inf"),0,0)
    for i,row in enumerate(rows):
        bounds=row["bounds"]
        parents=np.full((len(bounds),2),-1,dtype=int)
        local=row["score"]
        costs=local.copy()
        for step in range(1,min(i,RULES["max_skip_rows"]+1)+1):
            previous=rows[i-step]
            if not len(bounds) or not len(previous["bounds"]):
                continue
            movement=np.abs(bounds[:,None]-previous["bounds"])
            widths=np.abs(np.diff(bounds).ravel()[:,None]-np.diff(previous["bounds"]).ravel())
            transition=previous["total"][None,:]-.95*movement.sum(axis=2)-.60*widths-3.8*(step-1)
            if appearance:
                differences=(row["color"][:,None]-previous["color"])*np.array((.28,1.,1.))
                transition-=.26*np.linalg.norm(differences,axis=2)
            transition[np.max(movement,axis=2)>RULES["max_row_step"]*step]=-np.inf
            indices=transition.argmax(axis=1)
            score=transition[np.arange(len(bounds)),indices]+local
            better=score>costs
            costs[better]=score[better]
            parents[better,0]=i-step
            parents[better,1]=indices[better]
        row.update(total=costs,parent=parents)
        if len(costs) and costs.max()>best[0]:
            col=int(costs.argmax())
            best=(float(costs[col]),i,col)
    if not np.isfinite(best[0]):
        return dict(path=[],span=0,paired_fraction=0.,continuous_candidate=False)
    path=[]
    _,i,col=best
    while i>=0:
        row=rows[i]
        left,right=row["bounds"][col]
        path.append([int(i),int(left),int(right),bool(row["supported"][col])])
        i,col=row["parent"][col]
    path.reverse()
    span=1+path[-1][0]-path[0][0]
    paired=sum(row[-1] for row in path)/span
    enough=span>=max(14,RULES["minimum_span_fraction"]*total_height)
    return dict(path=path,span=span,paired_fraction=paired,coverage=len(path)/span,
                continuous_candidate=bool(enough and paired>=RULES["minimum_supported_fraction"]))


def draw_comparison(cases):
    board=Image.new("RGB",(1685,81+235*len(cases)),"#f4f1e8")
    draw=ImageDraw.Draw(board)
    label(draw,(14,12),"同一树干的外缘对照",25)
    label(draw,(14,46),"橙色为原图复核截面，青色表示有双侧证据的候选外缘，蓝色为单侧证据延伸。",16)
    for i,case in enumerate(cases):
        x0,y0,x1,y1=case["crop"]
        with Image.open(case["image"]) as f:
            source=f.convert("RGB").crop(case["crop"])
        zoom=min(5,165/source.height,226/source.width)
        view=source.resize((round(source.width*zoom),round(source.height*zoom)),Image.Resampling.NEAREST)
        y=80+i*235
        label(draw,(14,y),f"{case['id']}  {case['role']}",15)
        for j,(arm,name) in enumerate(zip(("reference",*ARMS),("原图边界参考",*DISPLAY))):
            tile=view.copy()
            pen=ImageDraw.Draw(tile)
            if arm=="reference":
                points=[[yy,a,b,True] for yy,a,b in case["sections"]]
                caption=name
            else:
                r=case["results"][arm]
                points=r["path"]
                correct=sum(x["both_within_tolerance"] for x in r["reference_sections"])
                caption=f"{name} · {r['span']} 行"
                if case["sections"]:
                    caption+=f" · {correct}/{len(case['sections'])} 处对齐"
                elif r["accepted"]:
                    caption+=" · 候选"
                else:
                    caption+=" · 待核"
            for yy,a,b,strong in points:
                radius=2 if arm=="reference" else 1
                color="#e48c3d" if arm=="reference" else "#0cc09e" if strong else "#4ca9df"
                ty=(yy-y0+.5)*zoom
                for x in (a,b):
                    tx=(x-x0+.5)*zoom
                    pen.ellipse((tx-radius,ty-radius,tx+radius,ty+radius),fill=color)
            px=14+j*336
            board.paste(tile,(px,y+28))
            label(draw,(px,y+205),caption,14)
    board.save(OUT/"trunk_envelope_comparison.png")


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
    weight=SCENE/"mapping_fixed.pt"
    assert sha(weight)==read(SCENE/"checkpoint_freeze.json")["mapping_fixed"]
    write("protocol.json",dict(script_sha256=sha(__file__),previous_sha256=sha(PREV/"results.json"),
                               checkpoint_sha256=sha(weight),settings=RULES,
                               comparison_tolerance=PREVIOUS_SETTINGS["tolerance"],
                               same_supplied_rois=True, reference_used_at_inference=False))
    model=SceneObjects()
    model.load_state_dict(torch.load(weight,map_location="cpu",weights_only=False)["model"])
    model.eval()
    cache={}
    final=[]
    for prior in old["cases"]:
        case={k:v for k,v in prior.items() if k!="results"}
        image=case["image"]
        assert sha(image)==case["image_sha256"]
        if image not in cache:
            with Image.open(image) as f:
                rgb=np.asarray(f.convert("RGB"))
            low,connected,_,_=scene_evidence(rgb,model,cv2,torch,MCP_Geometric)
            lab=cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
            cache[image]=cv2.GaussianBlur(lab,(0,0),.60),(low==2)&(connected==2)
        lab,tree=cache[image]
        x0,y0,x1,y1=case["crop"]
        sel=np.s_[y0:y1,x0:x1]
        profile=foreground_profile(lab[sel],cv2)
        results={"old":prior["results"]["tracked_color"]}
        for mode,color,body in (("envelope",False,False),("envelope_appearance",True,False),
                                ("luminous_body",True,True)):
            band=best_envelope(list(envelope_rows(profile,lab[sel],luminous_body=body)),
                               y1-y0,appearance=color)
            results[mode]=assess(case,band,tree[sel])
            if body:
                center=[(a+b)/2 for _,a,b,_ in band["path"]]
                drift=float(np.ptp(center)/band["span"]) if center else None
                results[mode]["lateral_drift_relative_to_span"]=drift
                results[mode]["rejected_by_near_vertical_scope"]=bool(
                    drift is not None and drift>RULES["maximum_lateral_drift"])
                results[mode]["accepted"] &= not results[mode]["rejected_by_near_vertical_scope"]
        case["results"]=results
        final.append(case)
        print(case["id"],[(k,v["span"],sum(r["both_within_tolerance"] for r in v["reference_sections"]),
                           v["accepted"]) for k,v in results.items()],flush=True)
    assert lock==verify_baselines()
    draw_comparison(final)
    write("results.json",dict(cases=final,painting_baselines=lock,adopted=False,
                               independent_validation=False,robot_connected=False))


if __name__=="__main__":
    main()
