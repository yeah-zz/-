# -*- coding: utf-8 -*-
"""Recognize pale trunks as two ordered contours surrounding one interior."""

import json

import numpy as np
from PIL import Image, ImageDraw

from gongbi_training.runtime import LAB, configure
from recognition_groups_20261123 import SCENE, label, read, sha, verify_baselines


OUT = LAB / "recognition_full_trunk_20261123"
PREV = LAB / "recognition_weak_branch_20261123"
SETTINGS = dict(min_width=3,max_width=25,minimum_dark_edge=7,minimum_single_edge=11,
                single_other_floor=-6,max_change_per_row=3,maximum_row_gap=3,
                minimum_span_fraction=.58,minimum_paired_fraction=.40,tolerance=2,
                color_flank_distance=3, minimum_color_contrast=2.5)
CASES = [
    ("T2","pale_trunk",(32,285,55,329),
     [(289,39,43),(297,39,42),(305,39,43),(313,38,46),(321,41,49),(327,40,50)]),
    ("T4","pale_trunk",(254,209,283,282),
     [(215,269,276),(224,264,274),(233,265,272),(242,264,271),
      (251,263,271),(260,262,272),(269,260,270),(278,258,268)]),
    ("T4","neighbor_trunk",(239,209,256,274),
     [(215,245,253),(224,244,251),(233,243,251),(242,242,251),
      (251,244,252),(260,243,252),(269,243,252)]),
    ("R1","rock_negative",(5,130,43,194),[]),
    ("R2","slope_negative",(277,303,314,335),[]),
    ("R5","pending_marks",(74,399,118,442),[]),
    ("T7","pending_diagonal",(368,284,421,327),[]),
]


def save(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),"utf-8")


def samples(gray, permissive, chroma=None):
    """Find dark left/right edge candidates with a brighter interior per row."""
    width=gray.shape[1]
    left,right=np.triu_indices(width,SETTINGS["min_width"])
    legal=right-left<=SETTINGS["max_width"]
    left,right=left[legal],right[legal]
    extent=right-left
    inner=np.linspace(.24,.76,4)
    interior=np.rint(left[:,None]+extent[:,None]*inner).astype(int)
    output=[]
    for y,line in enumerate(gray):
        center=line[interior].mean(axis=1)
        contrast_left=center-line[left]
        contrast_right=center-line[right]
        weakest=np.minimum(contrast_left,contrast_right)
        strongest=np.maximum(contrast_left,contrast_right)
        side=weakest>=SETTINGS["minimum_dark_edge"]
        one=(strongest>=SETTINGS["minimum_single_edge"]) & (weakest>=SETTINGS["single_other_floor"])
        accepted=(side|one) if permissive else side
        # Both flanks contribute. Saturation keeps one extremely dark accent from dominating.
        strength=np.clip(weakest,-6,16)*.54 + np.minimum(strongest,25)*.24
        strength+=2.0-.13*extent
        colors=None
        if chroma is not None:
            color=chroma[y]
            colors=color[interior].mean(axis=1)
            flank=SETTINGS["color_flank_distance"]
            left_colors=color[np.maximum(left[:,None]-np.arange(1,flank+1),0)].mean(axis=1)
            right_colors=color[np.minimum(right[:,None]+np.arange(1,flank+1),width-1)].mean(axis=1)
            difference=np.minimum(np.linalg.norm(colors-left_colors,axis=1),
                                  np.linalg.norm(colors-right_colors,axis=1))
            color_var=np.linalg.norm(color[interior]-colors[:,None,:],axis=2).mean(axis=1)
            supported=(difference>=SETTINGS["minimum_color_contrast"]) & (strongest>=7)
            accepted=accepted | supported
            strength+=.7*np.minimum(difference,12)-.30*color_var
        options=np.flatnonzero(accepted&(strength>0))
        order=options[np.argsort(-strength[options],kind="stable")[:100]]
        output.append(dict(y=y,ends=np.stack((left[order],right[order]),axis=1),
                           paired=side[order], weight=strength[order],
                           colors=None if colors is None else colors[order]))
    return output


def trace_pairs(gray, permissive, chroma=None):
    candidates=samples(gray,permissive,chroma)
    best=(-np.inf,0,0)
    for y,row in enumerate(candidates):
        coords=row["ends"]
        quality=row["weight"].copy()
        parents=np.full((len(coords),2),-1,dtype=int)
        for span in range(1,min(y,SETTINGS["maximum_row_gap"])+1):
            other=candidates[y-span]
            if not (len(coords) and len(other["ends"])):
                continue
            change=np.abs(coords[:,None,:]-other["ends"])
            continuity=np.max(change,axis=2)<=SETTINGS["max_change_per_row"]*span
            width_shift=np.abs(np.diff(coords).ravel()[:,None]-np.diff(other["ends"]).ravel())
            transfer=other["score"][None,:] - .9*change.sum(axis=2) - .75*width_shift - 4*(span-1)
            if chroma is not None:
                transfer-=.25*np.linalg.norm(row["colors"][:,None,:]-other["colors"][None,:,:],axis=2)
            transfer[~continuity]=-np.inf
            j=transfer.argmax(axis=1)
            total=row["weight"]+transfer[np.arange(len(coords)),j]
            improve=total>quality
            quality[improve]=total[improve]
            parents[improve,0]=y-span
            parents[improve,1]=j[improve]
        row.update(score=quality, parents=parents)
        if len(quality) and quality.max()>best[0]:
            j=int(quality.argmax())
            best=(float(quality[j]),y,j)
    if not np.isfinite(best[0]):
        return dict(path=[], span=0, paired_fraction=0., continuous_candidate=False)
    steps=[]
    _,y,j=best
    while y>=0:
        row=candidates[y]
        left,right=row["ends"][j]
        steps.append([int(y),int(left),int(right),bool(row["paired"][j])])
        y,j=row["parents"][j]
    steps.reverse()
    span=1+steps[-1][0]-steps[0][0]
    paired=sum(s[3] for s in steps)/span
    enough=span>=max(14,SETTINGS["minimum_span_fraction"]*gray.shape[0])
    return dict(path=steps,span=span,paired_fraction=paired,coverage=len(steps)/span,
                continuous_candidate=bool(enough and paired>=SETTINGS["minimum_paired_fraction"]))


def assess(case,result,tree):
    x0,y0,_,_=case["crop"]
    native=[[y+y0,left+x0,right+x0,paired] for y,left,right,paired in result["path"]]
    measured={y:(left,right,paired) for y,left,right,paired in native}
    checks=[]
    for y,left,right in case["sections"]:
        guess=measured.get(y)
        tol=SETTINGS["tolerance"]
        checks.append(dict(y=y,reference=[left,right],observed=list(guess[:2]) if guess else None,
                           both_within_tolerance=bool(guess and abs(guess[0]-left)<=tol
                                                      and abs(guess[1]-right)<=tol),
                           opposite_side_visible=bool(guess and guess[2])))
    anchor=[]
    for y,left,right,_ in native:
        a,b=left-x0+1,right-x0
        anchor.append(float(tree[y-y0,a:b].mean()))
    anchor_fraction=float(np.mean(anchor)) if anchor else 0.
    longest=0
    streak=0
    previous_y=None
    for (y,_,_,_),fraction in zip(native,anchor):
        streak=streak+1 if fraction>=.5 and previous_y==y-1 else int(fraction>=.5)
        longest=max(longest,streak)
        previous_y=y
    result.update(path=native,reference_sections=checks,anchor_fraction=anchor_fraction,
                  longest_local_anchor=longest,
                  accepted=bool(result["continuous_candidate"] and anchor_fraction>=.40))
    return result


def render(cases):
    board=Image.new("RGB",(1450,74+240*len(cases)),"#f4f1e8")
    draw=ImageDraw.Draw(board)
    label(draw,(12,11),"浅树干双侧轮廓与内部淡色",25)
    label(draw,(12,44),"原图橙点为核查边界。结果图绿色为两边均有墨迹，蓝色为单边有明确证据的估计区段。",15)
    for i,case in enumerate(cases):
        y=75+240*i
        x0,y0,x1,y1=case["crop"]
        with Image.open(case["image"]) as img:
            raw=img.convert("RGB").crop(case["crop"])
        z=min(5,168/raw.height,255/raw.width)
        tile=raw.resize((round(raw.width*z),round(raw.height*z)),Image.Resampling.NEAREST)
        label(draw,(12,y),f"{case['id']}  {case['role']}",15)
        for j,arm in enumerate(("reference","paired_only","tracked_side","tracked_color")):
            im=tile.copy()
            pen=ImageDraw.Draw(im)
            if arm=="reference":
                for yy,left,right in case["sections"]:
                    cy=(yy-y0+.5)*z
                    for p in (left,right):
                        cx=(p-x0+.5)*z
                        pen.ellipse((cx-2,cy-2,cx+2,cy+2),fill="#e68b42")
                caption="原图核查"
            else:
                row=case["results"][arm]
                for yy,left,right,paired in row["path"]:
                    cy=(yy-y0+.5)*z
                    color="#0aa77e" if paired else "#4c9fe0"
                    for edge in (left,right):
                        cx=(edge-x0+.5)*z
                        pen.ellipse((cx-1,cy-1,cx+1,cy+1),fill=color)
                tag={"paired_only":"双侧轮廓","tracked_side":"结合连续性","tracked_color":"加入内外色差"}[arm]
                count=sum(r["both_within_tolerance"] for r in row["reference_sections"])
                status="树干候选" if row["accepted"] else "待核"
                caption=f"{tag} · {status} · {row['span']} 行"
                if case["sections"]:
                    caption+=f" · {count}/{len(case['sections'])} 处边界"
            x=12+356*j
            board.paste(im,(x,y+31))
            label(draw,(x,y+207),caption,14)
    board.save(OUT/"paired_trunk_comparison.png")


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
    refs={r["key"]:r for r in read(PREV/"visual_reference_review.json")}
    records=[]
    for i,(key,role,crop,sections) in enumerate(CASES):
        source=refs[key]
        records.append(dict(id=f"{key}/{i+1}",role=role,crop=crop,sections=sections,
                            image=source["image"],image_sha256=source["image_sha256"],
                            reviewer="assistant measured visible source sections"))
    save("reference_freeze.json",dict(records=records,scope="known source; no expert or pixel-level ground truth",
                                      pair_coordinates_for_scoring_only=True))
    weight=SCENE/"mapping_fixed.pt"
    assert sha(weight)==read(SCENE/"checkpoint_freeze.json")["mapping_fixed"]
    save("protocol.json",dict(settings=SETTINGS,script_sha256=sha(__file__),
                              checkpoint_sha256=sha(weight),
                              reference_sha256=sha(OUT/"reference_freeze.json"),
                              input_scope="provided original-image ROI; no reference lines at inference"))
    model=SceneObjects()
    model.load_state_dict(torch.load(weight,map_location="cpu",weights_only=False)["model"])
    model.eval()
    cache={}
    for row in records:
        assert sha(row["image"])==row["image_sha256"]
        if row["image"] not in cache:
            with Image.open(row["image"]) as source:
                rgb=np.asarray(source.convert("RGB"))
            low,connected,_,_=scene_evidence(rgb,model,cv2,torch,MCP_Geometric)
            gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
            lab=cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
            chroma=cv2.GaussianBlur(lab[:,:,1:],(0,0),.65)
            cache[row["image"]]=cv2.GaussianBlur(gray,(0,0),.55),(low==2)&(connected==2),chroma
        im,tree,chroma=cache[row["image"]]
        x0,y0,x1,y1=row["crop"]
        zone=np.s_[y0:y1,x0:x1]
        row["results"]={arm:assess(row,trace_pairs(im[zone],permissive),tree[zone])
                        for arm,permissive in (("paired_only",False),("tracked_side",True))}
        row["results"]["tracked_color"]=assess(row,trace_pairs(im[zone],True,chroma[zone]),tree[zone])
        print(row["id"],[(k,v["span"],v["accepted"]) for k,v in row["results"].items()],flush=True)
    render(records)
    assert lock==verify_baselines()
    save("results.json",dict(cases=records,painting_baselines=lock,adopted=False,
                              independent_validation=False,robot_connected=False))


if __name__=="__main__":
    main()
