# -*- coding: utf-8 -*-
"""Recover short interruptions in a source-supported branch ridge."""

import json

import numpy as np
from PIL import Image, ImageDraw

from gongbi_training.runtime import LAB, configure
from recognition_attached_ink_20261123 import SETTINGS, STEPS, trace_attached
from recognition_branch_stop_20261123 import score_regions, stop_predictions
from recognition_groups_20261123 import SCENE, label, read, sha, summarize, verify_baselines
from recognition_role_gap_20261123 import scene_evidence, shifted
from recognition_structure_20261123 import derivatives


OUT = LAB / "recognition_weak_branch_20261123"
PREV = LAB / "recognition_branch_stop_20261123"
MODES = ("strict", "one_dip", "two_dips", "bending_dip")
NAMES = ("双侧墨脊", "放宽单点", "放宽两点", "允许单点小转折")
PARAMETERS = dict(minimum_original_ridge=.005, maximum_bend_degrees=45, audit_crops_per_label=9)


def save(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),"utf-8")


def bridge_points(confirmed, weak):
    single = np.zeros_like(confirmed)
    double = np.zeros_like(confirmed)
    bending = np.zeros_like(confirmed)
    cosine = np.cos(np.deg2rad(PARAMETERS["maximum_bend_degrees"])) - 1e-6
    for i,(dy,dx,length) in enumerate(STEPS):
        tail = shifted(confirmed,dx,dy) & shifted(confirmed,2*dx,2*dy)
        forward = shifted(confirmed,-dx,-dy) & shifted(confirmed,-2*dx,-2*dy)
        single |= weak & tail & forward
        pair = weak & shifted(weak,-dx,-dy) & tail
        pair &= shifted(confirmed,-2*dx,-2*dy) & shifted(confirmed,-3*dx,-3*dy)
        double |= pair | shifted(pair,dx,dy)
        for ey,ex,distance in STEPS[i+1:]:
            if (dy*ey+dx*ex) > -cosine*length*distance:
                continue
            if dy==-ey and dx==-ex:
                continue
            other = shifted(confirmed,ex,ey) & shifted(confirmed,2*ex,2*ey)
            bending |= weak & tail & other
    return single, single | double, single | bending


def candidates(rgb,low,connected,cv2,skeletonize):
    (extended,_,strict),rock_face,paired=stop_predictions(rgb,low,connected,cv2,skeletonize)
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    smooth=cv2.GaussianBlur(gray,(0,0),.7)
    ink=cv2.GaussianBlur(smooth,(0,0),7)-smooth>=SETTINGS["source_contrast"]
    skeleton=skeletonize(np.ascontiguousarray(ink))
    stable=skeleton & (low==2) & (connected==2)
    supported=skeleton & paired & ~rock_face
    ridge,_,_=derivatives(gray,1.1,cv2)
    weak=skeleton & ~supported & ~rock_face & (ridge>=PARAMETERS["minimum_original_ridge"])
    kernel=np.ones((3,3),np.uint8)
    outputs=[]
    for accepted in bridge_points(supported,weak):
        trial_allowed=stable | supported | accepted
        new=trace_attached(trial_allowed,stable,SETTINGS["maximum_length"])
        band=cv2.dilate(new.astype(np.uint8),kernel).astype(bool)
        # Use the old branch envelope, keeping other source ink and prior strict recovery fixed.
        add=band & extended & (paired | accepted) & ~rock_face
        outputs.append(strict | add)
    return strict,extended,outputs


def omitted_examples(row,gray,removed,truth,cv2):
    records=[]
    for category in (1,2):
        subset=removed & (truth==category)
        n,components,stats,centers=cv2.connectedComponentsWithStats(subset.astype(np.uint8),connectivity=8)
        ranked=sorted(range(1,n),key=lambda v:-stats[v,4])[:2]
        for idx in ranked:
            if stats[idx,4]<3:
                continue
            points=np.argwhere(components==idx)
            center=np.array(centers[idx][::-1])
            ry,rx=points[np.argmin(np.linalg.norm(points-center,axis=1))]
            width,height=gray.shape[::-1]
            crop=(max(0,int(rx)-25),max(0,int(ry)-25),min(width,int(rx)+26),min(height,int(ry)+26))
            region=components[crop[1]:crop[3],crop[0]:crop[2]]==idx
            records.append(dict(id=row["id"],image=row["image"],image_sha256=row["image_sha256"],
                                cohort=row["cohort"],coarse_category=int(category),size=int(stats[idx,4]),
                                center=[int(rx),int(ry)],crop=crop,
                                changed_yx=np.argwhere(region).tolist()))
    return records


def contact_sheet(references):
    w,h=325,205
    board=Image.new("RGB",(4*w,74+h*((len(references)+3)//4)),"#f4f1e8")
    draw=ImageDraw.Draw(board)
    label(draw,(14,12),"被双侧条件挡下的原图墨迹",24)
    label(draw,(14,43),"每组左侧为原图，右侧青色显示上轮宽松且本轮严格规则拒绝的区域。类型来自粗标。",15)
    for i,row in enumerate(references):
        assert sha(row["image"])==row["image_sha256"]
        with Image.open(row["image"]) as f:
            origin=f.convert("RGB").crop(row["crop"])
        overlay=np.asarray(origin).copy()
        yy,xx=np.asarray(row["changed_yx"]).T
        overlay[yy,xx]=(2,206,197)
        x,y=(i%4)*w+10,(i//4)*h+74
        for offset,tile in ((0,origin),(143,Image.fromarray(overlay))):
            width,height=tile.size
            board.paste(tile.resize((width*2,height*2),Image.Resampling.NEAREST),(x+offset,y+46))
        label(draw,(x,y),f"{row['key']}  {row['id']}",13)
        kind="粗标树区" if row["coarse_category"]==2 else "粗标山石"
        label(draw,(x,y+21),f"{kind}  中心 {row['center']}  {row['size']} 像素",14)
    board.save(OUT/"blocked_review_contact.png")


def main():
    configure()
    import cv2
    import torch
    from skimage.graph import MCP_Geometric
    from skimage.morphology import skeletonize
    from gongbi_training.scene_objects import SceneObjects
    from gongbi_training.source_scene_mapping import source_to_scene

    OUT.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(4)
    lock=verify_baselines()
    checkpoint=SCENE/"mapping_fixed.pt"
    assert sha(checkpoint)==read(SCENE/"checkpoint_freeze.json")["mapping_fixed"]
    rows=read(PREV/"protocol.json")["source_refs"]
    save("protocol.json",dict(script_sha256=sha(__file__),prior_protocol_sha256=sha(PREV/"protocol.json"),
                              parameters=PARAMETERS, source_groups="previously opened archives",
                              reference_used_for_tracking=False))
    model=SceneObjects()
    model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=False)["model"])
    model.eval()
    old={r["id"]:r for r in read(PREV/"results.json")["records"]}
    result=[]
    refs=[]
    detail=[]
    for i,row in enumerate(rows):
        assert sha(row["image"])==row["image_sha256"] and sha(row["mask"])==row["mask_sha256"]
        with Image.open(row["image"]) as f:
            rgb=np.asarray(f.convert("RGB"))
        with Image.open(row["mask"]) as f:
            truth=source_to_scene(np.asarray(f))
        low,connected,_,_=scene_evidence(rgb,model,cv2,torch,MCP_Geometric)
        strict,extended,variants=candidates(rgb,low,connected,cv2,skeletonize)
        masks=[strict,*variants]
        predictions=[np.where(changed,2,connected) for changed in masks]
        metric=score_regions(truth,predictions,cv2)
        # The imported scorer binds names for the four previous arms.
        metric=dict(zip(MODES,metric.values()))
        assert metric["strict"]==old[row["id"]]["metrics"]["paired_sides"]
        result.append(dict(id=row["id"],cohort=row["cohort"],score=metric,
                           additional={mode:{str(k):int(np.count_nonzero((fill & ~strict)&(truth==k)))
                                             for k in (0,1,2,3,255)}
                                       for mode,fill in zip(MODES,masks)}))
        if row["cohort"]=="additional_groups":
            refs.extend(omitted_examples(row,rgb[:,:,0],extended & ~strict,truth,cv2))
        if row["id"]=="zouzhe_xiangjiaoqiuwutu_43":
            detail=[dict(x=x,y=y,tree={mode:bool(pred[y,x]==2) for mode,pred in zip(MODES,predictions)})
                    for x,y in ((134,418),(138,416),(142,413))]
        if (i+1)%6==0:
            print("SCORED",i+1,flush=True)
    groups={}
    for cohort in sorted({r["cohort"] for r in result}):
        data=[r for r in result if r["cohort"]==cohort]
        groups[cohort]={mode:{loc:summarize(sum(np.array(r["score"][mode][loc]) for r in data))
                              for loc in ("all","boundary")} for mode in MODES}
    selected=[]
    for code,tag in ((2,"T"),(1,"R")):
        ordered=sorted((r for r in refs if r["coarse_category"]==code),
                       key=lambda r:(-r["size"],r["id"]))
        seen=set()
        for r in ordered:
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            selected.append(dict(**r,key=f"{tag}{len(seen)}"))
            if len(seen)==PARAMETERS["audit_crops_per_label"]:
                break
    save("blocked_candidate_regions.json",dict(records=selected, coarse_mask_for_candidate_selection=True, expert_labels=False))
    contact_sheet(selected)
    assert verify_baselines()==lock
    save("results.json",dict(records=result,cohorts=groups,branch=detail,painting_baselines=lock,
                             independent_validation=False,adopted=False,robot_connected=False))
    print("DONE",len(result),len(selected),flush=True)


if __name__=="__main__":
    main()
