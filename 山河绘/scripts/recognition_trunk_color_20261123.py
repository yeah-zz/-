# -*- coding: utf-8 -*-
"""Constrain supplemental tree-ink recovery to an attached woody color band."""

import json

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from gongbi_training.runtime import LAB, configure
from recognition_attached_ink_20261123 import attachment_maps
from recognition_branch_stop_20261123 import stop_predictions
from recognition_groups_20261123 import SCENE, label, read, sha, verify_baselines
from recognition_role_gap_20261123 import scene_evidence


OUT = LAB / "recognition_weak_branch_20261123"
PREV = LAB / "recognition_branch_stop_20261123"
MODELS = ("strict", "connected_trunk", "connected_trunk_color")
POLICY = dict(maximum_trunk_distance=9, maximum_color_chroma_difference=7, neighborhood_sigma=1.2)
REVIEW = {
    "T2": ("tree_trunk", "淡色竖干从上方叶丛下接，候选落在一侧树皮线旁。"),
    "T4": ("tree_trunk", "赭色分干向上连续。候选落在同一根干的受光一侧。"),
    "T7": ("uncertain", "斜线靠近叶簇与坡边。局部枝条、树根与山面勾线需要进一步区分。"),
    "R1": ("rock", "墨迹延续到山体内部的斜向纹理，周围没有可接续的树干。"),
    "R2": ("ground_edge", "候选沿坡面脚部的绿色横边，与侧面的竖干走向不同。"),
    "R5": ("uncertain", "树根附近有点状墨迹，当前无法确认是点苔、地表纹理还是植被。"),
    "T1": ("tree_group", "候选处于成组树冠与树干附近，单个补点归属未复核。"),
    "T8": ("uncertain", "点状墨迹在树冠与浅岩之间，局部归属待核。"),
    "R8": ("uncertain", "候选靠近浅岩和短暗笔，无法核实每点归属。"),
}


def save(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),"utf-8")


def color_match(rgb,known,cv2,distance_transform_edt):
    if not known.any():
        return np.zeros_like(known)
    lab=cv2.cvtColor(rgb.astype(np.float32) / 255,cv2.COLOR_RGB2LAB)
    chroma=cv2.GaussianBlur(lab[:,:,1:],(0,0),POLICY["neighborhood_sigma"])
    distance,(y,x)=distance_transform_edt(~known,return_indices=True)
    different=np.linalg.norm(chroma-chroma[y,x],axis=2)
    return (distance<=POLICY["maximum_trunk_distance"]) & (
        different<=POLICY["maximum_color_chroma_difference"])


def main():
    configure()
    import cv2
    import torch
    from scipy.ndimage import distance_transform_edt
    from skimage.graph import MCP_Geometric
    from skimage.morphology import skeletonize
    from gongbi_training.scene_objects import SceneObjects
    from gongbi_training.source_scene_mapping import source_to_scene

    torch.set_num_threads(4)
    lock=verify_baselines()
    raw=read(OUT/"blocked_candidate_regions.json")["records"]
    refs=[dict(r,visual_role=REVIEW[r["key"]][0],basis=REVIEW[r["key"]][1],
               reference_scope="assistant source-region review; no precise pixel truth")
          for r in raw if r["key"] in REVIEW]
    save("visual_reference_review.json",refs)
    rows=read(PREV/"protocol.json")["source_refs"]
    checkpoint=SCENE/"mapping_fixed.pt"
    assert sha(checkpoint)==read(SCENE/"checkpoint_freeze.json")["mapping_fixed"]
    save("color_protocol.json",dict(script_sha256=sha(__file__),policy=POLICY,reference_sha256=sha(OUT/"visual_reference_review.json"),
                                    checkpoint_sha256=sha(checkpoint),coarse_labels_only_for_evaluation=True))
    net=SceneObjects()
    net.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=False)["model"])
    net.eval()
    results=[]
    local=[]
    board=Image.new("RGB",(1180,75+151*len(refs)),"#f4f1e8")
    draw=ImageDraw.Draw(board)
    label(draw,(15,11),"从连续树干出发的补回位置",25)
    label(draw,(15,44),"青色显示严格方案之外的新增墨迹，所有组沿用同一张原图。",16)
    ref_order={r["key"]:i for i,r in enumerate(refs)}
    for i,row in enumerate(rows):
        assert sha(row["image"])==row["image_sha256"] and sha(row["mask"])==row["mask_sha256"]
        with Image.open(row["image"]) as img:
            rgb=np.asarray(img.convert("RGB"))
        low,base,_,_=scene_evidence(rgb,net,cv2,torch,MCP_Geometric)
        _,rooted,(_,trunk),_=attachment_maps(rgb,low,base,cv2,skeletonize)
        (_,_,strict),rock_edge,_=stop_predictions(rgb,low,base,cv2,skeletonize)
        added=trunk & ~strict & ~rock_edge
        color=added & color_match(rgb,rooted & (base==2),cv2,distance_transform_edt)
        with Image.open(row["mask"]) as src:
            truth=source_to_scene(np.asarray(src))
        classifications={}
        for name,mask in zip(MODELS,(np.zeros_like(added),added,color)):
            classifications[name]={str(k):int(np.count_nonzero(mask & (truth==k))) for k in (0,1,2,3,255)}
        results.append(dict(id=row["id"],cohort=row["cohort"],additional=classifications))
        for ref in (r for r in refs if r["id"]==row["id"]):
            left,top,right,bottom=ref["crop"]
            region=np.s_[top:bottom,left:right]
            y=75+151*ref_order[ref["key"]]
            label(draw,(14,y),f"{ref['key']}  {ref['id']}",15)
            label(draw,(14,y+29),ref["basis"],14)
            points=np.asarray(ref["changed_yx"])
            local.append(dict(key=ref["key"],visual_role=ref["visual_role"],
                              candidate_region_pixels=len(points),
                              tracked=[int(mask[region][points[:,0],points[:,1]].sum())
                                       for mask in (strict,added,color)]))
            for j,(mask,name) in enumerate(((np.zeros_like(added),"原图"),(added,"连续树干"),(color,"连续树干与颜色"))):
                tile=rgb[region].copy()
                tile[mask[region]]=(5,190,180)
                x=14+350*j
                board.paste(ImageOps.contain(Image.fromarray(tile),(155,86)),(x,y+51))
                label(draw,(x+170,y+72),name,15)
        if (i+1)%8==0:
            print("COHORT",i+1,flush=True)
    board.save(OUT/"trunk_color_comparison.png")
    totals={}
    for cohort in sorted({r["cohort"] for r in rows}):
        sample=[r for r in results if r["cohort"]==cohort]
        totals[cohort]={mode:{k:sum(r["additional"][mode][k] for r in sample) for k in ("0","1","2","3","255")}
                        for mode in MODELS}
    assert lock==verify_baselines()
    save("color_results.json",dict(records=results,cohorts=totals,reviewed_regions=local,
                                    painting_baselines=lock,adopted=False,independent_validation=False))
    print(totals,flush=True)


if __name__=="__main__":
    main()
