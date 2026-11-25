# -*- coding: utf-8 -*-
"""Check retained trunk candidates for competing rock evidence at scene scales."""

import json

import numpy as np
from PIL import Image, ImageDraw

from gongbi_training.runtime import LAB, configure
from recognition_groups_20261123 import SCENE, label, read, sha, verify_baselines
from recognition_trunk_archive_transfer_20261124 import color_overlay
from recognition_trunk_entrance_20261124 import alignment


OUT = LAB / "trunk_rock_context_20261124"
PREV = LAB / "trunk_archive_transfer_20261124"
EARLIER = LAB / "recognition_trunk_entrance_20261124"
REFERENCE = LAB / "recognition_trunk_source_background_20261123"
SIZES = (128,192,256,512)
RULE = dict(strong_rock=.55, rock_advantage=.10, bilateral_rows=.50,
            rock_flank_threshold=.58, separation_lightness=8., separate_rows=.40)
MIXED = dict(coarse_rock=.45, coarse_rock_disadvantage=.08, detail_both_flanks=.60,
             detail_one_flank=.72, same_coarse_flank=.50)


def write(name, value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),"utf-8")


def scene_probabilities(rgb,model,torch,cv2):
    from guohua_objects_train_20261112 import pack

    result={}
    for size in SIZES:
        data,_,(left,top,width,height,original_w,original_h)=pack(rgb,size=size)
        with torch.inference_mode():
            pred=model(torch.from_numpy(data)[None]).softmax(1)[0].numpy()
        result[size]=np.stack([cv2.resize(channel[top:top+height,left:left+width],(original_w,original_h))
                               for channel in pred])
    return result


def measure(path,probabilities,gray):
    if not path:
        return {}
    support={}
    clear=[]
    width=gray.shape[1]
    for size,prob in probabilities.items():
        rows=[]
        for y,left,right,_ in path:
            span=right-left
            margin=max(10,min(25,span*2))
            a=slice(max(0,left-margin),max(0,left-1))
            b=slice(min(width,right+2),min(width,right+margin+1))
            if a.stop<=a.start or b.stop<=b.start:
                continue
            body=slice(left+1,right)
            core=prob[:,y,body].mean(axis=1)
            west=prob[1,y,a].mean()
            east=prob[1,y,b].mean()
            rows.append((core[1],core[2],west,east))
            if size==128:
                middle=float(np.median(gray[y,body]))
                light=[float(np.median(gray[y,k]))-middle for k in (a,b)]
                clear.append(min(light)>=RULE["separation_lightness"])
        values=np.asarray(rows)
        if not rows:
            continue
        threshold=RULE["rock_flank_threshold"]
        support[str(size)]=dict(rock=float(np.median(values[:,0])),
                                tree=float(np.median(values[:,1])),
                                rock_left=float(np.median(values[:,2])),
                                rock_right=float(np.median(values[:,3])),
                                bilateral=float(np.mean((values[:,2]>=threshold)&(values[:,3]>=threshold))))
    support["visibly_separate_both_flanks"]=float(np.mean(clear)) if clear else None
    return support


def conflicts(support):
    coarse=[support[str(scale)] for scale in (128,192) if str(scale) in support]
    core=sum(x["rock"]>=RULE["strong_rock"] and x["rock"]-x["tree"]>=RULE["rock_advantage"] for x in coarse)
    flanks=sum(x["bilateral"]>=RULE["bilateral_rows"] for x in coarse)
    separated=support.get("visibly_separate_both_flanks")
    scores=dict(coarse_core=core==2,
                coarse_context=(core==2 or (flanks==2 and separated is not None and separated<RULE["separate_rows"])),
                mixed_bilateral=False,mixed_same_side=False)
    wide,detail=support.get("128"),support.get("512")
    if not (wide and detail):
        return scores
    plausible_rock=wide["rock"]>=MIXED["coarse_rock"]
    plausible_rock &= wide["rock"]-wide["tree"]>=-MIXED["coarse_rock_disadvantage"]
    scores["mixed_bilateral"]=bool(plausible_rock and min(detail["rock_left"],detail["rock_right"])>=MIXED["detail_both_flanks"])
    scores["mixed_same_side"]=bool(plausible_rock and any(
        wide[flank]>=MIXED["same_coarse_flank"] and detail[flank]>=MIXED["detail_one_flank"]
        for flank in ("rock_left","rock_right")))
    return scores


def render(rows):
    picked=[r for r in rows if r["view"]=="原框" and r["path"] and r["cohort"]=="transfer"]
    board=Image.new("RGB",(1190,78+254*len(picked)),"#f4f1e8")
    pen=ImageDraw.Draw(board)
    label(pen,(14,12),"树干候选和两侧岩面证据",25)
    label(pen,(14,47),"青色为候选外缘，橙点为原图参考。右列比较不同场景尺度的岩面和树木分数。",16)
    for i,row in enumerate(picked):
        y=78+i*254
        label(pen,(14,y),row["title"]+"　"+row["id"],15)
        with Image.open(row["image"]) as source:
            original=source.convert("RGB")
        x0,y0,x1,y1=row["crop"]
        region=(max(0,x0-19),max(0,y0-8),min(original.width,x1+19),min(original.height,y1+8))
        tile=color_overlay(original,region,row["path"],row["sections"])
        tile.thumbnail((275,215),Image.Resampling.LANCZOS)
        board.paste(tile,(14,y+26))
        for j,size in enumerate(SIZES):
            stat=row["evidence"].get(str(size))
            if stat:
                msg=f"{size} 输入　内部岩面 {stat['rock']:.0%}　树木 {stat['tree']:.0%}　两侧岩面 {stat['rock_left']:.0%}／{stat['rock_right']:.0%}"
                label(pen,(370,y+35+j*41),msg,17)
        status="岩面冲突，待核" if row["withheld"]["mixed_same_side"] else "保留候选"
        label(pen,(370,y+200),status,17)
    board.save(OUT/"rock_context_comparison.png")


def main():
    configure()
    import cv2
    import torch
    from gongbi_training.scene_objects import SceneObjects

    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True,parents=True)
    previous=read(PREV/"results.json")
    sources={r["id"]:r for r in read(REFERENCE/"results.json")["cases"]}
    earlier=read(EARLIER/"results.json")["cases"]
    archive=[]
    for case in earlier:
        reference=sources[case["id"]]
        archive.append(dict(id=case["id"],title=case["id"],role=case["role"],image=case["image"],
                            image_sha256=reference["image_sha256"],sections=reference["sections"],
                            views=[dict(view=row["view"],crop=row["crop"],candidate=row["after"])
                                   for row in case["views"]]))
    baselines=verify_baselines()
    checkpoint=SCENE/"mapping_fixed.pt"
    assert sha(checkpoint)==read(SCENE/"checkpoint_freeze.json")["mapping_fixed"]
    write("protocol.json",dict(previous_sha256=sha(PREV/"results.json"),script_sha256=sha(__file__),
                              scales=SIZES,rule=RULE,checkpoint_sha256=sha(checkpoint),
                              mixed_scale_rule=MIXED, mixed_rule_chosen_after_first_known_case_diagnostic=True,
                              additional_regression_sha256=sha(EARLIER/"results.json"),
                              annotations_for_scoring_only=True))
    model=SceneObjects()
    model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=False)["model"])
    model.eval()
    cache={}
    results=[]
    cases=[(cohort,case) for cohort,group in (("transfer",previous["cases"]),("earlier",archive)) for case in group]
    for cohort,case in cases:
        assert sha(case["image"])==case["image_sha256"]
        if case["image"] not in cache:
            with Image.open(case["image"]) as source:
                rgb=np.asarray(source.convert("RGB"))
            cache[case["image"]]=scene_probabilities(rgb,model,torch,cv2),cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        prob,gray=cache[case["image"]]
        for view in case["views"]:
            path=view["candidate"]["retained"]
            evidence=measure(path,prob,gray)
            withheld=conflicts(evidence)
            results.append(dict(cohort=cohort,id=case["id"],title=case["title"],role=case["role"],view=view["view"],
                                image=case["image"],crop=view["crop"],path=path,sections=case["sections"],
                                evidence=evidence,withheld=withheld,
                                retained=dict(previous=bool(path),**{k:bool(path) and not v for k,v in withheld.items()}),
                                aligned=alignment(path,case["sections"])))
    render(results)
    assert baselines==verify_baselines()
    write("results.json",dict(records=results,painting_baselines=baselines,adopted=False,
                               independent_validation=False,robot_connected=False))
    for row in results:
        if row["view"]=="原框" and row["path"]:
            print(row["title"],row["withheld"],flush=True)


if __name__=="__main__":
    main()
