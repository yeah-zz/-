# -*- coding: utf-8 -*-
"""Run source-guided trunk and cunfa studies using replayable digital strokes."""

from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from gongbi_training.runtime import LAB, configure
from gongbi_training.scene_topology import ink_paths
from master_studio.digital_strokes import DigitalStroke, render
from recognition_groups_20261123 import label, read, sha, verify_baselines
from recognition_scene_topology_20261124 import write


OUT = LAB/"digital_specialty_round_20261124"
GABOR = LAB/"cunfa_gabor_baseline_20261110"
TRUNKS = LAB/"recognition_trunk_source_background_20261123"/"results.json"
SIZE = (360, 300)


def observed_marks(image, cv2, skeletonize, polygon=None):
    gray = np.asarray(image.convert("L"), dtype=np.float32)/255
    smooth = cv2.GaussianBlur(gray,(0,0),.7)
    contrast = cv2.GaussianBlur(smooth,(0,0),5)-smooth
    support = contrast >= .033
    if polygon:
        region=np.zeros_like(gray,dtype=np.uint8)
        cv2.fillPoly(region,[np.asarray(polygon,np.int32)],1)
        support &= region.astype(bool)
    center = skeletonize(np.ascontiguousarray(support))
    radius = cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2,5)
    marks = []
    for path in ink_paths(center):
        if len(path) < 8:
            continue
        y,x = path.T
        curve = np.c_[x,y]
        length = float(np.linalg.norm(np.diff(curve,axis=0),axis=1).sum())
        direct = float(np.linalg.norm(curve[0]-curve[-1]))
        width = radius[y,x]*2
        darkness = 1-gray[y,x]
        marks.append((curve,width,length,direct,darkness))
    return marks, float(support.mean())


def recognize(rows, cv2, skeletonize):
    from train_cunfa_gabor_20261110 import CLASSES, grouped_cv
    baseline = read(GABOR/"full_gabor_evaluation.json")
    by_id = {r["id"]:r for r in rows}
    known = [by_id[p["id"]] for p in baseline["predictions"]]
    old_groups = {ident:fold["held_artist_group"] for fold in baseline["folds"] for ident in fold["test_ids"]}
    assert all(row["label"] == p["label"] for row,p in zip(known,baseline["predictions"]))
    other = [r for r in rows if r["label"] not in CLASSES]
    assert not {r["id"] for r in known} & {r["id"] for r in other}
    assert all(len({r["artist_group"] for r in known if r["work_group"]==group})==1
               for group in {r["work_group"] for r in known})
    def vector(r):
        with Image.open(r["image"]) as image:
            normalized=ImageOps.contain(image,(180,180))
            marks,density=observed_marks(normalized,cv2,skeletonize)
        if not marks:
            return [density,0.,0.,0.,0.,0.,0.,0.,0.,0.]
        widths=np.concatenate([m[1] for m in marks])
        lengths=np.array([m[2] for m in marks])
        straight=np.array([m[3]/m[2] for m in marks])
        directions=np.array([np.arctan2(*(m[0][-1]-m[0][0])[::-1]) for m in marks])
        direction_weights=np.abs(np.average(np.exp(2j*directions),weights=lengths))
        return [density, float(np.log1p(len(marks))), *np.quantile(widths,[.5,.85,.95]),
                *np.quantile(lengths,[.5,.85]),float(straight.mean()),float(direction_weights),
                float(np.average([m[4].mean() for m in marks],weights=lengths))]

    x=np.asarray([vector(r) for r in known])
    y=np.array([CLASSES.index(r["label"]) for r in known])
    group=np.array([old_groups[r["id"]] for r in known])
    report=grouped_cv(x,y,group,"stroke_statistics",known,np.asarray([vector(r) for r in other]))
    report["matched_gabor_label_agreement"]=baseline["label_agreement"]
    report["baseline_scope_sha256"]=sha(GABOR/"full_gabor_evaluation.json")
    if report["unknown_probe"]:
        report["unknown_probe"]["ids"]=[r["id"] for r in other]
    return report


def trunk_profile(case):
    yy,left,right=np.asarray(case["sections"]).T
    height=yy[-1]-yy[0]
    t=(yy[-1]-yy[::-1])/height
    center=((left+right)/2)[::-1]
    width=(right-left+1)[::-1]
    return t,(center-center[0])/height,width/width[0]


def draw_trunks(cases):
    prototype=next(r for r in cases if r["id"]=="T2/1")
    learn_t,learn_x,learn_w=trunk_profile(prototype)
    report=[]
    for case in cases:
        if not case["sections"] or case["image_sha256"]==prototype["image_sha256"]:
            continue
        t,x,w=trunk_profile(case)
        report.append({"id":case["id"],"source":case["image"],"sections":len(t),
                       "width_mae_flat":float(np.abs(w-1).mean()),
                       "width_mae_learned":float(np.abs(w-np.interp(t,learn_t,learn_w)).mean()),
                       "center_mae_straight":float(np.abs(x).mean()),
                       "center_mae_learned":float(np.abs(x-np.interp(t,learn_t,learn_x)).mean())})
    t=np.linspace(0,1,52)
    source={"role":"single reviewed trunk profile","fit_source":prototype["image"],
            "sha256":prototype["image_sha256"],"fit_sections":len(learn_t),
            "width_and_lateral_profile":"fit by interpolation from source sections",
            "layout_and_ink":"hand-set digital parameters"}
    examples=[]
    for example, sign in enumerate((1.,-1.)):
        root_id=f"tree{example}_trunk"
        body=np.c_[190+sign*(np.interp(t,learn_t,learn_x)*184+31*t*t),259-184*t]
        widths=14*np.interp(t,learn_t,learn_w)
        tree=[DigitalStroke(root_id,f"tree{example}","trunk","digital_taper",
                            body.tolist(),widths.tolist(),(57,48,36),.85,.09,.45,0,(),source)]
        for k,(at,way) in enumerate(((.46,-1),(.61,1),(.77,-1),(.86,1))):
            start=body[round(at*(len(body)-1))]
            extent=way*sign*(63-7*k)
            controls=[start,start+[extent*.25,-10],start+[extent*.65,-17],start+[extent,-37+2*k]]
            from master_studio.brush import catmull_rom
            curve=np.asarray(catmull_rom(controls,8))
            branch_w=np.linspace(4.0,1.1,len(curve))
            tree.append(DigitalStroke(f"tree{example}_branch{k}",f"tree{example}","branch","procedural_branch",
                                      curve.tolist(),branch_w.tolist(),(57,48,36),.83,.20,.32,1,(root_id,),
                                      {"layout_and_width":"hand-set branch controls","attachment":"trunk profile"}))
        examples.append(tree)
    return prototype,examples,report


def rock_passes(rows, cv2, skeletonize):
    reference=next(r for r in rows if r["id"]=="aam_long_hemp_2")
    with Image.open(reference["image"]) as opened:
        source=opened.convert("RGB")
    patch=[(55,28),(282,22),(261,212),(31,205)]
    marks,_=observed_marks(source,cv2,skeletonize,patch)
    selected=sorted(marks,key=lambda m:m[2],reverse=True)[:18]
    layouts=[
        ([139,58,36,60,69],[211,251,270,284,248]),
        ([182,142,103,39,63],[227,275,314,312,255])
    ]
    output=[]
    fit_geometry=[]
    for ex,(lx,rx) in enumerate(layouts):
        name=f"rock{ex}"
        y=np.array([42,88,151,220,252])
        border=np.vstack((np.c_[lx,y],np.c_[rx,y][::-1],[[lx[0],y[0]]]))
        stroke=DigitalStroke(f"{name}_contour",name,"rock_outline","digital_outline",border.tolist(),
                             [2.5]*len(border),(74,66,54),.85,.12,.38,0,(),{"outline":"hand-set new geometry"})
        strokes=[stroke]
        for rank,(curve,w,length,_,darkness) in enumerate(selected):
            uv=np.c_[(curve[:,0]-22)/270,(curve[:,1]-12)/220]
            v=np.clip(uv[:,1],.02,.98)
            cy=50+193*v
            left,right=np.interp(cy,y,lx),np.interp(cy,y,rx)
            cx=left+(right-left)*(.10+.8*np.clip(uv[:,0],0,1))
            cxy=np.c_[cx,cy]
            if np.linalg.norm(np.diff(cxy,axis=0),axis=1).sum() < 4:
                continue
            width=np.clip(w*(.72 if ex==0 else .8),1.05,8.5)
            opacity=float(np.clip(np.median(darkness)*1.15,.26,.86))
            # Geometry and relative ink come from the teaching image. Dryness and
            # spread are digital settings shared across both new rock faces.
            strokes.append(DigitalStroke(f"{name}_cun{rank}",name,"rock_texture","long_hemp_candidate",
                          cxy.tolist(),width.tolist(),(61,54,43),opacity,.31,.42,1,(stroke.id,),
                          {"source_url":reference["source_url"],"source_image":reference["image"],
                           "source_sha256":sha(reference["image"]),"roi":patch,"source_centerline":"image-derived",
                           "source_length_px":length,"original_order":"unknown","confidence":"exploratory"}))
            if ex==0:
                fit_geometry.append({"source_xy":curve.tolist(),"widths_px":w.tolist(),"opacity":opacity})
        output.append(strokes)
    return reference,source,output,fit_geometry


def main():
    configure()
    import cv2
    from skimage.morphology import skeletonize
    OUT.mkdir(exist_ok=True)
    lock=verify_baselines()
    manifest=GABOR/"dataset_manifest.json"
    rows=read(manifest)["records"]
    cases=read(TRUNKS)["cases"]
    for ref in cases:
        assert sha(ref["image"])==ref["image_sha256"]
    model=recognize(rows,cv2,skeletonize)
    prototype,trees,tree_check=draw_trunks(cases)
    rock_ref,rock_source,rocks,geometry=rock_passes(rows,cv2,skeletonize)
    mix = [
        replace(stroke, xy=(np.asarray(stroke.xy)*scale+[dx,dy]).tolist(),
                widths_px=(np.asarray(stroke.widths_px)*scale).tolist(), layer=stroke.layer+group*2)
        for group,(strokes,scale,dx,dy) in enumerate(((rocks[0],.75,8,56),(trees[1],.71,146,70)))
        for stroke in strokes
    ]
    sequences={"tree_replay":trees[0],"tree_new_form":trees[1],"rock_replay":rocks[0],
               "rock_new_form":rocks[1],"cunfa_only":[replace(s,after=()) for s in rocks[1][1:]],"composition":mix}
    renders={}
    for name,strokes in sequences.items():
        image,frames=render(strokes,SIZE,frames=True)
        renders[name]=image
        image.save(OUT/f"{name}.png")
        write(OUT/f"{name}.json",{"canvas_wh":SIZE,"coordinate_system":"xy pixels","sequence":"algorithmic",
                                  "strokes":[s.record() for s in strokes]})
        frames[0].save(OUT/f"{name}.gif",save_all=True,append_images=frames[1:],duration=165,loop=0)
    panel=Image.new("RGB",(1110,1010),"#f4f1e8")
    pen=ImageDraw.Draw(panel)
    label(pen,(14,10),"树干结构拟合与披麻皴数字笔触试验",23)
    with Image.open(prototype["image"]) as ref:
        tree_source=ref.convert("RGB").crop(prototype["crop"])
    captions=("树干截面来源","树干拟合＋规则分枝","变形树干＋规则分枝","有图注的长披麻局部",
              "新山石甲","新山石乙","生成的独立皴笔","树石合绘")
    panels=(tree_source,renders["tree_replay"],renders["tree_new_form"],rock_source,
            renders["rock_replay"],renders["rock_new_form"],renders["cunfa_only"],renders["composition"])
    for i,(caption,image) in enumerate(zip(captions,panels)):
        x,y=12+(i%3)*366,42+(i//3)*322
        label(pen,(x,y),caption,15)
        panel.paste(ImageOps.contain(image,(350,285)),(x,y+22))
    panel.save(OUT/"digital_comparison.png")
    write(OUT/"results.json", {"recognition":model,"trunk_transfer":tree_check,
        "trunk_fit_sections":len(prototype["sections"]),"cunfa_source":rock_ref["id"],
        "cunfa_source_traces":geometry,"sequence_sizes":{k:len(v) for k,v in sequences.items()},
        "independent_validation":False,"painting_baselines":lock,
        "input_hashes":{"recognition":sha(manifest),"trunk":sha(TRUNKS)}})
    assert verify_baselines()==lock


if __name__=="__main__":
    main()
