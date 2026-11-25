# -*- coding: utf-8 -*-
"""Measure candidate-level tree/rock fusion on the archived development references."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from gongbi_training.runtime import LAB, configure
from gongbi_training.structure_candidates import candidate_fusion
from gongbi_training.visual_structure import detect
from recognition_groups_20261123 import SCENE, label, read, sha, verify_baselines
from recognition_scene_topology_20261124 import write
from visual_structure_round import HOLDOUT, OUT as PREVIOUS, hits


OUT=LAB/"crosswork_structure_20261124"


def main():
    configure()
    import cv2
    import torch
    from skimage.morphology import skeletonize
    from gongbi_training import structure_candidates
    from gongbi_training.scene_objects import SceneObjects
    from gongbi_training.source_scene_mapping import source_to_scene
    from recognition_trunk_rock_context_20261124 import scene_probabilities

    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    old=read(PREVIOUS/"results.json")
    checkpoint=SCENE/"mapping_fixed.pt"
    assert sha(checkpoint)==read(PREVIOUS/"protocol.json")["checkpoint"]
    baselines=verify_baselines()
    assert baselines==old["painting_baselines"]
    refs=read(HOLDOUT/"reference_freeze.json")["cases"]
    windows=read(HOLDOUT/"scan_reference_freeze.json")["windows"]
    source_hashes={r["image"]:r["image_sha256"] for r in refs}|{r["image"]:r["sha256"] for r in windows}
    model=SceneObjects()
    model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=False)["model"])
    model.eval()
    root_id="wang-yuanqi_autumn-mountains_6"
    root_box=[8,32,52,104]
    record=[]
    example_paths={}
    spacing=173
    board=Image.new("RGB",(7*spacing+8,60+12*145),"#f4f1e8")
    pen=ImageDraw.Draw(board)
    label(pen,(10,8),"多源候选关联、双侧证据与树石分类",24)
    label(pen,(10,39),"青为树木，橙为山石，紫为待核。红圈为已核查树身位置。",15)
    row=0
    for path,digest in source_hashes.items():
        assert sha(path)==digest
        with Image.open(path) as image:
            source=image.convert("RGB")
        rgb=np.asarray(source)
        cues=detect(rgb,cv2,skeletonize)
        probabilities=scene_probabilities(rgb,model,torch,cv2)
        modes,geometry=candidate_fusion(rgb,cues,(probabilities[256]+probabilities[512])*.5,probabilities[512],cv2)
        slim=[]
        for ref in (r for r in refs if r["image"]==path):
            slim.append((ref["title"],ref["crop"],ref["points"],ref["role"],None))
        for win in (r for r in windows if r["image"]==path):
            assert sha(win["mask"])==win["mask_sha256"]
            with Image.open(win["mask"]) as image:
                truth=source_to_scene(np.asarray(image))
            for box in win["boxes"]:
                focus=win["id"]==root_id and box==root_box
                slim.append(("树根与岩脊交界" if focus else win["id"],box,[],
                             "root_pending" if focus else "rough_rock",truth))
        for name,box,points,kind,truth in slim:
            x0,y0,x1,y1=box
            area=np.s_[y0:y1,x0:x1]
            rough=truth[area]==1 if truth is not None else np.ones((y1-y0,x1-x0),bool)
            values={}
            for mode,(support,roles) in modes.items():
                d=roles[area]; s=support[area]
                lo=hits(support,points)
                tree=hits(support&(roles==2),points)
                rock=hits(support&(roles==1),points)
                values[mode]=dict(coverage=sum(lo),tree_hits=sum(tree),
                                  rock_without_tree=sum(r and not t for r,t in zip(rock,tree)),
                                  uncertain_without_tree_or_rock=sum(p and not r and not t for p,r,t in zip(lo,rock,tree)),
                                  area=int(s.sum()),
                                  rough_counts={k:int((s&rough&(d==v)).sum())
                                                for k,v in (("tree",2),("rock",1),("uncertain",255))})
            src={k:int((cues[k][area]&rough&(modes["A1"][1][area]==2)).sum()) for k in ("dark","light","paired")}
            record.append(dict(image=path,title=name,roi=box,kind=kind,points_yx=points,
                               rough_area=int(rough.sum()),prior_source_tree_load=src,modes=values))
            if truth is not None and kind!="root_pending":
                continue
            raw=source.crop(box)
            scale=min(4.,113/raw.height,160/raw.width)
            size=(round(raw.width*scale),round(raw.height*scale))
            y=60+145*row
            label(pen,(8,y),name,14)
            for j,key in enumerate((None,*modes)):
                tile=raw.resize(size,Image.Resampling.NEAREST)
                ink=ImageDraw.Draw(tile)
                if key:
                    support,roles=modes[key]
                    for dy,dx in np.argwhere(support[area]):
                        tx,ty=(dx+.5)*scale,(dy+.5)*scale
                        color={2:"#00c0a7",1:"#ea8c39",255:"#a775cf"}[int(roles[dy+y0,dx+x0])]
                        ink.ellipse((tx-1,ty-1,tx+1,ty+1),fill=color)
                for py,px in points:
                    tx,ty=(px-x0+.5)*scale,(py-y0+.5)*scale
                    ink.ellipse((tx-3,ty-3,tx+3,ty+3),outline="#fb321a")
                x=8+j*spacing
                board.paste(tile,(x,y+18))
                label(pen,(x,y+130),key or "原图",13)
            row+=1
            if points or kind=="root_pending":
                visible=[]
                for g in geometry:
                    xy=g["xy"]
                    inside=(xy[:,0]>=x0)&(xy[:,0]<x1)&(xy[:,1]>=y0)&(xy[:,1]<y1)
                    if not np.any(inside): continue
                    excluded={"path","xy","unit","local","duplicate","unique"}
                    visible.append(({k:v for k,v in g.items() if k not in excluded})|{"xy":xy.tolist()})
                example_paths[name]=dict(image=path,roi=box,paths=visible,
                    same_instance_status="unverified",occlusion_status="unverified")
        print(Path(path).name,len(geometry),flush=True)
    negatives=[r for r in record if r["kind"] in ("rough_rock","root_pending")]
    assert len(negatives)==40
    assert [sum(r["modes"][m]["rough_counts"]["tree"] for r in negatives) for m in ("A0","A1")]==[2246,3860]
    assert [sum(r["modes"][m]["coverage"] for r in record) for m in ("A0","A1")]==[9,12]
    board.save(OUT/"candidate_comparison.png")
    write(OUT/"paths.json",example_paths)
    write(OUT/"structure.json",dict(results=record,implementation=sha(structure_candidates.__file__),
              checkpoint=sha(checkpoint),baselines=baselines,independent_validation=False))
    assert verify_baselines()==baselines


if __name__=="__main__":
    main()
