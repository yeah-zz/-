# -*- coding: utf-8 -*-
"""Render cross-work tree and source-supported long-hemp study comparisons."""

from dataclasses import replace

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from gongbi_training.runtime import LAB, configure
from gongbi_training.cunfa_continuity import join_marks
from master_studio.brush import catmull_rom
from master_studio.digital_strokes import DigitalStroke, render, replay
from recognition_groups_20261123 import label, read, sha, verify_baselines
from recognition_scene_topology_20261124 import write
from crosswork_tree_samples import trees_from_references, widths_and_curvature, emit_tree, branch_validation
from digital_specialty_round import GABOR, OUT as PRIOR, observed_marks


OUT=LAB/"crosswork_drawing_20261124"
SIZE=(360,300)
SEED=24


def length(points):
    return float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())


def texture_metrics(curves, widths, scale):
    """Describe scale-normalized shape and spacing, without a style label."""
    lengths=np.array([length(x) for x in curves])
    ends=np.array([x[-1]-x[0] for x in curves])
    heading=np.arctan2(ends[:,1],ends[:,0])
    centers=np.array([np.mean(x,axis=0) for x in curves])
    spacing=np.linalg.norm(centers[:,None]-centers[None,:],axis=-1)
    np.fill_diagonal(spacing,np.inf)
    return dict(paths=len(curves),median_length_ratio=float(np.median(lengths)/scale),
                longest_ratio=float(lengths.max()/scale),
                axial_agreement=float(np.abs(np.average(np.exp(2j*heading),weights=lengths))),
                median_nearest_spacing_ratio=float(np.median(spacing.min(1))/scale) if len(curves)>1 else None,
                median_width_ratio=float(np.median(np.concatenate(widths))/scale))


def fit_hemp():
    import cv2
    from skimage.morphology import skeletonize

    rows=read(GABOR/"dataset_manifest.json")["records"]
    boxes={"aam_long_hemp_2":[(55,28),(282,22),(261,212),(31,205)],
           "aam_long_hemp_0":[(66,18),(305,24),(271,224),(63,214)],
           "aam_small_axe_2":[(28,26),(318,26),(318,207),(28,207)]}
    fit={}
    for ident,region in boxes.items():
        source=next(r for r in rows if r["id"]==ident)
        with Image.open(source["image"]) as image:
            im=image.convert("RGB")
        marks,_=observed_marks(im,cv2,skeletonize,region)
        gray=np.asarray(im.convert("L"),dtype=np.float32)/255
        paths,links=join_marks(marks,gray,cv2)
        fit[ident]=dict(image=source["image"],source_sha256=sha(source["image"]),
            source_url=source["source_url"],label=source["label"],image_group=source["work_group"],
            roi=region,marks=marks,paths=paths,links=links)
    scopes={k:{a:v for a,v in item.items() if a not in ("marks","paths")}|
        {"visible_fragments":[{"xy":m[0].tolist(),"width":m[1].tolist(),
                               "length":m[2],"darkness":m[4].tolist()} for m in item["marks"]],
         "inferred_routes":[{"xy":p[0].tolist(),"width":p[1].tolist(),"darkness":p[2].tolist(),
                             "origin":p[3]} for p in item["paths"]],
         "original_sequence":"unknown","annotation_scope":"named illustration; local paths are extracted hypotheses"}
        for k,item in fit.items()}
    write(OUT/"hemp_evidence.json",scopes)
    return fit


SHAPES=(
    ((160,113,62,35,66),(211,240,272,272,239),(186,174,160,151,135)),
    ((95,62,86,104,58),(156,224,276,320,283),(132,141,171,214,233)),
)
ROCK_Y=np.array([62,93,146,194,241])


def digital_line(id,object_id,role,points,width,opacity,dry,after=(),layer=0,technique="manual",meta=None):
    xy=np.asarray(points,dtype=float).tolist()
    return DigitalStroke(id,object_id,role,technique,xy,[width]*len(xy),(61,57,48),opacity,dry,.26,layer,after,
                         {"geometry":"manual design"} if meta is None else meta)


def warp(curve, shape, directed):
    left,right,ridge=tuple(np.asarray(v) for v in shape)
    curve=np.asarray(curve)
    u=np.clip((curve[:,0]-22)/270,.025,.975)
    v=np.clip((curve[:,1]-12)/220,.02,.98)
    yy=66+170*v
    ll,rr,mid=(np.interp(yy,ROCK_Y,p) for p in (left,right,ridge))
    s=.1+.8*u
    xx=ll+s*(rr-ll)
    if directed:
        xx=np.where(u<.5,ll+(mid-ll)*(2*s),mid+(rr-mid)*(2*s-1))
    return np.c_[xx,yy]


def stone(fit,index,directed,fragments=False):
    shape=SHAPES[index]
    left,right,ridge=shape
    tag=f"rock{index}"
    corner=np.vstack((np.c_[left,ROCK_Y],np.c_[right,ROCK_Y][::-1],[[left[0],ROCK_Y[0]]]))
    edge=digital_line(f"{tag}_outline",tag,"rock_contour",corner,1.75,.71,.29)
    result=[edge]
    if directed:
        inner=catmull_rom(np.c_[ridge,ROCK_Y],7)
        result.append(digital_line(f"{tag}_ridge",tag,"face_boundary",inner,1.75,.58,.37,
                                   (edge.id,),1,"rock_face"))
    item=fit["aam_long_hemp_2"]
    if fragments:
        paths=sorted(((m[0],m[1],m[4],[i]) for i,m in enumerate(item["marks"])),
                     key=lambda p:length(p[0]),reverse=True)[:18]
    else:
        paths=sorted(item["paths"],key=lambda p:length(p[0]),reverse=True)[:12]
    for i,(curve,ws,ink,origins) in enumerate(paths):
        xy=warp(curve,shape,directed)
        width=np.clip(ws*.53,.85,5.2)
        dependence=result[1].id if directed else edge.id
        opacity_profile=np.clip(.53+.72*ink,.49,1)
        result.append(DigitalStroke(f"{tag}_cun{i}",tag,"rock_texture","hemp_route_hypothesis",
            xy.tolist(),width.tolist(),(58,53,43),.64,.46,.25,2,(dependence,),
            {"geometry":"individual source fragment" if fragments else "source extracted route with evidenced continuations","fragments":origins,
             "source_work":item["image_group"],"source_sha256":item["source_sha256"],
             "field":"hand-set two rock faces" if directed else "hand-set bounding face",
             "sequence":"algorithmic inference","virtual_dryness":"manual"},tuple(opacity_profile.tolist())))
    return result


def render_variant(name, strokes, renderer="ribbon",gif=False):
    canvas,frames=render(strokes,SIZE,seed=SEED,frames=gif,renderer=renderer)
    canvas.save(OUT/f"{name}.png")
    sequence=OUT/f"{name}.json"
    write(sequence,dict(canvas_wh=SIZE,coordinate_system="xy pixels",seed=SEED,renderer=renderer,
                        sequence="algorithmic",strokes=[s.record() for s in strokes]))
    assert replay(sequence)[0].tobytes()==canvas.tobytes()
    if frames:
        frames[0].save(OUT/f"{name}.gif",save_all=True,append_images=frames[1:],duration=145,loop=0)
        for i,frame in enumerate(frames):
            if i in (0,1,len(frames)//2,len(frames)-1):
                frame.save(OUT/f"{name}_step_{i:02d}.png")
    return canvas


def composition(fit,trees,grid,taper):
    rock=stone(fit,1,True)
    # Attach the tree to the upper shoulder of the rear face. Its lowest
    # point shares a small ledge with the ridge, so the objects overlap.
    tree=emit_tree(trees,grid,taper,((133.,119.),83.,.06),1,"learned")
    tree[0]=replace(tree[0],after=(rock[1].id,),layer=3)
    tree[1:]=[replace(s,layer=4) for s in tree[1:]]
    root=catmull_rom([[133,117],[135,121],[128,125],[121,128],[114,133]],5)
    base=digital_line("root_ledge",tree[0].object_id,"root",root,1.6,.71,.27,(tree[0].id,),4,"manual_root_contact")
    layers=[("upper_shelf",[(118,129),(131,130),(149,124),(172,128)],.41),
            ("talus",[(30,254),(58,247),(92,248),(112,250)],.32),
            ("water_edge",[(219,256),(248,255),(273,256),(299,255)],.27),
            ("water_edge_low",[(81,268),(103,267),(129,269),(166,268)],.23)]
    accents=[digital_line(f"setting_{name}","ground",name,catmull_rom(points,6),1.2,alpha,.38,
                          (rock[0].id,),5,"manual_space") for name,points,alpha in layers]
    return rock+tree+[base]+accents


def show_grid(name,heading,caption_images,label_size=15):
    width=1110
    height=55+330*((len(caption_images)+2)//3)
    board=Image.new("RGB",(width,height),"#f4f1e8")
    pen=ImageDraw.Draw(board)
    label(pen,(12,9),heading,23)
    for i,(caption,image) in enumerate(caption_images):
        x,y=10+(i%3)*366,52+(i//3)*330
        label(pen,(x,y),caption,label_size)
        board.paste(ImageOps.contain(image,(350,300)),(x,y+25))
    board.save(OUT/f"{name}.png")


def main():
    configure()
    OUT.mkdir(exist_ok=True)
    lock=verify_baselines()
    trees=trees_from_references()
    grid,profile,widths=widths_and_curvature()
    branch=branch_validation(trees)
    fit=fit_hemp()
    write(OUT/"tree_references.json",dict(samples=trees,width_reference=widths,branch_metrics=branch,
          explicitly_independent_works=[],usage="historical development; split by whole work",
          fit_source="assistant-reviewed partial tree lines; no definite isolated leafless silhouettes"))
    tree_tiles=[]
    for i,tree in enumerate(trees):
        with Image.open(tree["image"]) as image:
            raw=image.convert("RGB").crop(tree["crop"])
        annotated=raw.copy()
        pen=ImageDraw.Draw(annotated)
        for path in tree["paths"]:
            xy=(np.asarray(path["xy"])-tree["crop"][:2]).tolist()
            pen.line([tuple(p) for p in xy],fill="#58b090" if path["parent"] else "#ee8c57",width=1)
        layout=([178.,258.],172.,.09 if i%2==0 else -.11)
        rule=emit_tree(trees,grid,profile,layout,i,"rule")
        learned=emit_tree(trees,grid,profile,layout,i,"learned")
        tree_tiles.extend([(f"作品 {i+1} 原图与可见局部轨迹",annotated),
                           ("T0 规则分枝",render_variant(f"tree{i}_rule",rule)),
                           ("T3 多作品宽度和关联分枝",render_variant(f"tree{i}_learned",learned))])
    show_grid("tree_comparison","跨作品树干与可见枝条图对照",tree_tiles)
    rock_tiles=[]
    primary=fit["aam_long_hemp_2"]
    target=sorted(primary["paths"],key=lambda p:length(p[0]),reverse=True)[:12]
    texture={"source_12_routes":texture_metrics([r[0] for r in target],[r[1] for r in target],220)}
    for i in range(2):
        comparisons=[("R0","R0 旧 18 个碎片迁到相同轮廓",stone(fit,i,False,fragments=True)),
                     ("R1","R1 结合原图证据接续",stone(fit,i,False)),
                     ("R2","R2 加岩面方向和脊线",stone(fit,i,True))]
        for variant,caption,strokes in comparisons:
            key=f"rock{i}_{variant}"
            pieces=[s for s in strokes if s.role=="rock_texture"]
            texture[key]=texture_metrics([np.asarray(s.xy) for s in pieces],
                                         [np.asarray(s.widths_px) for s in pieces],170)
            rock_tiles.append((caption,render_variant(key,strokes)))
    source_panel=[]
    for row in fit.values():
        with Image.open(row["image"]) as image:
            raw=image.convert("RGB")
        fragments=raw.copy()
        ink=ImageDraw.Draw(fragments)
        for mark in row["marks"]:
            ink.line([tuple(p) for p in mark[0]],fill="#68c6b7",width=2)
        joins=fragments.copy()
        ink=ImageDraw.Draw(joins)
        for link in row["links"]:
            if link["selected"]:
                ink.line([tuple(p) for p in link["path"]],fill="#fa9248",width=2)
        source_panel.extend([(row["label"]+" · 有图注的原图",raw),("可见墨迹片段",fragments),("橙线为推断接续",joins)])
    show_grid("hemp_sources","披麻、斧劈原图与有图像证据的接续",source_panel)
    show_grid("hemp_transfer","同一披麻参考在两种石形上的迁移",rock_tiles)
    painting=composition(fit,trees,grid,profile)
    assert len(painting)==24
    current=render_variant("tree_rock",painting,gif=True)
    legacy=render_variant("same_strokes_legacy",painting,"legacy")
    with Image.open(PRIOR/"composition.png") as image:
        old=image.convert("RGB")
    show_grid("painting_comparison","同为 24 笔的组合试验与笔刷对照",
              [("上一轮并排规则对照",old),("本轮结构 · 旧笔刷",legacy),("本轮结构 · 新笔刷",current)])
    def summarize(item):
        marks=item["marks"];paths=item["paths"]
        longest=max(p[2] for p in marks)
        return dict(fragments=len(marks),routes=len(paths),
                    hypothesized_links=sum(row["selected"] for row in item["links"]),
                    unsupported_pairs=sum(not row["evidence_pass"] for row in item["links"]),
                    max_fragment_px=longest,max_route_px=max(length(r[0]) for r in paths),
                    visual_mark_groups=1,original_order_known=False)
    write(OUT/"drawing_results.json",dict(hemp={k:summarize(v) for k,v in fit.items()},branch=branch,widths=widths,
          texture_shape_metrics=texture,painting_strokes=len(painting),same_budget=24,
          worked_comparisons=["T0","T1","T1_taper","T2","T3","R0","R1","R2"],
          independent_validation=False,baseline_hashes=lock))
    assert verify_baselines()==lock


if __name__=="__main__":
    main()
