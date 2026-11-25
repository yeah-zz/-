"""Replay predicted scene silhouettes using the existing Stroke IR and brush."""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from master_studio.digital_strokes import DigitalStroke, render, replay
from professional_clpd import write


def render_pair(out, row, rgb, s0, candidate, candidate_name="s2"):
    size=(400,400)
    def candidates(pred, kind):
        contours,_=cv2.findContours((pred==kind).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        paths=[]
        for contour in sorted(contours,key=cv2.contourArea,reverse=True):
            if cv2.contourArea(contour)<40:
                continue
            points=cv2.approxPolyDP(contour,max(1.5,.004*cv2.arcLength(contour,True)),True)[:,0]
            if len(points)>=3:
                paths.append(points)
        return paths

    masks={"s0":s0,candidate_name:candidate}
    paths={method:{kind:candidates(p,kind) for kind in (1,2)} for method,p in masks.items()}
    budget={k:min(5,*(len(p[k]) for p in paths.values())) for k in (1,2)}
    if not all(budget.values()):
        return
    scale=364/max(rgb.shape[:2])
    offset=np.array(size)/2 - np.array(rgb.shape[:2][::-1])*scale/2
    board=Image.new("RGB",(1240,458),"#f7f3e9")
    font=ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",16)
    pen=ImageDraw.Draw(board)
    source=Image.fromarray(rgb)
    source.thumbnail(size)
    board.paste(source,(12,55))
    pen.text((12,8),f"各 {sum(budget.values())} 笔的预测轮廓 · {row['id']}",font=font,fill="#204335")
    pen.text((12,32),"原图",font=font,fill="#204335")
    for i,(method, polygons) in enumerate(paths.items(),1):
        strokes=[]
        for kind in (1,2):
            role="mountain_rock" if kind==1 else "trees"
            for index,points in enumerate(polygons[kind][:budget[kind]]):
                xy=(np.concatenate([points,points[:1]])*scale+offset).tolist()
                tag=f"{role}-{index}"
                width=2 if kind==1 else 1.3
                strokes.append(DigitalStroke(tag,tag,role,"predicted silhouette",xy,[width]*len(xy),
                  (62,52,43) if kind==1 else (38,62,48),.80,.18,.30,kind,(),
                  {"model":method,"source_image":row["image_sha256"],"geometry":"predicted semantic boundary",
                   "ink_parameters":"fixed same-budget drawing comparison"}))
        path=out/f"{method}_drawing.json"
        write(path,dict(canvas_wh=size,coordinate_system="xy pixels",seed=24,renderer="ribbon",
                        strokes=[s.record() for s in strokes]))
        canvas, frames=render(strokes,size,frames=True,renderer="ribbon")
        assert replay(path)[0].tobytes()==canvas.tobytes()
        canvas.save(out/f"{method}_drawing.png")
        frames[0].save(out/f"{method}_drawing.gif",save_all=True,append_images=frames[1:],duration=220,loop=0)
        board.paste(canvas,(420+(i-1)*408,55))
        pen.text((420+(i-1)*408,32),method.upper(),font=font,fill="#204335")
    board.save(out/"drawing_comparison.png")
    protocol = dict(source=row["id"],source_sha256=row["image_sha256"],
                    budget=budget,scope="only semantic silhouette extraction is learned; brush is rule-based")
    write(out/"drawing_protocol.json",protocol)
    return protocol
