"""Deterministic visual studies. No robot imports, connections or motion output.

Normalized width, pigment opacity and moisture are illustrative controls,
not measured pressure, physical ink concentration or a calibrated E6 model.
"""
from pathlib import Path
import sys
import json
import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from master_studio.brush import catmull_rom, resample, paper_base

OUT = ROOT / "offline_previews" / "20261021"
OUT.mkdir(parents=True, exist_ok=True)
FONT = "C:/Windows/Fonts/msyh.ttc"
KAI = "C:/Windows/Fonts/simkai.ttf"
INK = (38, 44, 42)
MUTED = (108, 115, 107)
GREEN = (49, 98, 84)
RECORDS = []


def label(im, pos, text, size=24, color=INK, kai=False):
    ImageDraw.Draw(im).text(pos, text, font=ImageFont.truetype(KAI if kai else FONT, size), fill=color)


def sheet(title, subtitle, h):
    im = Image.new("RGB", (1800, h), (245, 243, 236))
    label(im, (64, 28), "山河绘  /  BRUSH STUDIES  /  01", 18, GREEN)
    label(im, (60, 65), title, 43)
    label(im, (64, 128), subtitle, 22, MUTED)
    ImageDraw.Draw(im).line((64, 175, 1736, 175), fill=(208, 212, 201), width=2)
    label(im, (64, h - 48), "离线视觉示意 · 未经实机标定 · 参数为归一化视觉量，不是下压毫米值或接触力", 20, MUTED)
    return im


def envelope(n, kind="expressive", seed=0):
    t = np.linspace(0, 1, n)
    if kind == "constant":
        return np.full(n, .65)
    if kind == "taper":
        return .025 + .94 * np.sin(np.pi * t) ** .72 * (1 - .60 * t)
    if kind == "pulse":
        p = .12 + .64 * np.sin(np.pi * t) ** .6
        p *= .73 + .24 * np.sin(t * np.pi * 5 - .4)
    else:
        p = .045 + .90 * np.sin(np.pi * t) ** .65
        p *= .82 + .16 * np.sin(t * np.pi * 3 + .5)
    return np.maximum(.012, p * np.clip((1 - t) / .12, .01, 1))


def stroke(im, controls, width=18, kind="expressive", tone=.9, water=.35,
           dry=.0, seed=0, color=INK, record=False):
    pts = np.asarray(resample(catmull_rom(controls, 12), 1.15), float)
    if len(pts) < 2:
        return
    n = len(pts)
    p = envelope(n, kind, seed)
    widths = width * (.04 + .96 * p)
    if kind == "constant":
        widths[:] = width * .65
    tang = np.gradient(pts, axis=0)
    tang /= np.maximum(1e-6, np.linalg.norm(tang, axis=1))[:, None]
    normal = np.column_stack((-tang[:, 1], tang[:, 0]))
    margin = width + 10
    x0 = max(0, int(pts[:, 0].min() - margin))
    y0 = max(0, int(pts[:, 1].min() - margin))
    x1 = min(im.width, int(pts[:, 0].max() + margin + 1))
    y1 = min(im.height, int(pts[:, 1].max() + margin + 1))
    if x1 <= x0 or y1 <= y0:
        return
    local = pts - (x0, y0)
    sz = (x1 - x0, y1 - y0)
    S = 2
    mask = Image.new("L", (sz[0] * S, sz[1] * S))
    d = ImageDraw.Draw(mask)
    left = (local + normal * widths[:, None] / 2) * S
    right = (local - normal * widths[:, None] / 2) * S
    d.polygon([tuple(v) for v in np.vstack((left, right[::-1]))], fill=255)
    if kind == "constant":
        for q, w in ((local[0], widths[0]), (local[-1], widths[-1])):
            x, y = q * S
            r = w * S / 2
            d.ellipse((x-r, y-r, x+r, y+r), fill=255)
    rng = np.random.default_rng(seed)
    if dry > .01:
        # Correlated longitudinal gaps represent separated bristle bundles.
        for j in range(max(5, int(width * .42))):
            off = rng.uniform(-.49, .49)
            q = (local + normal * widths[:, None] * off) * S
            noise = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, 12), rng.random(12))
            on = noise < dry * .78
            start = None
            for i in range(n):
                if on[i] and start is None:
                    start = i
                if start is not None and (not on[i] or i == n-1):
                    if i-start > 1:
                        d.line([tuple(v) for v in q[start:i+1]], fill=int(255*(1-dry)),
                               width=max(1, int(rng.uniform(.6, 1.4)*S)))
                    start = None
    mask = mask.resize(sz, Image.Resampling.LANCZOS)
    a = np.asarray(mask, float) / 255
    grain = rng.random(a.shape)
    a *= np.clip(1 - .10*grain - dry * .20 * (grain > .68), 0, 1)
    if water > .1:
        blur = np.asarray(mask.filter(ImageFilter.GaussianBlur(.6+water*2.3)), float)/255
        a = np.maximum(a * (1-water*.09), blur * water * .37)
    a = np.clip(a*tone, 0, 1)
    crop = np.asarray(im.crop((x0,y0,x1,y1)), float)
    comp = crop*(1-a[...,None]) + np.asarray(color)*a[...,None]
    im.paste(Image.fromarray(np.uint8(np.clip(comp,0,255))), (x0,y0))
    if record:
        RECORDS.append(dict(points_px=pts.round(2).tolist(),
                            width_envelope_normalized=p.round(4).tolist(),
                            max_width_px=width, opacity_visual=tone,
                            moisture_visual=water, dry_visual=dry, seed=seed))


def plot(im, box, values, caption, color=GREEN):
    x,y,w,h = box
    d = ImageDraw.Draw(im)
    d.line((x,y+h,x+w,y+h), fill=(207,209,198), width=1)
    pts = [(x+t*w,y+h-v*h) for t,v in zip(np.linspace(0,1,len(values)), values)]
    d.line(pts, fill=color, width=3)
    label(im, (x,y+h+9), caption, 18, MUTED)


def comparison():
    im = sheet("同一条路径，改变运笔方式", "几何路径保持一致；分别观察线宽、起收、墨色与飞白。左侧为理想化对照，并非原画测量。", 1240)
    label(im, (115,202), "A  等宽、等色对照", 26)
    label(im, (940,202), "B  目标笔触示意", 26, GREEN)
    rows = [("提按与出锋", "constant", "taper", .94,.94,0,0,.0,.0),
            ("转折与顿挫", "constant", "pulse", .94,.94,0,0,.0,.0),
            ("墨色层次", "expressive", "expressive", .94,.38,0,0,.2,.7),
            ("枯笔与飞白", "expressive", "expressive", .94,.90,0,.88,.3,.02)]
    for i,(title,ka,kb,ta,tb,da,db,wa,wb) in enumerate(rows):
        y = 270+i*214
        for x in (88,913):
            ImageDraw.Draw(im).rounded_rectangle((x,y,x+798,y+190), radius=13, fill=(251,249,242))
        label(im,(108,y+10),f"0{i+1}  {title}",22,MUTED)
        controls = [(0,0),(100,-18),(215,8),(340,-27),(520,-5),(680,-25)]
        if i == 1:
            controls = [(0,0),(130,-15),(220,-60),(300,-20),(400,25),(520,7),(680,-18)]
        for x,kind,tone,dry,water in ((125,ka,ta,da,wa),(950,kb,tb,db,wb)):
            pts = [(x+a,y+104+b*.65) for a,b in controls]
            stroke(im,pts,width=30 if i!=1 else 24,kind=kind,tone=tone,dry=dry,water=water,seed=180+i,record=True)
        desc = ["轻入 → 渐按 → 行笔 → 边走边提，末端出锋",
                "按笔和提笔随段落变化；实际速度曲线需另行标定",
                "右侧为淡湿目标：降低墨色不等于减少含水量",
                "右侧为浓干目标：保留纵向断白与分毫纹理"][i]
        label(im,(940,y+149),desc,19,GREEN)
        label(im,(112,y+149),["固定线宽，起收钝圆","缺少段落性的提按","浓墨湿笔参考","浓湿、连续落墨参考"][i],19,MUTED)
    im.save(OUT/"01_same_path_comparison.png")


def gallery():
    im = sheet("先选你希望机械臂画出的线质", "六种目标笔触。浓淡、干湿分别变化；飞白形态为程序纹理示意。", 1030)
    cards = [("01  浓湿勾线",.95,.85,0,"expressive","饱满连续 · 用于近景主轮廓"),
             ("02  淡湿勾线",.33,.85,0,"expressive","浅而湿润 · 用于远景与轻线"),
             ("03  浓干皴擦",.94,.03,.90,"expressive","墨色深、断白多 · 用于粗涩山石"),
             ("04  淡干皴线",.40,.02,.82,"expressive","轻、松、枯 · 用于疏淡纹理"),
             ("05  提按出锋",.88,.25,.12,"taper","粗细连续变化 · 用于枝干与长线"),
             ("06  顿挫折转",.86,.18,.25,"pulse","段落起伏 · 用于山骨转折")]
    for i,(title,tone,water,dry,kind,desc) in enumerate(cards):
        x = 64+(i%2)*855
        y = 205+(i//2)*245
        ImageDraw.Draw(im).rounded_rectangle((x,y,x+817,y+221),radius=14,fill=(251,249,242))
        label(im,(x+24,y+16),title,26,GREEN)
        pts = [(x+40,y+114),(x+160,y+99),(x+295,y+118),(x+440,y+87),(x+600,y+113),(x+755,y+93)]
        stroke(im,pts,width=40,kind=kind,tone=tone,water=water,dry=dry,seed=300+i,record=True)
        label(im,(x+24,y+163),desc,21,MUTED)
    im.save(OUT/"02_brush_gallery.png")


def wash(im, poly, color, strength, seed, flat=False):
    mask = Image.new("L", im.size)
    ImageDraw.Draw(mask).polygon(poly,fill=255)
    if not flat:
        mask=mask.filter(ImageFilter.GaussianBlur(1.3))
    a=np.asarray(mask,float)/255
    rng=np.random.default_rng(seed)
    if not flat:
        cloud=Image.fromarray(np.uint8(rng.uniform(60,255,(18,26)))).resize(im.size,Image.Resampling.BICUBIC)
        cloud=np.asarray(cloud,float)/255
        yy=np.arange(im.height)[:,None]
        ys=[q[1] for q in poly]
        fade=np.clip((max(ys)-yy)/max(1,max(ys)-min(ys)),0,1)**.45
        a*= (.25+.75*cloud)*fade
    a*=strength
    arr=np.asarray(im,float)
    im.paste(Image.fromarray(np.uint8(arr*(1-a[...,None])+np.asarray(color)*a[...,None])))


def landscape(improved=True):
    W,H=1500,900
    im=paper_base(W,H,random.Random(271),tone=(248,244,232))
    blue=(44,103,129); green=(65,112,82)
    geometry=[]
    def line(points,width=4,tone=.78,dry=.0,seed=1,color=INK):
        geometry.append(points)
        stroke(im,points,width=width,kind="expressive" if improved else "constant",
               tone=tone if improved else .9,water=.30 if improved else .02,
               dry=dry if improved else 0,seed=seed,color=color)
    # Coordinates echo the supplied composition; this is a new schematic.
    mountains=[([(420,380),(505,325),(565,340),(630,275),(690,290),(750,150),(805,125),(850,190),(915,235),(965,225),(1020,270),(1105,300),(1175,225),(1210,160),(1240,350),(1240,465),(420,465)],.40),
               ([(560,475),(650,420),(735,430),(825,315),(870,270),(930,345),(1000,365),(1030,435),(1030,510),(560,510)],.60),
               ([(990,545),(1055,505),(1125,510),(1185,540),(1250,545),(1285,590),(990,590)],.72),
               ([(85,820),(95,665),(150,595),(170,500),(235,460),(270,525),(320,545),(370,675),(415,705),(450,825)],.78)]
    for k,(poly,strength) in enumerate(mountains):
        if improved:
            wash(im,poly,blue,strength,60+k)
            wash(im,poly,green,strength*.14,80+k)
        else:
            mask=Image.new("L",im.size);ImageDraw.Draw(mask).polygon(poly,fill=255)
            stripe=Image.new("RGB",im.size,(248,244,232));sd=ImageDraw.Draw(stripe)
            for x in range(0,W,9):sd.line((x,0,x,H),fill=(64+(x%3)*7,133,158),width=8)
            im.paste(stripe,(0,0),mask)
        ridge=poly[:-2] if k<3 else poly[:-1]
        line(ridge,width=5 if k==3 else 3.6,tone=.55 if k==0 else .83,dry=.22,seed=50+k)
    # Broken rock structure follows slope, with ample open paper.
    paths=[[(756,178),(742,235),(725,282),(688,340)],[(817,169),(839,225),(882,260)],
           [(1188,243),(1170,295),(1165,349)],[(859,329),(835,373),(820,425)],
           [(913,375),(941,398),(970,450)],[(151,663),(188,684),(225,752),(241,788)],
           [(260,566),(285,618),(321,649),(340,710)],[(118,710),(147,752),(162,801)]]
    for k,p in enumerate(paths):line(p,6 if k>4 else 4,.76,.65,100+k)
    rng=random.Random(401)
    for k in range(52):
        x=rng.uniform(730,825);y=rng.uniform(210,300)
        if x < 750-(y-210)*.4:continue
        line([(x,y),(x-8,y+23),(x-14,y+43)],rng.uniform(2,4),.40,.72,k+400)
    # Old pine: tapered trunk and layered branches.
    line([(168,636),(190,574),(246,520),(269,447),(308,393),(285,357)],30,.94,.30,202)
    line([(235,526),(289,484),(360,467),(438,482),(520,500)],16,.93,.23,203)
    line([(283,443),(354,408),(417,416),(478,400)],12,.87,.25,204)
    line([(301,395),(243,367),(212,333)],10,.90,.15,205)
    for k,(cx,cy,rx) in enumerate([(211,337,57),(264,326,65),(320,350,68),(366,386,83),(441,388,66),(493,449,85),(542,473,65),(201,400,44)]):
        if improved:
            wash_rng=random.Random(1000+k)
            poly=[(cx+rx*math.cos(t),cy+rx*.24*math.sin(t)+wash_rng.uniform(-7,7)) for t in np.linspace(0,2*math.pi,30)]
            wash(im,poly,green,.42,550+k)
        for j in range(14):
            x=cx+rng.uniform(-rx*.85,rx*.85);y=cy+rng.uniform(-9,9)
            line([(x-10,y-2),(x,y-8),(x+8,y+1)],2.8,.72,.40,600+k*20+j,
                 INK if improved else green)
    # Pavilion, bank and boat.
    roof=[(1100,494),(1148,481),(1186,452),(1220,479),(1268,491),(1227,504),(1132,505),(1100,494)]
    line(roof,5,.93,.08,710)
    for p in [[(1131,503),(1133,558)],[(1234,504),(1234,558)],[(1186,452),(1185,494)],[(1130,550),(1237,550)],[(1127,561),(1245,561)]]:line(p,4,.86,.12,711)
    line([(1038,587),(1083,575),(1127,575),(1190,580),(1273,581)],5,.75,.35,715)
    line([(732,677),(770,686),(815,687),(857,677)],5,.92,.1,720)
    line([(764,678),(774,653),(791,646),(812,651),(827,675)],4,.90,.08,721)
    line([(791,650),(792,676)],3,.88,.05,722)
    line([(759,674),(756,637)],3,.85,.1,723)
    for k,(x,y,l) in enumerate([(590,624,85),(925,645,135),(1050,686,140),(520,761,150),(705,799,290),(825,835,330),(1110,724,120)]):
        line([(x,y),(x+l*.5,y-2),(x+l,y-4)],5 if improved else 7,.40,.35,k+730,blue)
    label(im,(135,110),"山河绘",70,INK,kai=True)
    label(im,(142,194),"笔意研究 · 离线示意",22,MUTED,kai=True)
    return im,geometry


def scene_plate():
    before,before_geometry=landscape(False)
    after,after_geometry=landscape(True)
    assert before_geometry==after_geometry, "Comparison centerlines must be identical"
    before.save(OUT/"03a_uniform_landscape.png")
    after.save(OUT/"03b_expressive_landscape.png")
    im=sheet("把笔触变化放回山水画中", "沿用左松、群山、右亭、孤舟的构图关系；两侧采用相同几何骨架，并非对原照片的修复。", 930)
    label(im,(76,204),"A  等宽线与规则排线示意",26)
    label(im,(930,204),"B  提按、分层淡染与枯笔示意",26,GREEN)
    im.paste(before.resize((816,490),Image.Resampling.LANCZOS),(64,258))
    im.paste(after.resize((816,490),Image.Resampling.LANCZOS),(920,258))
    label(im,(76,774),"对照：线色一致，色块的规则排线清晰可见。",21,MUTED)
    label(im,(932,774),"目标：线条有提按，近浓远淡，山体保留透气的留白。",21,GREEN)
    label(im,(76,826),"整体示意同时改变了线质、颜色铺法和层次；不能用于单独评估某一项算法的提升。",21,MUTED)
    im.save(OUT/"03_landscape_comparison.png")


def main():
    comparison();gallery();scene_plate()
    data=dict(format="offline-brush-visual-study-v1",calibrated=False,robot_executable=False,
              note="Visual envelopes only. Not physical pressure, force, concentration or an E6 prediction.",strokes=RECORDS)
    (OUT/"visual_stroke_samples.json").write_text(json.dumps(data,ensure_ascii=False),encoding="utf-8")
    (OUT/"README.md").write_text("# 离线毛笔预览\n\n这些图是程序生成的目标视觉研究，不是机械臂实画或经过标定的物理预测。\n\n01 同一路径：提按、顿挫、浓淡和飞白对比。\n02 六种线质：浓湿、淡湿、浓干、淡干、提按和折转。\n03 同构图整体示意；另附左右两幅高清单图。\n\n参数全部是归一化视觉量。下压毫米值、实际速度、墨液配方、刮笔动作和姿态补偿，需之后实机标定。\n\n运行：在项目目录执行 `python scripts/offline_brush_preview.py`。脚本只导入图像工具，不导入机械臂驱动。现有工作台、现场配置及执行轨迹均未修改。\n",encoding="utf-8")
    print(json.dumps(dict(output=str(OUT),images=sorted(p.name for p in OUT.glob("*.png")),samples=len(RECORDS)),ensure_ascii=False))


if __name__ == "__main__":
    main()
