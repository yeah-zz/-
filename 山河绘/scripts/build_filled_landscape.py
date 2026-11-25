"""Offline dense blue/green area coverage from original art, preserving reviewed ink geometry."""
from pathlib import Path
from datetime import datetime
import sys,json,shutil,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np
import vision,imgio,watercolor,job,ink_control,shuimo
from geometry import CanvasMap,check_waypoints
from run_mountain_v4_robot import safe_transfers
from run_color_experiment import module_lines
from color_profile import ColorProfile
from config import ArmConfig

# 接桥条件: 允许连接的最长跳距(mm), 以及跳线上必须落在区域内的最低比例。
# 原行为 = 2.0mm 且 100% 在区域内。对凹形/多孔区域过严 -> 一片色块被切成很多碎笔。
BRIDGE_MAX_MM=2.0
BRIDGE_MIN_INSIDE=1.0
# 单笔长度上限(mm)。这是笔数的直接决定因素,但不能超过一次蘸色的墨量预算
# (上色的 needs_color_reload 用 50mm;写超过 50 就会画到干笔)。
MAX_STROKE_MM=40.0


PRESS_RANGE = None    # (淡处下压, 浓处下压) mm; None=整笔用 profile 的统一值
SHRINK_PX = 0         # 上色前把蒙版内缩的像素数: 保证颜色不超出山体轮廓
DAB_MM = 0.0          # >0 时用"点染"填充: 抖动网格 + 随机方向短笔


def dab_fill(mask, scale, color, pitch_px, width_mm, target_opacity, seed=7):
    """点染填充: 在色块内按 pitch 网格取点(加抖动), 每点画一条随机方向的短笔。"""
    rng = np.random.RandomState(seed)
    pitch = max(2, int(pitch_px))
    dab_len_mm = DAB_MM
    dl = max(2.0, dab_len_mm/scale)
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    Hh, Ww = mask.shape
    strokes = []
    coverage = np.zeros_like(mask)
    jit = max(1, pitch//3)
    for k in range(1, nlab):
        if stats[k, cv2.CC_STAT_AREA]*scale*scale < 1:
            continue
        region = (labels == k)
        x, y, w, h = stats[k, :4]
        for yy in range(y, y+h, pitch):
            for xx in range(x, x+w, pitch):
                cx = int(np.clip(xx+rng.randint(-jit, jit+1), 0, Ww-1))
                cy = int(np.clip(yy+rng.randint(-jit, jit+1), 0, Hh-1))
                if not region[cy, cx]:
                    continue
                ang = rng.uniform(0.0, np.pi)
                dx, dy = dl*np.cos(ang), dl*np.sin(ang)
                p1 = [cx-dx/2, cy-dy/2]
                p2 = [cx+dx/2, cy+dy/2]
                strokes.append(dict(
                    id="%s-%d-%d-%d" % (color, k, xx, yy), color=color,
                    region_id=str(k), points=[p1, p2], recipe="dab-fill", layer=0,
                    target_opacity=[round(target_opacity, 3)]*2,
                    target_width_mm=[round(width_mm, 3)]*2,
                    brush_width_mm_estimate=round(width_mm, 3),
                    length_mm=round(dab_len_mm, 3), path_style="dab",
                    pressure_mm=None, pigment_recipe_calibrated=False))
                cv2.line(coverage, [tuple(map(int, p1)), tuple(map(int, p2))],
                         k, max(1, round(width_mm/scale)))
    return strokes, coverage


def fill_paths(mask, scale, color, pitch_px, width_mm, target_opacity):
    """Boustrophedon segments joined only when the connector stays in-region.

    pitch_px / width_mm 由 main() 从下压推出(见 color_width_and_pitch),不再各自写死
    0.65 / 1.0 —— 那三个数互相矛盾:行距只有模型笔宽的 53%,必然铺成实心块。
    """
    pitch=max(1,int(pitch_px))
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    strokes=[];coverage=np.zeros_like(mask)
    def save(points,region):
        if len(points)<2:return
        pts=np.asarray(points,float)
        length=float(np.linalg.norm(np.diff(pts,axis=0),axis=1).sum()*scale)
        if length<.6:return
        n=len(pts)
        strokes.append(dict(id=f'{color}-fill-{len(strokes):05d}',color=color,region_id=f'{color}-{region}',
            points=pts.tolist(),recipe='area-fill',layer=0,target_opacity=[target_opacity]*n,
            target_width_mm=[round(width_mm,3)]*n,brush_width_mm_estimate=round(width_mm,3),length_mm=length,
            path_style='in-region-serpentine',pressure_mm=None,pigment_recipe_calibrated=False))
        cv2.polylines(coverage,[np.rint(pts).astype(np.int32)],False,1,max(1,round(width_mm/scale)))
    for k in range(1,count):
        if stats[k,cv2.CC_STAT_AREA]*scale*scale<1:continue
        region=(labels==k)
        x,y,w,h=stats[k,:4];current=[];length=0.
        for rowi,yy in enumerate(range(y,y+h,pitch)):
            changes=np.diff(np.r_[False,region[yy],False].astype(int))
            runs=list(zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)-1))
            if rowi%2:runs=runs[::-1]
            for a,b in runs:
                if b<=a:continue
                # Limit each loaded stroke to at most 40 mm including row turns.
                chunks=max(1,int(np.ceil((b-a)*scale/max(4.0,MAX_STROKE_MM*0.8))))
                edges=np.linspace(a,b,chunks+1)
                pairs=list(zip(edges[:-1],edges[1:]))
                if rowi%2:pairs=[(v,u) for u,v in pairs[::-1]]
                for start,end in pairs:
                    q=np.array([start,yy]);z=np.array([end,yy]);seg=abs(end-start)*scale
                    bridge=0.;valid=False
                    if current:
                        d=np.linalg.norm(np.array(current[-1])-q);bridge=d*scale
                        samples=np.rint(np.linspace(current[-1],q,max(2,int(np.ceil(d))+1))).astype(int)
                        hit=float(np.mean(region[samples[:,1],samples[:,0]])) if len(samples) else 0.0
                        valid=bridge<=BRIDGE_MAX_MM and hit>=BRIDGE_MIN_INSIDE
                    if current and (not valid or length+bridge+seg>MAX_STROKE_MM):
                        save(current,k);current=[];length=0.
                    if not current:current=[q.tolist()]
                    else:current.append(q.tolist());length+=bridge
                    current.append(z.tolist());length+=seg
        save(current,k)
    return strokes,float(np.sum((coverage>0)&(mask>0))/max(1,np.sum(mask>0)))


def tone_thin(strokes, strength, pale_keep, seed=12345):
    """按参考图局部的浓淡抽稀笔触: 参考图越淡的地方保留越少。

    strength = 255 - 灰度, 越大代表参考图那里越浓。
    pale_keep=1.0 表示不抽稀(保持原行为)。
    这样远山(参考图本来就淡)会自动变疏 -> 颜色更淡, 而且总笔数下降。
    """
    if pale_keep >= 0.999 or not strokes:
        return strokes
    h, w = strength.shape
    vals = []
    for s in strokes:
        pts = np.asarray(s["points"], float).astype(int)
        ys = np.clip(pts[:, 1], 0, h - 1)
        xs = np.clip(pts[:, 0], 0, w - 1)
        vals.append(float(strength[ys, xs].mean()))
    vals = np.asarray(vals)
    if len(vals) < 2 or (vals.max() - vals.min()) < 1e-6:
        return strokes
    t = (vals - vals.min()) / (vals.max() - vals.min())      # 0=最淡, 1=最浓
    keep = pale_keep + (1.0 - pale_keep) * t                 # 淡处保留 pale_keep
    rng = np.random.RandomState(seed)
    r = rng.rand(len(strokes))
    return [s for s, k, rr in zip(strokes, keep, r) if rr < k]


def color_width_and_pitch(press_mm, scale, pitch_factor, pitch_mm_override=None):
    """由下压推出"模型笔宽"和行距,保证两者一致。

    机器人不能命令笔宽,只能命令下压量。纸上线宽由 press_to_width_mm
    (= ShuimoConfig.BRUSHES 实测插值表)换算。行距必须 >= 该宽度才有留白。
    """
    width_mm=float(shuimo.press_to_width_mm(press_mm))
    pitch_mm=float(pitch_mm_override) if pitch_mm_override else float(pitch_factor)*width_mm
    pitch_px=max(1,int(round(pitch_mm/scale)))
    return width_mm,pitch_mm,pitch_px


def fill_paths_rot(mask, scale, color, pitch_px, width_mm, target_opacity, angle_deg):
    """按指定角度铺色: 把 mask 旋转后再走同一套扫描行算法, 最后把点旋回原坐标系。

    angle_deg=0 就是原来的横向扫描(左右刷); 90 相当于纵向扫描(从上往下)。
    旋转只为改变"扫描行相对于形状的方向" —— 形状不变, 落点仍会被
    module_lines 的 20mm 边距检查兜住。
    """
    if abs(float(angle_deg)) < 1e-6:
        return fill_paths(mask, scale, color, pitch_px, width_mm, target_opacity)
    h, w = mask.shape
    M = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), float(angle_deg), 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    M[0, 2] += (nw - 1) / 2.0 - (w - 1) / 2.0
    M[1, 2] += (nh - 1) / 2.0 - (h - 1) / 2.0
    rot = cv2.warpAffine(mask, M, (nw, nh), flags=cv2.INTER_NEAREST)
    fn = dab_fill if DAB_MM > 0 else fill_paths
    strokes, coverage = fn(rot, scale, color, pitch_px, width_mm, target_opacity)
    Minv = cv2.invertAffineTransform(M)
    for s in strokes:
        pts = np.asarray(s["points"], float)
        homo = np.hstack([pts, np.ones((len(pts), 1))]) @ Minv.T
        s["points"] = homo[:, :2].tolist()
    return strokes, coverage


def main(argv=None):
    ap=argparse.ArgumentParser(description='密集铺色: 由下压推出笔宽与行距, 保证内部自洽')
    ap.add_argument('--src',default=None,help='来源画稿目录(含 _计划.json / _黑线任务.json / 参考原图.png)')
    ap.add_argument('--out-name',default='层峦松水_蓝绿铺色',help='输出目录前缀')
    ap.add_argument('--press-mm',type=float,default=None,help='色块逐点下压;默认沿用计划 motion_candidate.press_mm')
    ap.add_argument('--pitch-factor',type=float,default=1.3,
                    help='行距 = 该系数 x 模型笔宽。>1 才有留白;=1 恰好相接;<1 必然实心')
    ap.add_argument('--pitch-mm',type=float,default=None,help='直接指定行距,覆盖 --pitch-factor')
    ap.add_argument('--reload-strokes',type=int,default=8,help='每蘸色最多画几笔(1~8)')
    ap.add_argument('--colour-press-range',default=None,
                    help='彩色色阶: "淡处,浓处" 两个下压mm(如 0.10,0.30)。'
                         '按参考图局部浓淡给每个点单独下压, 远山淡/近石浓。留空=全图统一下压')
    ap.add_argument('--shrink-mm',type=float,default=0.0,
                    help='上色前把色区蒙版内缩多少 mm, 保证颜色不超出轮廓(建议 1.0~1.5)')
    ap.add_argument('--dab-mm',type=float,default=0.0,
                    help='点染模式的单笔长度(mm), >0 时启用: 抖动网格+随机方向短笔, 不排平行线')
    ap.add_argument('--width-mm',type=float,default=None,
                    help='覆盖笔宽(mm): 模拟铺色大笔/深压的宽笔痕; 影响元数据/预览,不影响行距(行距用 --pitch-mm)')
    ap.add_argument('--tone-pale-keep',type=float,default=1.0,
                    help='参考图最淡处保留的笔触比例(0~1)。1.0=不抽稀; 0.45=远山明显变淡且笔数下降')
    ap.add_argument('--travel-velocity',type=int,default=40,
                    help='纸面上方移动速度(%%): 只影响空走与杯位往返, 不影响落墨; 合法 1~60')
    ap.add_argument('--touchdown-velocity',type=int,default=None,help='落笔速度(%%, 留空用默认)')
    ap.add_argument('--lift-velocity',type=int,default=None,help='抬笔速度(%%, 留空用默认)')
    ap.add_argument('--scan-angle',default='0',
                    help='铺色扫描行角度(度): 单一数值=所有颜色; 或按颜色 blue:83,green:0。0=左右刷, 90=从上往下')
    ap.add_argument('--bridge-mm',type=float,default=2.0,help='接桥允许的最长跳距(mm), 越大笔数越少')
    ap.add_argument('--bridge-inside',type=float,default=1.0,
                    help='接桥跳线上必须落在区域内的最低比例(0~1), 越小笔数越少但可能略微出界')
    ap.add_argument('--max-stroke-mm',type=float,default=40.0,
                    help='单笔长度上限(mm), 默认40。调大直接减少笔数; 上限 300 = 一次蘸色的墨量预算')
    a=ap.parse_args(argv)
    if not 1<=a.reload_strokes<=8:raise SystemExit('--reload-strokes 必须在 1~8')
    if not 0.0<=a.bridge_inside<=1.0:raise SystemExit('--bridge-inside 必须在 0~1')
    global BRIDGE_MAX_MM,BRIDGE_MIN_INSIDE,MAX_STROKE_MM
    BRIDGE_MAX_MM=float(a.bridge_mm);BRIDGE_MIN_INSIDE=float(a.bridge_inside)
    MAX_STROKE_MM=float(a.max_stroke_mm)
    if not 5.0<=MAX_STROKE_MM<=300:raise SystemExit('--max-stroke-mm 必须在 5~300(上限=一次蘸色的墨量上限)')
    SCAN={}
    for _part in str(a.scan_angle).split(','):
        _part=_part.strip()
        if not _part:continue
        if ':' in _part:
            _k,_v=_part.split(':',1);SCAN[_k.strip()]=float(_v)
        else:
            SCAN={'blue':float(_part),'green':float(_part)}
    def scan_for(color):return float(SCAN.get(color,0.0))
    root=Path(__file__).resolve().parents[1]
    src=Path(a.src) if a.src else root/'输出'/'层峦松水_粗细分明_20260929_132355'
    if not src.is_absolute():src=root/src
    if not src.is_dir():raise SystemExit('来源目录不存在: %s'%src)
    out=root/'输出'/(a.out_name+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir(exist_ok=False)
    plans=sorted(src.glob('*_计划.json')); inks=sorted(src.glob('*_黑线任务.json'))
    if len(plans)!=1 or len(inks)!=1:
        raise SystemExit('来源目录必须恰好含一个 *_计划.json 和一个 *_黑线任务.json: %s'%src)
    p=json.loads(plans[0].read_text(encoding='utf8'))
    _,meta=job.load_job(str(inks[0]))
    ref_src=Path(p['reference_image'])
    if not ref_src.is_file():ref_src=src/'参考原图.png'
    shutil.copy2(ref_src,out/'参考原图.png')
    p['reference_image']=str(out/'参考原图.png');meta['image']=p['reference_image']
    work,info=vision.prepare_image(vision.load_image(p['reference_image']),work_size=1800)
    scale=p['mm_per_px'];b,g,r=cv2.split(work.astype(float));gray=cv2.cvtColor(work,cv2.COLOR_BGR2GRAY)
    strength=255.0-gray      # 参考图局部浓度(远山淡 -> 数值小)
    green=(g-r>3)&(g-b>2)&(gray<210)&(gray>55)
    blue=(b-r>3)&(gray<247)&(gray>100)&~green
    dark=(gray<85)
    mapper=CanvasMap(p['image_size'],p['image_corners'])
    yy,xx=np.indices(gray.shape)
    # Existing exact executor margin is applied below; conservatively pad masks here.
    margin=int(np.ceil(21/scale))
    m=p['modules']['color']
    profile=dict(m['motion_candidate'])
    profile.update(press_mm=(a.press_mm if a.press_mm is not None else float(profile['press_mm'])),
                   draw_velocity_percent=30,travel_velocity_percent=int(a.travel_velocity))
    if a.touchdown_velocity is not None: profile['touchdown_velocity_percent']=int(a.touchdown_velocity)
    if a.lift_velocity is not None: profile['lift_velocity_percent']=int(a.lift_velocity)
    ColorProfile(**profile).to_dict()
    global DAB_MM, SHRINK_PX
    global PRESS_RANGE
    if a.colour_press_range:
        _lo,_hi=[float(v) for v in a.colour_press_range.split(',')]
        assert 0<=_lo<=_hi, '色阶下压须为 0<=淡<=浓'
        PRESS_RANGE=(_lo,_hi)
        print('  色阶: 每点下压 %.3f~%.3f mm (按参考图局部浓淡)'%(_lo,_hi))
    DAB_MM=float(a.dab_mm)
    SHRINK_PX=int(round(float(a.shrink_mm)/scale))
    if SHRINK_PX>0: print('  色区蒙版内缩 %.2f mm (%d px): 颜色不超出轮廓'%(a.shrink_mm,SHRINK_PX))
    if PRESS_RANGE is not None: profile['press_mm']=float(PRESS_RANGE[1])   # 上限=最浓那档
    WIDTH_MM,PITCH_MM,PITCH_PX=color_width_and_pitch(profile['press_mm'],scale,a.pitch_factor,a.pitch_mm)
    if a.width_mm is not None: WIDTH_MM=float(a.width_mm)   # 铺色大笔假设
    p['color_geometry']=dict(press_mm=float(profile['press_mm']),model_width_mm=round(WIDTH_MM,4),
        pitch_mm=round(PITCH_MM,4),pitch_px=int(PITCH_PX),pitch_over_width=round(PITCH_MM/WIDTH_MM,3),
        scan_angle_deg={c:scan_for(c) for c in ('blue','green')},
        bridge_max_mm=BRIDGE_MAX_MM,bridge_min_inside=BRIDGE_MIN_INSIDE,max_stroke_mm=MAX_STROKE_MM,
        source='shuimo.press_to_width_mm(ShuimoConfig.BRUSHES)',
        note='行距由模型笔宽推出;笔宽本身来自 BRUSHES 标定表,未在当前纸面/湿笔状态下重测。')
    print('下压 %.3f mm -> 模型笔宽 %.3f mm -> 行距 %.3f mm (%d px) -> 行距/笔宽 %.2f'%(
        profile['press_mm'],WIDTH_MM,PITCH_MM,PITCH_PX,PITCH_MM/WIDTH_MM))
    allstrokes=[];report={}
    for color,raw in [('blue',blue),('green',green)]:
        rad=max(1,round((.65 if color=='blue' else .4)/scale))
        mask=cv2.morphologyEx(raw.astype(np.uint8),cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*rad+1,)*2))
        mask[dark]=0
        mask=cv2.erode(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        if SHRINK_PX>0:
            k=2*SHRINK_PX+1
            mask=cv2.erode(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)))

        mask[:margin]=0;mask[-margin:]=0;mask[:,:margin]=0;mask[:,-margin:]=0
        strokes,coverage=fill_paths_rot(mask,scale,color,PITCH_PX,WIDTH_MM,
                                        (.50 if color=='blue' else .70),scan_for(color))
        raw_n=len(strokes)
        strokes=tone_thin(strokes,strength,a.tone_pale_keep)
        print('  %s: 铺色 %d 笔 -> 按浓淡抽稀后 %d 笔 (淡处保留 %.0f%%)'%(
            color,raw_n,len(strokes),100*float(a.tone_pale_keep)))
        if PRESS_RANGE is not None:
            lo,hi=PRESS_RANGE
            v=strength[mask>0]
            if v.size:
                vlo,vhi=float(np.percentile(v,5)),float(np.percentile(v,95))
                if vhi-vlo<1e-6: vhi=vlo+1.0
                for st in strokes:
                    pts=np.asarray(st['points'],float)
                    yy=np.clip(pts[:,1].astype(int),0,strength.shape[0]-1)
                    xx=np.clip(pts[:,0].astype(int),0,strength.shape[1]-1)
                    t=np.clip((strength[yy,xx]-vlo)/(vhi-vlo),0.0,1.0)
                    st['press']=[round(float(lo+(hi-lo)*float(q)),4) for q in t]
                ps=np.concatenate([np.asarray(st['press']) for st in strokes])
                print('  %s 色阶: 下压 p10 %.3f / p50 %.3f / p90 %.3f mm'%(
                    color,np.percentile(ps,10),np.percentile(ps,50),np.percentile(ps,90)))
        allstrokes+=strokes
        report[color]=dict(strokes=len(strokes),estimated_mask_coverage=coverage)
    m['motion_candidate']=profile   # 必须写回! executor 从 module['motion_candidate'] 读参数
    m['strokes']=allstrokes;m['regions']=[]
    m['reload_strokes_per_load']=int(a.reload_strokes)
    allowed=set()
    for color in ('blue','green'):
        lines,skipped=module_lines(m,color,profile);allowed.update(s['id'] for s in lines)
        report[color]['margin_skipped']=len(skipped)
    m['strokes']=[s for s in allstrokes if s['id'] in allowed];m['stroke_count']=len(m['strokes'])
    assert all(v['margin_skipped']==0 for v in report.values()),report
    snap=meta['ink_execution'];base=snap['base_strokes']
    for s in base:s['speed']=[30.]*len(s['points'])
    steps,_=ink_control.build_plan(base,shuimo.ink_palette_index(),scale,ink_control.InkPolicy(**snap['policy']))
    p['modules']['outline']['strokes']=[s['stroke'] for s in steps if s['kind']=='stroke']
    wp,_=job.steps_to_waypoints(steps,mapper);wp=safe_transfers(wp);assert check_waypoints(wp)[0]
    p['note']=('色块蛇形铺色: 下压 %.3f mm -> 模型笔宽 %.3f mm -> 行距 %.3f mm (行距/笔宽 %.2f)。'
        '黑线几何不变, speed 30, travel 40。笔宽来自 BRUSHES 标定表, 未在当前纸面/湿笔状态重测;'
        '未做物理覆盖率或颜料浓度标定。')%(profile['press_mm'],WIDTH_MM,PITCH_MM,PITCH_MM/WIDTH_MM)
    p['fill_report']=report
    prefix=out/a.out_name
    Path(str(prefix)+'_计划.json').write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf8')
    # 重新生成时把纸面基准刷新成当前值,否则 load_reviewed 会以
    # 'Calibration/settings changed; regenerate and review' 拒绝执行。
    meta['paper_z']=float(ArmConfig.PAPER_Z)
    job.save_job(str(prefix)+'_黑线任务.json',wp,meta)
    canvas,_=watercolor.render_image(work,m,include_ink=False)
    for s in p['modules']['outline']['strokes']:
        pts=np.rint(s['points']).astype(int)
        for i,(a,z) in enumerate(zip(pts,pts[1:])):
            width=(shuimo.press_to_width_mm(s['press'][i])+shuimo.press_to_width_mm(s['press'][i+1]))/2
            cv2.line(canvas,tuple(a),tuple(z),(35,35,35),max(1,round(width/scale)),cv2.LINE_AA)
    imgio.imwrite(str(prefix)+'_预览.png',canvas)
    print('OUTPUT',out);print('REPORT',report)


if __name__=='__main__':main()
