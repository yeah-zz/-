"""Compose four editable robot-native landscapes from the demonstrated line vocabulary.

No raster retracing: calibrated mountain/pine strokes are reused as geometric
motifs; bridges, boats, cottages and bamboo are explicit brush trajectories.
"""
from pathlib import Path
import sys,json,copy,math
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2,numpy as np
import job,shuimo,ink_control,imgio,watercolor
from geometry import CanvasMap,check_waypoints
from config import ArmConfig,CanvasConfig,PALETTE
from color_profile import ColorProfile
from run_mountain_v4_robot import safe_transfers
from run_color_experiment import module_lines
from run_json_painting import make_bundle,load_artwork,manifest_for
from painting_session import save

OUT=ROOT/'输出'/'蓝绿山水四幅_按展示图_20261001'
DONOR=ROOT/'输出'/'层峦松水_双蓝长笔版_20260930_172520'
INK=json.loads(next(DONOR.glob('*_黑线任务.json')).read_text(encoding='utf-8'))
PLAN=json.loads(next(DONOR.glob('*_计划.json')).read_text(encoding='utf-8'))
SIZE=tuple(PLAN['image_size'])
MAPPER=CanvasMap(SIZE)
SCALE=MAPPER.mm_per_px
PROFILE=ColorProfile(**PLAN['modules']['color']['motion_candidate']).to_dict()


def clip(a,b,box):
    x0,y0,x1,y1=box;d=b-a;lo,hi=0.,1.
    for p,q in [(-d[0],a[0]-x0),(d[0],x1-a[0]),(-d[1],a[1]-y0),(d[1],y1-a[1])]:
        if abs(p)<1e-12:
            if q<0:return None
        elif p<0:lo=max(lo,q/p)
        else:hi=min(hi,q/p)
        if lo>hi:return None
    return lo,hi


def clipped_parts(stroke, box, target, color=False, flip=False):
    xy=np.asarray(stroke['points'],float)
    fields=['press','speed'] if not color else ['press','speed','target_opacity','target_width_mm']
    fields=[k for k in fields if k in stroke]
    x0,y0,x1,y1=box;tx0,ty0,tx1,ty1=target
    factors=np.array([(tx1-tx0)/(x1-x0),(ty1-ty0)/(y1-y0)])
    active=[];attrs={k:[] for k in fields};parts=[]
    def finish():
        nonlocal active,attrs
        if len(active)>1:
            pts=(np.asarray(active)-[x0,y0])*factors+[tx0,ty0]
            if flip:pts[:,0]=tx0+tx1-pts[:,0]
            length=float(np.linalg.norm(np.diff(pts,axis=0),axis=1).sum()*SCALE)
            if length>=.75:
                s=copy.deepcopy(stroke);s['points']=pts.tolist();s.update(attrs)
                s['length_mm']=length
                if color:
                    s['color']='blue' if s['color']=='blue2' else s['color']
                    s['id']='pending';s['region_id']='composed-motif'
                    s['target_width_mm']=[shuimo.press_to_width_mm(p) for p in s.get('press',[PROFILE['press_mm']]*len(pts))]
                    s['brush_width_mm_estimate']=max(s['target_width_mm'])
                else:
                    # Keep demonstrated pressure variation and remove cut-end jumps.
                    # Smaller composed motifs use lighter pressure, rather than
                    # rendering artificially thin lines with unchanged commands.
                    s['press']=[float(p)*.45 for p in s['press']]
                    s['press'][0]=min(s['press'][0],.10);s['press'][-1]=min(s['press'][-1],.10)
                    ink_control.refresh_stats(s,SCALE)
                parts.append(s)
        active=[];attrs={k:[] for k in fields}
    for i,(a,b) in enumerate(zip(xy,xy[1:])):
        hit=clip(a,b,box)
        if hit is None:finish();continue
        lo,hi=hit;p=a+(b-a)*lo;q=a+(b-a)*hi
        if active and np.linalg.norm(np.array(active[-1])-p)>.01:finish()
        if not active:
            active.append(p.tolist())
            for k in fields:attrs[k].append(float(stroke[k][i]+(stroke[k][i+1]-stroke[k][i])*lo))
        active.append(q.tolist())
        for k in fields:attrs[k].append(float(stroke[k][i]+(stroke[k][i+1]-stroke[k][i])*hi))
        if hi<1-1e-9:finish()
    finish();return parts


class Drawing:
    def __init__(self,name):self.name=name;self.ink=[];self.colors=[]
    def motif(self,source,target,flip=False):
        for s in INK['meta']['ink_execution']['base_strokes']:
            self.ink.extend(clipped_parts(s,source,target,flip=flip))
        for s in PLAN['modules']['color']['strokes']:
            if s['color'] in ('blue','blue2','green'):
                self.colors.extend(clipped_parts(s,source,target,color=True,flip=flip))
    def line(self,controls,press=.18,color=None):
        pts=[]
        for a,b in zip(controls,controls[1:]):
            n=max(1,int(np.ceil(np.linalg.norm(np.array(b)-a)/3)))
            pts.extend(np.linspace(a,b,n,endpoint=False))
        pts.append(controls[-1]);pts=np.asarray(pts,float)
        t=np.linspace(0,1,len(pts));ramp=np.minimum(np.clip(t/.13,0,1),np.clip((1-t)/.13,0,1))
        pressure=(.06+(press-.06)*ramp).tolist()
        s=dict(points=pts.tolist(),press=pressure,speed=[30.]*len(pts),
               layer=1,tail_mm=0.,lift_mm=4.,pal_index=shuimo.ink_palette_index(),
               name='authored brush line',role='structure',design_origin='robot-native-composition')
        if color:
            widths=[shuimo.press_to_width_mm(p) for p in pressure]
            s.update(color=color,recipe='accent',region_id='authored',
                     target_opacity=[.62]*len(pts),target_width_mm=widths,
                     brush_width_mm_estimate=max(widths),length_mm=float(np.linalg.norm(np.diff(pts,axis=0),axis=1).sum()*SCALE),
                     path_style='authored-brush',pressure_mm=None,pigment_recipe_calibrated=False)
            self.colors.append(s)
        else:self.ink.append(ink_control.refresh_stats(s,SCALE))
    def curve(self,ctrl,press=.16,color=None):
        # Catmull-Rom interpolation through intentionally sparse design controls.
        p=np.asarray([ctrl[0],*ctrl,ctrl[-1]],float);points=[]
        for i in range(1,len(p)-2):
            a,b,c,d=p[i-1:i+3]
            for t in np.linspace(0,1,12,endpoint=False):
                points.append((.5*((2*b)+(-a+c)*t+(2*a-5*b+4*c-d)*t*t+(-a+3*b-3*c+d)*t*t*t)).tolist())
        points.append(ctrl[-1]);self.line(points,press,color)
    def bridge(self,x,y,w=220):
        t=np.linspace(0,np.pi,30)
        for dx,dy,ww,hh in [(0,0,w,62),(17,8,w-34,43)]:
            self.line([(x+dx+(1-np.cos(q))*ww/2,y+dy-np.sin(q)*hh) for q in t],.18)
        self.line([(x-12,y+5),(x+w+12,y+5)],.12)
        for q in np.linspace(.25,np.pi-.25,8):
            xx=x+(1-np.cos(q))*w/2;yy=y-np.sin(q)*62
            self.line([(xx,yy-16),(xx,yy)],.12)
        self.curve([(x-8,y-10),(x+w*.25,y-60),(x+w*.55,y-78),(x+w*.8,y-54),(x+w+8,y-10)],.13)
    def boat(self,x,y,w=175):
        self.curve([(x,y),(x+w*.2,y+15),(x+w*.7,y+17),(x+w,y-5)],.22)
        self.curve([(x+6,y+6),(x+w*.2,y+32),(x+w*.75,y+31),(x+w,y-5)],.18)
        self.curve([(x+w*.30,y+10),(x+w*.37,y-29),(x+w*.57,y-33),(x+w*.7,y+10)],.18)
        self.line([(x+w*.37,y-29),(x+w*.6,y-29)],.12)
        self.line([(x+w*.86,y+1),(x+w*.88,y-52)],.15)
        for dx in [0,24,65]:self.curve([(x-20+dx,y+45),(x+40+dx,y+49),(x+90+dx,y+45)],.10,'blue')
    def cottage(self,x,y,w=165):
        self.line([(x,y),(x+w*.45,y-55),(x+w,y),(x+w*.9,y-2),(x+w*.45,y-37),(x+w*.08,y+7),(x,y)],.19)
        self.line([(x+18,y+2),(x+18,y+75),(x+w-15,y+75),(x+w-15,y-1)],.15)
        self.line([(x+w*.48,y+75),(x+w*.48,y+27),(x+w*.66,y+27),(x+w*.66,y+75)],.14)
        self.line([(x+35,y+25),(x+57,y+25),(x+57,y+43),(x+35,y+43),(x+35,y+25)],.12)
        for d in [0,14,28,42,56]:self.line([(x+25+d,y-4-d*.4),(x+44+d,y+2-d*.4)],.10)
    def bamboo(self,x,y,h=260):
        for dx,lean,factor in [(-55,-40,.8),(-20,5,1),(20,42,.9),(55,72,.7)]:
            height=h*factor
            self.curve([(x+dx,y),(x+dx+lean*.25,y-height*.5),(x+dx+lean,y-height)],.15)
            for level in [.35,.57,.78,.95]:
                cx=x+dx+lean*level;cy=y-height*level
                self.line([(cx-4,cy),(cx+5,cy+1)],.14)
                for sign in [-1,1]:
                    self.curve([(cx,cy),(cx+sign*26,cy-18),(cx+sign*50,cy-17)],.11)
                    for j in [0,1,2]:
                        sx=cx+sign*(15+j*13);sy=cy-10-j*3
                        self.curve([(sx,sy),(sx+sign*13,sy-20-j*3),(sx+sign*19,sy-34-j*2)],.26,'green')

    def export(self):
        folder=OUT/self.name;folder.mkdir(parents=True,exist_ok=True)
        policy=ink_control.InkPolicy()
        base=ink_control.order_for_ink(ink_control.split_long_strokes(self.ink,SCALE,policy),SCALE,policy)
        for i,s in enumerate(base):s['id']=self.name+'-ink-%04d'%i
        steps,stats=ink_control.build_plan(base,shuimo.ink_palette_index(),SCALE,policy)
        conditioned=[s['stroke'] for s in steps if s['kind']=='stroke']
        wp,_=job.steps_to_waypoints(steps,MAPPER);wp=safe_transfers(wp)
        ok,errors=check_waypoints(wp)
        if not ok:raise ValueError(errors)
        # A scaled motif may exceed the current 100 mm loading budget. Split its
        # actual robot path at shared endpoints; never lengthen a single load.
        bounded=[]
        for stroke in self.colors:
            pixels=np.asarray(stroke['points'],float)
            robot=MAPPER.to_robot(pixels)
            cumulative=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(robot,axis=0),axis=1))]
            if cumulative[-1]<=90:
                bounded.append(stroke);continue
            for lo in np.arange(0,cumulative[-1],90.):
                hi=min(lo+90.,cumulative[-1])
                samples=np.r_[lo,cumulative[(cumulative>lo)&(cumulative<hi)],hi]
                part=copy.deepcopy(stroke)
                part['points']=np.column_stack([np.interp(samples,cumulative,pixels[:,j]) for j in range(2)]).tolist()
                for key in ['press','speed','target_opacity','target_width_mm']:
                    if key in stroke:part[key]=np.interp(samples,cumulative,stroke[key]).tolist()
                part['length_mm']=float(hi-lo);bounded.append(part)
        self.colors=bounded
        for i,s in enumerate(self.colors):s['id']=self.name+'-color-%04d'%i
        geometry=dict(image_size=list(SIZE),mm_per_px=SCALE,image_corners=MAPPER.image_corners)
        module=dict(format='sparse-color-design-v1',strokes=self.colors,regions=[],
                    palette_bgr=watercolor.PALETTE,motion_candidate=PROFILE,stroke_count=len(self.colors),
                    reload_strokes_per_load=8,**geometry)
        for color in ['blue','green']:
            lines,skipped=module_lines(module,color,PROFILE)
            if skipped or not lines:raise ValueError((self.name,color,len(lines),skipped))
            if any(np.linalg.norm(np.diff(np.asarray(line['xy']),axis=0),axis=1).sum()>100 for line in lines):
                raise ValueError('Color stroke exceeds the existing loading budget')
        reference=str((folder/(self.name+'_实际路径预览.png')).resolve())
        plan=dict(format='painting-workflow-design-v1',ready_for_execution=False,subject='reference',
                  title=self.name,reference_image=reference,**geometry,stages=['outline','color'],
                  modules=dict(outline=dict(format='outline-design-v1',strokes=conditioned,**geometry),color=module),
                  sequence=[dict(kind='run_module',module='outline'),dict(kind='manual_wash_wipe_and_load',next_stage='color'),dict(kind='run_module',module='color')],
                  note='Robot-native composition based on the user demonstrated contour style. Blue and green only. Preview renders brush paths; no raster texture is pasted.')
        meta=dict(mode='shuimo-framework',profile='mountain',ink_policy='balanced',image=reference,
                  image_size=list(SIZE),canvas_corners=[list(p) for p in CanvasConfig.ROBOT_CORNERS],
                  paper_z=ArmConfig.PAPER_Z,strokes=len(base),
                  ink_execution=dict(base_strokes=base,policy=stats['ink_policy'],mm_per_px=SCALE,
                     calibration=dict(rpy=list(ArmConfig.RPY),user=ArmConfig.USER_INDEX,tool=ArmConfig.TOOL_INDEX,
                                      speed_factor=ArmConfig.SPEED_FACTOR,palette=PALETTE)))
        ink=dict(format='arm-painting job',version=1,meta=meta,waypoints=wp)
        save(folder/(self.name+'_计划.json'),plan);save(folder/(self.name+'_黑线任务.json'),ink)
        bundle=folder/(self.name+'_完整绘画.json');save(bundle,make_bundle(plan,ink))
        _,checked,strokes,checked_meta=load_artwork(bundle);manifest_for(strokes,checked,checked_meta)
        work=np.full((SIZE[1],SIZE[0],3),255,np.uint8)
        preview,center=watercolor.render_image(work,module,include_ink=False)
        inkview=work.copy()
        for s in conditioned:
            pts=np.rint(s['points']).astype(int)
            for i,(a,b) in enumerate(zip(pts,pts[1:])):
                width=max(1,round((shuimo.press_to_width_mm(s['press'][i])+shuimo.press_to_width_mm(s['press'][i+1]))/(2*SCALE)))
                for canvas in [preview,inkview]:cv2.line(canvas,tuple(a),tuple(b),(35,35,35),width,cv2.LINE_AA)
                cv2.line(center,tuple(a),tuple(b),(35,35,35),1,cv2.LINE_AA)
        for tag,canvas in [('实际路径预览',preview),('黑线预览',inkview),('中心路径',center)]:
            imgio.imwrite(str(folder/(self.name+'_'+tag+'.png')),canvas)
        counts=Counter(s['color'] for s in self.colors)
        result=dict(name=self.name,black=len(base),blue=counts['blue'],green=counts['green'],bundle=str(bundle),
                    validation='passed; no robot connection',canvas_mm=MAPPER.canvas_size_mm(),paper_z=ArmConfig.PAPER_Z,
                    source='Recomposed demonstrated landscape strokes + authored bridge/boat/cottage/bamboo')
        save(folder/'离线检查.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
        return result


def scenes():
    far=(20,40,1510,235);peak=(75,250,550,455);peak2=(535,240,1005,510)
    pine=(32,410,725,980);shore=(870,505,1500,685);rocks=(40,735,805,995)
    a=Drawing('01_云峰溪桥')
    a.motif(far,(145,125,1390,275))
    a.motif(peak,(160,310,650,510));a.motif(peak2,(795,285,1390,520))
    a.motif(pine,(145,520,725,925));a.motif(shore,(985,635,1390,790))
    a.curve([(865,490),(804,526),(855,565),(806,612),(920,650)],.13)
    a.bridge(740,721,225)
    a.curve([(780,752),(823,799),(772,838),(923,881),(1080,918)],.15)
    a.curve([(958,758),(933,802),(1000,833),(1145,864),(1240,921)],.16)
    b=Drawing('02_松崖飞瀑')
    b.motif(far,(150,145,1050,305))
    b.motif(peak2,(770,260,1120,690));b.motif(peak,(1180,325,1390,685),True)
    b.motif(pine,(145,500,705,915))
    b.motif(rocks,(960,780,1390,940),True)
    b.curve([(1103,450),(1131,443),(1163,450),(1188,436)],.15)
    b.curve([(1125,448),(1134,500),(1117,574),(1129,635),(1113,705),(1121,751)],.12)
    b.curve([(1162,450),(1151,515),(1160,590),(1148,672),(1151,737)],.10)
    for offset in [0,22,45]:b.curve([(1090-offset,775+offset/3),(1140,783+offset/4),(1210+offset,774+offset/3)],.11,'blue')
    c=Drawing('03_湖山孤舟')
    c.motif(far,(150,145,1390,310))
    c.motif(peak,(620,350,1010,535));c.motif(peak2,(1060,305,1390,510))
    c.motif(pine,(145,450,650,865));c.motif(shore,(995,570,1390,720))
    c.curve([(700,590),(865,607),(942,594),(1032,612)],.12)
    c.boat(773,793,178)
    c.curve([(405,880),(600,893),(709,875)],.10,'blue')
    d=Drawing('04_竹岸山居')
    d.motif(far,(145,120,1390,265))
    d.motif(peak,(170,300,685,545));d.motif(peak2,(790,270,1385,550))
    d.motif(rocks,(145,735,640,935));d.motif(shore,(900,655,1390,830))
    d.bamboo(330,775,250);d.cottage(950,590,170)
    d.curve([(802,514),(738,557),(811,609),(736,659),(779,722),(692,786),(780,887)],.14)
    d.curve([(895,650),(855,697),(904,765),(846,819),(990,924)],.13)
    return [a,b,c,d]


if __name__=='__main__':
    from unittest.mock import patch
    OUT.mkdir(parents=True,exist_ok=True)
    with patch('robot.E6Robot.__init__',side_effect=AssertionError('Offline composition only')):
        results=[drawing.export() for drawing in scenes()]
    save(OUT/'四幅清单.json',results)
