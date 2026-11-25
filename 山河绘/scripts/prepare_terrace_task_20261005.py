"""Prepare an independent offline task builder for the latest supplied reference."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / '输出/横题山河绘_云山亭阶_20261004'
OUT = ROOT / '输出/横题山河绘_青绿梯田云山_20261005'
SOURCE = Path('C:/Users/admin/AppData/Local/Temp/codex-clipboard-a704e71c-e197-4147-94c0-c311596be2d0.png')
OUT.mkdir(exist_ok=False)
shutil.copy2(SOURCE, OUT / '参考原图.png')
for name in ['task_defaults.json', 'speed_profile.py', 'join_paths.py', 'paint_this.py', 'color_this.py', 'verify_this.py', 'render_paths.py', 'audit_paths.py']:
    content = (TEMPLATE / name).read_text(encoding='utf8')
    content = content.replace('横题山河绘_云山亭阶', '横题山河绘_青绿梯田云山')
    if name == 'audit_paths.py':
        content = content.replace("1040<t['source_center'][0]<1235", "955<t['source_center'][0]<1140")
    (OUT / name).write_text(content, encoding='utf8')

code = (TEMPLATE / 'build_this.py').read_text(encoding='utf8')
code = code.replace('横题山河绘_云山亭阶', '横题山河绘_青绿梯田云山').replace('yun-shan-ting-jie', 'qing-lv-ti-tian')
code = code.replace('title=(sx>805)&(sx<1525)&(sy>95)&(sy<405)', 'title=(sx>775)&(sx<1410)&(sy>145)&(sy<385)')
start = code.index('far=polygon_mask([')
end = code.index('\ndef clean(', start)
code = code[:start] + '''# Semantic regions measured on the new 1491 x 1055 reference.
far=polygon_mask([
 [[185,280],[475,230],[850,420],[900,495],[740,568],[470,510],[330,438],[220,380]],
 [[780,545],[1010,469],[1290,420],[1490,420],[1490,704],[1090,735],[865,654]],
 [[1030,740],[1210,758],[1390,796],[1490,828],[1490,881],[1120,836]]])
# Colored distant peaks remain color, not synthetic black ridges.
wm=((sy>890)&(b-r>24)&(gray<235)&(gray>110)&(local>10)).astype(np.uint8)
water=wm.copy();mask[water>0]=0
pavilion=np.zeros((h,w),bool);stairs=pavilion.copy()
''' + code[end:]
start = code.index(' if np.mean(pavilion[')
end = code.index(' # Subpixel simplification', start)
code = code[:start] + ''' if np.mean(far[q[:,1],q[:,0]])>.60:role='far_outline'
 elif cy>890:role='water'
 elif cy<530 and cx<340:role='mountain_outline' if width>3.5 else 'texture'
 else:role='near_outline' if width>3.5 else 'texture'
''' + code[end:]
start = code.index('# Source foliage')
end = code.index('join_paths.MASK=mask', start)
code = code[:start] + 'leaf=np.zeros((h,w),np.uint8)\n' + code[end:]
start = code.index('# Sparse diagonal colour')
end = code.index("colors.sort(key=", start)
code = code[:start] + '''# Cyan/teal reference: blue on rock shoulders, green on terrace pools;
# source hue determines assignment, pale clouds and paths remain white.
b,g,r=cv2.split(work.astype(float))
chroma=work.max(2).astype(float)-work.min(2)
colored_source=(np.maximum(b,g)-r>12)&(gray<246)&(gray>65)&(chroma>16)&~title
terrace=(sx<845)&(sy>510)&(sy<840)
green=colored_source&((g>=b-8)|terrace)
blue=colored_source&~green
color=plan['modules']['color'];color.update(image_size=size,image_corners=corners,mm_per_px=scale,regions=[])
fill.MAX_STROKE_MM=45.;fill.BRIDGE_MAX_MM=.6;fill.BRIDGE_MIN_INSIDE=1.
dark=cv2.dilate((gray<100).astype(np.uint8),np.ones((5,5),np.uint8))>0
def maskcolor(raw):
 m=cv2.morphologyEx(raw.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
 m=cv2.erode(m,np.ones((3,3),np.uint8));m[dark|title]=0
 return clean(m,25)
bm,gm=maskcolor(blue),maskcolor(green)
imgio.imwrite(str(HERE/'蓝色色区.png'),255-bm*180)
imgio.imwrite(str(HERE/'青绿色区.png'),255-gm*180)
# Disjoint regions: mountain slopes, terrace pools, foreground rocks, water.
zones=[(0,185,0,510,78),(185,780,0,510,35),(780,1491,0,740,35),
       (0,845,510,840,0),(0,320,840,890,75),(320,1491,740,890,65),
       (0,1491,890,1055,0)]
# Zone 4/5 overlap zone 3 only in x,y outside its ranges; explicitly assign
# each source pixel once to prevent overlapping color strokes.
zone_map=np.full((h,w),-1,int)
for zi,(a,c,bb,d,ang) in enumerate(zones):
 inside=(sx>=a)&(sx<c)&(sy>=bb)&(sy<d)
 zone_map[inside&(zone_map<0)]=zi
def colored(pitch):
 out=[]
 for cname,maskc in [('blue',bm),('green',gm)]:
  for zi,(_,_,_,_,ang) in enumerate(zones):
   m=maskc.copy();m[zone_map!=zi]=0
   ss,_=fill.fill_paths_rot(m,scale,cname,max(2,round(pitch/scale)),shuimo.press_to_width_mm(.22),.5,ang)
   for st in ss:
    if st['length_mm']<1.3:continue
    st['region_id']=f'{cname}-source-zone-{zi}'
    st['role']='water_color' if zi==6 else 'terrace_color' if zi==3 else 'mountain_color'
    out.append(st)
 return out
ls=[]
candidates=[]
for pitch in [1.1,1.3,1.5,1.7]:
 cs=colored(pitch);candidates.append((abs(len(cs)-650),pitch,cs))
_,pitch,colors=min(candidates,key=lambda v:v[0])
for i,st in enumerate(colors):
 p=resample(st['points'],.4);q=np.rint(p).astype(int);n=len(p)
 strength=np.maximum(b[q[:,1],q[:,0]],g[q[:,1],q[:,0]])-r[q[:,1],q[:,0]]
 remote=(far[q[:,1],q[:,0]]>0)
 press=np.clip(.17+strength/160*.11,.17,.28)
 press[remote]=np.minimum(press[remote],.21)
 if st['role']=='water_color':press=np.minimum(press,.22)
 opacity=np.clip(.25+strength/150*.38,.25,.68)
 st.update(id=f'{st["color"]}-terrace-source-{i:05}',points=p.tolist(),press=np.round(press,4).tolist(),target_width_mm=[float(shuimo.press_to_width_mm(v)) for v in press],target_opacity=np.round(opacity,3).tolist(),brush_width_mm_estimate=float(shuimo.press_to_width_mm(max(press))),length_mm=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()*scale))
''' + code[end:]
code = code.replace("source='new reference masks only',leaf_pressure_mm=.19,pressure_range_mm=[.17,.29]", "source='latest cyan terrace reference masks only',pressure_range_mm=[.17,.28]")
start = code.index("plan['note']=")
end = code.index('\nvalidate_plan', start)
code = code[:start] + "plan['note']='按20261005最新青绿梯田原图独立提线及蓝绿配色；横题保留内外轮廓；梯田横向疏铺、山石顺坡、水纹横向；云雾和田埂留白。所有预览由最终任务路径生成，配色和线宽为估计。'" + code[end:]
(OUT / 'build_this.py').write_text(code, encoding='utf8')
print(OUT)
