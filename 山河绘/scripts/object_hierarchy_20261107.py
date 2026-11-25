# -*- coding: utf-8 -*-
"""An explicitly curated hierarchy for ONE accepted scene, not semantic training.

SAM part keys and an assistant-corrected tree extent are source-hash bound.
All 489 paths are preserved exactly; only ownership, roles and order change.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,copy,hashlib,shutil,colorsys
import numpy as np,cv2
from scipy.ndimage import binary_fill_holes
from PIL import Image,ImageDraw,ImageFont
from module_stroke_core_20261104 import samples,length
from hierarchical_paint_core_20261105 import order_metrics
ROOT=LAB/'object_hierarchy_20261107';PRIOR=LAB/'approved_execution_refine_20261107/G03_01_500'
MASKS=LAB/'module_stroke_planner_20261104/outputs/G03_01'
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18);SMALL=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)

def read(p):return json.loads(p.read_text('utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item(),allow_nan=False),encoding='utf8')
def color(i):return tuple(round(v*190) for v in colorsys.hsv_to_rgb((i*.618)%1,.7,1))

TREE=[[185,305],[199,275],[207,222],[219,188],[186,186],[153,199],[151,170],[198,154],[233,151],[235,113],[258,105],[293,113],[315,136],[352,137],[384,150],[393,176],[425,184],[478,203],[480,247],[438,243],[403,220],[374,211],[346,199],[308,187],[289,172],[274,180],[249,195],[238,222],[239,259],[221,281],[228,302]]
# Source-specific object membership reviewed against the saved SAM contact sheet.
OBJECTS=[
 dict(name='左侧坡岸山石',kind='rock',parts=['m040','m009','m001','m038','m021','m033']),
 dict(name='左中部坡石',kind='rock',parts=['m029','m031','m034','m037']),
 dict(name='中央主山石（含右侧石面）',kind='rock',parts=['m025','m043','m004','m020']),
 dict(name='前景左侧石组',kind='rock',parts=['m024','m007','m011']),
 dict(name='前景中部主石',kind='rock',parts=['m006','m005','m010']),
 dict(name='前景近岸小石',kind='rock',parts=['m002','m036','m030']),
 dict(name='右侧近岸石',kind='rock',parts=['m015','m023']),
 dict(name='亭子整体',kind='architecture',parts=['m003','m022','m039','m041']),
 dict(name='松树整体：树干→枝条→树冠细线',kind='tree',parts=['m016','m032','m035','m012'],polygon=TREE),
 dict(name='远山与岸线',kind='distant',parts=['m000','m042']),
 dict(name='水面线与余下水际细线',kind='water',parts=['m008','m027','m028','m044']),
 dict(name='亭子左侧坡上树丛',kind='tree',parts=['m013']),
 dict(name='主石下方独立小石',kind='rock',parts=['m017','m019','m014']),
]
DRAW_ORDER=[0,1,2,3,4,12,5,6,11,7,8,9,10]

def extent(mask):
 y,x=np.where(mask);return [float(x.min()),float(y.min()),float(x.max()),float(y.max())] if len(x) else [0.,0.,0.,0.]

def object_masks():
 z=np.load(MASKS/'sam_masks_seeded.npz');out=[]
 for obj in OBJECTS:
  m=np.zeros((496,496),np.uint8)
  for part in obj['parts']:m|=z[part].astype(np.uint8)
  if obj.get('polygon'):cv2.fillPoly(m,[np.array(obj['polygon'],np.int32)],1)
  m=binary_fill_holes(m).astype(np.uint8)
  out.append(m)
 # Architecture owns its structural interior, not surrounding tree space.
 out[7]=cv2.dilate(out[7],np.ones((3,3),np.uint8));out[8][out[7]>0]=0
 return out

def assign_objects(rows,masks):
 distances=[cv2.distanceTransform((1-m).astype(np.uint8),cv2.DIST_L2,5) for m in masks]
 assignments=[]
 for i,r in enumerate(rows):
  q=samples(np.asarray(r['points']),.4);xy=np.clip(np.rint(q).astype(int),0,495)
  fits=[]
  for mid,dist in enumerate(distances):
   d=dist[xy[:,1],xy[:,0]];fraction=float((d<=2.).mean());mean=float(d.mean())
   # Strong full-object regions override small accidentally intersecting parts.
   fits.append((fraction,mean,mid))
  strong=[s for s in fits if s[0]>=.72]
  if strong:
   # Whole pavilion/tree ownership wins only where most of the curve fits it.
   semantic=[s for s in strong if s[2] in (7,8,11)]
   best=min(semantic or strong,key=lambda a:(-a[0],a[1],int(masks[a[2]].sum())))
  else:
   best=min(fits,key=lambda a:(a[1],-a[0]))
   p=np.asarray(r['points']);axis=p[-1]-p[0]
   if q[:,1].mean()>365 and abs(axis[0])>abs(axis[1])*3 and best[0]<.55:best=fits[10]
  assignments.append(dict(curve=i,module=best[2],whole_curve_fit_fraction=best[0],mean_distance_px=best[1],
    boundary_or_uncertain=best[0]<.72,method='source-specific assistant-reviewed object masks + full-curve assignment'))
 return assignments

def schedule(rows,modules):
 """Hard object barrier. A later object cannot start before this one is done."""
 route=[];cursor=np.array([0.,495.]);order=[]
 for mid in DRAW_ORDER:
  m=modules[mid]
  ids=[i for i,r in enumerate(rows) if r['module']==m['id']]
  if not ids:continue
  order.append(m['id'])
  for phase in range(3):
   remaining=[i for i in ids if rows[i]['phase']==phase]
   while remaining:
    def priority(i):
     p=np.asarray(rows[i]['points']);d=min(np.linalg.norm(p[0]-cursor),np.linalg.norm(p[-1]-cursor))
     return length(p)**.7/(1+d/50)
    i=max(remaining,key=priority);remaining.remove(i);p=np.asarray(rows[i]['points'])
    reverse=bool(np.linalg.norm(p[-1]-cursor)<np.linalg.norm(p[0]-cursor));q=p[::-1] if reverse else p
    route.append(dict(sequence=len(route)+1,stroke_id=f's{i:05}',curve=i,module=m['id'],phase=phase,reverse=reverse,
      air_travel_px=float(np.linalg.norm(q[0]-cursor)),length_px=length(p),texture_bundle=None));cursor=q[-1]
 return route,order

def main():
 ROOT.mkdir(exist_ok=True);folder=ROOT/'G03_01';folder.mkdir(exist_ok=True);parent=PRIOR/'plan.json';digest=sha(parent);old=read(parent);rows=copy.deepcopy(old['curves'])
 manifest=read(MASKS/'sam_manifest_seeded.json');assert sha(PRIOR/'source.png')==manifest['source_sha256']
 masks=object_masks();assignment=assign_objects(rows,masks);profiles=read(PRIOR/'width_targets.json')['profiles'];modules=[]
 for mid,(obj,mask) in enumerate(zip(OBJECTS,masks)):
  contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);bbox=extent(mask)
  modules.append(dict(id=mid,name=obj['name'],kind=obj['kind'],sam_key='reviewed-parts',verified=False,
    scene_hierarchy_reviewed_by_assistant=True,semantic_model_verified=False,bbox=bbox,center=[(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2],
    outlines=[c.reshape(-1,2).tolist() for c in contours if len(c)>2],source_parts=obj['parts'],order=mid+1))
 for i,(r,a) in enumerate(zip(rows,assignment)):
  r['previous_module']=r['module'];r['module']=a['module'];r['object_boundary_uncertain']=a['boundary_or_uncertain'];p=np.asarray(r['points']);arc=length(p);m=modules[r['module']]
  profile=profiles[i];valid=np.asarray(profile['observed']);width=np.asarray(profile['width_px']);observed=float(np.median(width[valid])) if valid.any() else 0.
  if m['kind']=='tree':phase=0 if arc>=18 and observed>=1.5 else 1 if arc>=9 else 2
  elif m['kind']=='architecture':phase=0 if arc>=18 else 1 if arc>=6 else 2
  elif m['kind'] in ('water','distant'):phase=0 if arc>=18 else 1
  else:
   inner=cv2.distanceTransform(masks[r['module']],cv2.DIST_L2,5);q=samples(p,.5);xy=np.clip(np.rint(q).astype(int),0,495)
   boundary=float((inner[xy[:,1],xy[:,0]]<3).mean());phase=0 if arc>=10 and boundary>=.35 else 1 if arc>=6 else 2
  r.update(phase=phase,phase_label=['对象轮廓／骨架候选','对象内部长线候选','对象细节候选'][phase],role_verified=False)
 route,order=schedule(rows,modules)
 previous_in_new_objects=[{**a,'module':rows[a['curve']]['module']} for a in old['actions']]
 for m in modules:
  m['planned_strokes']=sum(r['module']==m['id'] for r in rows);m['order']=order.index(m['id'])+1 if m['id'] in order else 0
  m['uncertain_curves']=sum(r['module']==m['id'] and r['object_boundary_uncertain'] for r in rows)
 plan=copy.deepcopy(old);plan.update(id='G03_01',parent=str(parent),parent_sha256=digest,curves=rows,modules=modules,actions=route,module_order=order,
  hierarchy_method='assistant-curated scene-specific object hierarchy, not general autonomous segmentation',
  source_specific_hierarchy=True,automatic_semantic_generalization=False,adopted=False,ready_for_robot=False)
 plan['stats']={**old['stats'],'before':{'count':len(rows)},'after':{'count':len(rows)},'accepted_continuations':0,
  'object_proposals':len(modules),'unresolved_regions':sum(a['boundary_or_uncertain'] for a in assignment),'baseline_order':order_metrics(previous_in_new_objects),'hierarchy_order':order_metrics(route)}
 save(folder/'plan.json',plan);save(folder/'object_assignment.json',assignment)
 save(ROOT/'scene_hierarchy.json',dict(source_sha256=sha(PRIOR/'source.png'),objects=OBJECTS,
  assistant_visual_correction=True,manual_extent_tree=True,semantic_model_not_trained=True,
  new_weights_trained=False,curves_split_or_deleted=False,uncertain_curve_ids=[a['curve'] for a in assignment if a['boundary_or_uncertain']]))
 for file in ('source.png','width_target.png'):shutil.copy2(PRIOR/file,folder/file)
 rgb=np.array(Image.open(PRIOR/'source.png').convert('RGB'));base=np.full((992,992,3),255,np.uint8);colored=base.copy()
 for r in rows:
  p=np.rint(np.asarray(r['points'])*2).astype(np.int32);cv2.polylines(base,[p],False,(30,42,35),1,cv2.LINE_AA);cv2.polylines(colored,[p],False,color(r['module']),2,cv2.LINE_AA)
 overlay=(rgb*.55+255*.45).astype(np.uint8)
 for mid,m in enumerate(modules):
  for c in m['outlines']:cv2.polylines(overlay,[np.asarray(c,np.int32)],True,color(mid),1,cv2.LINE_AA)
  x,y=m['center'];cv2.putText(overlay,str(mid+1),(round(x),round(y)),cv2.FONT_HERSHEY_SIMPLEX,.45,color(mid),1,cv2.LINE_AA)
 Image.fromarray(overlay).save(folder/'modules.png')
 board=Image.new('RGB',(1560,595),'#f4f1e8');d=ImageDraw.Draw(board)
 for j,(im,label) in enumerate([(Image.fromarray(rgb),'原图'),(Image.fromarray(base),'线稿保持489笔 · 几何不变'),(Image.fromarray(colored),'完整对象分组 · 每种颜色为一个模块')]):
  board.paste(im.resize((496,496)),(j*520+10,45));d.text((j*520+10,10),label,font=FONT,fill='#263e34')
 d.text((10,555),'本幅对象归属由助手校正：松树整体、亭子整体、山石与水面依次锁定；不是新模型自动语义识别，也未实画。',font=SMALL,fill='#263e34')
 board.save(folder/'comparison.jpg',quality=95);shutil.copy2(folder/'comparison.jpg',folder/'sequence_comparison.jpg')
 gallery=Image.new('RGB',(1560,5*555),'#f4f1e8');gd=ImageDraw.Draw(gallery)
 for rank,mid in enumerate(order):
  m=modules[mid]
  im=np.uint8(base*.2+255*.8)
  for r in rows:
   if r['module']==mid:cv2.polylines(im,[np.rint(np.asarray(r['points'])*2).astype(np.int32)],False,color(mid),2,cv2.LINE_AA)
  x=rank%3*520;y=rank//3*555;gallery.paste(Image.fromarray(im).resize((496,496)),(x+10,y+45));gd.text((x+10,y+10),f'{rank+1}. {m["name"]} · {m["planned_strokes"]}笔',font=FONT,fill='#263e34')
 gallery.save(folder/'rock.jpg',quality=94)
 examples=Image.new('RGB',(1560,555),'#f4f1e8');ed=ImageDraw.Draw(examples)
 for j,mid in enumerate((2,7,8)):
  im=np.uint8(base*.2+255*.8)
  for r in rows:
   if r['module']==mid:cv2.polylines(im,[np.rint(np.asarray(r['points'])*2).astype(np.int32)],False,color(mid),2,cv2.LINE_AA)
  examples.paste(Image.fromarray(im).resize((496,496)),(j*520+10,45));ed.text((j*520+10,10),modules[mid]['name'],font=FONT,fill='#263e34')
 examples.save(ROOT/'object_examples.jpg',quality=96)
 save(ROOT/'catalog.json',dict(scenes=[dict(id='G03_01',label='G03_01 · 完整对象笔序校正（489笔）')],default='G03_01',robot_connected=False));save(ROOT/'execution_binding.json',dict(scene=None,journal=None,plan_sha256=None))
 page=Path(__file__).with_name('hierarchical_paint_platform_20261105.html').read_text('utf8')
 page=page.replace('o.textContent=({G03_01:','o.textContent=s.label||({G03_01:')
 page=page.replace("m.sam_key?'自动对象候选 · 未验证为独立山石':'仅空间分组 · 对象归属待定'","m.scene_hierarchy_reviewed_by_assistant?'本幅归属由助手校正 · 非自动识别验证':'归属待定'")
 page=page.replace('按块完成 · 块内分层 · 连续笔画候选','按完整对象完成 · 本幅归属校正 · 线稿不变')
 page=page.replace('输入细段 → 连续候选','校正前后笔数（只改归属与顺序）').replace('原图支持的断段连接','本轮新增或删除笔画')
 page=page.replace('对象候选 / 归属待定区','本幅对象数 / 边界待核对笔数')
 page=page.replace("?'对象候选':'归属待定'","?'本幅校正对象':'归属待定'")
 page=page.replace('轮廓／骨架候选','本对象轮廓／骨架候选').replace('内部长线／皴线候选','本对象内部长线候选').replace('细节／待定','本对象细节候选')
 page=page.replace('区域是自动分割候选，尚未验证为独立山石；“内部长线”不等于已识别具体皴法。归属待定区保留标识。回放速度只影响动画，不是机械臂速度。','本幅对象归属由助手校正：完成当前对象全部笔画后才切换对象。轮廓/内部/细节仍为启发式分层；跨对象共享边只绘制一次，疑似跨界笔画保留记录。不是模型自主语义识别验证。回放速度不是机械臂速度。')
 page=page.replace('<footer>','<footer><p><a href="/report.html">查看每个完整对象的笔画范围与对照</a></p>')
 (ROOT/'index.html').write_text(page,encoding='utf8')
 html='''<!doctype html><meta charset="utf-8"><title>完整对象笔序校正</title><style>body{font:16px/1.7 Microsoft YaHei,system-ui;background:#f4f1e8;color:#243e33;padding:24px;max-width:1560px;margin:auto}img{width:100%}a{color:#286246}li{padding:4px}</style><h1>把分割碎片恢复为绘画对象</h1><p>此前零回访统计针对零散分割区，不能证明一块石头或一棵树已经画完。现在对这幅G03_01的对象归属做了助手视觉校正：SAM部件合并、树干与树冠归为完整松树、亭子结构归为整体，并为山石设置明确绘画模块。没有训练新模型，也不代表其他构图已经能自动这样分组。</p><p>全部489笔坐标、形状与数量保持不变。只改归属、分层和笔序。每个模块先处理轮廓/骨架候选，再内部线，最后细节；本模块全部结束后才允许下一个模块开始。共享轮廓只分配一个绘画归属，不重画、不拆笔。</p><p><a href="/">打开逐对象回放</a></p><ol>'''
 for mid in order:
  m=modules[mid];html+=f'<li>{m["name"]}：{m["planned_strokes"]}笔</li>'
 html+='</ol><img src="/G03_01/comparison.jpg"><h2>逐对象核查范围</h2><p>着色线是当前模块完整笔画；淡灰线是其他模块。本幅对象定义含若干石组，并未声称每一块小石头都已独立识别。边界归属不确定的曲线记录在object_assignment.json；未标为准确识别。</p><img src="/G03_01/rock.jpg">'
 (ROOT/'report.html').write_text(html,encoding='utf8')
 assert sha(parent)==digest
 print('OBJECT_HIERARCHY',[(m['name'],m['planned_strokes']) for m in modules], 'uncertain',sum(a['boundary_or_uncertain'] for a in assignment),flush=True)

if __name__=='__main__':main()
