# -*- coding: utf-8 -*-
"""Image-reviewed sparse object evidence, kept separate from stroke/ink truth."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from guohua_objects_train_20261112 import ContextObjects,predict
ROOT=LAB/'guohua_boundary_reference_20261116';V2=LAB/'guohua_objects_v2_20261113';V3=LAB/'guohua_objects_v3_20261116'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf8'))
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def reg(name,obj,kind,points,note,width=None):return dict(name=name,object=obj,kind=kind,points=points,note=note,corridor_width_px=width)
def prepare():
 ROOT.mkdir(exist_ok=True);assert not (ROOT/'references.json').exists(),'References frozen; do not revise after predictions.'
 ds=read(V2/'dataset_manifest.json');byid={r['id']:r for r in ds['records']}
 def case(ident,regions):
  r=byid[ident];return dict(id=ident,image=r['image'],image_sha256=r['image_sha256'],size=r['size'],work_group=r['work_group'],prior_use='old_model_training_work',regions=regions)
 cases=[dict(id='H_14',image=str(LAB/'cunfa_roles_v2_20261112/references/H_14_input.png'),prior_use='repeated_known_user_regression',work_group='known_tutorial',regions=[
  reg('faint_mountain_interior','rock','core_polygon',[[213,52],[239,39],[264,35],[277,49],[251,67],[234,77]],'淡轮廓围出的山体内部；是对象上下文，不要求内部有墨'),
  reg('faint_mountain_interior_right','rock','core_polygon',[[326,101],[357,89],[383,94],[396,111],[368,124],[341,129]],'右侧山面内部，避开毛笔遮挡'),
  reg('foreground_rock','rock','core_polygon',[[111,198],[135,179],[156,177],[185,190],[192,207],[175,216],[138,214]],'前景有皴线的山石内部，不推断皴法类别'),
  reg('foliage_core','tree','core_polygon',[[182,249],[199,237],[220,237],[238,247],[241,258],[177,259]],'用户确认底部为树叶；此更小多边形由助手选择，非精确叶片边界'),
  reg('faint_outline_left','rock','contour_corridor',[[57,143],[91,120],[118,103],[144,83],[166,62],[183,43],[205,26],[235,18],[266,14],[287,20],[302,34]],'原图淡轮廓附近的粗定位走廊，不是中心线或逐像素墨带真值',4),
  reg('faint_outline_right','rock','contour_corridor',[[247,121],[276,103],[300,84],[327,66],[348,62],[368,65],[390,78],[403,94]],'右侧上方淡轮廓附近，归属山石；不判断完整分笔',4),
  reg('foliage_rock_overlap','unknown','uncertain_polygon',[[147,248],[164,226],[186,218],[213,223],[243,232],[264,251],[264,264],[148,264]],'树叶/山石边缘交叠，不强制对象类别；不覆盖内部单独确认区域')]),
 case('dai-jin_returning-late-from-a-spring-outing_6',[
  reg('pine_foliage_left','tree','core_polygon',[[81,433],[100,423],[114,430],[108,448],[89,451]],'松针簇可确认树木，内部留白仍属于对象区域'),
  reg('pine_foliage_middle','tree','core_polygon',[[148,406],[161,396],[177,401],[181,417],[170,427],[155,421]],'独立松针簇'),
  reg('pine_foliage_right','tree','core_polygon',[[295,403],[310,399],[324,408],[325,421],[308,425],[296,417]],'前景右端松针簇'),
  reg('upper_branches','tree','contour_corridor',[[301,182],[315,169],[332,158],[348,155],[366,149]],'可见树枝方向附近；不是可直接绘制的完整轨迹',4),
  reg('foreground_rock_face','rock','core_polygon',[[218,457],[235,449],[247,458],[240,478],[222,486],[210,477]],'前景岩面保守内部，避开上方树干'),
  reg('trunk_rock_contact','unknown','uncertain_polygon',[[174,433],[206,416],[246,421],[265,448],[242,457],[217,446],[202,462],[180,455]],'树干与岩石重叠，保留待定')]),
 case('qiu-ying_fairyland-of-peach-blossoms_8',[
  reg('rock_left','rock','core_polygon',[[86,73],[118,53],[145,78],[157,132],[140,174],[110,188],[87,151]],'左侧青绿山石内部'),
  reg('rock_right','rock','core_polygon',[[433,30],[474,14],[496,29],[480,62],[449,79],[425,61]],'上方右侧山体内部，避开树冠'),
  reg('crown_top','tree','core_polygon',[[391,88],[409,77],[426,82],[434,97],[419,111],[398,108]],'顶部树冠保守内部'),
  reg('crown_middle','tree','core_polygon',[[275,210],[289,207],[309,217],[302,232],[282,237],[268,229]],'伸向空处的树冠内部'),
  reg('trunk','tree','core_polygon',[[316,354],[326,350],[319,380],[307,402],[297,420],[289,417],[304,388]],'可见树干较宽内部，不推断主枝笔顺'),
  reg('right_interlace','unknown','uncertain_polygon',[[342,143],[369,143],[387,165],[372,189],[347,198],[326,186]],'树枝、青绿山体与墨线交叠，保持未知')]),
 case('wang-shimin_clearing-of-rivers-and-mountains-after-snow_6',[
  reg('faint_rock_left','rock','core_polygon',[[36,98],[64,81],[100,94],[112,119],[93,147],[57,150],[30,128]],'淡设色山体内部，不将设色边界当皴笔'),
  reg('rock_upper_right','rock','core_polygon',[[432,33],[469,23],[500,36],[492,87],[475,130],[461,155],[440,147],[430,106]],'右上山面内部'),
  reg('tree_trunks','tree','core_polygon',[[158,340],[169,334],[179,346],[176,392],[163,405],[155,389]],'前景树林内部，不能由对象标签确定每根树干'),
  reg('pine_crown_right','tree','core_polygon',[[378,203],[398,198],[419,212],[420,231],[405,244],[386,231]],'右侧松树冠内部'),
  reg('pine_trunks_right','tree','core_polygon',[[411,234],[420,231],[418,259],[407,277],[398,278],[403,257]],'右侧树林中的可辨枝干区域'),
  reg('tree_rock_boundary','unknown','uncertain_polygon',[[326,294],[352,276],[380,268],[400,285],[374,309],[345,324]],'树冠、枝干和山面交界需精细复核')])]
 for c in cases:
  c['image_sha256']=sha(c['image']);c['size']=list(Image.open(c['image']).size)
 refs=dict(cases=cases,annotation_author='assistant_visual_review',expert_reviewed=False,new_independent_test=False,used_for_training=False,source_labels_not_copied=True,policy='Sparse object evidence only; contour corridors are coarse location, not ink masks, centers or complete strokes. Unknown areas have no negative label; confirmed internal regions take precedence over broad uncertain zones.',frozen_before_new_predictions=True,already_seen_case_status_disclosed=True,robot_connected=False)
 write('references.json',refs);print('Frozen',len(cases),'cases',sum(len(c['regions']) for c in cases),'regions',flush=True)
def render():
 refs=read(ROOT/'references.json');assert not (ROOT/'results.json').exists(),'Do not overwrite reviewed results';(ROOT/'views').mkdir(exist_ok=True);(ROOT/'masks').mkdir(exist_ok=True);records=[];color={'rock':(183,134,75),'tree':(45,161,91),'unknown':(196,64,162)};font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
 torch.set_num_threads(4);models={}
 paths={'pilot':LAB/'guohua_objects_pilot_20261112/context_object_head.pt','expanded':V2/'expanded_control.pt','core_loss':V3/'core_balanced.pt'}
 for name,path in paths.items():
  net=ContextObjects().cuda();net.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['model']);net.eval();models[name]=net
 board=Image.new('RGB',(1280,len(refs['cases'])*360+75),'#f4f1e8');draw=ImageDraw.Draw(board)
 for i,c in enumerate(refs['cases']):
  assert sha(c['image'])==c['image_sha256'];im=Image.open(c['image']).convert('RGB');size=im.size;layers={};overlay=im.convert('RGBA');allcore=Image.new('L',size,255)
  # Unknown is explicit on the overlay, but never becomes paper/background.
  for r in sorted(c['regions'],key=lambda r:r['object']!='unknown'):
   mask=Image.new('L',size,0);d=ImageDraw.Draw(mask)
   if r['kind']=='contour_corridor':d.line([tuple(p) for p in r['points']],fill=255,width=r['corridor_width_px'],joint='curve')
   else:d.polygon([tuple(p) for p in r['points']],fill=255)
   layers[r['name']]=np.array(mask)>0;rgba=Image.new('RGBA',size,(*color[r['object']],0));rgba.putalpha(mask.point(lambda a:a*85//255));overlay=Image.alpha_composite(overlay,rgba);od=ImageDraw.Draw(overlay);od.line([tuple(p) for p in r['points']]+([] if r['kind']=='contour_corridor' else [tuple(r['points'][0])]),fill=(*color[r['object']],230),width=2)
   if r['kind']=='core_polygon' and r['object']!='unknown':allcore.paste(1 if r['object']=='rock' else 2,mask=mask)
  allcore.save(ROOT/'masks'/(c['id']+'_sparse_core.png'));assert set(np.unique(allcore))<={1,2,255}
  Image.fromarray(np.array(im)).save(ROOT/'views'/(c['id']+'_input.png'));overlay.convert('RGB').save(ROOT/'views'/(c['id']+'_reference.jpg'),quality=95);preds={};scores=[]
  for name,net in models.items():
   prob,pred=predict(net,np.array(im));np.savez_compressed(ROOT/'masks'/(c['id']+'_'+name+'.npz'),prediction=pred,probability=prob.astype('float16'));palette=np.array([[145,157,170],[183,134,75],[65,149,91]],'uint8');Image.fromarray(palette[pred]).save(ROOT/'views'/(c['id']+'_'+name+'.png'));preds[name]='views/'+c['id']+'_'+name+'.png'
  for r in c['regions']:
   region=layers[r['name']];row=dict(name=r['name'],object=r['object'],kind=r['kind'],pixels=int(region.sum()),note=r['note'],scores=None)
   if r['object']!='unknown':
    expected=1 if r['object']=='rock' else 2;row['scores']={name:float((np.load(ROOT/'masks'/(c['id']+'_'+name+'.npz'))['prediction'][region]==expected).mean()) for name in models}
   scores.append(row)
  record=dict(id=c['id'],prior_use=c['prior_use'],input='views/'+c['id']+'_input.png',reference='views/'+c['id']+'_reference.jpg',predictions=preds,regions=scores);records.append(record)
  for j,(path,title) in enumerate([(record['input'],'原图'),(record['reference'],'助手参考：棕山石／绿树木／粉待定'),(preds['expanded'],'冻结模型：上轮扩充数据'),(preds['core_loss'],'冻结模型：失败的新约束')]):
   panel=Image.open(ROOT/path);panel.thumbnail((308,280));board.paste(panel,(j*320+5,i*360+42));draw.text((j*320+5,i*360+8),title,font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15),fill='#294735')
  draw.text((5,i*360+327),c['id'][:95],font=font,fill='#79523a')
 draw.text((5,len(refs['cases'])*360+6),'参考用于对象归属诊断，非专家真值、精确墨带或模型提取结果；三例来自旧训练作品。',font=font,fill='#79523a');board.save(ROOT/'reference_comparison.jpg',quality=94)
 result=dict(records=records,reference_sha256=sha(ROOT/'references.json'),checkpoint_hashes={n:sha(p) for n,p in paths.items()},new_training=False,expert_accuracy_claimed=False,independent_test=False,robot_connected=False);write('results.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':
 if '--prepare' in sys.argv:prepare()
 else:render()
