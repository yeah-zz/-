# -*- coding: utf-8 -*-
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from gongbi_training.scene_object_reference import sparse_scene_reference,legacy_object_to_scene
from guohua_objects_train_20261112 import ContextObjects,predict,confusion,metrics
ROOT=LAB/'guohua_faint_sources_20261116'
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def R(name,obj,points,note,kind='core_polygon'):return dict(name=name,object=obj,points=points,note=note,kind=kind)
def prepare():
 assert not (ROOT/'references.json').exists(),'References already frozen';ds=read(ROOT/'dataset_manifest.json');byid={r['id']:r for r in ds['records']};cases=[]
 def add(ident,regions):
  r=byid[ident];cases.append(dict(id=ident,image=r['image'],image_sha256=r['image_sha256'],size=r['size'],work_group=r['work_group'],regions=regions))
 add('tang-yin_making-tea_2',[
 R('foreground_rock','rock',[[80,100],[129,77],[173,112],[182,160],[152,181],[105,165]],'浓墨山石对照，不以浅色区域数量替代质量'),
 R('rock_lower','rock',[[103,260],[134,236],[165,251],[174,294],[155,317],[117,305]],'可辨识山石内部'),
 R('open_upper_right','open_scene',[[338,20],[460,20],[471,102],[415,119],[348,85]],'轮廓外较空旷场景，非精确无墨纸面'),
 R('rock_tree_mix','unknown',[[270,330],[323,313],[350,350],[344,409],[293,428],[259,384]],'山石与植被交叠，保持待定','uncertain_polygon')])
 add('tang-yin_making-tea_3',[
 R('faint_distant_peak','rock',[[159,115],[180,105],[201,115],[201,139],[174,147],[154,133]],'远山淡墨内部，避开左前枝条'),
 R('distant_peak_middle','rock',[[247,76],[280,58],[312,65],[335,89],[320,121],[278,124],[248,104]],'中间远山内部'),
 R('open_sky','open_scene',[[147,12],[199,10],[208,43],[170,53],[146,35]],'山体轮廓外天空；不是可见墨迹标签'),
 R('building_roof','other_object',[[54,289],[112,270],[145,285],[161,316],[107,328],[64,316]],'建筑独立对象，不能以其他对象类别替代空景'),
 R('fading_mountain_base','unknown',[[222,150],[280,146],[341,154],[361,193],[295,221],[235,205]],'淡山底部融入雾气，边界无法确定','uncertain_polygon')])
 add('yun-shouping_sunset-along-the-floral-embankment_part_1',[
 R('near_faint_face','rock',[[164,118],[180,99],[196,107],[206,137],[222,172],[204,187],[182,162]],'浅墨山面内部，和对象外空景分开'),
 R('distant_faint_face','rock',[[347,138],[363,124],[377,144],[400,175],[388,184],[365,164]],'淡远山内部，不据此推断皴法'),
 R('open_sky','open_scene',[[352,16],[471,16],[477,67],[429,86],[361,68]],'远山之外空景'),
 R('dissolving_slope','unknown',[[298,177],[337,180],[383,190],[426,206],[410,227],[349,220]],'远山坡脚晕淡区，保留未知','uncertain_polygon')])
 add('yun-shouping_sunset-along-the-floral-embankment_part_4',[
 R('willow_crown','tree',[[300,190],[334,171],[362,181],[371,217],[354,241],[317,235],[293,215]],'淡墨柳叶簇内部，对象范围不等于叶片墨迹'),
 R('willow_trunk','tree',[[330,305],[340,299],[346,336],[343,363],[356,389],[346,404],[333,376],[328,346]],'可辨柳树主干内部，不推断每笔顺序'),
 R('left_crown','tree',[[118,349],[134,335],[152,349],[155,378],[141,391],[120,378]],'左侧独立柳叶簇'),
 R('open_sky','open_scene',[[37,30],[206,28],[212,103],[157,118],[44,101]],'树冠外空景，不能误作山石'),
 R('crown_behind_trunk','unknown',[[363,241],[405,234],[430,273],[417,325],[384,338],[354,309]],'细叶、枝条、树后空隙难分，不强制归属','uncertain_polygon')])
 add('zhao-ji_autumn-colors_2',[
 R('faint_back_peak','rock',[[153,222],[182,212],[204,230],[197,257],[170,267],[148,246]],'左后方淡墨山体内部'),
 R('tall_peak_face','rock',[[355,159],[379,144],[397,164],[399,210],[389,240],[360,233],[351,196]],'高峰浅墨内部，避开峰顶密集点墨'),
 R('open_sky_left','open_scene',[[31,29],[260,29],[260,139],[76,160],[28,126]],'山体外天空大块空景'),
 R('mist_peak_base','unknown',[[329,242],[370,240],[405,252],[421,284],[393,310],[346,298]],'峰脚融入雾中，边界不硬补','uncertain_polygon')])
 add('zhao-ji_autumn-colors_3',[
 R('distant_peak_face','rock',[[88,69],[115,60],[138,75],[147,96],[127,113],[96,103]],'远山较淡内部，避开上缘植被不确定位置'),
 R('right_distant_face','rock',[[332,81],[349,73],[370,85],[370,102],[350,112],[331,102]],'右后远山内部'),
 R('tree_group','tree',[[347,199],[370,185],[395,203],[399,236],[379,257],[351,245]],'近处树林保守内部，不标每根细枝'),
 R('mist_gap','unknown',[[212,105],[258,96],[302,104],[319,127],[294,150],[247,151]],'山体与雾气交界，没有可靠完整边界','uncertain_polygon')])
 for c in cases:
  assert sha(c['image'])==c['image_sha256'];assert all(0<=x<c['size'][0] and 0<=y<c['size'][1] for r in c['regions'] for x,y in r['points'])
 refs=dict(cases=cases,annotation_author='assistant_visual_review',expert_reviewed=False,used_for_training=False,predictions_not_seen_before_reference_freeze=True,status='New development works, not formal independent test',source='Pinned CLPD; filename artist/work attribution not authenticated',taxonomy='scene0=open_scene,1=rock,2=tree,3=other_object,255=unknown; legacy0 means other_object and must map to3',ink_labels='All255 unknown, no object polygon becomes an ink target',coverage='Pale tonal/colored mountains and trees; not yet a diverse set of unfilled line-only mountain drawings',robot_connected=False);write('references.json',refs);print('FROZEN',len(cases),sum(len(c['regions']) for c in cases),flush=True)
def evaluate():
 assert not (ROOT/'results.json').exists(),'Do not overwrite results';refs=read(ROOT/'references.json');ds=read(ROOT/'dataset_manifest.json');select=read(ROOT/'selection_before_download.json');(ROOT/'views').mkdir(exist_ok=True);(ROOT/'masks').mkdir(exist_ok=True)
 # The reserved works stay metadata-only, no preview or inference.
 reserved={Path(p).stem for p in select['reserved_paths']};assert not any(p.stem in reserved for p in (ROOT/'raw').glob('*.jpg'))
 paths={'previous':LAB/'guohua_objects_v2_20261113/expanded_control.pt','sparse_weak':LAB/'guohua_objects_v4_sparse_20261116/sparse_weak.pt'};torch.set_num_threads(4);models={}
 for name,path in paths.items():
  net=ContextObjects().cuda();net.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['model']);net.eval();models[name]=net
 out=[];palette=np.array([[145,157,170],[183,134,75],[65,149,91]],'uint8');colors={'rock':(183,134,75),'tree':(45,161,91),'open_scene':(27,159,174),'other_object':(112,112,190),'unknown':(196,64,162)}
 for c in refs['cases']:
  im=Image.open(c['image']).convert('RGB');scene,ink=sparse_scene_reference(c['size'],c['regions']);assert np.all(ink==255);Image.fromarray(scene).save(ROOT/'masks'/(c['id']+'_scene_sparse.png'));Image.fromarray(ink).save(ROOT/'masks'/(c['id']+'_ink_unknown.png'));overlay=im.convert('RGBA');masks={};preds={}
  for r in sorted(c['regions'],key=lambda r:r['object']!='unknown'):
   mask=Image.new('L',im.size,0);ImageDraw.Draw(mask).polygon([tuple(p) for p in r['points']],fill=255);masks[r['name']]=np.array(mask)>0;layer=Image.new('RGBA',im.size,(*colors[r['object']],0));layer.putalpha(mask.point(lambda a:a*85//255));overlay=Image.alpha_composite(overlay,layer);ImageDraw.Draw(overlay).line([tuple(p) for p in r['points']]+[tuple(r['points'][0])],fill=(*colors[r['object']],230),width=2)
  im.save(ROOT/'views'/(c['id']+'_input.png'));overlay.convert('RGB').save(ROOT/'views'/(c['id']+'_reference.jpg'),quality=95)
  for name,net in models.items():
   prob,pred=predict(net,np.array(im));np.savez_compressed(ROOT/'masks'/(c['id']+'_'+name+'.npz'),prediction=pred,probability=prob.astype('float16'));preds[name]=pred;Image.fromarray(palette[pred]).save(ROOT/'views'/(c['id']+'_'+name+'.png'))
  regions=[]
  for r in c['regions']:
   m=masks[r['name']];entry=dict(name=r['name'],object=r['object'],note=r['note'],pixels=int(m.sum()),scores=None,distribution=None)
   if r['object'] in ['rock','tree','other_object']:
    expected={'rock':1,'tree':2,'other_object':0}[r['object']];entry['scores']={a:float((pred[m]==expected).mean()) for a,pred in preds.items()}
   elif r['object']=='open_scene':entry['distribution']={a:np.bincount(pred[m],minlength=3).tolist() for a,pred in preds.items()}
   regions.append(entry)
  out.append(dict(id=c['id'],work_group=c['work_group'],input='views/'+c['id']+'_input.png',reference='views/'+c['id']+'_reference.jpg',predictions={a:'views/'+c['id']+'_'+a+'.png' for a in models},regions=regions))
 write('results.json',dict(records=out,reference_sha256=sha(ROOT/'references.json'),model_hashes={a:sha(p) for a,p in paths.items()},used_for_training=False,expert_accuracy=False,new_independent_validation=False,reserved_not_downloaded=True,robot_connected=False));print('DIAGNOSTIC_READY',len(out),flush=True)
if __name__=='__main__':
 if '--prepare' in sys.argv:prepare()
 else:evaluate()
