# -*- coding: utf-8 -*-
"""Record user's foliage correction without claiming the model learned it."""
import sys,json,hashlib,base64
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from gongbi_training.cunfa_object_roles import OBJECT_IDS,constrain_rock_roles,forbidden_cun_reference
ROOT=LAB/'cunfa_object_correction_20261112';PREV=LAB/'cunfa_retention_v4_20261112';REF=LAB/'cunfa_band_checked_20261112'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ROOT.mkdir(exist_ok=True);ds=json.loads((REF/'dataset_manifest.json').read_text('utf8'));r=next(x for x in ds['records'] if x['id']=='H_14');z=np.load(r['path']);image=z['image'];shape=image.shape[:2]
 # Conservative core of the user-identified crown. Outer foliage, the rock/tree
 # overlap and cut-off branches remain unlabelled. This is an object REGION;
 # it is not a complete tree boundary or a per-pixel ink mask.
 polygon=[[174,257],[181,247],[198,239],[216,237],[232,241],[243,248],[245,264],[169,264]]
 region=np.zeros(shape,'uint8');cv2.fillPoly(region,[np.array(polygon,'int32')],1);objects=np.full(shape,255,'uint8');objects[region>0]=OBJECT_IDS['tree_foliage'];verified=region>0
 np.savez_compressed(ROOT/'H_14_object_reference.npz',object_targets=objects,object_verified=verified,reference_region=region)
 oldmetrics_sha=sha(PREV/'results.json');olddata_sha=sha(REF/'dataset_manifest.json');rows=[];font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n);overlay=image.copy();overlay[verified]=(.45*image[verified]+.55*np.array([135,65,165])).astype('uint8');Image.fromarray(overlay).save(ROOT/'user_foliage_reference.png')
 board=Image.new('RGB',(1200,460),'#f4f1e8');d=ImageDraw.Draw(board);cmap=np.array([[234,239,226],[192,128,40],[40,135,180]],'uint8')
 raw=np.load(PREV/'predictions/H_14_known_ink_retention.npz')['prediction'];marked=cmap[raw];marked=cv2.polylines(marked.copy(),[np.array(polygon,'int32')],True,(170,45,150),2)
 for j,(im,title) in enumerate([(Image.fromarray(image),'原图：底部是松树枝叶'),(Image.fromarray(marked),'模型原始误判 / 紫框指出错区'),(Image.fromarray(overlay),'用户确认的树叶参考核心')]):
  im=im.resize((385,249),Image.Resampling.NEAREST);board.paste(im,(j*400+6,45));d.text((j*400+6,8),title,font=font(19),fill='#243e32')
 d.text((8,335),'紫色是参考标记，不是模型已经学会的树叶类别。原始预测和旧指标全部保留。',font=font(20),fill='#815238');d.text((8,378),'先分山石 / 树木等对象，再在山石中分结构和皴笔；未知对象应保留待定。',font=font(20),fill='#243e32');board.save(ROOT/'foliage_error.jpg',quality=95)
 for name in ['v2_frozen','previous_refiner','matched_continuation','known_ink_retention']:
  path=PREV/'predictions'/('H_14_'+name+'.npz');prediction=np.load(path)['prediction'];values=prediction[verified];rows.append(dict(model=name,reference_region_pixels=int(verified.sum()),assigned_paper=int((values==0).sum()),assigned_structure=int((values==1).sum()),assigned_cun=int((values==2).sum()),regional_cun_assignment_fraction=float((values==2).mean()),not_a_full_image_or_expert_pixel_accuracy=True,raw_prediction_sha256=sha(path)))
 counts={str(k):int((z['core_targets'][verified]==k).sum()) for k in [0,1,2,255]};record=dict(id='H_14_foliage_user_correction',source='user explicitly identified the bottom area as foliage',source_image_sha256=sha(LAB/'cunfa_roles_v2_20261112/references/H_14_input.png'),polygon_native_context_xy=polygon,object_type='tree_foliage',object_level_confirmed_by_user=True,polygon_boundary_selected_by_assistant=True,exact_pixel_or_full_object_boundary_verified=False,object_region_is_not_ink_mask=True,old_core_label_counts_inside_region=counts,raw_model_predictions=rows,new_object_detector_trained=False,model_correction_claimed=False,rock_cun_forbidden_in_confirmed_region=True,unlabelled_region_must_not_become_background=True,old_results_sha256=oldmetrics_sha,old_dataset_sha256=olddata_sha,robot_connected=False)
 (ROOT/'correction.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
 # Apply the new API only for a reference-guided regression check. It is not
 # reported as model inference or a successful newly trained classifier.
 guarded=constrain_rock_roles(raw,objects,verified);assert np.all(guarded==255);assert np.all(forbidden_cun_reference(objects,verified)[verified]);assert np.array_equal(raw,np.load(PREV/'predictions/H_14_known_ink_retention.npz')['prediction'])
 synthetic=np.array([[2,2,2,1,2]],'uint8');obj=np.array([[1,3,255,1,1]],'uint8');v=np.array([[1,1,1,0,0]],bool);expected=np.array([[2,255,255,255,255]],'uint8');assert np.array_equal(constrain_rock_roles(synthetic,obj,v),expected);assert sha(PREV/'results.json')==oldmetrics_sha and sha(REF/'dataset_manifest.json')==olddata_sha
 (ROOT/'checks.json').write_text(json.dumps(dict(confirmed_foliage_cannot_emit_cun=True,unknown_object_abstains=True,unverified_rock_abstains=True,confirmed_rock_role_retained=True,unknown_not_converted_to_background=True,raw_predictions_unchanged=True,old_metrics_and_targets_unchanged=True,reference_guided_guard_not_model_learning=True,robot_connected=False),indent=2),encoding='utf8')
 # Correct the existing page's interpretation, while leaving all experimental
 # numbers and raw comparison images intact. Inline media needs no server restart.
 notice='''<aside id="object-correction" style="margin:18px 0;padding:18px;background:#fff1e9;border-left:5px solid #a63a3a"><b>用户已指出明确误判：底部是松树枝叶，模型将其分成了皴笔。</b><p>这块树叶没有被旧的局部角色参考覆盖。旧表中的召回和纸面误检没有检验树叶误分，因此不能用它们证明整张图识别正确。原始预测保留为错例。</p><p>后续需先区分山石、树干枝条、树叶等对象；仅在确认的山石内判断结构与皴笔。当前未训练出树木对象模型，不能把手工标出树叶说成模型已经修复。</p><img style="max-width:100%;height:auto" alt="底部树叶误判与用户参考" src="data:image/jpeg;base64,IMAGE_DATA"></aside>'''
 notice=notice.replace('IMAGE_DATA',base64.b64encode((ROOT/'foliage_error.jpg').read_bytes()).decode());(ROOT/'notice.html').write_text(notice,encoding='utf8');page=PREV/'index.html';s=page.read_text('utf8');start=s.find('<aside id="object-correction"')
 if start>=0:s=s[:start]+s[s.index('</aside>',start)+8:]
 s=s.replace('<h2>1. 先核对遗漏原因</h2>',notice+'<h2>1. 先核对遗漏原因</h2>');page.write_text(s,encoding='utf8')
 (ROOT/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>树叶对象误判复核</title><main style="max-width:1200px;margin:auto;font:18px/1.7 sans-serif">'+notice+'<p>对象核心内各模型的分类计数见correction.json；此处只记录用户确认的错例，不报告训练成功。</p></main></html>',encoding='utf8')
 print(json.dumps(dict(old_core_label_counts=counts,model_assignments=rows,new_model_trained=False,guard_checks_passed=True),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
