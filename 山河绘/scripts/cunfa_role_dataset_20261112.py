# -*- coding: utf-8 -*-
import sys,json,hashlib,datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from gongbi_training.cunfa_supervision import style_target,role_targets_checked
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'cunfa_role_reference_20261112';BASE=LAB/'cunfa_gabor_baseline_20261110';PREV=LAB/'cunfa_fresh_validation_20261111'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d):(ROOT/n).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 assert not (ROOT/'source_probe_results.json').exists(),'Reference admission is frozen before training.'
 (ROOT/'references').mkdir(exist_ok=True);(ROOT/'roles').mkdir(exist_ok=True);records=[];font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
 def add(ident,path,label,publisher,group,evidence,url,box=None,scope='named_teaching_example'):
  image=Image.open(path).convert('RGB');box=box or [0,0,image.width,image.height];dest=ROOT/'references'/(ident+'.png');image.crop(box).save(dest)
  r=dict(id=ident,image=str(dest),source_image=str(path),source_box=box,source_sha256=sha(path),crop_sha256=sha(dest),local_style=label,label_scope=scope,source_url=url,source_evidence=evidence,
   publisher=publisher,source_group=group,source_named_label=True,expert_local_verified=False,role_mask_path=None,role_reference_checked=False,automatic_prediction_as_label=False)
  r['style_target']=style_target(r);r['role_supervision_eligible']=role_targets_checked(r);assert r['style_target']>=0;records.append(r)
 aamurl='https://education.asianart.org/wp-content/uploads/sites/6/2019/09/Chinese-Brushstrokes-Vocabulary-AAM_0.pdf'
 for ident,file,label in [('aam_short_hemp','aam_2_img3.png','披麻皴'),('aam_long_hemp','aam_2_img13.png','披麻皴'),('aam_small_axe','aam_2_img1.png','斧劈皴'),('aam_large_axe','aam_2_img7.png','斧劈皴')]:
  add(ident,BASE/'sources'/file,label,'AAM','Lampo_Leong_all_examples','AAM PDF page2 captions name this texture illustration; illustrations credited to Lampo Leong.',aamurl)
 hk=ROOT/'sources/hkma_p4_img10.jpeg';hkurl='https://hk.art.museum/Archive/resources/S_Wucius_WONG_Suggested_Class_Activities_CHIN.pdf'
 add('hkma_axe',hk,'斧劈皴','HKMA','HKMA_illustrator_unverified','HKMA PDF page4 labels left practice photograph axe-cut. Caption/newsprint outside selected example excluded.',hkurl,[18,85,302,296])
 add('hkma_hemp',hk,'披麻皴','HKMA','HKMA_illustrator_unverified','HKMA PDF page4 labels right practice photograph hemp-fibre. Crop retains stone, excludes hand and most newspaper.',hkurl,[372,130,547,276])
 npm=BASE/'sources/npm_article_13_img1.jpeg';u='https://www.npm.gov.tw/NewChineseArtDownload.ashx?bid=4360'
 add('npm_named_axe',npm,'斧劈皴','NPM','Li_Tang_same_work','NPM article PDF page13, printed28, Fig17-1 rectangle1 explicitly labels axe-cut; using underlying photograph without labels.',u,[137,106,339,259],'named_local_source_region')
 add('npm_named_hemp',npm,'披麻皴','NPM','Li_Tang_same_work','NPM article PDF page13, printed28, Fig17-1 rectangle3 explicitly labels hemp-fibre; using underlying photograph without labels.',u,[499,140,558,300],'named_local_source_region')
 fresh=json.loads((PREV/'test_manifest.json').read_text('utf8'));audit=[]
 for r in fresh['records']:
  candidate=dict(local_style=r['label'],label_scope='work_level',source_url=r['source_url'],source_evidence=r['label_scope']);audit.append(dict(id=r['id'],old_label=r['label'],policy_target=style_target(candidate),reason='work-level description is insufficient for local supervised label',old_result_unchanged=True))
 assert all(r['policy_target']==-100 for r in audit)
 # Reference regions are review candidates, never pixel masks or complete strokes.
 specs={
  'fresh_230757_0': [('structure_candidate',[0,145,208,240]),('cun_candidate',[105,263,149,417]),('cun_candidate',[309,261,347,381]),('wash_or_paper_unknown',[191,310,280,425])],
  'fresh_230757_1': [('structure_candidate',[0,104,117,454]),('cun_candidate',[307,90,368,347]),('cun_candidate',[211,91,268,328]),('wash_or_paper_unknown',[120,315,202,425])],
  'fresh_230757_2': [('structure_candidate',[15,185,245,275]),('cun_candidate',[170,70,269,189]),('wash_or_merged_ink_unknown',[250,274,341,350])],
  'fresh_230757_3': [('structure_candidate',[150,283,315,405]),('cun_candidate',[80,175,158,265]),('wash_or_paper_unknown',[14,40,139,144])]
 }
 native=json.loads((ROOT/'native_manifest.json').read_text('utf8'));roles=[];colours={'structure_candidate':'#966821','cun_candidate':'#227e9a'}
 for r in native['records']:
  image=Image.open(r['native_image']).convert('RGB');overlay=image.copy();dr=ImageDraw.Draw(overlay);rs=[]
  for i,(role,box) in enumerate(specs[r['id']]):
   colour=colours.get(role,'#8c7a62');dr.rectangle(box,outline=colour,width=3);dr.text((box[0]+3,box[1]+3),str(i+1),font=font(17),fill=colour)
   rs.append(dict(id=r['id']+'_region'+str(i+1),role_candidate=role,box=box,checked_role=None,local_style=None,role_reference_checked=False,eligible_for_role_loss=False,eligible_for_style_loss=False,ink_band_truth=None,complete_curve=None,stroke_order=None,pressure=None))
  dest=ROOT/'roles'/(r['id']+'.png');overlay.save(dest);roles.append(dict(id=r['id'],source_image=r['native_image'],source_sha256=sha(r['native_image']),overlay=str(dest),regions=rs,
   source_work_label='small axe-cut plus short blunt-brush marks',confirmed_local_named_style=None,source_work_not_new_independent_test=True,expert_verified=False))
 save('source_references.json',dict(records=records,original_reference_units=len(records),local_named_example_labels=True,not_expert_pixel_gold=True,role_masks_available=0,posterior_classifier_predictions_not_used=True,
  no_rotated_views_counted_as_new_data=True,source_groups=3,publishers=3,known_illustrator_attributions=['Lampo Leong','Li Tang (museum attribution)'],HKMA_illustrator_identity_unknown=True,admission_fixed_before_probe=True))
 save('partial_roles.json',dict(records=roles,known_pixel_masks=0,local_named_style_labels=0,unknown_is_not_negative=True,ready_for_role_training=False))
 save('legacy_policy_audit.json',dict(records=audit,legacy_manifest_sha256=sha(PREV/'test_manifest.json'),old_test_results_modified=False,all16_work_labels_blocked_for_new_local_style_training=True))
 canvas=Image.new('RGB',(1440,1000),'#f4f1e8');d=ImageDraw.Draw(canvas);d.text((18,10),'直接标注技法的来源参考：8个原始示例单位，不是8位作者或像素真值',font=font(25),fill='#243e32')
 for i,r in enumerate(records):
  x=i%4*360;y=i//4*465+55;image=Image.open(r['image']);image.thumbnail((330,340));canvas.paste(image,(x+12,y));d.text((x+12,y+349),r['id'],font=font(18),fill='#243e32');d.text((x+12,y+379),r['local_style']+' · '+r['publisher'],font=font(22),fill='#243e32');d.text((x+12,y+415),'角色/完整笔画归属：未标注',font=font(17),fill='#815d45')
 canvas.save(ROOT/'source_references.jpg',quality=93);print('REFERENCES',len(records),'PARTIAL_ROLE_CANDIDATES',sum(len(r['regions']) for r in roles),flush=True)
if __name__=='__main__':main()
