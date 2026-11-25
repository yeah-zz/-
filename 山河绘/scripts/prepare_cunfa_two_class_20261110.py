# -*- coding: utf-8 -*-
"""Source-evidenced two-class pilot, with explicit label scopes and exclusions.

Museum work-level descriptions do not certify every local crop. Region labels
are provisional where only the whole work is described. No expert gold claimed.
"""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'cunfa_gabor_baseline_20261110'
PIMA='披麻皴';AXE='斧劈皴'
# Crops chosen for visible mountains rather than seals, letters, boats or trees.
# Pixel rectangles are source-image coordinates and are retained for auditing.
MUSEUM={
 '234540':dict(label=PIMA,artist='Shen_Zhou',work='Cangzhou_Qu',boxes=[
  [25,40,180,180],[220,25,365,175],[400,40,545,185],[565,65,700,200],[760,60,900,200],[990,85,1130,220],[1080,280,1240,440]]),
 '259387':dict(label=PIMA,artist='Dai_Xi',work='Yi_Song',boxes=[
  [45,135,190,280],[205,170,350,315],[370,160,515,305],[565,205,710,350],[740,75,885,220],[940,80,1085,225],[1090,110,1235,255],[955,300,1100,445]]),
 '229314':dict(label=PIMA,artist='Huang_Gongwang',work='Dan_Ya_Yu_Shu',boxes=[
  [140,310,260,430],[260,330,380,450],[355,475,475,595],[180,580,300,700],[350,650,470,770],[270,790,390,910],
  [215,425,295,505],[335,620,405,700],[300,755,380,840]]),
 '233992':dict(label=AXE,artist='Xie_Bin',work='Tan_Mei',boxes=[
  [90,300,210,420],[210,265,330,385],[305,350,425,470],[110,415,215,520]]),
 '233929':dict(label=AXE,artist='Zhou_Chen',work='Shan_Shui_Ren_Wu',boxes=[
  [580,125,700,245],[705,145,825,265],[490,155,590,255],[790,235,890,335]]),
 '234408':dict(label=AXE,artist='Lan_Ying',work='Yun_He_Cang_Yu',boxes=[
  [75,125,165,215],[110,220,200,310],[95,320,185,410],[35,620,125,710],[110,705,200,795],[165,805,255,895]]),
 '234074':dict(label=AXE,artist='Yuan_Jiang',work='Shan_Shui_Fan',boxes=[
  [665,200,765,300],[765,235,865,335],[825,315,925,415],[600,330,700,430]])}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,v):(ROOT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def make():
 (ROOT/'crops').mkdir(exist_ok=True)
 sources={r['id']:r for r in json.loads((ROOT/'museum_manifest.json').read_text('utf8'))}
 records=[]
 def add(identity,path,box,label,artist,work,publisher,url,evidence,scope,grade='source_evidenced_provisional',challenge=False):
  im=Image.open(path).convert('RGB');w,h=im.size;x0,y0,x1,y1=box
  assert 0<=x0<x1<=w and 0<=y0<y1<=h,(identity,box,(w,h))
  crop=im.crop(box);dest=ROOT/'crops'/(identity+'.png');crop.save(dest)
  records.append(dict(id=identity,image=str(dest),source_image=str(path),source_box=box,label=label,
   artist_group=artist,artist_basis='museum attribution or explicitly credited illustration',work_group=work,publisher_group=publisher,
   source_url=url,evidence=evidence,label_scope=scope,label_grade=grade,assistant_roi_review='pending_visual_check',
   expert_local_annotation=False,source_sha256=sha(path),crop_sha256=sha(dest),challenge_only=challenge,
   original_source_pixels=[x1-x0,y1-y0],is_augmentation=False,robot_data=False))
 for key,m in MUSEUM.items():
  src=sources[key];image=Path(src['images'][0]['path'])
  evidence=src['main_text']
  assert ('披麻皴' if m['label']==PIMA else '斧劈皴') in evidence
  for i,box in enumerate(m['boxes']):
   add(f'dpm_{key}_{i:02d}',image,box,m['label'],m['artist'],m['work'],'DPM',src['url'],evidence,
    'work-level museum description + assistant mountain ROI; local style not independently certified')
 # Museum teaching sheet credits all illustrations to Lampo Leong. All examples
 # and crops, including unknown-type probes, remain one artist group.
 aamurl='https://education.asianart.org/wp-content/uploads/sites/6/2019/09/Chinese-Brushstrokes-Vocabulary-AAM_0.pdf'
 aam=[('aam_2_img1.png',AXE,'small_axe'),('aam_2_img7.png',AXE,'large_axe'),('aam_2_img13.png',PIMA,'long_hemp'),('aam_2_img3.png',PIMA,'short_hemp')]
 for name,label,work in aam:
  path=ROOT/'sources'/name;im=Image.open(path);w,h=im.size
  # Four non-overlapping regions, each still carries multiple texture strokes.
  boxes=[[10,10,w//2-5,h//2-5],[w//2+5,10,w-10,h//2-5],[10,h//2+5,w//2-5,h-10],[w//2+5,h//2+5,w-10,h-10]]
  for i,box in enumerate(boxes):add(f'aam_{work}_{i}',path,box,label,'Lampo_Leong','aam_'+work,'AAM',aamurl,
   f'PDF page 2 labelled {work}; illustrations credited to Lampo Leong. Label text excluded from crop.',
   'explicit caption for this example; crop remains assistant-selected',grade='explicit_source_example')
 for name,work in [('aam_2_img9.png','raveled_rope'),('aam_2_img15.png','cloud_head'),('aam_2_img5.png','mi_dot'),('aam_2_img11.png','bands_dragged'),('aam_2_img17.png','nail_head')]:
  path=ROOT/'sources'/name;im=Image.open(path);w,h=im.size
  add('probe_'+work,path,[10,10,w-10,h-10],'其他／无法确认','Lampo_Leong','aam_'+work,'AAM',aamurl,
   f'PDF page 2 explicitly names another texture: {work}.','explicit other-type example; not a trained third class',grade='explicit_source_example',challenge=True)
 # Underlying PDF photograph excludes overlaid labels and white rectangles.
 # Mapping from the article's numbered rectangles, with a margin inside each.
 npmurl='https://www.npm.gov.tw/NewChineseArtDownload.ashx?bid=4360'
 path=ROOT/'sources/npm_article_13_img1.jpeg'
 for i,box in enumerate([[137,106,237,180],[240,106,339,180],[137,183,237,259],[240,183,339,259]]):
  add(f'npm_axe_{i}',path,box,AXE,'Li_Tang','Wan_He_Song_Feng','NPM',npmurl,
   'Printed p28, PDF page13, Fig17-1 rectangle1 labelled 斧劈皴; using underlying unlabelled image.','explicit labelled subregion',grade='explicit_source_region')
 for i,box in enumerate([[499,140,558,218],[499,221,558,300]]):
  add(f'npm_hemp_{i}',path,box,PIMA,'Li_Tang','Wan_He_Song_Feng','NPM',npmurl,
   'Printed p28, PDF page13, Fig17-1 rectangle3 labelled 披麻皴; using underlying unlabelled image.','explicit labelled subregion',grade='explicit_source_region')
 # Frozen visual admission before feature extraction/model fitting. Sources and
 # exclusions stay in the manifest; no prediction-dependent relabelling.
 exclusions={
  'dpm_234540_00':'too much blank paper; tiny distant rock',
  'dpm_234540_01':'mostly blank/outline; insufficient texture',
  'dpm_234540_02':'tree/grass and shallow bank dominate texture',
  'dpm_234540_06':'mostly blank paper and vegetation',
  'dpm_259387_02':'valley/vegetation and small edge of rock',
  'dpm_259387_03':'architecture and ambiguous broad dark region',
  'dpm_259387_04':'pine canopy dominates',
  'dpm_259387_05':'vegetation dominates',
  'dpm_259387_07':'foreground boulders/vegetation; texture unclear',
  'dpm_229314_01':'architecture and vegetation dominate',
  'dpm_229314_02':'architecture and vegetation dominate',
  'dpm_229314_03':'pine canopy dominates',
  'dpm_229314_04':'forest dominates',
  'dpm_229314_05':'tree trunks and forest dominate',
  'dpm_233992_02':'branches and little clear rock texture',
  'dpm_233992_03':'fan border/background, not rock',
  'dpm_233929_02':'vegetation dominates',
  'dpm_233929_03':'vegetation dominates',
  'dpm_234408_02':'vegetation dominates',
  'dpm_234408_03':'blank/vegetation dominates',
  'dpm_234408_05':'vegetation and unclear local texture',
  'dpm_234074_01':'outline/wash; insufficient internal texture',
  'dpm_234074_03':'vegetation dominates',
  'aam_small_axe_1':'mostly blank/wash; insufficient texture'}
 for r in records:
  r['assistant_roi_review']='visually reviewed source crop, not expert local style certification'
  r['admission']='unknown_probe' if r['challenge_only'] else 'excluded' if r['id'] in exclusions else 'pilot_only'
  r['exclusion_reason']=exclusions.get(r['id'])
  r['formal_expert_test_admitted']=False
 save('region_candidates.json',records)
 save('dataset_manifest.json',dict(records=records,scope='source-evidenced provisional pilot, not expert gold',
  label_scope_limit='Museum work descriptions + assistant local crops; only AAM/NPM examples have explicit local style captions',
  independent_expert_local_annotations=0,selection_frozen_before_training=True,synthetic_training_images=0,
  old_weak_candidates_used_for_training=False,robot_connected=False))
 save('excluded_sources.json',[
  dict(id='dpm_234585',reason='EDB lecture associates this work with hemp fibre, but museum describes dot texturing and almost no lines; label conflict, excluded from both training and scoring'),
  dict(id='edb_page10',reason='Actual PDF contains image placeholders, not texture examples; excluded despite searchable headings'),
  dict(id='legacy_30',reason='Thirty old candidates include reused steps, uncertain authors, possible folded-band/axe confusion and visible watermarks; audit only, excluded from this source-backed pilot'),
  dict(id='cbern',reason='Public request returned403; no data acquired; not bypassed')])
 font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
 for start in range(0,len(records),30):
  subset=records[start:start+30];canvas=Image.new('RGB',(1250,((len(subset)+4)//5)*210),'#f4f1e9');d=ImageDraw.Draw(canvas)
  for i,r in enumerate(subset):
   im=Image.open(r['image']);im.thumbnail((240,155));x=i%5*250;y=i//5*210;canvas.paste(im,(x,y+45));d.text((x+3,y+3),r['id'],font=font,fill='black');d.text((x+3,y+23),r['label']+' '+r['artist_group'],font=font,fill='black')
  canvas.save(ROOT/f'region_review_{start//30}.jpg')
 print('REGION_CANDIDATES',len(records),flush=True)
if __name__=='__main__':make()
