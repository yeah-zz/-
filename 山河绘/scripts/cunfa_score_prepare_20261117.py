# -*- coding: utf-8 -*-
"""Freeze new shape references and comparisons before seeing their predictions."""
import sys,json,hashlib,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from PIL import Image,ImageDraw,ImageFont
from gongbi_training.cunfa_concept_supervision import CONCEPTS,concept_targets,validate_concept_split
ROOT=LAB/'cunfa_score_alignment_20261117';OLD=LAB/'cunfa_concept_supervision_20261116'
BASE=LAB/'cunfa_gabor_baseline_20261110';REF=LAB/'cunfa_role_reference_20261112'
URL='https://education.asianart.org/wp-content/uploads/sites/6/2019/09/Chinese-Brushstrokes-Vocabulary-AAM_0.pdf'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf8'))
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')

SPECS={
 'aam_extra_5':('aam_2_img5.png',[
  ([15,290,250,490],[0,0,0],'左下可见圆钝、横向短墨形，未见独立细长带或弯曲排线。'),
  ([335,20,590,220],[0,0,0],'中上以圆钝宽墨面为主，非角状楔面；未确认单笔。'),
  ([485,300,725,490],[None,0,0],'右下圆钝与横向墨形层叠，长线属性待定，不把叠墨边缘强行归类。'),
  ([145,240,310,355],[0,0,0],'横向短椭圆墨形；不由原图皴法名称产生形态标签。')]),
 'aam_extra_9':('aam_2_img9.png',[
  ([10,35,160,260],[1,0,None],'左上延伸墨带可见，断续与转向归组不确定。'),
  ([335,25,635,310],[1,0,0],'山石顶端和山面可见折向长边线，不是宽角墨面。'),
  ([250,300,550,500],[1,0,None],'下方长曲边及局部短墨线；是否同一弧组待定。'),
  ([480,230,730,495],[1,0,1],'右侧近邻墨带成组弯曲，角色与笔顺未知。')]),
 'aam_extra_11':('aam_2_img11.png',[
  ([115,60,590,205],[1,None,0],'上缘有长墨线，粗墨与染墨的宽部归属不确定。'),
  ([30,235,380,475],[None,None,0],'湿墨与擦笔混杂，不为线形或角块强行定标签。'),
  ([430,210,730,410],[1,None,0],'右部细长近直墨带可见，宽墨形态待定。')]),
 'aam_extra_15':('aam_2_img15.png',[
  ([10,80,295,285],[1,0,1],'左侧多条嵌套弧向墨带。'),
  ([285,10,620,180],[1,0,1],'上部墨带依相近弧向排列，不能由此推断笔顺。'),
  ([355,160,705,420],[1,0,1],'右部多条连续曲带成组，仍不提供完整分笔。'),
  ([20,280,300,485],[1,0,1],'左下邻近弯曲墨带，保留可见间隙。')]),
 'aam_extra_17':('aam_2_img17.png',[
  ([105,250,340,470],[1,1,0],'可见角状宽笔头与细长拖尾同时存在。'),
  ([435,100,630,345],[1,1,0],'右侧宽角笔头与细长墨带并存。'),
  ([240,10,580,200],[1,None,0],'顶缘延伸折线可见，但框内也有宽部交叠，角块属性待定。'),
  ([58,340,112,385],[None,None,0],'浅灰墨迹被裁切、形态不充分，线形与角块均保持待定。')])}

NATIVE_SPECS=[
 ('fresh_230757_0',[30,70,64,106],[0,0,0],'小圆钝墨点；原作局部皴法和点苔作用均未据此确定。'),
 ('fresh_230757_0',[184,279,239,322],[0,0,0],'孤立圆钝墨形及底纹，不作为细长线或弧形笔组。'),
 ('fresh_230757_3',[120,17,163,56],[0,0,0],'圆钝墨点，形态负例不等于背景或非墨迹。'),
 ('fresh_230757_0',[55,260,135,435],[1,None,0],'下方延伸近直墨带可见，形态角色及宽部待定。')]

def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'views').mkdir(exist_ok=True)
 assert not (ROOT/'protocol.json').exists(),'Frozen before fitting'
 ds=read(OLD/'reference_manifest_corrected.json');rows=copy.deepcopy(ds['records']);previous_ids={r['id'] for r in rows}
 for r in rows:
  r['evaluation_scope']='known_HKMA_regression' if r['split']=='test' else 'original_training'
  r['new_region_reference']=False
 # New regions have NO final pima/axe labels. All five other teaching images
 # remain one source/illustrator family; never called five independent authors.
 def add(unit,path,boxes,split,group,url,kind,evalscope):
  im=Image.open(path).convert('RGB')
  for j,(box,values,note) in enumerate(boxes):
   assert 0<=box[0]<box[2]<=im.width and 0<=box[1]<box[3]<=im.height,(unit,box,im.size)
   ident=unit+'_shape_'+str(j);dest=ROOT/'views'/(ident+'.png');im.crop(box).save(dest)
   r=dict(id=ident,image=str(dest),image_sha256=sha(dest),reference_image=str(path),reference_image_sha256=sha(path),
    reference_unit=unit,source_sha256=sha(path),source_group=group,source_url=url,
    source_evidence=kind,box=box,split=split,object_type='rock',reference_kind='rock_surface_region',
    object_reference_checked=True,object_detector_input=False,label_scope='observable_region_morphology',
    concept_origin='visual_region_review',concepts=dict(zip(CONCEPTS,values)),visual_evidence=note,
    reviewer='assistant_visual_review_20261117',concept_reference_checked=True,expert_verified=False,
    automatic_prediction_as_label=False,role='unknown',local_style_target=None,pixel_mask=None,
    complete_curve=None,stroke_instance=None,stroke_order=None,brush_pressure=None,
    evaluation_scope=evalscope,new_region_reference=True,reference_is_not_pixel_truth=True)
   r['targets']=concept_targets(r).tolist();rows.append(r)
 for unit,(file,boxes) in SPECS.items():
  add(unit,BASE/'sources'/file,boxes,'train','Lampo_Leong_all_examples',URL,
   'Existing AAM teaching-sheet image; morphology reviewed locally; no pima/axe target inferred from its caption','new_training_counterexamples')
 native={r['id']:r for r in read(REF/'native_manifest.json')['records']}
 grouped={}
 for ident,box,values,note in NATIVE_SPECS:grouped.setdefault(ident,[]).append((box,values,note))
 for ident,boxes in grouped.items():
  r=native[ident];assert sha(r['native_image'])==r['native_sha256']
  add('xia_native_'+ident[-1],r['native_image'],boxes,'test','Xia_Gui_230757_whole_work',r['source_url'],
   'Previously reviewed work and native source assets; newly fixed shape references, NOT a new independent work','known_work_new_shape_probe')
 validate_concept_split(rows)
 assert {r['id'] for r in rows if not r['new_region_reference']}==previous_ids
 for old in ds['records']:
  new=next(r for r in rows if r['id']==old['id']);assert old['targets']==new['targets'] and old['box']==new['box'] and old['split']==new['split']
 write('reference_manifest.json',dict(records=rows,old_reference_sha256=sha(OLD/'reference_manifest_corrected.json'),
  original_25_targets_unchanged=True,new_final_style_targets=0,new_expert_labels=0,new_works=0,
  training_illustrator_families=1,new_training_illustration_units=5,known_HKMA_test_source_photos=1,
  known_Xia_Gui_test_work_groups=1,full_stroke_truth_added=0,robot_connected=False))
 font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n);new=[r for r in rows if r['new_region_reference']]
 board=Image.new('RGB',(1500,((len(new)+4)//5)*245+100),'#f3f0e7');d=ImageDraw.Draw(board)
 d.text((14,8),'新增形态参考：先固定观察，再看模型预测',font=font(27),fill='#243f35')
 d.text((14,52),'复用旧图例与已知作品，不是新增独立作者；下方数值是助手参考，不是模型输出。',font=font(20),fill='#805333')
 for i,r in enumerate(new):
  x=i%5*300+10;y=i//5*245+103;im=Image.open(r['image']);scale=min(275/im.width,150/im.height);im=im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))))
  board.paste(im,(x,y));d.text((x,y+158),r['id'],font=font(15),fill='#243f35')
  d.text((x,y+184),' / '.join('?' if v is None else str(v) for v in r['concepts'].values())+' · '+r['split'],font=font(18),fill='#805333')
 board.save(ROOT/'new_references.jpg',quality=94)
 print(json.dumps(dict(train=sum(r['split']=='train' for r in rows),test=sum(r['split']=='test' for r in rows),new_regions=len(new),train_units=len({r['reference_unit'] for r in rows if r['split']=='train'})),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
