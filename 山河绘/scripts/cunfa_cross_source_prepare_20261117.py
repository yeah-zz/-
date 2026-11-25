# -*- coding: utf-8 -*-
"""New source groups, conservative local shape AND contextual-role references."""
import sys,json,hashlib,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from PIL import Image,ImageDraw,ImageFont
from gongbi_training.cunfa_concept_supervision import CONCEPTS,concept_targets,validate_concept_split
ROOT=LAB/'cunfa_cross_source_20261117';OLD=LAB/'cunfa_score_alignment_20261117'
ROLE_NAMES=['outline_context','interior_mark_context','diffuse_tone','unmarked_surface']

# Roles describe checked REGION CONTEXT, not every pixel/true artist contacts.
# None targets must remain unknown; captions do not become local labels.
QP={
 'qingpu_18_0':[
  ([40,92,160,140],[1,0,0],'outline_context','上部外缘折向线；非逐笔起止。','rock'),
  ([265,135,400,190],[1,0,0],'outline_context','中石上缘与外部留白交界，作为结构上下文参考。','rock'),
  ([440,255,550,340],[1,0,0],'outline_context','右侧小石外缘线，内部基本留白。','rock'),
  ([295,185,330,215],[None,None,None],'unmarked_surface','石内未着墨面，外观留白不等于对象外空景。','rock')],
 'qingpu_19_1':[
  ([60,40,156,148],[1,0,0],'outline_context','第一步石形以外轮廓为主。','rock'),
  ([242,76,283,124],[1,0,None],'interior_mark_context','加皴阶段石内新出现的短墨线；无逐笔或像素真值。','rock'),
  ([15,155,45,185],[None,None,None],'unmarked_surface','步骤图间未着墨背景，非从自动预测产生的负例。','paper')],
 'qingpu_20_0':[
  ([180,65,282,175],[1,0,1],'interior_mark_context','山体内部成组长曲带，与下方树木分开。','rock'),
  ([305,75,350,150],[1,0,None],'interior_mark_context','石内细长墨迹；交叠弧组数量待定。','rock'),
  ([480,180,555,285],[1,0,1],'interior_mark_context','右侧山面内部近邻长曲线。','rock'),
  ([128,36,180,145],[1,None,0],'outline_context','左上山体外缘，宽墨与淡染混合部分待定。','rock'),
  ([350,142,390,180],[None,None,None],None,'山体内部较空面仍夹有淡线，不能作为干净留白或完整墨带真值。','rock'),
  ([12,20,100,100],[None,None,None],'unmarked_surface','图例外的未着墨背景。','paper'),
  ([160,242,225,290],[None,None,None],None,'明确为树丛；仅作对象保护参考，不给山石形态或皴笔目标。','tree')],
 'qingpu_21_0':[
  ([235,110,285,240],[None,None,None],'interior_mark_context','独立石面内部墨迹，局部线／块形态混合，属性待定。','rock'),
  ([450,180,505,290],[1,None,0],'interior_mark_context','右侧内部可见延伸墨带，宽部角形与交叠归属待定。','rock'),
  ([335,150,410,200],[1,None,0],'outline_context','石面与中部留白交界的结构边。','rock')],
 'qingpu_22_0':[
  ([80,295,165,385],[1,0,1],None,'相邻弧带可见，结构边与皴线的具体归属不确定。','rock'),
  ([175,205,260,305],[1,0,1],None,'墨带成弧组；没有将图名直接当作局部角色。','rock')],
 'qingpu_23_0':[
  ([445,215,540,310],[0,0,0],'interior_mark_context','山面内部圆钝短墨形成组，不是长细线或弧组。','rock'),
  ([300,160,405,250],[None,None,0],'interior_mark_context','山面点状墨迹与淡墨叠加，线与块形态不够明确。','rock'),
  ([430,90,470,120],[None,None,None],None,'山脊上部墨色模糊，线形和角色均不作确定标签。','rock'),
  ([375,108,415,140],[None,None,None],'diffuse_tone','山面浅灰连续墨色过渡；不能确定单笔边界。','rock')],
 'qingpu_25_0':[
  ([198,105,278,180],[None,None,None],'diffuse_tone','远坡宽而柔的浅墨区域，无清楚单笔边界。','rock'),
  ([290,120,410,175],[None,None,None],'diffuse_tone','相同远坡另一段浅墨面；同图分组不拆到测试。','rock'),
  ([30,322,150,365],[None,None,None],'diffuse_tone','远坡淡灰连续区域，作为区域染墨外观参考。','rock'),
  ([465,325,510,345],[None,None,None],'diffuse_tone','淡灰面内部，不把微弱纸纹当完整线条。','rock'),
  ([140,540,300,590],[1,None,0],None,'近坡深墨与留白混合，结构／皴笔具体角色待定。','rock'),
  ([325,590,520,645],[1,None,0],None,'近坡长墨带可见，单笔与上下文角色待定。','rock'),
  ([20,200,190,240],[None,None,None],'unmarked_surface','文字与墨迹以外的未着墨间隔。','paper')]
}
NT={
 'ntm_huang':[
  ([170,130,365,360],[1,0,1],'interior_mark_context','山体内部多条可区分长曲墨带；未分单笔。','rock'),
  ([420,160,575,345],[1,0,1],'interior_mark_context','主山面内部成组弯曲长线。','rock'),
  ([170,120,285,290],[1,None,None],None,'主山左侧可见延伸墨带，外缘与内部曲带的具体归属待定。','rock'),
  ([445,328,485,365],[0,None,0],None,'孤立短墨点；不由作品皴法名推断其点苔或皴笔作用。','rock'),
  ([520,369,560,409],[0,None,0],None,'另一处短墨点，具体笔法作用未知。','rock'),
  ([35,75,110,175],[None,None,None],'unmarked_surface','作品中较干净的浅色空景，避免印章与标注。','paper'),
  ([33,385,85,435],[None,None,None],'diffuse_tone','左侧远山内部连续平缓灰墨面。','rock')],
 'ntm_mi':[
  ([180,270,330,395],[0,0,0],'interior_mark_context','山面密集圆钝墨点，属性不继承其他皴法图名。','rock'),
  ([500,170,655,310],[0,0,0],'interior_mark_context','右山点状墨迹组合，不是长曲带或宽角面。','rock'),
  ([650,45,780,100],[None,None,None],'unmarked_surface','右上浅色空景，未含题字或印章。','paper')],
 'ntm_xia':[
  ([1580,1100,1710,1490],[1,None,0],'outline_context','大崖右侧外缘与空景交界；未确定单笔。','rock'),
  ([250,1400,465,1750],[1,None,0],'interior_mark_context','左下崖面内延伸墨迹，宽角形与擦染归属待定。','rock'),
  ([1615,2705,1875,2815],[1,None,0],'outline_context','近景小石顶边与周围空景的结构关系；宽墨块归属待定。','rock'),
  ([1800,600,2000,750],[None,None,None],'unmarked_surface','右侧空景，避开树林、印章和显著叠字。','paper'),
  ([150,190,200,285],[None,None,None],'diffuse_tone','上方山面相对连续的灰色区域，仍非像素级墨迹真值。','rock')]
}

def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 (ROOT/'views').mkdir(exist_ok=True);assert not (ROOT/'protocol.json').exists(),'Frozen reference selection'
 ds=read(OLD/'reference_manifest.json');rows=copy.deepcopy(ds['records'])
 for r in rows:r['context_role_target']=None;r['role_reference_checked']=False
 inv={r['id']:r for r in read(ROOT/'source_image_inventory.json')};nt={r['id']:r for r in read(ROOT/'new_test_units.json')}
 for source,specs in [(QP,False),(NT,True)]:
  for unit,regions in source.items():
   if specs:
    u=nt[unit];path=Path(u['image']);url=u['source_url'];group=u['source_group'];pdfsha=u['source_pdf_sha256'];page=6
   else:
    u=inv[unit];path=Path(u['path']);url='https://m.qplll.net/File/Course/QP_COURSE_PDF/中国山水画法基础.pdf';group='Qingpu_book_illustrator_unverified';pdfsha=sha(ROOT/'sources/qingpu.pdf');page=u['page']
   image=Image.open(path).convert('RGB')
   for j,(box,values,role,note,obj) in enumerate(regions):
    assert 0<=box[0]<box[2]<=image.width and 0<=box[1]<box[3]<=image.height,(unit,box,image.size)
    ident=unit+'_region_'+str(j);dest=ROOT/'views'/(ident+'.png');image.crop(box).save(dest)
    r=dict(id=ident,image=str(dest),image_sha256=sha(dest),reference_image=str(path),reference_image_sha256=sha(path),reference_unit=unit,
     source_sha256=sha(path),source_group=group,source_url=url,source_pdf_sha256=pdfsha,source_pdf_page=page,
     source_evidence='Source captions and step text checked; region properties and role separately reviewed by assistant',
     box=box,split='test' if specs else 'train',object_type=obj,reference_kind='rock_surface_region' if obj=='rock' else 'object_region_reference',
     object_reference_checked=True,object_detector_input=False,label_scope='observable_region_morphology',
     concept_origin='visual_region_review',concepts=dict(zip(CONCEPTS,values)),visual_evidence=note,
     reviewer='assistant_context_review_20261117',concept_reference_checked=True,expert_verified=False,
     automatic_prediction_as_label=False,context_role_target=role,role_reference_checked=role is not None,
     role_reference_scope='region_context_not_pixel_or_stroke_truth',role='unknown',local_style_target=None,
     pixel_mask=None,complete_curve=None,stroke_order=None,brush_pressure=None,
     evaluation_scope='new_NT_source_probe' if specs else 'new_Qingpu_development',new_region_reference=True)
    r['targets']=concept_targets(r).tolist();rows.append(r)
 validate_concept_split(rows)
 # Source evidence cannot create a role label on leaves or an unknown object.
 for r in rows:
  if r['object_type']=='tree':assert r['context_role_target'] is None and all(v==-100 for v in r['targets'])
 # Named illustration units, not final-style labels on each chosen region.
 oldstyle=read(LAB/'cunfa_role_reference_20261112/source_references.json')['records']
 styles=[dict(id=r['id'],image=r['image'],reference_unit=r['id'],target=r['style_target'],split='train',source_url=r['source_url']) for r in oldstyle if r['publisher']=='AAM']
 for unit,target,box in [('qingpu_20_0',0,[45,0,600,320]),('qingpu_21_0',1,[140,50,580,380])]:
  im=Image.open(inv[unit]['path']);dest=ROOT/'units'/(unit+'_named.png');im.crop(box).save(dest)
  styles.append(dict(id=unit,image=str(dest),reference_unit=unit,target=target,split='train',source_url='https://m.qplll.net/File/Course/QP_COURSE_PDF/中国山水画法基础.pdf',label_scope='named_teaching_illustration_not_per_region',tree_intersections_not_given_local_style_targets=True))
 for unit in nt.values():
  styles.append(dict(id=unit['id'],image=unit['image'],reference_unit=unit['id'],target={'pima':0,'fupi':1}.get(unit['method_caption']),split='test',source_url=unit['source_url'],label_scope='caption_named_example',outside_two_class_scope=unit['method_caption'].startswith('mi_dot')))
 write('reference_manifest.json',dict(records=rows,old_manifest_sha256=sha(OLD/'reference_manifest.json'),
  role_names=ROLE_NAMES,style_units=styles,old48_targets_unchanged=True,new_expert_labels=0,
  source_split='AAM and Qingpu development / NTM source holdout; old HKMA and XueTang checks retained',
  Qingpu_book_compiler='Qiu_Kerong stated in preface; individual image illustrator not independently established',
  source_independence_not_proof_of_author_independence=True,complete_stroke_truth_added=0,robot_connected=False))
 font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
 new=[r for r in rows if r['evaluation_scope'] in {'new_Qingpu_development','new_NT_source_probe'}]
 board=Image.new('RGB',(1500,((len(new)+4)//5)*240+105),'#f3f0e7');d=ImageDraw.Draw(board)
 d.text((14,10),'新来源局部参考：属性、对象、角色分开，仍非专家或逐笔真值',font=font(26),fill='#243f35')
 d.text((14,55),'青浦教材仅用于开发；美术馆资料用于冻结检查。图名不会自动成为局部皴法标签。',font=font(20),fill='#805333')
 for i,r in enumerate(new):
  x=i%5*300+10;y=i//5*240+105;im=Image.open(r['image']);scale=min(270/im.width,140/im.height);im=im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))))
  board.paste(im,(x,y));d.text((x,y+150),r['id'],font=font(14),fill='#243f35')
  d.text((x,y+175),' / '.join('?' if v is None else str(v) for v in r['concepts'].values())+' · '+r['split'],font=font(18),fill='#805333')
  d.text((x,y+202),r['context_role_target'] or 'role unknown / '+r['object_type'],font=font(14),fill='#286a79')
 board.save(ROOT/'new_reference_board.jpg',quality=94)
 print(json.dumps(dict(total=len(rows),new=len(new),new_role_targets=sum(r['role_reference_checked'] for r in new),
  new_train=sum(r['split']=='train' for r in new),new_test=sum(r['split']=='test' for r in new)),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
