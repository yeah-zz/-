# -*- coding: utf-8 -*-
"""Assistant visible-structure annotation; unknown stroke ownership stays null."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'cunfa_context_reference_20261111';SOURCE=LAB/'cunfa_structure_20261111/sources/228140_0.jpg'
BOX=[365,145,815,660]
LINES=[
 dict(id='C01',role='visible_outer_contour',points=[[400,251],[414,209],[447,174],[476,169],[514,177],[543,188],[584,218],[617,229],[649,244],[676,247],[709,269],[739,288],[763,319],[773,351],[772,381]]),
 dict(id='C02',role='visible_outer_contour',points=[[772,395],[781,429],[777,460],[759,490],[738,516],[718,529]]),
 dict(id='P01',role='visible_plane_boundary',points=[[594,341],[620,346],[653,354],[687,361],[718,365]]),
 dict(id='P02',role='visible_plane_boundary',points=[[716,369],[703,398],[690,425],[678,451],[664,479],[642,511],[617,547],[598,582],[580,614]])]
GROUPS=[
 dict(id='G01',role='cun_group_candidate',polygon=[[444,212],[505,195],[560,227],[573,262],[529,306],[479,300],[448,266]],interpretation='Upper dark angled brush areas: candidate common rock plane, individual stroke boundaries unknown.'),
 dict(id='G02',role='cun_group_candidate',polygon=[[418,289],[469,277],[524,318],[545,393],[514,444],[484,526],[446,536],[425,468],[407,392]],interpretation='Left-side elongated dark and dry-brush areas; candidate texture group, not expert certified.'),
 dict(id='U01',role='unknown_wash_or_cun',polygon=[[593,379],[688,378],[671,425],[642,470],[611,510],[578,549],[550,580]],interpretation='Faint marks on broad face cannot be separated confidently into cun versus wash; ignored for stroke labels.')]
def main():
 (ROOT/'reference').mkdir(exist_ok=True);src=Image.open(SOURCE).convert('RGB');base=src.crop(BOX);x0,y0,x1,y1=BOX;font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
 gray=np.array(src.convert('L')).astype('float32');bg=cv2.GaussianBlur(gray,(0,0),8);response=np.clip(bg-gray,0,None);response=cv2.dilate(response,np.ones((7,7),'uint8'))
 for line in LINES:
  p=np.asarray(line['points'],float);samples=[]
  for a,b in zip(p[:-1],p[1:]):samples.extend(np.linspace(a,b,max(2,int(np.linalg.norm(b-a)))).tolist())
  q=np.round(samples).astype(int);line['near_visible_dark_mark_fraction']=float(np.mean(response[q[:,1],q[:,0]]>8));line['reference_kind']='assistant visible boundary guide, not a recovered original contact';line['single_stroke_identity']=None;line['draw_direction']=None;line['expert_verified']=False;line['eligible_for_supervision']=False;line['review_status']='uncertain_boundary' if line['near_visible_dark_mark_fraction']<.8 else 'visible_mark_near_guide_not_identity_verified'
 for g in GROUPS:g.update(individual_strokes=None,stroke_ownership_mask=None,stroke_order=None,expert_verified=False,eligible_for_supervised_stroke_loss=False)
 artifact=dict(source_url='https://www.dpm.org.cn/collection/paint/228140.html',source_image=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),author='Wang_Lu',source_work='Hua_Shan_album',source_technique_label='small axe-cut, museum album-level description',
  object_id='right_main_rock_visible_surface',object_box=BOX,object_scope='Visible right rock mass; occluded boundaries and shared faces are not closed or completed.',
  outer_contours=[r for r in LINES if r['role']=='visible_outer_contour'],plane_boundaries=[r for r in LINES if r['role']=='visible_plane_boundary'],group_regions=GROUPS,
  topology=dict(phase_dependencies=[['visible_outer_contour','visible_plane_boundary'],['visible_plane_boundary','cun_group_candidate']],source_artist_actual_order_verified=False,interpretation='Proposed planning dependency for later validation, not recovered artist chronology'),
  verified_complete_strokes=0,expert_gold=False,used_as_training_target=False,unknown_is_not_negative=True,robot_parameters=dict(pressure=None,speed=None,ink_load=None),robot_connected=False)
 (ROOT/'object_reference.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2),encoding='utf8')
 base.save(ROOT/'reference/source.png');cont=base.copy();dc=ImageDraw.Draw(cont)
 for line in LINES:
  xy=[(p[0]-x0,p[1]-y0) for p in line['points']];dc.line(xy,fill='#916724' if line['role']=='visible_outer_contour' else '#30705b',width=3);dc.text(xy[0],line['id'],fill='#10291e',font=font(17))
 cont.save(ROOT/'reference/boundary_guides.png')
 layer=base.convert('RGBA');ov=Image.new('RGBA',base.size);dg=ImageDraw.Draw(ov)
 for g in GROUPS:
  xy=[(p[0]-x0,p[1]-y0) for p in g['polygon']];color=(30,115,145,60) if g['role']=='cun_group_candidate' else (160,125,70,65);dg.polygon(xy,fill=color,outline=color[:3]+(230,));dg.text(xy[0],g['id'],fill=(0,40,60,255),font=font(19))
 Image.alpha_composite(layer,ov).convert('RGB').save(ROOT/'reference/group_regions.png')
 # Preserve source ink-band shapes instead of replacing them with thin lines.
 # These are source crops, never generated or executed brush strokes.
 group_source=base.copy();group_source.save(ROOT/'reference/original_ink_bands.png')
 board=Image.new('RGB',(1380,680),'#f4f1e8');dd=ImageDraw.Draw(board);dd.text((20,14),'一块山石的分层参考：助手标记，非模型输出 / 非画家真实分笔',font=font(25),fill='#233e32')
 for i,(file,title) in enumerate([('source.png','原作：保留真实墨带'),('boundary_guides.png','可见外轮廓 / 山面转折引导'),('group_regions.png','G：笔组候选区；U：待定')]):
  im=Image.open(ROOT/'reference'/file);im.thumbnail((430,500));xx=i*460+12;board.paste(im,(xx,95));dd.text((xx,58),title,font=font(21),fill='#233e32')
 dd.text((20,612),'边界引导不等于单笔轨迹。笔组成员、逐笔起止、真实笔顺和下压量均未确认。',font=font(21),fill='#934330')
 dd.text((20,646),'这份参考未参与分类器训练；不把它作为机械臂绘画效果或监督真值。',font=font(20),fill='#40594a');board.save(ROOT/'object_reference.png')
 print('REFERENCE_READY',[(r['id'],round(r['near_visible_dark_mark_fraction'],3)) for r in LINES],flush=True)
if __name__=='__main__':main()
