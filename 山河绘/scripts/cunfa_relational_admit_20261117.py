import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'cunfa_relational_roles_20261117'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not (ROOT/'protocol.json').exists(),'Frozen training selection'
 d=json.loads((ROOT/'preliminary_manifest.json').read_text('utf8'));rows=d['records']
 # Inspection revealed actual ink in the proposed paper crop: retain unknown,
 # never train it as blank. This admission is fixed before model predictions.
 for r in rows:
  if r['id']=='T_cun_group1_new_3':
   r['context_role_target']=None;r['role_reference_checked']=False
   r['visual_evidence']='原拟留白框中实际夹有斜向墨迹；复核后改为待定，不参与角色损失。'
 image=Image.open(d['cliff_image']).convert('RGB')
 specs=[
  ([110,135,160,255],'outline_context','高石坡长边与外部留白相接。'),
  ([125,115,280,215],'outline_context','高坡台面外缘与邻近山面，保留完整转向上下文。'),
  ([235,615,480,675],'outline_context','低石坡上部延伸轮廓与外部留白。'),
  ([370,700,520,760],'outline_context','低坡前缘，附近附有点草，非框内每像素均为结构。'),
  ([153,285,178,330],None,'高坡浅灰区仍有细长墨迹，淡染与淡皴不能仅凭浓淡区分，保持待定。'),
  ([270,690,335,725],None,'低坡浅墨面夹有竖向纹理，暂不作为纯连续淡墨目标。'),
  ([390,800,520,825],'diffuse_tone','低坡下方连续灰墨带，无逐笔或物理浓度真值。'),
  ([350,450,450,520],'unmarked_surface','两坡之间的未着墨纸面，避开题字。'),
  ([30,610,125,675],'unmarked_surface','左下未着墨面，纸色与坡面来自同一照片。'),
  ([120,395,160,455],None,'高坡浅墨处仍夹细线，染墨与笔迹角色待定。')]
 for j,(box,role,note) in enumerate(specs):
  ident='qingpu_26_new_'+str(j);dest=ROOT/'views'/(ident+'.png');image.crop(box).save(dest)
  rows.append(dict(id=ident,image=str(dest),image_sha256=sha(dest),reference_image=d['cliff_image'],reference_image_sha256=sha(d['cliff_image']),
   reference_unit='qingpu_26',role_fold_group='qingpu_26',source_group='Qingpu_book_illustrator_unverified',
   source_url='https://m.qplll.net/File/Course/QP_COURSE_PDF/中国山水画法基础.pdf',source_pdf_sha256=d['cliff_source_pdf_sha256'],source_pdf_page=26,
   source_evidence='Existing book cliff-slope diagram; local role reviewed separately from caption',
   box=box,split='train',object_type='paper' if role=='unmarked_surface' else 'rock',object_reference_checked=True,
   context_role_target=role,role_reference_checked=role is not None,role_reference_scope='region_context_not_pixel_or_stroke_truth',
   visual_evidence=note,reviewer='assistant_cliff_context_review_20261117',expert_verified=False,
   automatic_prediction_as_label=False,evaluation_scope='same_book_cliff_development',added_this_round=True,
   local_style_target=None,complete_curve=None,stroke_order=None,pressure=None))
 out=dict(records=rows,source_manifest_sha256=sha(ROOT/'preliminary_manifest.json'),
  added_role_regions=sum(r['added_this_round'] and r['role_reference_checked'] for r in rows),
  reused_sources_not_new_independent_works=True,all_video_phases_share_one_fold_group=True,
  uncertain_regions_not_negative=True,role_scope='given-region context only',robot_connected=False)
 (ROOT/'reference_manifest.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
 font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n);new=[r for r in rows if r['added_this_round']]
 board=Image.new('RGB',(1500,((len(new)+4)//5)*260+100),'#f3f0e7');dr=ImageDraw.Draw(board)
 dr.text((15,10),'补充结构与淡墨参考：同一来源合组，夹有墨迹的留白候选改为待定',font=font(25),fill='#243f35')
 for i,r in enumerate(new):
  x=i%5*300+10;y=i//5*260+85;im=Image.open(r['image']);s=min(270/im.width,155/im.height);im=im.resize((max(1,round(im.width*s)),max(1,round(im.height*s))))
  board.paste(im,(x,y));dr.text((x,y+165),r['id'],font=font(15),fill='#243f35');dr.text((x,y+195),r['context_role_target'] or 'unknown / excluded',font=font(16),fill='#805333')
 board.save(ROOT/'added_references.jpg',quality=94)
 print(json.dumps(dict(total=len(rows),added_views=len(new),added_role_targets=out['added_role_regions'])),flush=True)
if __name__=='__main__':main()
