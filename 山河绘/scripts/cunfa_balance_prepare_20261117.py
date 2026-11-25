# -*- coding: utf-8 -*-
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from PIL import Image,ImageDraw,ImageFont
from gongbi_training.cunfa_boundary_relations import canvas_points
from gongbi_training.cunfa_context_roles import context_box
ROOT=LAB/'cunfa_boundary_balance_20261117';OLD=LAB/'cunfa_boundary_relations_20261117'
def main():
 assert not (ROOT/'reference_manifest.json').exists();d=json.loads((OLD/'reference_manifest.json').read_text('utf8'));rows=d['records'];points={
 'wanghui_2':[(.30,.30),(.60,.42),(.65,.75)],'gongxian_2':[(.65,.20),(.70,.50),(.45,.70)],
 'gongxian_4':[(.25,.30),(.50,.50),(.75,.70)],'gongxian_5':[(.25,.30),(.50,.50),(.75,.70)],
 'gongxian_6':[(.25,.30),(.50,.50),(.75,.70)],'gongxian_7':[(.25,.30),(.50,.50),(.75,.70)]}
 negatives=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17);board=Image.new('RGB',(1200,590),'#f3f0e7');dr=ImageDraw.Draw(board)
 for i,(ident,ps) in enumerate(points.items()):
  row=next(r for r in rows if r['id']==ident);assert row['split']=='train';x0,y0,x1,y1=row['box'];p=[[round(x0+x*(x1-x0)),round(y0+y*(y1-y0))] for x,y in ps]
  a=dict(id=ident,line=[],rock=[],open=[],negative_points=p,split='train',group=row['role_fold_group'],source_image=row['reference_image'],source_url=row['source_url'],scope='assistant checked non-contour points; neither non-ink labels nor paper labels; no entire-window targets',checked=True,negative_kind=row['context_role_target']);negatives.append(a)
  im=Image.open(row['image']).convert('RGB');s=min(370/im.width,200/im.height);im=im.resize((round(im.width*s),round(im.height*s)));pd=ImageDraw.Draw(im)
  for x,y in p:
   xx=(x-x0)*s;yy=(y-y0)*s;pd.ellipse((xx-3,yy-3,xx+3,yy+3),fill='#db4136')
  xx=(i%3)*400+8;yy=(i//3)*295+5;board.paste(im,(xx,yy));dr.text((xx,yy+210),ident,font=font,fill='#243f35');dr.text((xx,yy+239),a['negative_kind'],font=font,fill='#805333')
 out=dict(records=rows,curves=d['curves'],negative_references=negatives,old_manifest_sha256=hashlib.sha256((OLD/'reference_manifest.json').read_bytes()).hexdigest(),reference_scope='Same146 existing cases, all checks previously opened;18 new train non-contour points in6 windows, no new works or expert labels',targets_frozen_before_fit=True,production_replaced=False,robot_connected=False)
 ROOT.mkdir(exist_ok=True);(ROOT/'reference_manifest.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');board.save(ROOT/'negative_references.jpg');print('NEGATIVE_WINDOWS',len(negatives),'POINTS',sum(len(a['negative_points']) for a in negatives))
if __name__=='__main__':main()
