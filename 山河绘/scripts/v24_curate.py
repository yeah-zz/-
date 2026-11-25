# -*- coding: utf-8 -*-
from v24_sources import ROOT
from rock_v15_common import read,jsave,digest,FONT,SMALL
from PIL import Image,ImageDraw
import math

# A whole work/step sequence has one split; crop count is not artwork count.
CROPS={
 'U01':('three_faces_body_00',[.06,.03,.98,.91],'train','three_faces_single'),
 'U02':('three_faces_body_01',[0,.015,1,.96],'train','three_faces_shapes'),
 'U03':('three_faces_body_02',[0,.015,1,.94],'train','three_faces_facets'),
 'U04':('wang_xingtang_04',[.04,.02,.81,.97],'train','wang_linear_rock'),
 'U05':('zhuge_body_14',[.015,.17,.985,.98],'train','zhuge_small_fupi'),
 'U06':('cai_fupi_body_00',[0,0,1,1],'train','cai_small_fupi'),
 'C01':('zhuge_body_08',[.10,.30,.87,.96],'calibration','zhuge_large_fupi'),
 'H01':('zhuge_body_01',[.04,.25,.99,.98],'new_work_development','zhuge_basic_sequence'),
 'H02':('wang_xingtang_01',[0,.12,1,.98],'new_work_development','wang_step_sequence'),
}

def main():
    lookup={r['id']:r for r in read(ROOT/'source_manifest.json')['records']};out=[];(ROOT/'images').mkdir(exist_ok=True)
    for key,(sid,b,split,work) in CROPS.items():
        r=lookup[sid];im=Image.open(r['image']).convert('RGB');box=[round(v*(im.width if i%2==0 else im.height)) for i,v in enumerate(b)];im=im.crop(box);file=ROOT/'images'/(key+'.png');im.save(file);out.append(dict(id=key,image=str(file),image_sha256=digest(file),source_id=sid,source_image=r['image'],source_image_sha256=r['sha256'],source_page=r['source_page'],source_group=r['source_group'],work_group=work,crop_box=box,split=split,shape=[im.height,im.width]))
    jsave(ROOT/'selected.json',dict(records=out,split_before_predictions=True,not_expert_reference=True,all_steps_of_each_work_kept_together=True))
    for start in range(0,len(out),4):
        sub=out[start:start+4];board=Image.new('RGB',(1200,620*math.ceil(len(sub)/2)),'#f1f0e9');d=ImageDraw.Draw(board)
        for i,r in enumerate(sub):
            x=i%2*600+45;y=i//2*620+42;im=Image.open(r['image']).resize((500,500));board.paste(im,(x,y));d.text((x,y-30),r['id']+' '+r['source_id']+' '+r['split'],font=FONT,fill='black')
            for q in range(0,101,10):
                xx=x+q*5;yy=y+q*5;d.line((xx,y,xx,y+500),fill='#c2c2c2');d.line((x,yy,x+500,yy),fill='#c2c2c2');d.text((xx-5,y+504),str(q),font=SMALL,fill='black');d.text((x-24,yy-7),str(q),font=SMALL,fill='black')
        board.save(ROOT/('GRID_'+str(start//4)+'.jpg'),quality=97)
    print('V24_CURATED',len(out),flush=True)

if __name__=='__main__':main()
