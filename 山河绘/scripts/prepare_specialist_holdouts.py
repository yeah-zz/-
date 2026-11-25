# -*- coding: utf-8 -*-
from simple_landscape_pairs import ROOT
from rock_v15_common import jsave,read,digest
from PIL import Image,ImageDraw,ImageFont
if __name__=='__main__':
    records=[];folder=ROOT/'imagegen_holdouts';folder.mkdir(exist_ok=True)
    for sheet in range(1,7):
        source=ROOT/'imagegen_sheets'/f'sheet_{sheet:02d}.png';im=Image.open(source).convert('RGB');w,h=im.size
        assert w%3==0 and h%2==0
        for row in range(2):
            for col in range(3):
                # Fixed grid split and fixed 8px gutter inset, no content-dependent crop.
                box=(col*w//3+8,row*h//2+8,(col+1)*w//3-8,(row+1)*h//2-8)
                sid=f'G{sheet:02d}_{row*3+col+1:02d}';dest=folder/(sid+'.png');im.crop(box).save(dest)
                records.append(dict(id=sid,image=str(dest),sha256=digest(dest),sheet=str(source),sheet_sha256=digest(source),group=sheet,crop_box=list(box),reference=None,split='unlabeled_imagegen_holdout',used_for_training=False,used_for_parameter_selection=False))
    jsave(ROOT/'imagegen_holdout_manifest.json',dict(records=records,scene_count=36,imagegen_call_count=6,scenes_within_sheet_correlated=True,not_36_independent_artworks=True,no_human_stroke_gold=True,source_prompts_sha256=digest(ROOT/'imagegen_prompts.json')))
    board=Image.new('RGB',(1488,6*280),'#f3f1eb');d=ImageDraw.Draw(board);f=ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc',17)
    for i,r in enumerate(records):
        x=i%6*248;y=i//6*280;d.text((x+3,y+2),r['id'],font=f,fill='black');v=Image.open(r['image']);v.thumbnail((242,242));board.paste(v,(x+3,y+29))
    board.save(ROOT/'IMAGEGEN_HOLDOUT_CONTACT.jpg',quality=94)
    print('HOLDOUTS_READY',len(records),flush=True)
