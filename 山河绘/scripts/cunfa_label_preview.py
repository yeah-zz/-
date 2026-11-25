import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure
configure()
from gongbi_training.cunfa_prepare import ink_target,CORPUS
from PIL import Image,ImageDraw,ImageFont
import numpy as np
names=['ref_02_00','ref_12_00','ref_11_04','ref_16_01']
out=Image.new('RGB',(1200,len(names)*300),'#f3f3ef');draw=ImageDraw.Draw(out)
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
for i,name in enumerate(names):
    im=Image.open(CORPUS/'crops'/(name+'.png')).convert('RGB');t,w=ink_target(np.asarray(im))
    for j,img in enumerate((im,Image.fromarray(255-t[1]).convert('RGB'))):
        img.thumbnail((560,265));out.paste(img,(j*600+(560-img.width)//2,30+i*300))
    draw.text((5,i*300),name+' / 自动弱标签检查，非模型预测',font=font,fill='black')
out.save(CORPUS/'review'/'label_revision.jpg',quality=93)
print(CORPUS/'review'/'label_revision.jpg')
