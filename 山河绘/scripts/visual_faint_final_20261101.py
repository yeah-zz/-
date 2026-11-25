from faint_diagnostics_20261101 import *
from PIL import Image,ImageDraw,ImageFont
import numpy as np
def main():
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17);boxes={'N01':(500,300,900,650),'N02':(600,660,1000,1010),'N03':(650,480,1050,830)};summary=Image.new('RGB',(1488,1650),'#f3f1eb');d=ImageDraw.Draw(summary)
    for i,ident in enumerate(['N01','N02','N03']):
        folder=ROOT/'fresh_outputs'/ident
        for j,(file,title) in enumerate([('input.png','新原图'),('previous.png','冻结的上轮流程'),('repaired.png','本轮淡线修正候选')]):im=Image.open(folder/file).convert('RGB');im.thumbnail((496,496),Image.Resampling.LANCZOS);d.text((j*496+4,i*550+5),ident+' '+title,font=font,fill='black');summary.paste(im,(j*496,i*550+45))
        box=boxes[ident];w=box[2]-box[0];h=box[3]-box[1];board=Image.new('RGB',(w*3,h+80),'#f3f1eb');dd=ImageDraw.Draw(board)
        for j,(file,title) in enumerate([('input.png','原图局部'),('previous.png','上轮同尺度'),('repaired.png','本轮候选')]):dd.text((j*w+4,8),ident+' '+title,font=font,fill='black');board.paste(Image.open(folder/file).convert('RGB').crop(box),(j*w,44))
        dd.text((4,h+49),'固定展示框仅用于同尺度查看，不参与推理或参数选择。',font=font,fill='#444444');board.save(folder/'fixed_detail.jpg',quality=97);jsave(folder/'display_roi.json',dict(box=list(box),used_for_inference=False,selection='Fixed same-scale visual display, not a label or accuracy proof'))
    summary.save(ROOT/'FRESH_COMPARE.jpg',quality=96)
    local=Image.new('RGB',(1536,860),'#f3f1eb');dd=ImageDraw.Draw(local)
    for i,key in enumerate(['T01_1','T16_3']):
        folder=ROOT/'local_repair'/key
        for j,(path,title) in enumerate([(ROOT/'diagnostics'/(key+'.png'),'原图'),(folder/'previous.png','上轮'),(folder/'repaired.png','本轮候选'),(folder/'changes.png','蓝新增 / 红替换')]):dd.text((j*384+4,i*430+5),key+' '+title,font=font,fill='black');local.paste(Image.open(path).convert('RGB').resize((384,384),Image.Resampling.NEAREST),(j*384,i*430+42))
    local.save(ROOT/'LOCAL_COMPARE.jpg',quality=97)
    print('FAINT_FINAL_VISUALS',flush=True)
if __name__=='__main__':main()
