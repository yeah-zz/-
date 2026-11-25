# -*- coding: utf-8 -*-
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'cunfa_branch_isolation_20261117';OLD=LAB/'cunfa_boundary_relations_20261117'
def main():
 notes=[dict(id='nizan_1',outer=[133,100,245,190],inside=[145,126,265,218],note='内侧斜线可能是山面结构转折；不能仅因位于内部就当作皴笔负例。'),dict(id='nizan_0',outer=[123,82,236,151],inside=[216,141,340,238],note='外缘与内部短墨迹共存；短墨迹的皴笔／点苔归属待定。'),dict(id='qingpu_26_new_1',outer=[126,104,267,166],inside=[183,170,292,242],note='岩缘下方有草状墨迹；对象归属需独立核实，不能全部叫皴笔。')]
 board=Image.new('RGB',(1420,900),'#f3f0e7');font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n);dr=ImageDraw.Draw(board);dr.text((12,10),'邻近结构与内部墨迹复核：定位框为助手观察，不是模型输出',font=font(26),fill='#243f35')
 for i,n in enumerate(notes):
  src=OLD/'views'/(n['id']+'_near.png');im=Image.open(src).convert('RGB');n['image']=str(src);n['native_size']=im.size;anno=im.copy();pd=ImageDraw.Draw(anno);pd.rectangle(n['outer'],outline='#d75b31',width=2);pd.rectangle(n['inside'],outline='#2b71c9',width=2);scale=min(390/im.width,235/im.height);size=(round(im.width*scale),round(im.height*scale));y=65+i*270;board.paste(im.resize(size),(12,y));board.paste(anno.resize(size),(420,y));dr.text((850,y+10),n['id'],font=font(21),fill='#243f35')
  text=n['note'];dr.text((850,y+55),text[:20],font=font(21),fill='#805333');dr.text((850,y+95),text[20:],font=font(21),fill='#805333');dr.text((850,y+145),'待定区域不进入本轮监督。',font=font(20),fill='#99513b')
 (ROOT/'pair_review.json').write_text(json.dumps(dict(notes=notes,reviewed_after_predictions=True,not_new_independent_data=True,not_training_labels=True,automatic_predictions_not_truth=True),ensure_ascii=False,indent=2),encoding='utf8');board.save(ROOT/'pair_review.jpg',quality=94);print('PAIR_REVIEW',len(notes))
if __name__=='__main__':main()
