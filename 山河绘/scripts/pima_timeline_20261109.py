import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import cv2
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'pima_demonstration_20261109'
def main():
 font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
 for lesson,start,end in [(36,260,786),(37,0,501)]:
  cap=cv2.VideoCapture(str(ROOT/'sources'/str(lesson)/'video.mp4'));times=list(range(start,end,20));board=Image.new('RGB',(1280,205*((len(times)+3)//4)),'#eeeae1');d=ImageDraw.Draw(board)
  folder=ROOT/'sources'/str(lesson)/'timeline';folder.mkdir(exist_ok=True)
  for i,t in enumerate(times):
   cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,bgr=cap.read()
   if not ok:continue
   im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));im.save(folder/f'{t:04d}.jpg',quality=95);im=im.resize((320,180));x=(i%4)*320;y=(i//4)*205;board.paste(im,(x,y+25));d.text((x+4,y+2),f'{lesson}课 {t}秒',font=font,fill='#222')
  cap.release();board.save(ROOT/f'timeline_{lesson}.jpg',quality=95)
if __name__=='__main__':main()
