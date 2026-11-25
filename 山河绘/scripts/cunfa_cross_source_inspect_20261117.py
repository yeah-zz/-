import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
sys.path.insert(0,str(LAB/'guohua_system_review_20261110/deps'))
import pymupdf
ROOT=LAB/'cunfa_cross_source_20261117'
def main():
 (ROOT/'original_images').mkdir(exist_ok=True);records=[]
 for ident,numbers in [('qingpu',[13,15,18,19,20,21,22,23,25]),('ntmofa',[6])]:
  doc=pymupdf.open(ROOT/'sources'/(ident+'.pdf'))
  for number in numbers:
   page=doc[number-1];page.get_pixmap(matrix=pymupdf.Matrix(2,2)).save(ROOT/'pages'/(ident+'_'+str(number)+'_large.png'))
   for j,entry in enumerate(page.get_images(full=True)):
    xref=entry[0];data=doc.extract_image(xref)
    if data['width']<100 or data['height']<50:continue
    dest=ROOT/'original_images'/(ident+'_'+str(number)+'_'+str(j)+'.'+data['ext']);dest.write_bytes(data['image'])
    records.append(dict(id=dest.stem,path=str(dest),pdf=ident,page=number,xref=xref,size=[data['width'],data['height']],
     sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),page_rects=[list(r) for r in page.get_image_rects(xref)]))
 (ROOT/'source_image_inventory.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps(records,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
