import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from PIL import Image
ROOT=LAB/'cunfa_cross_source_20261117'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 (ROOT/'units').mkdir(exist_ok=True)
 inventory=json.loads((ROOT/'source_image_inventory.json').read_text('utf8'))
 src=next(r for r in inventory if r['id']=='ntmofa_6_0');image=Image.open(src['path']).convert('RGB');rows=[]
 for ident,box,artist,work,method in [
  ('ntm_huang',[1055,325,1890,1415],'Huang_Gongwang','Fuchun_Mountains','pima'),
  ('ntm_mi',[1055,1500,1890,2020],'Mi_Fu','Chunshan_Ruisong','mi_dot_outside_two_class_scope'),
  ('ntm_xia',[2150,0,4299,3035],'Xia_Gui','Xishan_Qingyuan','fupi')]:
  dest=ROOT/'units'/(ident+'.png');image.crop(box).save(dest)
  rows.append(dict(id=ident,image=str(dest),sha256=sha(dest),source_image=src['path'],source_sha256=src['sha256'],
   source_box=box,source_pdf_sha256=sha(ROOT/'sources/ntmofa.pdf'),source_url='https://twfineartsarchive.ntmofa.gov.tw/QuarterlyFile/B000702.pdf',
   pdf_page=6,printed_page=36,artist=artist,work=work,source_group='ntmofa_'+artist+'_'+work,
   method_caption=method,local_region_method_not_inherited=True,split='test',
   source_caption_reviewed_by_assistant=True,expert_pixel_or_stroke_truth=False,
   publisher_watermarks_preserved=True,annotations_and_seals_not_used_as_morphology_labels=True))
 (ROOT/'new_test_units.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps([dict(id=r['id'],size=Image.open(r['image']).size,artist=r['artist']) for r in rows]),flush=True)
if __name__=='__main__':main()
