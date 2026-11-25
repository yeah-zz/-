"""Freeze existing generated-image v4 outputs; diagnostic gallery, no training labels."""
import sys,json,hashlib,shutil
from pathlib import Path
sys.dont_write_bytecode=True
from PIL import Image,ImageDraw,ImageFont
LAB=Path(r'E:\Codex_Workspace_Data\Gongbi_Line_Model_20261022')
R2=LAB/'simple_landscape_specialist_20261029/round2'
OUT=LAB/'v4_local_repairs_20261029'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    OUT.mkdir(exist_ok=True)
    records=json.loads((R2/'imagegen_holdout_manifest.json').read_text(encoding='utf-8'))['records']
    frozen=[]
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    for group in range(1,6):
        canvas=Image.new('RGB',(1500,1040),'#f3f1eb');d=ImageDraw.Draw(canvas)
        for i,r in enumerate([r for r in records if r['group']==group]):
            src=R2/'outputs'/r['id']/'v4';target=OUT/'baseline'/r['id'];target.mkdir(parents=True,exist_ok=True)
            hashes={}
            for name in ['input.png','lines.png','paths.json','statistics.json','raw_skeleton.png','ink.png','dense_regions.png','pending.png']:
                s=src/name;t=target/name
                if not t.exists():shutil.copy2(s,t)
                assert sha(s)==sha(t)
                hashes[name]=sha(t)
            assert sha(target/'input.png')==r['sha256']
            split={1:'development',2:'development',3:'development',4:'calibration',5:'locked_reused_validation'}[group]
            frozen.append(dict(id=r['id'],group=group,split=split,source=r['image'],baseline=str(target),hashes=hashes))
            x=(i%3)*500;y=(i//3)*520
            d.text((x+8,y+4),r['id']+' 原图 / v4',font=font,fill='black')
            for j,name in enumerate(['input.png','lines.png']):
                im=Image.open(target/name).convert('RGB');im.thumbnail((248,480))
                canvas.paste(im,(x+j*250,y+30))
        canvas.save(OUT/f'audit_sheet_{group:02}.jpg',quality=92)
    dump(OUT/'baseline_manifest.json',dict(records=frozen,scope='30 existing generated crops from five parent sheets; NOT thirty independent works or a fresh blind test',split_rule='whole parent sheet separated; group 5 reserved from this repair tuning',reference_labels=None,baseline_source='split_detail_v4_frozen.py',baseline_code_sha256=sha(Path(__file__).with_name('split_detail_v4_frozen.py'))))
    print(json.dumps(dict(frozen=len(frozen),output=str(OUT)),ensure_ascii=False))
if __name__=='__main__':main()
