import sys
sys.dont_write_bytecode=True
from mountain_structure_v4_20261101 import ROOT,LAB,run,CFG,read,jsave,digest
from pathlib import Path
import argparse,json,shutil
from PIL import Image,ImageDraw,ImageFont
def main():
    p=argparse.ArgumentParser();p.add_argument('--ids',default='G01_01,G02_01,G05_01');args=p.parse_args();ids=args.ids.split(',')
    lookup={r['id']:r for r in read(LAB/'v4_local_repairs_20261029/baseline_manifest.json')['records']}
    lookup['N02']=dict(source=str(LAB/'simple_landscape_specialist_20261029/round2/native_holdouts/N02.png'),group='native_N02',baseline=str(LAB/'simple_landscape_specialist_20261029/round2/native_outputs/N02/v4'))
    if args.ids=='all':ids=list(lookup)
    records=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    protocol=ROOT/('frozen_protocol_'+CFG['version']+'.json')
    if args.ids=='all':
        frozen=dict(config=CFG,source_code_sha256=digest(Path(__file__).with_name('mountain_structure_v4_20261101.py')),development_examples=['G01_01','G02_01','G05_01'],dataset='30 reused crops from five parent sheets plus accepted native N02; not blind test',new_training_steps=0)
        if protocol.exists():assert read(protocol)==frozen,'Do not silently change frozen protocol'
        else:jsave(protocol,frozen)
    for ident in ids:
        r=lookup[ident];base=Path(r['baseline']) if ident=='N02' else LAB/'v4_local_repairs_20261029/candidate'/ident;out=ROOT/'outputs'/ident
        stats=run(r['source'],base,out);records.append(dict(id=ident,source_group=r['group'],stats=stats))
        board=Image.new('RGB',(2000,445),'#f3f1eb');draw=ImageDraw.Draw(board)
        for j,(name,title) in enumerate([('input','原图'),('baseline','v4与短线补回对照'),('routing','蓝：山体内部；绿：细节保护'),('lines','主轮廓与少量长皴线'),('changes','红：转入待定；黑：保留')]):
            draw.text((j*400+4,5),ident+' '+title,font=font,fill='black');im=Image.open(out/(name+'.png')).convert('RGB');im.thumbnail((394,394));board.paste(im,(j*400,38))
        board.save(out/'comparison.jpg',quality=95);print('MOUNTAIN_STRUCTURE',ident,json.dumps(stats),flush=True)
    jsave(ROOT/('results_'+args.ids.replace(',','_')+'.json'),dict(records=records,config=CFG,code_sha256=digest(Path(__file__).with_name('mountain_structure_v4_20261101.py')),reference_accuracy=None,reused_images_not_blind_test=True,new_training_steps=0))
if __name__=='__main__':main()
