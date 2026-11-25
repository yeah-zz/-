from faint_joint_model_20261101 import *
from pathlib import Path
EXTRA_NEGATIVES={'T01_1':[[118,150,133,175]],'T06_1':[[30,110,48,164]],'T19_3':[[42,45,78,110]]}
def main():
    original=read(ROOT/'paired_manifest.json');folder=ROOT/'paired_data_r2';folder.mkdir(exist_ok=True);groups={}
    for kind,rows in [('synthetic',original['synthetic']),('real',original['real'])]:
        new=[]
        for r in rows:
            z=np.load(r['file']);rgb=z['rgb'];t=z['target'].astype(np.float32);k=z['known'].astype(np.float32)
            if kind=='synthetic':t[1]=(t[1]>.075).astype(np.float32)
            else:
                for x,y,xx,yy in EXTRA_NEGATIVES.get(r['id'],[]):t[:,y:yy,x:xx]=0;k[0,y:yy,x:xx]=1;k[1,y:yy,x:xx]=1;k[2,y:yy,x:xx]=0;k[3,y:yy,x:xx]=1
            dest=folder/Path(r['file']).name;np.savez_compressed(dest,rgb=rgb,target=t.astype(np.float16),known=k.astype(np.float16));assert np.array_equal(np.load(dest)['rgb'],rgb);new.append(dict(**{k:v for k,v in r.items() if k not in ['file','sha256']},file=str(dest),sha256=digest(dest),parent_data_file=r['file'],rgb_unchanged=True))
        groups[kind]=new
    jsave(ROOT/'paired_manifest_r2.json',dict(**groups,parent_manifest_sha256=digest(ROOT/'paired_manifest.json'),correction='Synthetic ink head is binary visible-ink occupancy, not scaled ink density. Darkness is not an extraction-confidence target.',extra_real_negative_boxes=EXTRA_NEGATIVES,extra_negatives_are_manually_reviewed_not_crown_or_stroke_gold=True,real_center_positive_supervision='Neighborhood presence only; exact centers remain synthetic vector supervision',all_rgb_unchanged=True,original_data_preserved=True));print('FAINT_R2_PAIRS_READY',flush=True)
if __name__=='__main__':main()
