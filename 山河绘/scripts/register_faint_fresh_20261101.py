from faint_diagnostics_20261101 import *
from extract_faint_joint_20261101 import verify_round
import argparse,shutil,re
GENERATED=Path('E:/Codex_Workspace_Data/CLIPascene_20261021/generated_images/01a15e88-26fa-7a12-9622-3be075148c5d')
def main():
    p=argparse.ArgumentParser();p.add_argument('--id',required=True);p.add_argument('--filename',required=True);a=p.parse_args();assert a.id in ['N01','N02','N03'];assert re.fullmatch(r'exec-[a-f0-9-]+\.png',a.filename);verify_round();source=(GENERATED/a.filename).resolve();assert source.parent==GENERATED.resolve();folder=ROOT/'fresh_inputs';folder.mkdir(exist_ok=True);destination=folder/(a.id+'.png')
    if destination.exists():assert digest(destination)==digest(source)
    else:shutil.copy2(source,destination)
    with Image.open(destination) as im:size=list(im.size)
    record=dict(id=a.id,path=str(destination),source_generated_file=str(source),sha256=digest(source),size=size,independent_generation=True,used_for_training=False,tuned_on_this_image=False,selective_regeneration=False,experiment_lock_sha256=digest(ROOT/'frozen_experiment.json'));jsave(folder/(a.id+'.json'),record);print('FRESH_REGISTERED',a.id,size,flush=True)
if __name__=='__main__':main()
