"""Independent image entry; existing stable drawing launchers remain untouched."""
import sys
sys.dont_write_bytecode=True
from mountain_structure_v4_20261101 import ROOT,run,read,jsave,v4
from v4_local_repair_candidates import recover,CFG as SHORT_CFG
from pathlib import Path
from PIL import Image
import numpy as np,argparse,datetime

def extract(image,output):
    image=Path(image).resolve();output=Path(output)
    if not output.is_absolute() or output.drive.upper() not in ['E:','F:']:raise ValueError('Use absolute E/F output')
    if (output/'structure/paths.json').exists():raise FileExistsError('Preserve completed extraction')
    v4.run(image,output/'v4_baseline')
    rgb=np.asarray(Image.open(output/'v4_baseline/input.png').convert('RGB'));raw=np.asarray(Image.open(output/'v4_baseline/raw_skeleton.png').convert('L'))<128;doc=read(output/'v4_baseline/paths.json');added,_,_=recover(rgb,doc['paths'],raw)
    doc['paths']+=added;doc['short_recovery_config']=SHORT_CFG;doc['ready_for_robot']=False;jsave(output/'detail_baseline/paths.json',doc)
    stats=run(output/'v4_baseline/input.png',output/'detail_baseline',output/'structure');jsave(output/'entry.json',dict(image=str(image),output=str(output),steps=['frozen v4','source-supported short-detail stage','mountain-interior subset selection'],new_training_steps=0,robot_connected=False,statistics=stats))
    print('MOUNTAIN_STRUCTURE_READY',str(output/'structure/lines.png'),flush=True)
    return stats
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('image');parser.add_argument('--output');a=parser.parse_args();out=Path(a.output) if a.output else ROOT/'manual_outputs'/(Path(a.image).stem+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));extract(a.image,out)
