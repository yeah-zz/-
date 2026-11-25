# -*- coding: utf-8 -*-
"""Arbitrary-image offline entry; completeness and sparse view saved separately."""
import sys
sys.dont_write_bytecode=True
from mountain_complete_lines_20261101 import *
from extract_mainline_recovery_20261101 import extract as prepare_previous
from extract_mainline_recovery_20261101 import describe
import datetime

def extract(image,output):
    image=Path(image).resolve();output=Path(output)
    if not output.is_absolute() or output.drive.upper() not in ['E:','F:']:raise ValueError('Use absolute E/F output')
    if (output/'complete/complete_paths.json').exists():raise FileExistsError('Do not overwrite completed extraction')
    prepare_previous(image,output/'upstream');source=output/'upstream/upstream/structure';rgb=np.asarray(Image.open(source/'input.png').convert('RGB'));base=np.asarray(Image.open(source/'baseline.png'))<128;previous=np.asarray(Image.open(output/'upstream/recovery/lines.png'))<128;r7=np.asarray(Image.open(source/'lines.png'))<128;z=np.load(source/'routing_masks.npz');routing={k:z[k] for k in z.files};doc=read(output/'upstream/upstream/detail_baseline/paths.json');oldrows=describe(rgb,base,r7,routing,doc,read(source/'decisions.json'));rows,field,masks=build(rgb,base,previous,routing,oldrows)
    artifact=torch.load(FIT/'model.pt',map_location='cpu',weights_only=False);artifact['bucket_thresholds']=read(ROOT/'bucket_calibration.json')['thresholds'];data=extract_complete(rgb,previous,base,rows,field,masks,artifact);stats=save(image.stem,rgb,base,previous,rows,field,masks,data,output/'complete');jsave(output/'entry.json',dict(image=str(image),source_sha256=digest(image),model_sha256=digest(FIT/'model.pt'),calibration_sha256=digest(ROOT/'bucket_calibration.json'),entry_code_sha256=digest(__file__),manual_region_masks_used=False,annotation_files_read=False,robot_connected=False,statistics=stats))
    print('COMPLETE_IMAGE_READY',str(output/'complete/complete.png'),flush=True);return stats

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('image');p.add_argument('--output');a=p.parse_args();out=Path(a.output) if a.output else ROOT/'manual_outputs'/(Path(a.image).stem+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));extract(a.image,out)
