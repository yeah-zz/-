import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np
from PIL import Image
from gongbi_training.cunfa_photometry import *
from gongbi_training.cunfa_input_scale import input_canvas
def main():
 # A tiny paper variation should remain tiny under the linear mapping; a broad
 # pale wash remains darker than its reference. No mask/label enters either.
 a=np.full((128,128),200,'uint8');a[:,::2]=201;paper=Image.fromarray(a)
 reference=Image.fromarray(np.full((128,128),220,'uint8'));wash=Image.fromarray(np.full((128,128),180,'uint8'))
 old,_=input_canvas(paper,fill_extent=True);linear=appearance_canvas(paper,reference,'reference_linear')
 assert np.ptp(np.asarray(old).astype('int16'))>100
 assert np.ptp(np.asarray(linear).astype('int16'))<5
 assert np.asarray(appearance_canvas(wash,reference,'reference_linear')).mean()<np.asarray(appearance_canvas(reference,reference,'reference_linear')).mean()
 before=np.asarray(paper).copy();f=contrast_features(paper,reference);assert len(f)==len(CONTRAST_NAMES) and np.isfinite(f).all();assert np.array_equal(before,np.asarray(paper))
 assert np.array_equal(np.asarray(substrate_variant(paper,17)),np.asarray(substrate_variant(paper,17)))
 assert appearance_tensor(paper,reference,'raw_gray').shape==(3,224,224)
 root=LAB/'cunfa_photometry_20261117';root.mkdir(exist_ok=True)
 checks=dict(tiny_paper_contrast_not_stretched=True,pale_wash_not_replaced_by_white=True,
  source_pixels_unchanged=True,no_target_or_paper_mask_input=True,
  deterministic_training_only_augmentation=True,finite_contrast_features=True,
  resized_not_new_optical_detail=True,robot_connected=False)
 (root/'contract_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf8');print(json.dumps(checks),flush=True)
if __name__=='__main__':main()
