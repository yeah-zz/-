import sys
sys.dont_write_bytecode=True
from paper_center_refinement_data import ROOT,SPECS,PARENT
from rock_v15_common import read,jsave,digest
from train_paper_center_refinement import objective
import numpy as np,torch,cv2
from PIL import Image
records=read(ROOT/'manifest.json')['records'];checks=[]
def check(name,ok,**info):
    checks.append(dict(name=name,passed=bool(ok),**info));assert ok,name
for a,b in [('train','calibration'),('train','regression'),('calibration','regression')]:
    ra=[r for r in records if r['split']==a];rb=[r for r in records if r['split']==b]
    check(a+'_'+b+'_source_groups_disjoint',not {r['source_group'] for r in ra}&{r['source_group'] for r in rb})
    check(a+'_'+b+'_image_hashes_disjoint',not {r['sha256'] for r in ra}&{r['sha256'] for r in rb})
for r in [r for r in records if r['kind'] in ['actual_refined','paper_stress']]:
    z=np.load(r['data']);check(r['id']+'_paper_does_not_overlap_known_centers',not ((z['paper_mask']>0)&(z['target'][0]>.3)&(z['weight'][0]>0)).any())
    if r['id'] in SPECS:
        check(r['id']+'_measured_centers_present',int(((z['target'][0]>.3)&(z['weight'][0]>0)).sum())>100)
        for j,(x0,y0,x1,y1) in enumerate(SPECS[r['id']]['unknown']):
            check(r['id']+f'_unknown_junction_{j}_not_center_or_relation_gold',not z['weight'][0,y0:y1,x0:x1].any() and not z['relation_weight'][:,y0:y1,x0:x1].any())
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));out=rgb.copy();ink=(z['target'][1]>.5)&(z['weight'][1]>0);center=(z['target'][0]>.3)&(z['weight'][0]>0);out[ink]=(out[ink]*.45+np.array([20,130,230])*.55).astype(np.uint8);out[center]=[230,30,40]
        Image.fromarray(out).resize((out.shape[1]*2,out.shape[0]*2)).save(ROOT/(r['id']+'_MEASURED_REFERENCE.png'))
o=torch.zeros((1,18,16,16),device='cuda',requires_grad=True);t=torch.zeros((1,6,16,16),device='cuda');m=torch.zeros((1,4,16,16),device='cuda');role=torch.full((1,16,16),255,device='cuda',dtype=torch.long);rel=torch.zeros((1,8,16,16),device='cuda');rm=torch.zeros_like(rel);paper=torch.zeros((1,16,16),device='cuda')
value,_=objective(o,t,m,role,rel,rm,paper);value.backward();check('unknown_pixels_zero_loss_and_gradient',float(value.detach())==0 and not o.grad.any())
o2=torch.full((1,18,16,16),-12.,device='cuda');paper[:]=1;o2[0,0,5,5]=8.;_,parts=objective(o2,t,m,role,rel,rm,paper);check('sparse_paper_false_positive_receives_hard_loss',float(parts['paper_hard'])>.1)
jsave(ROOT/'preflight_checks.json',dict(checks=checks,count=len(checks),initial_checkpoint_sha256=digest(PARENT/'joint.pt'),new_annotations_are_not_expert_gold=True))
print('PAPER_CENTER_PREFLIGHT',len(checks),torch.cuda.get_device_name(),flush=True)
