"""Pre-training label/gradient checks; source hashes protect earlier checkpoints."""
import sys
sys.dont_write_bytecode=True
from joint_ink_structure_data import ROOT,OFFSETS,path_fields,hard_fixture
from rock_v15_common import read,jsave,digest
import numpy as np,torch
from train_joint_ink_structure import loss_fn,initialize

def main():
    manifest=read(ROOT/'manifest.json');records=manifest['records'];repairs=0
    # The preflight found that 2-vertex needle directions needed interpolation
    # along every raster segment, not only its two vertices. Correct before fit.
    for r in records:
        if (ROOT/'joint.pt').exists():
            assert digest(r['data'])==r['data_sha256'],'Training data changed: '+r['id']
            continue
        if r['kind'] not in ['hard_declared','actual_partial']:continue
        doc=read(r['reference']);paths=[np.asarray(s['points'],np.float32) for s in doc.get('strokes',[])]
        if 'partial_points' in doc:paths=[np.asarray(doc['partial_points'],np.float32)]
        if not paths:continue
        z=np.load(r['data']);data={k:z[k].copy() for k in z.files};_,co,si,dv,_,_,_=path_fields(paths,data['target'].shape[-2:])
        data['target'][2]=co;data['target'][3]=si
        np.savez_compressed(r['data'],**data);r['data_sha256']=digest(r['data']);repairs+=1
    if not (ROOT/'joint.pt').exists():
        manifest['source_sha256']=digest(__import__('joint_ink_structure_data').__file__);manifest['direction_interpolation_corrected_before_training']=True
        jsave(ROOT/'manifest.json',manifest)
    checks=[]
    def check(name,ok,**info):
        checks.append(dict(name=name,passed=bool(ok),**info));assert ok,name
    for a,b in [('train','calibration'),('train','test'),('calibration','test')]:
        ra=[r for r in records if r['split']==a];rb=[r for r in records if r['split']==b]
        check(a+'_'+b+'_source_groups_disjoint',not {r['source_group'] for r in ra}&{r['source_group'] for r in rb})
        check(a+'_'+b+'_image_hashes_disjoint',not {r['sha256'] for r in ra}&{r['sha256'] for r in rb})
    check('unknown_review_proposals_not_loaded',not set(manifest['unknown_proposals_ignored'])&{r['id'] for r in records})
    for kind in ['parallel','crossing','dry_continuation','needles','blank']:
        rgb,t,m,role,rel,rm,doc=hard_fixture(740100,kind)
        if kind=='parallel':check('close_parallel_strokes_supply_separate_labels',bool(((rel<.5)&(rm>0)).any()))
        if kind=='dry_continuation':check('same_stroke_supplies_join_labels',bool(((rel>.5)&(rm>0)).any()))
        if kind=='needles':
            norm=np.sqrt(t[2]**2+t[3]**2);check('two_vertex_needles_have_direction_along_length',float((norm[m[2]>0]>.99).mean())>.99)
        if kind=='blank':check('declared_blank_has_no_positive_centers',not t[0].any())
    for r in [r for r in records if r['kind']=='actual_partial']:
        z=np.load(r['data']);check(r['id']+'_real_unlabeled_pixels_remain_unknown',bool((z['weight'][0]==0).any()))
    device='cuda';torch.set_num_threads(4);o=torch.randn((1,18,16,16),device=device,requires_grad=True);t=torch.zeros((1,6,16,16),device=device);m=torch.zeros((1,4,16,16),device=device);role=torch.full((1,16,16),255,device=device,dtype=torch.long);rel=torch.zeros((1,8,16,16),device=device);rm=torch.zeros_like(rel)
    loss,_=loss_fn(o,t,m,role,rel,rm);loss.backward();check('unknown_pixels_produce_zero_loss_and_gradient',float(loss.detach())==0 and not o.grad.any())
    model=initialize();out=model(torch.zeros((1,7,64,64),device=device));check('gpu_joint_forward_18_channels',tuple(out.shape)==(1,18,64,64) and out.is_cuda)
    jsave(ROOT/'preflight_checks.json',dict(checks=checks,direction_records_checked=repairs,gpu=torch.cuda.get_device_name(),all_passed=True))
    print('JOINT_PREFLIGHT',len(checks),torch.cuda.get_device_name(),flush=True)
if __name__=='__main__':main()
