"""Rehearsal with corrected painted segments; old datasets remain immutable."""
import sys
sys.dont_write_bytecode=True
from paper_center_refinement_data import ROOT,PARENT
from preserve_faint_paper_refinement import train_second
from train_paper_center_refinement import crop,validation,JointModel,torch,np
from rock_v15_common import read,jsave,digest
import shutil
STAGE=ROOT/'corrected_renderer_stage3'
def train():
    if (STAGE/'fit/refined.pt').exists():raise FileExistsError('Preserve completed corrected refinement')
    torch.set_num_threads(4);rows=read(STAGE/'manifest.json')['records'];cal=[r for r in rows if r['split']=='calibration'];chosen=[r for r in cal if r['kind'] in ['actual_refined','paper_stress']]+[r for r in cal if r['kind']=='existing_declared'][:8]+[r for r in cal if r['kind']=='hard_declared'][:8];bank=[]
    for i,r in enumerate(chosen):
        bank.append(crop(r,np.random.default_rng(291100+i),'paper' if r['kind']=='paper_stress' else 'positive',False))
        if r['kind']=='actual_refined':bank.append(crop(r,np.random.default_rng(291900+i),'paper',False))
    model=JointModel().cuda();model.load_state_dict(torch.load(PARENT/'joint.pt',map_location='cpu',weights_only=False)['model']);baseline=validation(model,bank);jsave(STAGE/'calibration_initial.json',baseline)
    candidate=ROOT/'faint_preservation_stage2/refined.pt';model.load_state_dict(torch.load(candidate,map_location='cpu',weights_only=False)['model']);current=validation(model,bank);guard=current['center_recall']>=baseline['center_recall']-.05 and current['faint_recall']>=baseline['faint_recall']-.05
    fallback=candidate if guard else PARENT/'joint.pt';shutil.copy2(candidate,STAGE/'last.pt');shutil.copy2(fallback,STAGE/'refined.pt');assert digest(candidate)==digest(STAGE/'last.pt') and digest(fallback)==digest(STAGE/'refined.pt')
    values=current if guard else baseline;jsave(STAGE/'training_summary.json',dict(calibration_score=values['paper_hard']+.3*values['center']+.2*values['relation'],fallback_new_updates=800 if guard else 0,calibration_guard_pass=guard))
    jsave(STAGE/'corrected_calibration_comparison.json',dict(before=baseline,initial_candidate=current,candidate_guard=guard,new_updates_in_fallback=800 if guard else 0))
    train_second(experiment_root=STAGE,output_stage=STAGE/'fit',steps=200)
if __name__=='__main__':train()
