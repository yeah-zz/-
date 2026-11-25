"""Recover labeled faint strokes after hard-negative training; unchanged guards."""
import sys
sys.dont_write_bytecode=True
from train_paper_center_refinement import *
from pathlib import Path
STAGE=ROOT/'faint_preservation_stage2'

def train_second(experiment_root=ROOT,output_stage=STAGE,steps=300):
    ROOT=Path(experiment_root);STAGE=Path(output_stage)
    if (STAGE/'refined.pt').exists():raise FileExistsError('Preserve completed stage two')
    STAGE.mkdir(exist_ok=True);torch.set_num_threads(4);cv2.setNumThreads(2);rng=np.random.default_rng(291207);records=read(ROOT/'manifest.json')['records'];rows=[r for r in records if r['split']=='train'];cal=[r for r in records if r['split']=='calibration'];model=JointModel().cuda();model.load_state_dict(torch.load(ROOT/'last.pt',map_location='cpu',weights_only=False)['model'])
    old=[r for r in rows if r['kind']=='existing_declared'];hard=[r for r in rows if r['kind']=='hard_declared'];actual=[r for r in rows if r['kind']=='actual_refined' and np.load(r['data'])['target'][0].max()>.3];paper=[r for r in rows if r['kind'] in ['paper_stress','actual_refined'] and arrays(r['image'],r['data'])[1]['paper_mask'].any()]
    selected_cal=[r for r in cal if r['kind'] in ['actual_refined','paper_stress']]+[r for r in cal if r['kind']=='existing_declared'][:8]+[r for r in cal if r['kind']=='hard_declared'][:8];bank=[]
    for i,r in enumerate(selected_cal):
        bank.append(crop(r,np.random.default_rng(291100+i),'paper' if r['kind']=='paper_stress' else 'positive',False))
        if r['kind']=='actual_refined':bank.append(crop(r,np.random.default_rng(291900+i),'paper',False))
    baseline=read(ROOT/'calibration_initial.json');fallback=torch.load(ROOT/'refined.pt',map_location='cpu',weights_only=False);best_state=fallback['model'];best_step=0;best=read(ROOT/'training_summary.json')['calibration_score'];optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4);logs=[];start=time.perf_counter()
    config=dict(steps=steps,init=str(ROOT/'last.pt'),init_sha256=digest(ROOT/'last.pt'),guard_reference=str(ROOT/'calibration_initial.json'),guard_drop=.05,manifest_sha256=digest(ROOT/'manifest.json'),paper_hard_weight=.35,faint_center_bce_weight=1.2,faint_ink_weight=.5,test_used=False,robot_connected=False)
    jsave(STAGE/'training_config.json',config)
    for step in range(1,steps+1):
        chosen=[(old[int(rng.integers(len(old)))],'faint')]+[(hard[int(rng.integers(len(hard)))],'faint' if i==0 else 'positive') for i in range(2)]+[(actual[int(rng.integers(len(actual)))],'positive')]+[(paper[int(rng.integers(len(paper)))],'paper') for _ in range(2)]
        a=tensors([crop(r,rng,mode,True) for r,mode in chosen]);model.train();optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:group['lr']=1e-5*(.4+.6*.5*(1+math.cos(math.pi*step/steps)))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            o=model(a[0]);loss,parts=objective(o,*a[1:7]);faint=a[7]>0
            bce=F.binary_cross_entropy_with_logits(o[:,0].float(),a[1][:,0],reduction='none');lf=(bce*faint).sum()/faint.sum().clamp_min(1)
            fi=faint&(a[1][:,1]>.5);li=(F.softplus(-o[:,1].float())*fi).sum()/fi.sum().clamp_min(1)
            loss=loss-.65*parts['paper_hard']+1.2*lf+.5*li
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        if step%100==0:
            values=validation(model,bank);guard=values['center_recall']>=baseline['center_recall']-.05 and values['faint_recall']>=baseline['faint_recall']-.05;score=values['paper_hard']+.3*values['center']+.2*values['relation'];row=dict(step=step,loss=float(loss.detach()),seconds=time.perf_counter()-start,calibration=values,passes_preservation_guard=bool(guard),selection_score=score)
            if guard and score<best:best=score;best_step=step;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            logs.append(row);print('FAINT_PRESERVATION',json.dumps(row),flush=True)
    torch.save(dict(model=best_state,retained_step=best_step,completed_updates=steps,config=config),STAGE/'refined.pt');jsave(STAGE/'training_summary.json',dict(completed_updates=steps,retained_step=best_step,fallback_to_stage1_guarded_checkpoint=best_step==0,calibration_score=best,gpu=torch.cuda.get_device_name(),tensor_device=str(next(model.parameters()).device),elapsed_seconds=time.perf_counter()-start,checkpoint_sha256=digest(STAGE/'refined.pt'),logs=logs))
    print('FAINT_PRESERVATION_DONE',best_step,flush=True)
if __name__=='__main__':train_second()
