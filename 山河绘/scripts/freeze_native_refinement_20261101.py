from refine_native_fine_faint_20261101 import *
from run_stability_brush_20261101 import verify_lock
import shutil
def main():
    verify_lock();scripts=Path(__file__).parent;files=[scripts/n for n in ['native_fine_faint_20261101.py','select_native_fine_faint_20261101.py','train_native_fan_router_20261101.py','refine_native_fine_faint_20261101.py']]+[ROOT/'router/router.pt',ROOT/'router/manifest.json',ROOT/'router/training_config.json'];lock=dict(config=CFG3,source_field_config=CFG,files=[dict(path=str(p),sha256=digest(p)) for p in files],development_and_router_train_ids=['T01','T06','T17','T19'],regression_ids=[f'T{i:02}' for i in range(1,21) if i not in [1,6,17,19]],regression_images_are_previously_seen_not_new_blind_test=True,regression_labels_used_for_training=False,no_tuning_after_regression_started=True,accepted_baseline_lock_sha256=digest(BASE/'baseline_lock.json'),all_20_original_sha256={f'T{i:02}':digest(BASE/'originals'/f'T{i:02}.png') for i in range(1,21)},new_source_images=0,robot_connected=False)
    p=ROOT/'r3_lock.json'
    if p.exists():assert read(p)==lock,'Frozen experiment changed'
    else:jsave(p,lock)
    folder=ROOT/'source_snapshot';folder.mkdir(exist_ok=True)
    for p in files:
        dest=folder/p.name
        if dest.exists():assert digest(dest)==digest(p)
        else:shutil.copy2(p,dest)
    print('NATIVE_R3_FROZEN',len(files),flush=True)
if __name__=='__main__':main()
