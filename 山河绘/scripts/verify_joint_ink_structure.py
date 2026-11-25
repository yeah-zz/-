import sys
sys.dont_write_bytecode=True
from joint_ink_structure_data import ROOT,LAB,LOCAL
from rock_v15_common import read,jsave,digest
from pathlib import Path
from html.parser import HTMLParser
checks=[]
def check(name,ok,**info):
    checks.append(dict(name=name,passed=bool(ok),**info));assert ok,name
config=read(ROOT/'training_config.json');training=read(ROOT/'training_summary.json')
check('actual_900_gpu_updates',training['completed_updates']==900 and training['tensor_device']=='cuda:0')
check('training_manifest_unchanged',digest(ROOT/'manifest.json')==config['manifest_sha256'])
check('initial_neural_checkpoint_unchanged',digest(config['init'])==config['init_sha256'])
check('trained_checkpoint_matches_log',digest(ROOT/'joint.pt')==training['checkpoint_sha256'])
manifest=read(ROOT/'manifest.json');fresh=read(ROOT/'ink_gate_stage2/fresh_test_manifest.json')
old_groups={r['source_group'] for r in manifest['records']};new_groups={r['source_group'] for r in fresh['records']}
check('fresh_test_seeds_separate_from_fit_calibration_and_stage1',not old_groups&new_groups)
check('fresh_test_image_hashes_disjoint',not {r['sha256'] for r in manifest['records']}&{r['sha256'] for r in fresh['records']})
for r in fresh['records']:check(r['id']+'_fresh_data_unchanged',digest(r['data'])==r['data_sha256'])
for r in read(LAB/'stable_line_extractor_20261028/baseline_lock.json')['files']:
    check('old_stable_lock_'+Path(r['path']).name,digest(r['path'])==r['sha256'])
check('v4_frozen_code_unchanged',digest(Path(__file__).with_name('split_detail_v4_frozen.py'))==read(LOCAL/'baseline_manifest.json')['baseline_code_sha256'])
for r in read(ROOT/'summary.json')['actual_graph_diagnostics']:
    p=ROOT/'ink_gate_stage2/evaluation'/r['id']/'joint/paths.json';doc=read(p)
    check(r['id']+'_offline_no_reference_no_fallback',doc['ready_for_robot'] is False and doc['actually_drawn'] is False and doc['reference_used'] is False and doc['v4_fallback'] is False and doc['physical_z'] is None)
    check(r['id']+'_prediction_reference_exists',Path(doc['meta']['prediction_source']).exists())
class Links(HTMLParser):
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ['href','src'] and value and not value.startswith(('http:','https:','#','data:')):
                check('report_link_'+value,(ROOT/value).exists())
Links().feed((ROOT/'index.html').read_text(encoding='utf-8'))
for p in (ROOT/'source_snapshot').glob('*.py'):compile(p.read_text(encoding='utf-8'),str(p),'exec')
summary=read(ROOT/'ink_gate_stage2/summary.json')
check('failed_blank_gate_is_disclosed',summary['fresh_blank_safety_pass'] is False and summary['adoption']=='research only; not promoted')
jsave(ROOT/'FINAL_VERIFICATION.json',dict(checks=checks,all_passed=True,check_count=len(checks),robot_connected=False))
print('JOINT_FINAL_VERIFIED',len(checks))
