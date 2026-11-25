import sys
sys.dont_write_bytecode=True
from paper_center_refinement_data import ROOT,PARENT,LAB,LOCAL
from rock_v15_common import read,jsave,digest
from pathlib import Path
from html.parser import HTMLParser
STAGE=ROOT/'corrected_renderer_stage3';checks=[]
def check(name,ok):
    checks.append(dict(name=name,passed=bool(ok)));assert ok,name
for directory,config_path in [(ROOT,ROOT/'training_config.json'),(ROOT/'faint_preservation_stage2',ROOT/'faint_preservation_stage2/training_config.json'),(STAGE/'fit',STAGE/'fit/training_config.json')]:
    config=read(config_path);manifest=STAGE/'manifest.json' if directory==STAGE/'fit' else ROOT/'manifest.json'
    check(str(directory.relative_to(ROOT))+'_manifest_unchanged',digest(manifest)==config['manifest_sha256'])
    check(str(directory.relative_to(ROOT))+'_init_checkpoint_unchanged',digest(config['init'])==config['init_sha256'])
for r in read(STAGE/'manifest.json')['records']:check('data_'+r['id'],digest(r['data'])==r['data_sha256'])
summary=read(ROOT/'summary.json');check('actual_gpu_updates_match',summary['executed_updates']==1100 and summary['selected_new_updates']==900)
check('full_v4_default_unchanged',summary['full_image_v4_default_unchanged'] is True)
check('focused_gates_passed_without_claiming_full_accuracy',summary['focused_gates_all_pass'] is True and summary['actual_full_image_accuracy'] is None)
train_groups={r['source_group'] for r in read(STAGE/'manifest.json')['records'] if r['split'] in ['train','calibration']};fresh=read(STAGE/'fresh_test_manifest.json')['records']
check('fresh_test_groups_disjoint',not train_groups&{r['source_group'] for r in fresh})
check('fresh_seeds_are_950000_range',all(950000<=read(r['reference'])['seed']<=950047 for r in fresh))
for r in fresh:
    check(r['id']+'_renderer2',read(r['reference'])['renderer_version']==2)
    check(r['id']+'_test_data_unchanged',digest(r['data'])==r['data_sha256'])
audit=read(STAGE/'renderer_rgb_checks.json');check('80_RGB_midpoint_checks_pass',audit['all_passed'] and audit['count']==80)
check('legacy_metrics_explicitly_invalidated',read(ROOT/'LEGACY_RENDERER_METRICS_INVALID.json')['legacy_aggregate_scores_invalidated'] is True)
for r in read(LAB/'stable_line_extractor_20261028/baseline_lock.json')['files']:check('stable_lock_'+Path(r['path']).name,digest(r['path'])==r['sha256'])
check('frozen_v4_source_unchanged',digest(Path(__file__).with_name('split_detail_v4_frozen.py'))==read(LOCAL/'baseline_manifest.json')['baseline_code_sha256'])
for p in (STAGE/'evaluation').glob('*/after/joint/paths.json'):
    doc=read(p);check('offline_'+p.parents[2].name,doc['ready_for_robot'] is False and doc['actually_drawn'] is False and doc['reference_used'] is False and doc['v4_fallback'] is False and doc['physical_z'] is None)
class Links(HTMLParser):
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ['href','src'] and value and not value.startswith(('http:','https:','#','data:')):check('link_'+value,(ROOT/value).exists())
Links().feed((ROOT/'index.html').read_text(encoding='utf-8'))
for p in (ROOT/'source_snapshot').glob('*.py'):compile(p.read_text(encoding='utf-8'),str(p),'exec')
jsave(ROOT/'FINAL_VERIFICATION.json',dict(all_passed=True,count=len(checks),checks=checks,model_sha256=digest(summary['checkpoint']),robot_connected=False))
print('PAPER_CENTER_FINAL_VERIFIED',len(checks))
