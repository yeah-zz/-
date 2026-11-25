import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
ROOT=LAB/'local_rock_patch_20261117'
def read(p):return json.loads(Path(p).read_text('utf8'))
def main():
 lock=read(ROOT/'baseline_lock.json');verify=read(ROOT/'verification.json');browser=read(ROOT/'browser_checks.json');assert all(verify['engineering_checks'].values());assert all(v for k,v in browser.items() if k not in ['url','robot_connected']);assert not browser['robot_connected']
 for k in ['approved500','working489']:assert hashlib.sha256(Path(lock[k]).read_bytes()).hexdigest()==lock[k+'_sha256']
 p=LAB/'guohua_hierarchy_20261112/current_milestones.json';s=read(p);assert not any(s['capability_state'].values());url=read(ROOT/'server.json')['url'];s.update(latest_local_rock_patch=str(ROOT/'metrics.json'),latest_local_rock_patch_report=url+'report.html',latest_local_rock_patch_replay=url,latest_local_rock_patch_verification=str(ROOT/'verification.json'),local_patch_production_replaced=False,local_patch_new_neural_training=False,local_patch_selected_path_count=21,local_patch_other_path_count_preserved=468,robot_connected=False);p.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf8')
 (ROOT/'postexperiment_checks.json').write_text(json.dumps(dict(approved500_preserved=True,working489_preserved=True,local_patch_checks_passed=True,browser_checks_passed=True,production_replaced=False,robot_connected=False),indent=2),encoding='utf8');print(url);print(url+'report.html')
if __name__=='__main__':main()
