import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
def main():
 s=Path(__file__).with_name('check_guohua_objects_v2_view_20261113.py').read_text('utf8')
 s=s.replace('guohua_objects_v2_20261113','cunfa_branch_isolation_20261117').replace(')==27',')==146').replace('range(27)','range(146)')
 s=s.replace("assert js(\"document.querySelectorAll('#metrics tr').length\")==6", "assert js(\"document.querySelectorAll('#metrics tr').length\")==4").replace('#works tr','#aux tr')
 s=s.replace("['expanded_control.pt','line_aware_aug.pt','../expanded_control.pt','predictions/H_14_expanded_control.npz']", "['isolation_models.pt','dense_outputs.npz','../isolation_models.pt','reference_targets.npz']")
 s=s.replace("['protocol.json','results.json','selection_before_download.json','source_checked_cunfa_concepts.json','verification.json']", "['protocol.json','results.json','reference_manifest.json','training_selection.json','verification.json','gradient_diagnostics.json','pair_review.json']")
 s=s.replace('assert requests.post(url,timeout=5).status_code==405', '''assert js('document.querySelectorAll("#failures p").length')==9
 assert js('document.querySelectorAll("#predictions tr").length')==5
 assert js('document.querySelectorAll("#model option").length')==4
 assert js('document.querySelectorAll("#gradients tr").length')==5
 for i in range(4):
  js(f'document.getElementById("model").selectedIndex={i};document.getElementById("model").dispatchEvent(new Event("change"))');loaded()
 assert js('document.body.textContent.includes("没有新来源独立验收")')
 assert js('document.body.textContent.includes("不是新增自主辨识能力")')
 assert requests.post(url,timeout=5).status_code==405''')
 s=s.replace('all27_observations_switch_and_load','all146_regions_switch_and_load').replace('metrics_six_rows','metrics_four_rows').replace('work_metrics_four_rows','sparse_metrics_four_rows')
 exec(compile(s,str(Path(__file__).resolve()),'exec'),{'__name__':'__main__','__file__':str(Path(__file__).resolve())})
if __name__=='__main__':main()
