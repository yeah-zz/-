import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
def main():
 s=Path(__file__).with_name('check_guohua_objects_v2_view_20261113.py').read_text('utf8')
 s=s.replace('guohua_objects_v2_20261113','cunfa_cross_source_20261117').replace(')==27',')==93').replace('range(27)','range(93)').replace(')==6',')==3')
 s=s.replace("assert js(\"document.querySelectorAll('#works tr').length\")==4", "assert js(\"document.querySelectorAll('#styles tr').length\")==3")
 s=s.replace("['expanded_control.pt','line_aware_aug.pt','../expanded_control.pt','predictions/H_14_expanded_control.npz']", "['cross_source_models.joblib','../cross_source_models.joblib','features_and_scores.npz','sources/qingpu.pdf','source_inventory.json']")
 s=s.replace("['protocol.json','results.json','selection_before_download.json','source_checked_cunfa_concepts.json','verification.json']", "['protocol.json','results.json','reference_manifest.json','contract_checks.json','verification.json','source_summary.json']")
 s=s.replace('assert requests.post(url,timeout=5).status_code==405', '''assert js('document.querySelectorAll("#rolemetrics tr").length')==2
 assert js('document.querySelectorAll("#roles_by_class tr").length')==4
 assert js('document.querySelectorAll("#sources tr").length')==2
 assert js('document.querySelectorAll("#role_failures p").length')==7
 for scope in ['new_NT_source_probe','known_HKMA_regression','known_work_new_shape_probe']:
  js('document.getElementById("metric_scope").value='+json.dumps(scope)+';document.getElementById("metric_scope").dispatchEvent(new Event("change"))')
  assert js('document.querySelectorAll("#metrics tr").length')==3
 assert requests.post(url,timeout=5).status_code==405''')
 s=s.replace('all27_observations_switch_and_load','all93_regions_switch_and_load').replace('metrics_six_rows','metrics_three_rows').replace('work_metrics_four_rows','style_probe_three_rows')
 exec(compile(s,str(Path(__file__).resolve()),'exec'),{'__name__':'__main__','__file__':str(Path(__file__).resolve())})
if __name__=='__main__':main()
