import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
def main():
 s=Path(__file__).with_name('check_guohua_objects_v2_view_20261113.py').read_text('utf8')
 s=s.replace('guohua_objects_v2_20261113','cunfa_score_alignment_20261117')
 s=s.replace(')==27',')==48').replace('range(27)','range(48)').replace(')==6',')==3')
 s=s.replace("assert js(\"document.querySelectorAll('#works tr').length\")==4", "assert js(\"document.querySelectorAll('#styles tr').length\")==2")
 s=s.replace("['expanded_control.pt','line_aware_aug.pt','../expanded_control.pt','predictions/H_14_expanded_control.npz']", "['score_alignment_models.joblib','../score_alignment_models.joblib','features_and_scores.npz']")
 s=s.replace("['protocol.json','results.json','selection_before_download.json','source_checked_cunfa_concepts.json','verification.json']", "['protocol.json','results.json','reference_manifest.json','contract_checks.json','verification.json']")
 s=s.replace('assert requests.post(url,timeout=5).status_code==405', '''assert js('document.querySelectorAll("#styles_summary tr").length')==4
 assert js('document.querySelectorAll("#failures p").length')==5
 for scope in ['known_HKMA_regression','known_work_new_shape_probe','all_known_shape_probes']:
  js('document.getElementById("metric_scope").value='+json.dumps(scope)+';document.getElementById("metric_scope").dispatchEvent(new Event("change"))')
  assert js('document.querySelectorAll("#metrics tr").length')==3
  assert js('document.querySelectorAll("#brier tr").length')==3
 assert requests.post(url,timeout=5).status_code==405''')
 s=s.replace('all27_observations_switch_and_load','all48_regions_switch_and_load').replace('metrics_six_rows','metrics_three_rows').replace('work_metrics_four_rows','style_probe_two_rows')
 exec(compile(s,str(Path(__file__).resolve()),'exec'),{'__name__':'__main__','__file__':str(Path(__file__).resolve())})
if __name__=='__main__':main()
