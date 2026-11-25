import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
def main():
 s=Path(__file__).with_name('check_guohua_objects_v2_view_20261113.py').read_text('utf8')
 s=s.replace('guohua_objects_v2_20261113','cunfa_photometry_20261117').replace(')==27',')==93').replace('range(27)','range(93)')
 s=s.replace("assert js(\"document.querySelectorAll('#works tr').length\")==4", "assert js(\"document.querySelectorAll('#roles tr').length\")==4")
 s=s.replace("['expanded_control.pt','line_aware_aug.pt','../expanded_control.pt','predictions/H_14_expanded_control.npz']", "['photometry_models.joblib','rejection_scope_model.joblib','../photometry_models.joblib','features.npz']")
 s=s.replace("['protocol.json','results.json','selection_before_download.json','source_checked_cunfa_concepts.json','verification.json']", "['protocol.json','results.json','training_selection.json','rejection_scope_results.json','rejection_scope_protocol.json','verification.json','contract_checks.json']")
 s=s.replace('assert requests.post(url,timeout=5).status_code==405', '''assert js('document.querySelectorAll("#rejection tr").length')==5
 assert js('document.querySelectorAll("#failures p").length')==5
 assert requests.post(url,timeout=5).status_code==405''')
 s=s.replace('all27_observations_switch_and_load','all93_regions_switch_and_load').replace('work_metrics_four_rows','role_metrics_four_rows')
 exec(compile(s,str(Path(__file__).resolve()),'exec'),{'__name__':'__main__','__file__':str(Path(__file__).resolve())})
if __name__=='__main__':main()
