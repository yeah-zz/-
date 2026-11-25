import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
def main():
 s=Path(__file__).with_name('check_guohua_objects_v2_view_20261113.py').read_text('utf8')
 s=s.replace('guohua_objects_v2_20261113','cunfa_relational_roles_20261117').replace(')==27',')==112').replace('range(27)','range(112)')
 s=s.replace("assert js(\"document.querySelectorAll('#metrics tr').length\")==6", "assert js(\"document.querySelectorAll('#metrics tr').length\")==5")
 s=s.replace("assert js(\"document.querySelectorAll('#works tr').length\")==4", "assert js(\"document.querySelectorAll('#roles tr').length\")==4")
 s=s.replace("['expanded_control.pt','line_aware_aug.pt','../expanded_control.pt','predictions/H_14_expanded_control.npz']", "['relational_models.joblib','../relational_models.joblib','features.npz','sources/T_structure.png']")
 s=s.replace("['protocol.json','results.json','selection_before_download.json','source_checked_cunfa_concepts.json','verification.json']", "['protocol.json','results.json','reference_manifest.json','training_selection.json','contract_checks.json','verification.json']")
 s=s.replace('assert requests.post(url,timeout=5).status_code==405', '''assert js('document.querySelectorAll("#failures p").length')==6
 assert js('document.querySelectorAll("#predictions tr").length')==6
 assert requests.post(url,timeout=5).status_code==405''')
 s=s.replace('all27_observations_switch_and_load','all112_regions_switch_and_load').replace('metrics_six_rows','metrics_five_rows').replace('work_metrics_four_rows','role_metrics_four_rows')
 exec(compile(s,str(Path(__file__).resolve()),'exec'),{'__name__':'__main__','__file__':str(Path(__file__).resolve())})
if __name__=='__main__':main()
