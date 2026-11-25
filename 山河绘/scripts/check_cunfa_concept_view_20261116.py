"""Use the existing headless-browser harness for every region, not just a cover."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()

def main():
 source=Path(__file__).with_name('check_guohua_objects_v2_view_20261113.py').read_text('utf8')
 source=source.replace('guohua_objects_v2_20261113','cunfa_concept_supervision_20261116')
 source=source.replace(')==27',')==25').replace('range(27)','range(25)').replace(')==6',')==3')
 source=source.replace("#works tr", "#styles tr").replace(')==4',')==2')
 source=source.replace("document.getElementById(\"refnote\").textContent.includes(\"无完整对象边界\")", "document.getElementById(\"role\").textContent.includes(\"未知\")")
 source=source.replace("['expanded_control.pt','line_aware_aug.pt','../expanded_control.pt','predictions/H_14_expanded_control.npz']", "['concept_heads.joblib','concept_style_head.joblib','../concept_heads.joblib','features_and_scores.npz']")
 source=source.replace("['protocol.json','results.json','selection_before_download.json','source_checked_cunfa_concepts.json','verification.json']", "['protocol.json','results.json','reference_manifest.json','contract_checks.json','verification.json','native_asset_reconciliation.json']")
 source=source.replace("['protocol.json','results.json','reference_manifest.json','contract_checks.json','verification.json','native_asset_reconciliation.json']", "['protocol.json','results.json','reference_manifest.json','reference_manifest_corrected.json','reference_metadata_correction.json','contract_checks.json','verification.json','native_asset_reconciliation.json']")
 source=source.replace("assert requests.post(url,timeout=5).status_code==405", "assert js('document.querySelectorAll(\"#native details\").length')==4\n assert js('document.querySelectorAll(\"#failures p\").length')==3\n assert requests.post(url,timeout=5).status_code==405")
 source=source.replace('all27_observations_switch_and_load','all25_regions_switch_and_load').replace('metrics_six_rows','metrics_three_rows').replace('work_metrics_four_rows','style_probe_two_rows')
 exec(compile(source,str(Path(__file__).resolve()),'exec'),{'__name__':'__main__','__file__':str(Path(__file__).resolve())})

if __name__=='__main__':main()
