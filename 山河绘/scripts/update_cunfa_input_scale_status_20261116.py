import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
r=LAB/'cunfa_input_scale_20261116';p=LAB/'guohua_hierarchy_20261112/current_milestones.json';s=json.loads(p.read_text('utf8'));assert not any(s['capability_state'].values());s.update(latest_cunfa_result=str(r/'results.json'),latest_cunfa_report=json.loads((r/'server.json').read_text('utf8'))['url'],latest_cunfa_verification=str(r/'verification.json'),latest_cunfa_browser_checks=str(r/'browser_checks.json'),latest_cunfa_context_probe=str(r/'context_probe_results.json'),latest_cunfa_status='Input extent standardization: same63 training rows, known13 agreement7to8; Xia_Gui small-axe4cases still fail; no independent acceptance',new_cunfa_model_trained=True,new_cunfa_training_scope='Two linear classification heads with frozen encoder for input-scale ablation; no new concepts, stroke generation or unknown recognition learned',new_model_trained_this_round=True,production_replaced=False,robot_connected=False)
p.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf8')
test=[r for r in json.loads((LAB/'cunfa_fresh_validation_20261111/test_manifest.json').read_text('utf8'))['records'] if r['admission']=='source_label_test'];result=json.loads((r/'results.json').read_text('utf8'));preds={q['id']:q for q in result['results']['fixed_extent']['records']};queue=[]
for q in test:
 if preds[q['id']]['agreement']:continue
 queue.append(dict(id=q['id'],image=q['image'],source_image=q['source_image'],source_box=q['source_box'],source_url=q['source_url'],work_scope_method=q['label'],source_documented_variant='小斧劈与短线条秃笔' if q['work_id']=='230757' else '披麻皴；局部具体变体未知',local_method_confirmed=False,local_stroke_groups=None,visible_concepts={k:None for k in ['short_mark_or_long_fibre','line_or_face','width_variation','edge_character','group_flow_on_surface','overlap_and_wash','paper_texture_interference']},training_target=None,prediction_is_not_label=True,review_required=True))
(r/'next_cunfa_evidence_queue.json').write_text(json.dumps(dict(records=queue,scope='Verify local mark/group evidence; no new labels invented from model or work title',new_expert_labels=0),ensure_ascii=False,indent=2),encoding='utf8');print(s['latest_cunfa_report'])
