# -*- coding: utf-8 -*-
"""Source-backed illustration concepts; actions are not image-measured controls."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
ROOT=LAB/'guohua_objects_pilot_20261112'
def main():
 old=json.loads((LAB/'cunfa_role_reference_20261112/source_references.json').read_text('utf8'));pdf=LAB/'cunfa_gabor_baseline_20261110/sources/aam.pdf'
 descriptions={
  'aam_short_hemp':dict(visible_concepts={'relative_length':'shorter_than_long_hemp','edge_character':'more_ragged_than_long_hemp'},instruction_only={'brush_contact':'tip'},basis='PDF第2页短披麻图示及说明'),
  'aam_long_hemp':dict(visible_concepts={'relative_length':'long','curve_character':'slightly_wavy'},instruction_only={'brush_orientation':'upright'},basis='PDF第2页长披麻图示及说明'),
  'aam_small_axe':dict(visible_concepts={'mark_shape':'triangular'},instruction_only={'brush_orientation':'oblique'},basis='PDF第2页小斧劈图示及说明'),
  'aam_large_axe':dict(visible_concepts={'relative_mark_size':'larger_than_small_axe','relative_mark_count':'fewer_than_small_axe'},instruction_only={},basis='PDF第2页大斧劈图示及与小斧劈的比较')}
 records=[]
 for r in old['records']:
  if r['id'] not in descriptions:continue
  d=descriptions[r['id']];records.append(dict(id=r['id'],image=r['image'],image_sha256=hashlib.sha256(Path(r['image']).read_bytes()).hexdigest(),named_cunfa=r['local_style'],source_url=r['source_url'],source_page=2,source_pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),artist_group=r['source_group'],scope='named_teaching_illustration_as_whole_not_individual_stroke',concept_basis=d['basis'],source_supported_visual_concepts=d['visible_concepts'],documentary_action_knowledge=d['instruction_only'],action_not_observed_or_robot_calibrated=True,review_level='source_caption_and_illustration_crosschecked_by_assistant',expert_pixel_or_stroke_labels=False,newly_collected_image=False,ready_for_independent_concept_test=False))
 assert len(records)==4 and len({r['artist_group'] for r in records})==1
 out=dict(records=records,illustrations=4,original_illustrators=1,new_expert_annotations=0,new_independent_test=0,concept_model_trained=False,tree_skill_training_started=False,unobserved_concepts_remain_unknown=True,robot_connected=False)
 (ROOT/'source_checked_cunfa_concepts.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print('SOURCE_CHECKED_CONCEPTS',len(records),'ILLUSTRATORS',1,flush=True)
if __name__=='__main__':main()
