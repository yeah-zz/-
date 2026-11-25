# -*- coding: utf-8 -*-
"""Freeze exploratory external works and distinguish real labels from stress data."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,hashlib,shutil
ROOT=LAB/'pima_static_discovery_20261109';EXTERNAL=LAB/'pima_independent_20261109'
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'external').mkdir(exist_ok=True)
 sources=[]
 for key in ['new_howto','new_fuchun']:
  p=EXTERNAL/key;meta=json.loads((p/'metadata.json').read_text('utf8'))
  sources.append(dict(id=key,page=meta['url'],uploader=meta['author']['name'],title=meta['name'],
   source_video_sha256=sha(p/'video.mp4'),visually_distinct_work_from_liu_qiushan=True,
   original_painter_identity_verified=False,author_disjointness_verified=False,expert_labels=False,
   allowed_use='exploratory external-work evaluation only; no training or checkpoint selection',
   quality='edited/speed uncertain; static visible curves only' if key=='new_howto' else 'wide view and mixed painting stages; qualitative only'))
 for key,name in [('new_howto','frame_00.jpg'),('new_howto','frame_01.jpg'),('new_fuchun','frame_01.jpg')]:
  shutil.copy2(EXTERNAL/key/name,ROOT/'external'/(key+'_'+name))
 save(ROOT/'external_sources.json',sources+[dict(id='tong_museum_index',page='https://www.youtube.com/watch?v=ZAMisiskqcA',
   official_context='https://www.artmuseum.cuhk.edu.hk/en/exhibition/current/detail/66',
   artist='唐锦腾',author_verified_from_official_museum=True,media_status='LOGIN_REQUIRED: sign in to confirm not a bot; not downloaded or bypassed',used_for_training=False,used_for_evaluation=False)])
 refs=[dict(id='H01',image='new_howto_frame_00.jpg',role='visible_main_contour',
   points=[[409,212],[400,214],[390,220],[387,230],[387,235],[382,242],[379,251],[374,260],[371,271],[365,286]],
   provenance='assistant traced visible static curve; artist stroke identity/order unknown',annotation_uncertainty_px=3.,expert_verified=False),
  dict(id='H02',image='new_howto_frame_01.jpg',role='visible_bending_main_contour',
   points=[[365,193],[387,177],[406,167],[426,151],[445,137],[460,116],[473,96],[487,76],[498,64],[516,58],[537,51],[553,52],[566,47],[580,49],[596,56],[607,61],[615,72],[620,82]],
   provenance='assistant traced visible static curve; may comprise multiple artist strokes',annotation_uncertainty_px=3.,expert_verified=False)]
 save(ROOT/'external_visible_curve_references.json',refs)
 protocol=dict(offline_only=True,robot_connected=False,training_real_candidate_ids=['S01','S02'],development_same_rock_ids=['S03'],
  external_work_ids=['new_howto','new_fuchun'],external_used_for_training=False,external_used_for_checkpoint_selection=False,
  external_blinded=False,external_artist_disjointness_verified=False,external_curve_metrics_scope='partial, approximate visible-curve recall; unmarked curves are unknown, not negatives',
  synthetic_interventions=['known-identity crossing overlay','interior dry-gap erasure','spatial bend warp'],
  synthetic_interventions_are_not_real_demonstrations=True,input_contract='single static image; automatic ink endpoint proposals, no temporal difference or reference seed supplied at external inference',
  training_input_contract='static appearance patches; supervised by existing video-derived candidate paths',
  updates=1000,seed=1109,max_endpoint_seeds_per_image=80,max_trace_steps=400,
  automated_proposals_are_not_approved_strokes=True,physical_parameters_trained=False,accepted_489_drawing_replacement=False)
 save(ROOT/'protocol.json',protocol);print('STATIC_PROTOCOL_FROZEN',ROOT,flush=True)
if __name__=='__main__':main()
