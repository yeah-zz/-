# -*- coding: utf-8 -*-
"""Predeclared follow-up: keep spatial decoder fixed, update final classifier only.

Known-case exploration after the first trial, never an independent validation.
"""
import os,sys,json,hashlib,shutil
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
import torch
import regional_presence_trial_20261117 as experiment
ROOT=LAB/'regional_presence_headonly_20261117';PREV=LAB/'regional_presence_trial_20261117'
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,v):(ROOT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def prepare():
    ROOT.mkdir(exist_ok=True);assert not (ROOT/'protocol.json').exists(),'Follow-up already frozen'
    p=read(PREV/'protocol.json')
    p.update(followup_wrapper_sha256=sha(__file__),followup_after_known_results=True,
             preceding_result_sha256=sha(PREV/'results.json'),
             trainable='Only head[-1] 1x1 four-class classifier; all spatial decoder and encoder parameters fixed',
             optimizer='AdamW lr0.0001 wd0.0001; same128 steps/schedule/loss, only final classifier trainable',
             hypothesis='Can frozen spatial features reduce tree spill into rocks caused by decoder adaptation?',
             evaluation='All previously inspected regressions, no new independent examples or acceptance')
    for name in ['admission.json','split_retirement.json','pretrain_checks.json']:shutil.copyfile(PREV/name,ROOT/name)
    write('protocol.json',p);print('PREPARED_HEAD_ONLY',flush=True)
def train():
    p=read(ROOT/'protocol.json');assert p['followup_wrapper_sha256']==sha(__file__)
    original=experiment.SceneObjects
    def classifier_only(*args,**kwargs):
        net=original(*args,**kwargs);net.requires_grad_(False);net.head[-1].requires_grad_(True)
        return net
    experiment.SceneObjects=classifier_only;experiment.ROOT=ROOT
    experiment.train()
    state=torch.load(p['initialization'],map_location='cpu',weights_only=False)['model'];changes={}
    for arm in p['arms']:
        current=torch.load(ROOT/(arm+'.pt'),map_location='cpu',weights_only=False)['model']
        changed=[k for k in state if not torch.equal(state[k],current[k])]
        assert changed and all(k.startswith('head.6.') for k in changed)
        changes[arm]=changed
    write('freeze_checks.json',dict(only_final_classifier_changed=True,changed_state_keys=changes,
        identical_schedule_to_decoder_trial=sha(ROOT/'schedule.json')==sha(PREV/'schedule.json'),
        known_exploration_only=True,independent_validation=False))
    print('HEAD_ONLY_VERIFIED',changes,flush=True)
if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    else:train()
