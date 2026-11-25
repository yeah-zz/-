import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,torch
from gongbi_training.cunfa_boundary_balance import *
from gongbi_training.cunfa_context_roles import context_role_target
ROOT=LAB/'cunfa_boundary_balance_20261117'
def main():
 l=torch.tensor([1.,-1.],requires_grad=True);t=torch.tensor([1.,0.]);a=balanced_binary_loss(l,t);b=balanced_binary_loss(torch.cat([l[:1],l[1:].repeat(20)]),torch.tensor([1.]+[0.]*20));assert torch.allclose(a,b)
 z=torch.tensor([1.,-1.,99.],requires_grad=True);loss=balanced_binary_loss(z,torch.tensor([1.,0.,-100.]));loss.backward();assert z.grad[2]==0
 d=json.loads((ROOT/'reference_manifest.json').read_text('utf8'));old=json.loads((LAB/'cunfa_boundary_relations_20261117/reference_manifest.json').read_text('utf8'))
 assert d['records']==old['records'] and d['curves']==old['curves'];assert all(r['split']=='train' for r in d['negative_references'])
 neg=d['negative_references'][0];e,s,g,o=targets_with_negatives(neg,[1100,1850,1350,2100]);assert (e==0).any() and (s==-100).all() and not o.any()
 assert context_role_target(next(r for r in d['records'] if r['id']=='nizan_8'))==-100
 p=ROOT/'contract_checks.json';p.write_text(json.dumps(dict(class_replication_invariance=True,unknown_zero_gradient=True,noncontour_not_paper_target=True,old_roles_and_curves_unchanged=True,negative_points_training_only=True,tree_still_ignored=True,robot_connected=False),indent=2),encoding='utf8');print(p.read_text())
if __name__=='__main__':main()
