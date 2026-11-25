import sys,json,os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import torch
from gongbi_training.cunfa_branch_isolation import IsolatedRoleHead
ROOT=LAB/'cunfa_branch_isolation_20261117'
def main():
 torch.set_num_threads(4);x=torch.randn(2,19,56,56);roi=torch.ones(2,56,56);initial={};checks={}
 for mode in ['shared','detached_heads','separate_trunks']:
  torch.manual_seed(0);m=IsolatedRoleHead(mode);o=m(x,roi);initial[mode]=o[0].detach();loss=o[1].square().mean()+o[2].square().mean()+o[3].square().mean();loss.backward();reach=any(p.grad is not None and p.grad.abs().sum()>0 for p in m.shared.parameters());assert reach==(mode=='shared');assert not any(p.grad is not None for p in m.role.parameters())
  if mode=='separate_trunks':assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.auxiliary_shared.parameters())
  checks[mode+'_gradient_routing']=True
 assert all(torch.equal(initial['shared'],v) for v in initial.values());checks['identical_initial_role_outputs']=True;checks['robot_connected']=False;ROOT.mkdir(exist_ok=True);(ROOT/'contract_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf8');print(json.dumps(checks))
if __name__=='__main__':main()
