# -*- coding: utf-8 -*-
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from gongbi_training.cunfa_context_roles import context_role_target
ROOT=LAB/'cunfa_spatial_pairs_20261117'
def main():
 assert not (ROOT/'reference_manifest.json').exists()
 d=json.loads((ROOT/'candidate_references.json').read_text('utf8'));changes=[]
 reasons={'wanghui_1':'外缘过淡，局部与空景界限难可靠划定','wanghui_3':'近树点与山面纹理混合，不能确认单一山石内部角色','wanghui_5':'灰面中仍有明显细短墨迹，不作为纯连续淡墨参考',
 'gongxian_0':'选区含岩面与下方粗横墨，单一结构角色待定','gongxian_3':'浓墨叠加与内部细笔难分，待定'}
 for r in d['records']:
  if r['id'] in reasons:
   changes.append(dict(id=r['id'],old=r['context_role_target'],new=None,reason=reasons[r['id']]))
   r['context_role_target']=None;r['role_reference_checked']=False;r['visual_evidence']+='；复核：'+reasons[r['id']]
 rows=d['records'];count=sum(context_role_target(r)!=-100 and r['split']=='train' for r in rows);fresh=sum(context_role_target(r)!=-100 and r['split']=='fresh_check' for r in rows)
 out=dict(records=rows,changes_before_model_predictions=changes,training_role_references=count,fresh_role_checks=fresh,
   reference_scope='assistant context references; not expert masks; fresh work attribution retained as attributed; not author-independent certification',
   targets_frozen_before_features=True,production_replaced=False,robot_connected=False)
 (ROOT/'reference_manifest.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
 print('TRAIN',count,'FRESH',fresh,'NEW_UNKNOWN',sum(r.get('added_this_round',False) and context_role_target(r)==-100 for r in rows))
if __name__=='__main__':main()
