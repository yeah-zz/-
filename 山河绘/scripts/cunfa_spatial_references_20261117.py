# -*- coding: utf-8 -*-
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'cunfa_spatial_pairs_20261117';OLD=LAB/'cunfa_relational_roles_20261117'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
# Coordinates on the displayed museum overview, scaled once into native pixels.
# Targets are contextual assistant references, not individual brush instances.
SPECS={
 'wanghui':((745,1400),[
  ((156,402,207,438),'outline_context','山脊轮廓从空景进入山面，邻域可见内外关系'),
  ((523,631,559,677),'outline_context','近山右侧斜向外缘，避开顶部树点'),
  ((407,671,450,716),'interior_mark_context','山面内可见重复长短皴笔，不把山脊树点当皴'),
  ((238,826,276,861),'interior_mark_context','前景山面内部的斜向纹理组'),
  ((477,357,508,381),'diffuse_tone','远山右坡连续灰调，独立线条不明显，需结合原图核查'),
  ((276,344,298,371),'diffuse_tone','远山坡面连续灰调局部，保留邻域'),
  ((172,271,219,316),'unmarked_surface','绘画内部空景纸面，避开装裱和题字'),
  ((177,521,207,551),'unmarked_surface','左部空景纸面，邻近山体但无清晰墨迹')]),
 'gongxian':((1400,741),[
  ((127,476,180,505),'outline_context','左下岩块顶部外缘，邻域有纸面及岩面'),
  ((240,545,273,586),'outline_context','左下竖向石壁转折边，非树干'),
  ((118,558,162,601),'interior_mark_context','石面内部深浅短笔群，避开树冠'),
  ((609,646,659,678),'interior_mark_context','下部岩面内部重复墨笔'),
  ((1047,168,1090,194),'diffuse_tone','远山上部连续灰面，避开树与轮廓'),
  ((1125,213,1160,235),'diffuse_tone','远山右侧连续淡灰面，无明显分离线条'),
  ((940,65,985,101),'unmarked_surface','山体外的画内未着墨纸面'),
  ((400,406,450,442),'unmarked_surface','左中空景，避开淡山边与文字')]),
 'shen_attributed':((1200,785),[
  ((373,74,422,108),'outline_context','上方主山轮廓转折，避开山顶叶点'),
  ((741,169,791,196),'outline_context','远山右端外缘，邻域可分山面与空景'),
  ((613,615,644,655),'interior_mark_context','近景岩块内部竖向纹理，避开外轮廓'),
  ((1102,541,1141,574),'interior_mark_context','右侧岩面内部斜皴笔组'),
  ((359,391,405,432),'unmarked_surface','水面空白参考，不称石内留白'),
  ((803,273,846,311),'unmarked_surface','画内空白水面，避开树枝'),
  ((196,92,239,114),None,'蓝灰远山色层，不能确认等同纯墨淡染'),
  ((789,434,840,478),None,'红色树叶，明确不提供山石角色或皴笔监督'),
  ((922,406,971,435),None,'亭子屋面，不属于山石，排除山石角色')])}
def main():
 assert not (ROOT/'reference_manifest.json').exists(),'Frozen references'
 (ROOT/'crops').mkdir(exist_ok=True)
 sources=json.loads((ROOT/'source_inventory.json').read_text('utf8'));rows=json.loads((OLD/'reference_manifest.json').read_text('utf8'))['records'];added=[]
 for src in sources:
  im=Image.open(src['image']).convert('RGB');size,entries=SPECS[src['id']]
  for j,(b,target,evidence) in enumerate(entries):
   box=[round(v*im.size[i%2]/size[i%2]) for i,v in enumerate(b)];p=ROOT/'crops'/f"{src['id']}_{j}.png";im.crop(box).save(p)
   obj='tree' if '树叶' in evidence else 'architecture' if '亭子' in evidence else 'paper' if target=='unmarked_surface' else 'rock'
   row=dict(id=f"{src['id']}_{j}",box=box,image=str(p),image_sha256=sha(p),reference_image=src['image'],source_url=src['source_url'],split=src['split'],role_fold_group=src['group'],object_type=obj,
    context_role_target=target,object_reference_checked=True,role_reference_checked=target is not None,reviewer='assistant_visual_context_reference',visual_evidence=evidence,
    role_reference_scope='region_context_not_pixel_or_stroke_truth',automatic_prediction_as_label=False,artist=src['artist'],work_title=src['title'],added_this_round=True)
   added.append(row)
 # Reference candidates must be visually rechecked before admission/training.
 (ROOT/'candidate_references.json').write_text(json.dumps(dict(records=rows+added,added=added,status='awaiting_visual_recheck'),ensure_ascii=False,indent=2),encoding='utf8')
 font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
 sheet=Image.new('RGB',(1250,((len(added)+4)//5)*260),'#f3f0e7');d=ImageDraw.Draw(sheet)
 for k,r in enumerate(added):
  x=(k%5)*250;y=(k//5)*260;im=Image.open(r['image']).convert('RGB');im.thumbnail((232,165));im=im.resize((int(im.width*min(232/im.width,165/im.height)),int(im.height*min(232/im.width,165/im.height))))
  sheet.paste(im,(x+5,y+5));d.text((x+5,y+175),r['id'],fill='#243f35',font=font);d.text((x+5,y+202),str(r['context_role_target']),fill='#243f35',font=small)
 sheet.save(ROOT/'reference_candidates.jpg');print('CANDIDATES',len(added),flush=True)
if __name__=='__main__':main()
