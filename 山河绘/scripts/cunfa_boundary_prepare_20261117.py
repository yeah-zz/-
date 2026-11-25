# -*- coding: utf-8 -*-
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'cunfa_boundary_relations_20261117';OLD=LAB/'cunfa_spatial_pairs_20261117'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not (ROOT/'reference_manifest.json').exists()
 d=json.loads((OLD/'reference_manifest.json').read_text('utf8'));rows=d['records'];src=json.loads((ROOT/'source_inventory.json').read_text('utf8'));im=Image.open(src['image']).convert('RGB');(ROOT/'crops').mkdir(exist_ok=True)
 specs=[([248,193,300,229],'outline_context','上方主山峰外缘转向，避开题字'),([552,409,603,451],'outline_context','右侧远山外缘，与空景相接'),
 ([319,266,347,298],'interior_mark_context','主山内部平行短墨笔，不把山顶外缘作为皴笔'),([499,570,538,592],'interior_mark_context','山脚岩面内部短线纹理'),
 ([432,478,451,500],'diffuse_tone','斜坡面较连续灰调候选，需近看排除细线'),([223,474,243,503],'diffuse_tone','左崖面连续浅灰调候选，需近看排除纸色'),
 ([452,680,485,708],'unmarked_surface','画内空白水面，非石内留白'),([577,686,608,713],'unmarked_surface','远近山之间空白水面，避开印章'),
 ([442,868,487,905],None,'前景树叶，明确不监督为山石皴笔')]
 for i,(box,target,evidence) in enumerate(specs):
  box=[round(v*im.size[j%2]/src['overview_size'][j%2]) for j,v in enumerate(box)];p=ROOT/'crops'/f'nizan_{i}.png';im.crop(box).save(p)
  rows.append(dict(id=f'nizan_{i}',box=box,image=str(p),image_sha256=sha(p),reference_image=src['image'],source_url=src['source_url'],role_fold_group=src['group'],split='fresh_check',object_type='tree' if i==8 else 'paper' if target=='unmarked_surface' else 'rock',context_role_target=target,object_reference_checked=True,role_reference_checked=target is not None,role_reference_scope='region_context_not_pixel_or_stroke_truth',reviewer='assistant_visual_context_reference',visual_evidence=evidence,automatic_prediction_as_label=False))
 # Coordinates refer to original source pixels except museum overview explicitly scaled below.
 curves=[
 dict(id='qingpu_18_0_region_0',line=[[37,149],[43,126],[55,112],[78,105],[100,116],[121,127]],rock=[[66,133],[94,142]],open=[[46,106],[101,96]]),
 dict(id='qingpu_18_0_region_1',line=[[278,178],[284,171],[310,163],[329,160],[350,158],[366,174],[387,187]],rock=[[312,188],[349,184]],open=[[312,143],[351,139]]),
 dict(id='qingpu_26_new_0',line=[[146,158],[139,194],[132,230],[135,252]],rock=[[158,199],[153,230]],open=[[115,193],[110,226]]),
 dict(id='qingpu_26_new_1',line=[[148,157],[185,161],[215,174],[249,193],[270,213]],rock=[[192,188],[230,215]],open=[[194,142],[238,158]]),
 dict(id='qingpu_26_new_2',line=[[254,656],[274,641],[303,632],[335,628],[366,627],[402,626]],rock=[[315,654],[360,653]],open=[[314,608],[370,605]]),
 dict(id='wanghui_0',line=[[154,430],[167,414],[180,406],[187,407],[200,417]],rock=[[177,430],[190,438]],open=[[156,397],[193,391]],overview_size=[745,1400]),
 dict(id='nizan_0',line=[[248,227],[256,217],[265,207],[275,202],[285,204],[296,211]],rock=[[269,230],[287,235]],open=[[253,197],[286,187]],overview_size=[772,1400]),
 dict(id='nizan_1',line=[[554,431],[567,419],[573,418],[583,426],[595,437]],rock=[[572,443],[588,450]],open=[[558,411],[588,410]],overview_size=[772,1400])]
 byid={r['id']:r for r in rows}
 for c in curves:
  r=byid[c['id']];s=Image.open(r['reference_image']).size
  if 'overview_size' in c:
   ow,oh=c.pop('overview_size')
   for k in ['line','rock','open']:c[k]=[[round(x*s[0]/ow),round(y*s[1]/oh)] for x,y in c[k]]
  c.update(split=r['split'],group=r['role_fold_group'],source_image=r['reference_image'],source_url=r['source_url'],scope='assistant sparse boundary centre/side scribbles; not full ink edge or brush instance',checked=False)
 candidate=dict(records=rows,curves=curves,status='candidate only; review before fit',old_manifest_sha256=sha(OLD/'reference_manifest.json'))
 (ROOT/'candidate_references.json').write_text(json.dumps(candidate,ensure_ascii=False,indent=2),encoding='utf8')
 font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17);sheet=Image.new('RGB',(1200,600),'#f3f0e7');draw=ImageDraw.Draw(sheet)
 from gongbi_training.cunfa_context_roles import context_box
 for i,c in enumerate(curves):
  r=byid[c['id']];ref=Image.open(r['reference_image']).convert('RGB');b=context_box(r['box'],ref.size,2.5);ref=ref.crop(b);z=min(280/ref.width,210/ref.height);ref=ref.resize((round(ref.width*z),round(ref.height*z)));pd=ImageDraw.Draw(ref)
  pts=lambda key:[((x-b[0])*z,(y-b[1])*z) for x,y in c[key]]
  pd.line(pts('line'),fill='#e75535',width=2)
  for key,col in [('rock','#1f65dc'),('open','#00a566')]:
   for x,y in pts(key):pd.ellipse((x-3,y-3,x+3,y+3),fill=col)
  x=(i%4)*300+5;y=(i//4)*300+5;sheet.paste(ref,(x,y));draw.text((x,y+220),c['id'],fill='#243f35',font=font);draw.text((x,y+246),c['split'],fill='#805333',font=font)
 sheet.save(ROOT/'curve_candidates.jpg')
 # Native crops on display, never sharpened or replaced by a drawing.
 board=Image.new('RGB',(1000,660),'#f3f0e7');dr=ImageDraw.Draw(board)
 for i,r in enumerate(rows[-9:]):
  z=Image.open(r['image']).convert('RGB');s=min(230/z.width,160/z.height);z=z.resize((round(z.width*s),round(z.height*s)),Image.Resampling.NEAREST);x=i%4*250+5;y=i//4*220+5;board.paste(z,(x,y));dr.text((x,y+166),r['id']+' '+str(r['context_role_target']),fill='#243f35',font=font)
 board.save(ROOT/'fresh_candidates.jpg');print('PREPARED',len(rows),len(curves))
if __name__=='__main__':main()
