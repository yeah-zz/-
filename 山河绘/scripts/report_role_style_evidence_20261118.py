# -*- coding: utf-8 -*-
import sys,json,html
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from role_style_evidence_20261118 import *

def main():
    examples=read(ROOT/'named_examples.json');result=read(ROOT/'results.json');refs=read(ROOT/'reviewed.json');freeze=read(ROOT/'reference_freeze.json')
    assert sha(ROOT/'reviewed.json')==freeze['reviewed_sha256']
    assert len({r['source_group'] for r in examples['records']})==1
    prior=read(LAB/'two_work_bands_20261118/checkpoint_freeze.json')
    assert all(sha(LAB/f'two_work_bands_20261118/{a}.pt')==h for a,h in prior.items())
    oldrefs=read(LAB/'cunfa_role_reference_20261112/source_references.json')['records'];matches={}
    for r in examples['records']:
        assert sha(r['image'])==r['image_sha256'] and sha(r['source_pdf'])==r['source_pdf_sha256']
        if r['id'] in ['small_axe','large_axe','long_hemp','short_hemp']:
            old=next(x for x in oldrefs if x['id']=='aam_'+r['id'])
            matches[r['id']]=np.array_equal(np.array(Image.open(old['image']).convert('RGB')),np.array(Image.open(r['image']).convert('RGB')))
    assert all(matches.values())
    for r in refs['records']:assert sha(r['partial_mask'])==r['partial_mask_sha256'] and r['assistant_reviewed']
    lock=read(LAB/'local_rock_patch_20261117/baseline_lock.json');bc={k:sha(lock[k])==lock[k+'_sha256'] for k in ['approved500','working489']};assert all(bc.values())
    cards=[];board=Image.new('RGB',(1260,960),'#f6f3eb');d=ImageDraw.Draw(board)
    for i,r in enumerate(examples['records']):
        im=Image.open(r['image']).convert('RGB');im.thumbnail((390,265));x=i%3*420;y=i//3*320;board.paste(im,(x+8,y+35))
        label=r['named_style']+(' · 初始类别' if r['initial_taxonomy']!='known_out_of_scope' else ' · 初始范围外')
        d.text((x+8,y+5),label,font=ImageFont.truetype(FONT,20),fill='#294136')
        cards.append('<article><h3>'+html.escape(label)+'</h3><img src="views/'+Path(r['image']).name+'" alt="原页具名示例"><p>示例级标签可用；逐像素角色、笔顺和真实分笔未知。</p></article>')
    board.save(ROOT/'named_examples_overview.png')
    table='<table><tr><th>新增片段</th><th>冻结旧模型</th><th>核心对照</th><th>核心＋纸面</th></tr>'
    for r in result['records']:
        table+='<tr><td>'+r['id']+'</td>'
        for a in ['teacher','core_control','band_paper']:
            s=r['scores'][a];table+=f'<td>墨迹 {s["ink_retained"]}/{s["ink_core"]}<br>纸面误检 {s["paper_false_ink"]}/{s["paper"]}</td>'
        table+='</tr>'
    table+='</table>'
    checks=dict(source_pdf_captions_and_illustrator_visually_checked=True,original_four_examples_pixel_match=matches,
        examples=9,initial_scope_examples=4,known_out_of_scope_examples=5,illustrators=1,source_documents=1,
        new_role_fragments=2,role_works=1,known_ink_pixels=48,near_paper_pixels=219,
        label_level_tests_passed=5,reference_hash_matches=True,frozen_models_unchanged=True,
        example_style_not_pixel_role=True,video_role_not_named_cunfa=True,neighbor_ink_not_paper=True,
        baseline_hashes=bc,new_model_trained=False,independent_validation=False,adopted=False,robot_connected=False)
    write('verification.json',checks)
    p='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>皴法名称与内部笔迹证据分级</title><style>body{font:17px/1.7 system-ui;max-width:1350px;margin:auto;padding:24px;background:#f6f3eb;color:#294136}.notice{padding:18px;background:#e8ddc7}img{display:block;max-width:100%;height:auto;margin:18px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:12px;white-space:nowrap}.scroll{overflow:auto}a{color:#17694f}</style><h1>皴法名称与内部笔迹：两级证据分别核查</h1><p class="notice">本轮补证据与监督接口，没有训练新模型。核对9个原页具名示例：4个属初始披麻/斧劈范围，5个为已知范围外类型；都来自同一文档和同一绘者，不能算9件独立作品。原先4个示例的像素与旧参考完全一致，本轮没有把它们冒充新增图片。</p><p>示例级皴法名称不自动下放为逐像素皴笔标签；视频前后阶段确认的内部墨迹也不自动获得皴法名称。标签契约已用5项逻辑检查验证这两条边界。初始范围外类型不等于无标签、混合或画家未知。</p><h2>教学原页及具名示例</h2><p><a href="pdf_page2.png">查看完整原页图注及署名</a></p><img src="named_examples_overview.png"><details><summary>逐个示例与适用范围</summary><div class="grid">CARDS</div></details><h2>两段新增内部笔迹（助手参考，非模型）</h2><img src="views/lower_left_review.png"><img src="views/lower_right_review.png"><p>这些是上一训练视频同一帧的其他可见片段，不是新来源或真实分笔答案。阶段前后支持其内部墨迹角色，来源仍不能给出可靠的局部皴法名称。邻近另一笔的可能墨迹已从纸面监督中剔除；第二段的断续以下区域不参与标签。</p><div class="scroll">TABLE</div><p>48个确定墨迹核心均保留，219个近邻纸面中有171个被三个冻结模型同样误检。上一轮局部改善未覆盖这些位置，存在窄线之间粘连的问题。仅为事后已知训练帧诊断，不能据此选模或证明泛化。独立验证未进行，模型不采用。</p><p><a href="named_examples.json">名称与来源范围</a> · <a href="reviewed.json">阶段与空间参考</a> · <a href="reference_freeze.json">当前计分前冻结</a> · <a href="results.json">全部诊断</a> · <a href="verification.json">完整性检查</a></p></html>'''
    (ROOT/'index.html').write_text(p.replace('CARDS',''.join(cards)).replace('TABLE',table),encoding='utf8')
    report=PROJECT/'reports/20261118_role_style_evidence.md'
    report.write_text('''# 皴法名称与内部笔迹证据分级

## 推进哪项目标
推进G1所需的可核查皴法名称与内部笔迹角色证据；保持对象、角色和类别监督层级，避免G2中的树木误入皴笔。G3/G4未提前启动。

## 实际新增什么能力
核对AAM PDF第2页的9个图像位置、图注和Lampo Leong署名。4个初始范围示例是原有图片的复核，另5个类型用于已知范围外参考；全部仅同一文档/同一绘者，不作为9个独立作品。新增标签分级接口：具名示例仅提供示例级类别，视频墨带仅提供空间角色，二者不能自动互相扩展；5项逻辑检查通过。
同一训练视频新增2段内部墨迹片段参考，48个确定核心像素、219个近邻纸面像素。来源前后帧支持角色，不能证明具名皴法；不是新增独立教程或完整笔画。没有新增模型识别能力。

## 还有哪些失败
三个冻结模型在新片段上均保留48个核心，但近邻纸面均误检171/219；上一轮改善未覆盖这里，窄线之间仍会粘连。第二件视频作品的可靠内部纹理角色仍缺少，不能建立跨作品角色与皴法联合监督。真实分笔、笔顺和物理参数未知。

## 是否通过独立验证
未进行。本轮是已有教学资料核查与已知训练帧事后参考。来源、像素一致性、冻结权重、参考与500/489笔基线哈希已检查，数据与逻辑检查不代表模型验收。未训练、未采用新模型，机械臂离线。

## 下一步
补第二件作品的内部笔迹阶段证据及不同绘者的具名皴法参考，训练/开发/验收按作品及绘者分组。固定本轮错例，避免只压纸面误检而删除真实细笔。尚不把示例级类别扩成像素级标签或完整笔触答案。
''',encoding='utf8')
    statepath=LAB/'guohua_hierarchy_20261112/current_milestones.json';s=read(statepath);assert not any(s['capability_state'].values())
    s.update(latest_round_report=str(report),latest_role_style_evidence=str(ROOT/'results.json'),new_model_trained_this_round=False,production_replaced=False,robot_connected=False)
    statepath.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf8')
    for name in ['project_plan_20261117.json','guohua_four_priorities_execution_20261116.json']:
        path=PROJECT/'research'/name;s=read(path);s.update(latest_round_report=str(report),latest_role_style_evidence=str(ROOT/'results.json'));path.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(checks,ensure_ascii=False))

if __name__=='__main__':main()
