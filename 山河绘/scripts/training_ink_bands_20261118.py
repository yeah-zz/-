# -*- coding: utf-8 -*-
"""Training-side, source-reviewed variable ink-band envelopes; not expert truth."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from gongbi_training.partial_ink_bands import rasterize_band,merge_bands
ROOT=LAB/'training_ink_bands_20261118';OLD=LAB/'cunfa_roles_v2_20261112'
FONT='C:/Windows/Fonts/msyh.ttc'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf8'))
def write(name,data):(ROOT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')


# Explicit assistant cross-section choices from the native source and row profiles.
# No image threshold, teacher prediction or skeleton is used to construct targets.
SPECS=[
 dict(id='outer_right',role=1,role_detail='visible_outer_contour',uncertain_rows=[],
      sections=[[80,128,129,127,130],[85,132,134,131,135],[90,138,140,136,141],[95,143,145,142,146],
                [100,147,147,146,148],[105,149,150,148,151],[110,152,153,151,154],[115,155,156,154,157],
                [120,157,158,156,159],[125,160,161,159,162],[130,162,163,161,164]]),
 dict(id='outer_left',role=1,role_detail='visible_outer_contour',uncertain_rows=[[193,197]],
      sections=[[145,60,62,59,63],[150,58,60,57,61],[155,57,59,56,60],[160,55,58,54,59],
                [165,54,55,53,57],[170,52,54,51,55],[175,51,52,50,53],[180,49,50,48,51],
                [185,47,49,46,50],[190,46,47,45,48],[195,44,45,43,46],[200,44,44,43,45],[205,43,43,42,44]]),
 dict(id='small_rock_side',role=1,role_detail='visible_structure_edge_object_instance_unconfirmed',uncertain_rows=[],
      sections=[[270,83,83,82,84],[275,81,82,80,83],[280,80,80,79,81],[285,79,79,78,80],
                [290,78,78,77,79],[295,76,77,75,78],[300,75,76,74,77],[305,74,75,73,76]]),
 dict(id='added_mark_left',role=2,role_detail='video_supported_added_interior_mark_not_named_cunfa',uncertain_rows=[[193,197]],
      sections=[[150,91,92,90,93],[155,86,87,85,88],[160,83,83,82,84],[165,79,80,78,81],
                [170,76,77,75,78],[175,74,75,73,76],[180,72,72,71,73],[185,70,70,69,71],
                [190,68,68,67,69],[195,66,67,65,68],[200,65,65,64,66]]),
 dict(id='added_mark_right',role=2,role_detail='video_supported_added_interior_mark_not_named_cunfa',uncertain_rows=[],
      sections=[[165,96,97,95,98],[170,93,94,92,95],[175,90,91,89,92],[180,88,88,87,89],
                [185,85,86,84,87],[190,83,84,82,85],[195,80,81,79,82],[200,78,79,76,80],[205,75,76,74,77]])
]


def build():
    assert not (ROOT/'dataset_manifest.json').exists(),'Keep scored references immutable'
    ROOT.mkdir(exist_ok=True);(ROOT/'views').mkdir(exist_ok=True)
    ds=read(OLD/'dataset_manifest.json');record=next(r for r in ds['records'] if r['id']=='L_149_3')
    assert record['split']=='train' and sha(record['path'])==record['sha256']
    z=dict(np.load(record['path']));image=z['image'];shape=image.shape[:2]
    assert np.array_equal(image,np.asarray(Image.open(OLD/'references/L_149_3_input.png')))
    rows=[]
    for spec in SPECS:
        b=rasterize_band(shape,spec['sections'],spec['role'],spec['uncertain_rows'])
        yy,xx=np.where(b['possible']|b['paper']);box=[max(0,int(xx.min())-5),max(0,int(yy.min())-5),min(shape[1],int(xx.max())+6),min(shape[0],int(yy.max())+6)]
        overlay=image.copy();overlay[b['fringe'] if 'fringe' in b else b['possible']&~b['core']]=[219,186,118]
        overlay[b['core']]=[195,116,38] if spec['role']==1 else [35,128,180]
        overlay[b['paper']]=[107,173,139]
        panels=[image,z['before'],overlay];cw,ch=410,390;board=Image.new('RGB',(cw*3,ch+65),'#f6f3eb');d=ImageDraw.Draw(board)
        for j,(a,label) in enumerate(zip(panels,['原图当前帧','已对齐的阶段前帧','橙/蓝=确定墨迹 绿=纸面'])):
            p=Image.fromarray(a).crop(box);scale=min((cw-15)/p.width,(ch-40)/p.height);p=p.resize((round(p.width*scale),round(p.height*scale)),Image.Resampling.NEAREST)
            board.paste(p,(j*cw+5,40));d.text((j*cw+5,8),label,font=ImageFont.truetype(FONT,19),fill='#263d36')
        d.text((8,ch+18),spec['id']+'：黄色边缘、断续和真实起收笔未知；不是模型输出。',font=ImageFont.truetype(FONT,20),fill='#79563d')
        board.save(ROOT/'views'/(spec['id']+'_draft.png'))
        rows.append(dict(**spec,view_box=box,assistant_reviewed=False,expert_verified=False,object_type='rock',
            source_work=record['source_url'],source_family=record['source_family'],source_frame='L_149_3',
            role_evidence='visible original/before stage pair; aligned before is context only, no pixel delta as target',
            before_seconds=130,after_seconds=149.3,complete_stroke=False,whole_rock_complete=False,
            true_endpoints=None,named_cunfa=None,brush_pressure=None,brush_speed=None,
            training_allowed_after_review=True,validation_scope='known_training_work_only'))
    write('draft_references.json',dict(records=rows,source=record,source_image_sha256=sha(OLD/'references/L_149_3_input.png'),
          before_image_sha256=sha(OLD/'references/L_149_3_before.png'),split='training_development',independent_validation=False,
          unique_works=1,current_images=1,paired_before_frames=1,not_independent_new_examples=True,
          target_construction='manually chosen variable cross-section bounds, interpolated only inside reviewed extents; fringe/ends/uncertain gaps ignored'))
    print('Built five draft band envelopes from one training-side work')


def admit():
    assert not (ROOT/'dataset_manifest.json').exists(), 'Admission frozen; use a new experiment for revisions'
    refs=read(ROOT/'reviewed_references.json');assert all(r['assistant_reviewed'] for r in refs['records'])
    z=dict(np.load(refs['source']['path']));bands=[rasterize_band(z['image'].shape[:2],r['sections'],r['role'],r['uncertain_rows']) for r in refs['records']]
    targets=merge_bands(bands);assert not targets['role_conflicts'].any()
    np.savez_compressed(ROOT/'partial_bands.npz',image=z['image'],before=z['before'],**targets)
    old=z['targets'];paper_on_old_ink=int(((targets['ink_targets']==0)&np.isin(old,[1,2])).sum())
    opposite_role=int(((targets['role_targets']!=255)&np.isin(old,[1,2])&(old!=targets['role_targets'])).sum())
    manifest=dict(reference_sha256=sha(ROOT/'reviewed_references.json'),data=str(ROOT/'partial_bands.npz'),data_sha256=sha(ROOT/'partial_bands.npz'),
        legacy_source_sha256=sha(refs['source']['path']),source_work=refs['source']['source_url'],source_family=refs['source']['source_family'],
        source_split='train',unique_works=1,current_images=1,paired_before_frames=1,reviewed_bands=len(bands),
        sure_ink_pixels=int((targets['ink_targets']==1).sum()),reviewed_paper_pixels=int((targets['ink_targets']==0).sum()),
        unknown_fringe_pixels=int(targets['fringe'].sum()),role_counts={str(k):int((targets['role_targets']==k).sum()) for k in [1,2]},
        conflicts_with_legacy=dict(paper_on_legacy_ink=paper_on_old_ink,opposite_role=opposite_role),
        label_scope='assistant variable-width partial ink envelopes, not exact full ink masks or artist strokes',
        source_reviewed=True,expert_verified=False,training_eligible=True,independent_validation=False,
        named_cunfa_supervision=False,tree_skill_training=False,new_model_trained=False,robot_connected=False)
    write('dataset_manifest.json',manifest);write('reference_freeze.json',dict(reference_sha256=manifest['reference_sha256'],data_sha256=manifest['data_sha256'],
        source_already_used_for_training=True,independent_validation=False,frozen_before_new_diagnostic_scoring=True))
    print(json.dumps(manifest,ensure_ascii=False))


if __name__=='__main__':{'build':build,'admit':admit}[sys.argv[1]]()
