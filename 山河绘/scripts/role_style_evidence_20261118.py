# -*- coding: utf-8 -*-
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
sys.path.insert(0,str(LAB/'guohua_system_review_20261110/deps'))
import fitz,numpy as np
from PIL import Image,ImageDraw,ImageFont
from gongbi_training.partial_ink_bands import rasterize_band,merge_bands
from gongbi_training.cunfa_evidence_levels import supervision_for
ROOT=LAB/'role_style_evidence_20261118';OLD=LAB/'cunfa_roles_v2_20261112';PROJECT=Path(__file__).resolve().parents[1]
FONT='C:/Windows/Fonts/msyh.ttc'
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):(ROOT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')

def prepare():
    assert not (ROOT/'draft.json').exists();(ROOT/'views').mkdir(exist_ok=True)
    pdf=LAB/'cunfa_gabor_baseline_20261110/sources/aam.pdf';doc=fitz.open(pdf);page=doc[1]
    imgs=[]
    for im in page.get_images(full=True):
        if im[2]<700 or im[3]<400:continue
        box=page.get_image_rects(im[0])[0];imgs.append((round(box.y0/100),box.x0,im[0],list(box)))
    imgs.sort();assert len(imgs)==9
    names=[('small_axe','小斧劈','axe'),('large_axe','大斧劈','axe'),('long_hemp','长披麻','hemp'),('short_hemp','短披麻','hemp'),('raveled_rope','解索','known_out_of_scope'),('cloud_head','云头','known_out_of_scope'),('mi_dot','米点','known_out_of_scope'),('dragged_mud','拖泥带水','known_out_of_scope'),('nail_head','钉头','known_out_of_scope')]
    rows=[]
    for (_,_,xref,box),(ident,label,tax) in zip(imgs,names):
        z=doc.extract_image(xref);p=ROOT/'views'/(ident+'.'+z['ext']);p.write_bytes(z['image'])
        r=dict(id=ident,kind='named_example',image=str(p),image_sha256=sha(p),publisher='Asian Art Museum',illustrator='Lampo Leong',source_group='AAM_Lampo_Leong_all_examples',
            source_pdf=str(pdf),source_pdf_sha256=sha(pdf),page_one_based=2,pdf_image_box=box,source_url='https://education.asianart.org/wp-content/uploads/sites/6/2019/09/Chinese-Brushstrokes-Vocabulary-AAM_0.pdf',
            named_style=label,initial_taxonomy=tax,label_scope='whole_teaching_example',caption_checked=True,source_seen_before=True,
            independent_validation=False,expert_pixel_verified=False,complete_stroke=False,pixel_roles=None)
        r['allowed_supervision']=supervision_for(r);rows.append(r)
    write('named_examples.json',dict(records=rows,example_units=9,illustrators=1,source_documents=1,not_nine_independent_works=True,new_independent_sources=0))
    data=dict(np.load(OLD/'references/L_149_3.npz'));im=data['image'];rows=[]
    specs=[('lower_left',[[255,46,47,45,48],[260,45,45,44,46],[265,44,44,43,45],[270,43,43,42,44],[275,41,42,40,43],[280,40,41,39,42],[285,39,40,38,41],[290,38,39,37,40],[295,37,38,36,39],[300,36,37,35,38]]),
        ('lower_right',[[255,55,56,53,57],[260,54,54,52,55],[265,53,53,51,54],[270,52,52,50,53]])]
    for ident,sections in specs:
        b=rasterize_band(im.shape[:2],sections,2)
        # Adjacent faint stroke is not a paper negative. Keep only left-side paper for first band.
        if ident=='lower_left':
            yy,xx=np.indices(im.shape[:2]);mid=np.interp(yy[:,0],[s[0] for s in sections],[(s[1]+s[2])/2 for s in sections]);remove=b['paper']&(xx>mid[:,None]);b['paper'][remove]=False;b['ink_targets'][remove]=255
        mask=ROOT/(ident+'.npz');np.savez_compressed(mask,**b)
        overlay=im.copy();overlay[b['possible']&~b['core']]=[220,188,122];overlay[b['core']]=[35,128,180];overlay[b['paper']]=[105,172,137]
        box=[25,245,70,310];board=Image.new('RGB',(1020,570),'#f6f3eb');d=ImageDraw.Draw(board)
        for i,(a,label) in enumerate([(data['before'],'阶段前'),(im,'当前原图'),(overlay,'蓝=内部墨迹 绿=纸面')]):
            p=Image.fromarray(a).crop(box).resize((315,455),Image.Resampling.NEAREST);board.paste(p,(i*340+5,45));d.text((i*340+5,8),label,font=ImageFont.truetype(FONT,20),fill='#294136')
        d.text((8,525),ident+'：局部片段，非完整分笔；边缘及邻近淡痕未知。',font=ImageFont.truetype(FONT,20),fill='#79563c');board.save(ROOT/'views'/(ident+'_review.png'))
        rows.append(dict(id=ident,kind='video_supported_band',object_type='rock',named_style=None,sections=sections,
            partial_mask=str(mask),partial_mask_sha256=sha(mask),source_npz=str(OLD/'references/L_149_3.npz'),source_sha256=sha(OLD/'references/L_149_3.npz'),
            source_work=read(OLD/'dataset_manifest.json')['records'][2]['source_url'],source_group='liupengkai_qiushan',
            stage_pair_checked=True,before_seconds=130,after_seconds=149.3,assistant_reviewed=False,
            complete_stroke=False,independent_validation=False,expert_pixel_verified=False,role='added_interior_ink_fragment',
            stage_evidence='Visible new interior marks absent in registered earlier frame; no direction or true stroke identity inferred'))
    write('draft.json',dict(records=rows,unique_role_works=1,no_named_cunfa_inferred_from_role=True))
    print('9 source-caption examples,2 video-supported fragment candidates')

def admit():
    assert not (ROOT/'results.json').exists();refs=read(ROOT/'reviewed.json');data=dict(np.load(OLD/'references/L_149_3.npz'))
    write('reference_freeze.json',dict(reviewed_sha256=sha(ROOT/'reviewed.json'),note='Frozen before current fragment scoring; source and earlier full predictions already seen'))
    rows=[]
    for r in refs['records']:
        assert sha(r['partial_mask'])==r['partial_mask_sha256'];assert sha(r['source_npz'])==r['source_sha256'];r['allowed_supervision']=supervision_for(r)
        m=dict(np.load(r['partial_mask']));scores={}
        for a in ['teacher','core_control','band_paper']:
            p=np.load(LAB/f'two_work_bands_20261118/predictions/L_{a}.npz')['prediction'];core=m['core'];paper=m['paper']
            scores[a]=dict(ink_retained=int((core&(p>0)).sum()),ink_core=int(core.sum()),paper_false_ink=int((paper&(p>0)).sum()),paper=int(paper.sum()))
        rows.append(dict(**r,scores=scores))
    write('results.json',dict(records=rows,scope='posthoc known training-frame references; no new model or independent test',
        new_model_trained=False,independent_validation=False,adopted=False,robot_connected=False))
    print(json.dumps([{r['id']:r['scores']} for r in rows],ensure_ascii=False))

if __name__=='__main__':{'prepare':prepare,'admit':admit}[sys.argv[1]]()
