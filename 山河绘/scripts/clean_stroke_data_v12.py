# -*- coding: utf-8 -*-
"""Sparse drafts from viewed tutorials, plus unchanged user-reviewed labels."""
import sys,json,math,shutil
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,cv2
from PIL import Image,ImageDraw
from clean_strokes_v12 import ROOT,FONT,jsave,digest
from human_mountain_v11 import fields

# Native-image coordinates, individually traced by the assistant. These are
# partial reference trajectories, not inferred historic brush lifts or gold.
DRAFTS={
'heye_01':[[[97,338],[105,300],[114,266],[123,236],[133,216],[142,196],[148,178],[158,171],[171,167],[179,173],[184,189],[188,211]]],
'heye_03':[
 [[84,398],[91,373],[108,357],[115,338],[123,311],[132,285],[141,255],[150,224],[161,194],[170,183],[186,177],[197,183],[205,201]],
 [[205,202],[208,214],[201,225],[193,227],[182,242],[169,264],[160,289],[153,316],[146,344],[146,376],[147,410]],
 [[111,342],[122,333],[133,320],[147,312]],[[208,211],[218,225],[225,245],[231,270],[238,290],[250,312]],
 [[227,269],[228,286],[218,310],[211,331],[206,350]],[[110,366],[115,384],[109,400]]],
'heye_04':[
 [[54,351],[68,337],[78,319],[85,287],[97,253],[106,218],[116,188],[131,165],[139,146],[161,138],[176,147]],
 [[175,146],[181,164],[187,177],[177,183],[169,198],[163,219],[151,232],[140,251],[131,273],[124,296],[115,318],[114,346],[113,389],[111,428]],
 [[176,147],[190,151],[204,163],[215,171],[224,187],[229,207],[236,216],[248,223],[259,238],[270,261],[280,281],[287,298],[301,306],[308,324],[315,352],[321,378],[333,405],[345,449],[355,487]],
 [[189,182],[201,190],[207,205],[219,222],[218,243],[225,265],[234,278],[244,292],[258,303]],
 [[201,190],[189,198],[178,213],[169,234],[165,258],[157,283],[150,309],[141,340]],
 [[237,216],[237,232],[228,248],[217,262],[205,280],[197,297]],
 [[255,233],[257,254],[253,275],[255,300],[269,327],[274,350],[269,379],[258,406]],
 [[270,263],[272,282],[264,300],[252,320],[237,340],[221,363],[206,387],[192,412],[181,439],[173,456]],
 [[288,299],[303,299],[316,310],[332,329],[343,350],[348,371]],
 [[321,377],[312,391],[309,416],[302,444],[297,472]],
 [[113,347],[122,355],[131,372],[142,400],[153,429]],
 [[81,371],[78,392],[75,419],[69,440],[57,455],[42,464]]],
'pima_01':[
 [[31,372],[40,345],[51,319],[67,299],[78,277],[98,266],[112,251],[124,241],[137,239],[150,252]],
 [[41,355],[60,332],[73,314],[84,293],[97,286],[112,292],[124,277],[131,260],[140,252],[150,260],[158,282],[166,299],[180,313],[191,340],[202,369],[213,404],[216,425]],
 [[149,253],[158,265],[164,284],[169,306],[175,331],[180,359],[183,394],[187,424]],
 [[80,376],[90,349],[98,320],[107,309],[116,313],[123,331],[131,352],[137,379],[141,405],[143,425]],
 [[127,329],[143,331],[153,346],[165,371],[175,401]],
 [[130,350],[144,357],[153,378],[159,408]]],
'pima_02':[
 [[30,325],[46,295],[63,267],[79,245],[94,235],[112,233],[129,216],[143,182],[157,179],[168,185],[182,208],[188,231]],
 [[79,334],[89,307],[100,275],[111,261],[120,262],[129,281],[136,304],[146,326],[151,356],[156,384]],
 [[134,281],[149,283],[158,298],[166,323],[174,354],[180,397]],
 [[146,185],[159,196],[166,216],[170,239],[181,265],[192,294],[204,329],[209,366],[217,423]],
 [[172,239],[185,241],[198,229],[212,230],[226,209],[241,202],[250,207],[255,224],[273,231],[282,239],[295,237],[310,232],[323,239],[337,240],[350,251],[362,269],[369,293],[383,324],[395,362]],
 [[287,239],[302,250],[313,270],[318,296],[322,330],[327,366],[332,405]],
 [[369,277],[381,279],[392,294],[401,317],[407,350],[413,386],[419,424]],
 [[183,251],[195,265],[202,285],[213,304],[219,337],[229,389],[234,422]]],
'pima_04':[
 [[135,157],[148,132],[162,112],[177,99],[189,76],[202,56],[213,53],[224,62],[231,79]],
 [[173,137],[186,124],[199,132],[210,111],[219,104],[232,113],[237,137],[245,160],[248,186],[254,212]],
 [[226,58],[239,47],[254,50],[264,66],[275,81],[288,90],[300,105],[312,130],[320,156],[326,176]],
 [[236,110],[249,87],[259,93],[269,109],[285,120],[303,132],[313,151],[324,172],[339,181],[353,193],[367,203],[381,217],[391,240],[401,266],[414,290],[427,316],[441,333]],
 [[237,139],[251,130],[265,127],[279,139],[293,154],[304,178],[316,207],[321,236],[324,260]],
 [[218,128],[204,147],[196,172],[192,199],[183,227],[176,247]],
 [[40,340],[58,315],[77,289],[98,272],[115,256],[133,244],[147,247],[158,258],[166,279],[173,297],[184,306],[198,286],[215,266],[228,266],[239,281],[250,298]],
 [[75,378],[87,348],[97,321],[105,307],[116,306],[125,322],[131,344],[140,366],[145,394],[151,424]],
 [[132,252],[145,264],[153,288],[163,310],[173,335],[184,369],[190,398],[197,425]],
 [[174,299],[189,304],[204,301],[220,303],[233,313],[249,300],[263,294],[276,297],[290,304],[301,324],[309,349],[319,380],[327,424]],
 [[302,327],[322,325],[336,333],[348,348],[361,372],[371,400],[379,425]],
 [[353,192],[363,214],[370,243],[379,271],[390,299],[403,328]],
 [[437,334],[439,356],[438,382],[438,407],[439,426]]],
'wang_left':[
 [[0,181],[22,161],[38,144],[54,137],[70,140],[85,134],[96,118],[109,106],[128,102],[143,105]],
 [[75,142],[74,156],[83,164],[81,181],[78,204],[69,227],[56,244],[30,247],[12,249],[0,232]],
 [[82,165],[98,153],[108,145],[112,151],[113,166],[106,185],[108,198],[101,220],[90,245],[74,253],[57,251]],
 [[143,105],[136,113],[135,130],[136,150],[133,173],[133,198],[121,219]],
 [[53,206],[45,224],[41,239]]],
'wang_right':[
 [[403,237],[412,219],[420,198],[430,174],[438,158],[450,150],[469,154],[488,149],[504,139],[521,125],[538,112],[558,100],[575,102],[593,116]],
 [[406,243],[421,249],[438,254],[451,265],[464,282],[480,291],[501,297],[521,289],[541,285],[557,274]],
 [[445,252],[444,269],[454,280],[467,289],[480,294]],
 [[474,179],[493,171],[512,163],[528,155],[550,149],[571,133],[584,120]],
 [[497,188],[518,179],[537,176],[550,165],[565,157],[576,142]],
 [[512,225],[529,214],[542,200],[549,185]],
 [[546,281],[553,259],[562,236],[572,215],[588,201],[599,187],[605,168]]]
}
SPECS=[('C01','heye_01',(65,125,225,365),'train'),('C02','heye_03',(35,135,275,418),'train'),('C03','heye_04',(25,110,385,505),'train'),('C04','pima_01',(15,205,245,427),'train'),('C05','pima_02',(15,140,445,427),'train'),('C06','pima_04',(15,20,450,427),'train'),('T01','wang_left',(0,70,149,278),'test'),('T02','wang_right',(390,90,620,307),'test')]

def build():
    data=ROOT/'data';data.mkdir(exist_ok=True);records=[];sources={r['id']:r for r in json.loads((ROOT/'web_sources.json').read_text(encoding='utf-8'))}
    board=Image.new('RGB',(1400,330*len(SPECS)),'#f5f3ec');draw=ImageDraw.Draw(board)
    for i,(identity,key,box,split) in enumerate(SPECS):
        source_key='wang_steps_00' if key.startswith('wang_') else key;r=sources[source_key];original=Image.open(r['image']).convert('RGB');crop=original.crop(box);w,h=crop.size;left=max(0,(320-w)//2);top=max(0,(320-h)//2);image=Image.new('RGB',(max(320,w),max(320,h)),tuple(np.asarray(crop)[0,0]));image.paste(crop,(left,top));path=data/(identity+'.png');image.save(path)
        strokes=[]
        for k,p in enumerate(DRAFTS[key]):
            q=[[round(x-box[0]+left,2),round(y-box[1]+top,2)] for x,y in p];q=[[min(image.width-1,max(0,x)),min(image.height-1,max(0,y))] for x,y in q]
            strokes.append(dict(id=f'draft_{k+1}',kind='stroke',points=q,role='contour',cunfa='not_applicable',width_profile=['unknown']*3,tone_profile=['unknown']*3,direction_known=False,note='assistant initial trace; not expert verified'))
        # The upper 8 native pixels of each selected training crop were viewed
        # as empty paper. Do not mark untraced ink elsewhere as background.
        regions=[]
        if split=='train':regions=[dict(id='paper_strip',kind='background',points=[[left,top],[left+w-1,top],[left+w-1,top+7],[left,top+7]],reason='blank',note='assistant visually judged paper, not expert gold')]
        doc=dict(width=image.width,height=image.height,sample_id=identity,strokes=strokes,regions=regions,review_state='draft',annotation_scope='partial',unlabeled='unknown',label_provenance='assistant_initial_visual_trace_not_expert')
        center,a,m,dist=fields(doc);jsave(data/(identity+'.json'),doc);np.savez_compressed(data/(identity+'_geometry.npz'),center=center,angles=a,mask=m,dist=dist)
        records.append(dict(id=identity,image=str(path),labels=str(data/(identity+'.json')),split=split,source_group=r['source_group'],source_page=r['source_page'],source_image_sha256=r['sha256'],image_sha256=digest(path),label_sha256=digest(data/(identity+'.json')),style_hint='heye' if key.startswith('heye') else 'pima' if key.startswith('pima') else 'rock_outline',style_hint_not_expert_verified=True,provenance=doc['label_provenance'],complete=False,source_box=list(box),padding=[left,top],paths=strokes))
        overlay=np.asarray(image).copy();overlay[cv2.dilate(center,np.ones((2,2),np.uint8))>0]=[235,80,20]
        for j,im in enumerate((image,Image.fromarray(overlay))):
            im=im.copy();im.thumbnail((680,290));board.paste(im,(j*700+10,i*330+30))
        draw.text((10,i*330+5),identity+' '+split+' | assistant drafts, NOT user/expert gold',font=FONT,fill='black')
    jsave(ROOT/'clean_manifest.json',dict(records=records,training_source_groups=['artist_steps'],test_source_groups=['wang_steps'],exact_sources_do_not_cross=True,underlying_book_family_verified=False,selected_source_illustrations=7,new_crops=8,notes='6 training illustrations are progressive views of 2 motifs; two test crops come from one illustration. Not 8 independent artworks. pima motif resembles an earlier user example; it is not counted as novel independent content.'))
    board.save(ROOT/'review'/'initial_drafts.jpg',quality=95);print('CLEAN_DRAFTS',len(records),'paths',sum(len(r['paths']) for r in records),flush=True)
if __name__=='__main__':build()
