from v20_sources import *
from human_mountain_v11 import paste
manifest=read(ROOT/'source_manifest.json');records=manifest['records'];known={r['id'] for r in records}
for r in read(ROOT/'caa_downloads.json'):
    im=Image.open(r['image']).convert('RGB');out=ROOT/'sources'/(r['id']+'.png');im.save(out)
    if r['id'] not in known:records.append(dict(r,image=str(out),width=im.width,height=im.height,source_page=SOURCES[0][1],source_group='caa_steps',sha256=digest(out),expert_verified=False,split='external_test',work_group='CAA_A' if int(r['id'][-2:])<=4 else 'CAA_B',independent_artwork_not_verified=True))
book='https://backend.sjhtbook.com/upload/file/2022/05/227447a6abc6438f878c5e05ed1c81c6.pdf'
crops=[('book_pima_juran',18,[112,827,645,1273],'pima','Juran'),('book_pima_wangmeng',18,[669,445,921,850],'pima','Wang Meng'),('book_pima_huang',19,[418,138,950,477],'pima','Huang Gongwang'),('book_fupi_diagram_a',22,[708,530,921,709],'fupi','Jieziyuan'),('book_fupi_diagram_b',22,[670,733,921,887],'fupi','Jieziyuan'),('book_fupi_litang',22,[112,634,646,1249],'fupi','Li Tang'),('book_fupi_mayuan',22,[669,1030,922,1250],'fupi','Ma Yuan'),('book_jiesuo',20,[649,919,923,1253],'jiesuo','Jieziyuan'),('book_heye',20,[553,291,922,597],'heye','Zhao Mengfu'),('book_yuntou',22,[669,171,922,460],'yuntou','Guo Xi')]
for identity,page,box,label,author in crops:
    im=Image.open(ROOT/'sources'/('book_page_'+str(page).zfill(3)+'.png')).convert('RGB').crop(box);out=ROOT/'sources'/(identity+'.png');im.save(out)
    if identity not in known:records.append(dict(id=identity,image=str(out),width=im.width,height=im.height,source_page=book,source_pdf_page_index=page,source_crop_box=box,source_group='publisher_book',work_group=author,label=label,label_basis='explicit textbook figure/section; dominant technique may be mixed',author=author,sha256=digest(out),expert_verified=False,split='classification_test_or_challenge',independent_artwork_not_verified=True))
class_labels={'mooc_gu_he_01':('pima','He Tianjian'),'mooc_gu_he_02':('pima','He Tianjian'),'mooc_gu_he_03':('fupi','Gu Kunbo'),'mooc_gu_he_04':('fupi','Gu Kunbo'),'mooc_gu_he_05':('fupi','He Tianjian')}
old=[]
for p in (LAB/'autonomous_strokes_v15_20261025'/'raw').glob('*'):
    try:old.append((str(p),phash(Image.open(p))))
    except Exception:pass
for r in records:
    if r['id'] in class_labels:r['label'],r['author']=class_labels[r['id']];r['label_basis']='explicit MOOC textbook section; inherited dominant technique, not per-stroke expert annotation'
    if r['id'].startswith('mooc'):r['work_group']=r.get('author','Gu/He mixed chapter')
    h=phash(Image.open(r['image']));distance,path=min((int(np.count_nonzero(h!=v)),p) for p,v in old);r.update(nearest_old_phash_hamming=distance,nearest_old_image=path,possible_duplicate_old=distance<=7)
manifest.update(records=records,classification_train_ids=list(class_labels),classification_test_ids=['book_pima_juran','book_pima_wangmeng','book_pima_huang','book_fupi_litang','book_fupi_mayuan'],source_split_policy='all new sources withheld from line-model training; classification separates publication and artist except explicit declared training duplication');jsave(ROOT/'source_manifest.json',manifest)
board=Image.new('RGB',(1500,300*int(np.ceil(len(records)/4))+40),'#f6f3ec');d=ImageDraw.Draw(board);d.text((6,4),'v20 新来源图和教材图例（不用于线条模型训练）；相同作品各阶段归为一组',font=FONT,fill='black')
for i,r in enumerate(records):
    x=(i%4)*375;y=40+(i//4)*300;d.text((x+5,y),r['id']+(' DUP?' if r['possible_duplicate_old'] else ''),font=SMALL,fill='black');paste(board,np.asarray(Image.open(r['image'])),x+5,y+25,365,270)
board.save(ROOT/'SOURCE_CONTACT.jpg',quality=94);print('V20_NEW_SOURCE_RECORDS',len(records),'DUP_FLAGS',[(r['id'],r['nearest_old_phash_hamming']) for r in records if r['possible_duplicate_old']],flush=True)
