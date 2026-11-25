from v20_sources import *
import importlib.util
print('PDF_RENDERER_AVAILABLE',bool(importlib.util.find_spec('fitz')),flush=True)
url='https://backend.sjhtbook.com/upload/file/2022/05/227447a6abc6438f878c5e05ed1c81c6.pdf';dest=ROOT/'sources'/'publisher_landscape_sample.pdf'
if not dest.exists():
    data,kind=fetch(url,max_bytes=25000000);dest.write_bytes(data)
import fitz
pdf=fitz.open(dest);found=[]
for i,page in enumerate(pdf):
    text=page.get_text()
    if any(k in text for k in ['皴法','披麻','斧劈','荷叶','山石法']):
        found.append(dict(index=i,text=text[:1400]));page.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(ROOT/'sources'/('book_page_'+str(i).zfill(3)+'.png'))
jsave(ROOT/'book_pages.json',dict(source=url,sha256=digest(dest),page_count=len(pdf),relevant_pages=found));print('PDF_PAGES',len(pdf),'RELEVANT',[(r['index'],r['text'][:100]) for r in found],flush=True)
manifest=read(ROOT/'source_manifest.json');print('CAA_ERRORS',json.dumps(manifest['errors'][-2:],ensure_ascii=False),flush=True)
