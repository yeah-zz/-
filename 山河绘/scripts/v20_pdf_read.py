import json,hashlib,subprocess
from pathlib import Path
from pypdf import PdfReader
root=Path(r'E:\Codex_Workspace_Data\Gongbi_Line_Model_20261022\structure_generalization_v20_20261025');file=root/'sources'/'publisher_landscape_sample.pdf';reader=PdfReader(file);found=[]
for i,page in enumerate(reader.pages):
    text=page.extract_text() or ''
    hits=[k for k in ['皴法','披麻','斧劈','荷叶','山石法'] if k in text]
    if hits:found.append(dict(index=i,keywords=hits,text=text))
selected=sorted(found,key=lambda r:sum(k in r['keywords'] for k in ['披麻','斧劈','荷叶']),reverse=True)[:10]
exe=r'C:\Users\Lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe'
for r in selected:
    prefix=root/'sources'/('book_page_'+str(r['index']).zfill(3));subprocess.run([exe,'-f',str(r['index']+1),'-l',str(r['index']+1),'-scale-to','1400','-singlefile','-png',str(file),str(prefix)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE);r['rendered_image']=str(prefix)+'.png'
(root/'book_pages.json').write_text(json.dumps(dict(source='https://backend.sjhtbook.com/upload/file/2022/05/227447a6abc6438f878c5e05ed1c81c6.pdf',sha256=hashlib.sha256(file.read_bytes()).hexdigest(),page_count=len(reader.pages),relevant_pages=found,rendered=[dict(index=r['index'],image=r['rendered_image'],keywords=r['keywords']) for r in selected]),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(page_count=len(reader.pages),hits=[dict(index=r['index'],keywords=r['keywords'],snippet=r['text'][:100]) for r in found],rendered=[r['index'] for r in selected]),ensure_ascii=False),flush=True)
