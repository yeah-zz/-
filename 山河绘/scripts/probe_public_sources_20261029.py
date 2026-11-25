import os,sys
from pathlib import Path
sys.dont_write_bytecode=True
BASE=Path(r'E:\Codex_Workspace_Data\Gongbi_Line_Model_20261022')
for k,v in {'TEMP':'temp','TMP':'temp','PIP_CACHE_DIR':'cache/pip','TORCH_HOME':'cache/torch','HF_HOME':'cache/huggingface','XDG_CACHE_HOME':'cache','CUDA_CACHE_PATH':'cache/cuda'}.items():os.environ[k]=str(BASE/v)
from rock_v15_common import jsave
import urllib.request,json,concurrent.futures
ROOT=BASE/'stroke_completeness_v5_20261029'
URLS={'loc':'https://www.loc.gov/resource/gdcwdl.wdl_13536/?fo=json','met':'https://collectionapi.metmuseum.org/public/collection/v1/objects/60760','cma':'https://openaccess-api.clevelandart.org/api/artworks/?q=Chinese%20landscape&has_image=1&limit=35'}
def one(pair):
    name,url=pair
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'ShanHeHuiResearch/1.0 (offline artwork analysis)'})
        with urllib.request.urlopen(req,timeout=25) as r:raw=r.read(16000000)
        data=json.loads(raw);jsave(ROOT/'source_metadata'/(name+'.json'),data)
        if name=='loc':small={'keys':list(data),'resources':str(data.get('resources'))[:1000],'item':str(data.get('item'))[:300]}
        elif name=='met':small={k:data.get(k) for k in ['title','objectID','isPublicDomain','primaryImage','additionalImages']}
        else:small={'total':data.get('info'),'works':[{k:r.get(k) for k in ['id','title','culture','images','url']} for r in data.get('data',[])[:3]]}
        return {'source':name,'ok':True,'summary':small}
    except Exception as e:return {'source':name,'ok':False,'error':str(e)}
if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for r in pool.map(one,URLS.items()):print(json.dumps(r,ensure_ascii=False),flush=True)
