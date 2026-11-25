import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure
configure()
from gongbi_training.cunfa_collect import fetch,Images,CORPUS
for url in ('https://m.sohu.com/a/259730799_744221','https://www.xmcdw.com/html/course-14.html'):
    try:
        raw=fetch(url);p=Images();p.feed(raw.decode('utf-8','replace'))
        print(url,'count',len(p.images),'samples',p.images[:3],flush=True)
        (CORPUS/'html'/('mobile.html' if 'sohu' in url else 'probe.html')).write_bytes(raw)
    except Exception as exc:print(type(exc).__name__,str(exc),flush=True)
