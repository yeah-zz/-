import sys,json,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import serve_regional_presence_20261117 as server
server.ROOT=server.LAB/'ink_run_review_20261118'
server.ALLOWED={'index.html','comparison.png','draft_references.json','reviewed_references.json','reference_freeze.json','results.json','review_queue.json','verification.json'}
server.ALLOWED.update('views/'+p.name for p in (server.ROOT/'views').iterdir() if p.suffix=='.png')
if __name__=='__main__':
    s=server.ThreadingHTTPServer(('127.0.0.1',0),server.Handler);url=f'http://127.0.0.1:{s.server_port}/'
    (server.ROOT/'server.json').write_text(json.dumps(dict(url=url,pid=os.getpid(),robot_connected=False)),encoding='utf8');print(url,flush=True);s.serve_forever()
