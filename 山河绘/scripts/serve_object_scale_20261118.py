import sys,json,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import serve_regional_presence_20261117 as server
server.ROOT=server.LAB/'object_scale_pairs_20261118'
server.ALLOWED={'index.html','comparison.jpg','pairs.json','protocol.json','results.json','regional_summary.json','verification.json','extended_H14_diagnostics.json'}
if __name__=='__main__':
    s=server.ThreadingHTTPServer(('127.0.0.1',0),server.Handler);url=f'http://127.0.0.1:{s.server_port}/'
    (server.ROOT/'server.json').write_text(json.dumps(dict(url=url,pid=os.getpid(),robot_connected=False)),encoding='utf8');print(url,flush=True);s.serve_forever()
