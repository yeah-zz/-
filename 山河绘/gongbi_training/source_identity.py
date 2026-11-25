"""Canonical identity checks before counting teaching sources as new works."""
from urllib.parse import urlsplit,unquote,parse_qs

def source_identity(url):
    parsed=urlsplit(url);host=parsed.netloc.lower().removeprefix('www.')
    path=unquote(parsed.path).rstrip('/')
    if host=='ganjingworld.com' and '/video/' in path:
        return 'ganjing_video:'+path.rsplit('/',1)[-1]
    if host in {'youtube.com','m.youtube.com','youtu.be'}:
        ident=path.lstrip('/') if host=='youtu.be' else parse_qs(parsed.query).get('v',[None])[0]
        if ident:return 'youtube_video:'+ident
    return host+path+'?'+parsed.query

def check_previously_seen(url,prior_records):
    ident=source_identity(url)
    matches=[r for r in prior_records if source_identity(r['url'])==ident]
    return dict(identity=ident,previously_seen=bool(matches),matches=matches,
                can_count_as_new_independent_work=False if matches else None)
