"""Restore a source-backed inner ridge; stage changes for the next idle run."""
from pathlib import Path
import sys, json, copy, hashlib
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from web.catalog import ARTWORKS
from painting_session import save
from scripts.consolidate_selected_artworks import digest, stroke_data

RIDGE = 'yun-shan-ting-jie-ink-00393'
WASH = 'pavilion-expanded-wash-16-green-008'

def audit(candidate, baseline):
    old = json.loads(Path(baseline).read_text(encoding='utf-8'))
    new = json.loads(Path(candidate).read_text(encoding='utf-8'))
    a, b = stroke_data(old), stroke_data(new)
    assert [s for s in a['black'] if s['id'] != RIDGE] == [s for s in b['black'] if s['id'] != RIDGE]
    assert [s for s in a['color'] if s['id'] != WASH] == b['color']
    prior = next(s for s in a['black'] if s['id'] == RIDGE)
    fixed = next(s for s in b['black'] if s['id'] == RIDGE)
    assert fixed['points'][:len(prior['points'])] == prior['points']
    assert set(fixed['speed']) == {40.0} and set(fixed['press']) == {.15}
    assert old['workflow']['speed_revision'] == new['workflow']['speed_revision']
    assert old['workflow']['modules']['color']['motion_candidate'] == new['workflow']['modules']['color']['motion_candidate']
    assert new['workflow']['source_revision']['stroke_data_sha256'] == digest(b)
    from web.execution import prepare
    from run_json_painting import load_artwork
    prepare(candidate)
    load_artwork(candidate)
    return dict(black_count_unchanged=True, other_black_exactly_preserved=True,
                removed_green_strokes=[WASH], ridge_length_mm=fixed['length_mm'],
                speed_and_acceleration_preserved=True, runtime_validation_passed=True,
                robot_connected=False)

def main():
    path = ROOT / ARTWORKS[2][3]
    old = json.loads(path.read_text(encoding='utf-8'))
    new = copy.deepcopy(old)
    tr = old['workflow']['source_revision']['source_transform']
    source = Image.open(path.parent.parent/'参考原图.png').convert('RGB')
    ref = Image.new('RGB', tuple(old['ink_job']['meta']['image_size']), 'white')
    ref.paste(source.resize((round(source.width*tr['scale']), round(source.height*tr['scale']))), tuple(tr['offset']))
    pixels = np.asarray(ref)
    # Follow the dark source contour below the colored summit, not the wash path.
    guides = np.array([[1465,547],[1475,548],[1485,553],[1495,559],[1505,563],
                       [1515,565],[1525,568],[1535,574],[1545,576],[1555,579],[1565,581],[1574,581]], float)
    continuation = []
    for x in range(1466,1575):
        guess = np.interp(x, guides[:,0], guides[:,1])
        lo, hi = int(guess)-5, int(guess)+6
        column = pixels[lo:hi,x].astype(float)
        ys = np.flatnonzero((column.max(1)<110) & (np.ptp(column,axis=1)<65))+lo
        if not len(ys): raise ValueError('No dark source contour at x=%d'%x)
        continuation.append([float(x), float(np.median(ys))])
    ink = new['ink_job']['meta']['ink_execution']['base_strokes']
    ridge = next(s for s in ink if s['id'] == RIDGE)
    ridge['points'].extend(continuation)
    ridge['press'].extend([.15]*len(continuation))
    ridge['speed'].extend([40.]*len(continuation))
    from ink_control import refresh_stats
    refresh_stats(ridge, new['ink_job']['meta']['ink_execution']['mm_per_px'])
    from web.execution import prepare
    prepare(path)
    from artwork_order import ink_payload
    new['ink_job'] = ink_payload(old['ink_job'],ink)
    plan = new['workflow']
    plan['modules']['outline'].update(strokes=copy.deepcopy(ink),stroke_count=len(ink))
    color = plan['modules']['color']
    color['strokes'] = [s for s in color['strokes'] if s['id'] != WASH]
    color['stroke_count'] = len(color['strokes'])
    plan['source_revision'].update(inner_right_ridge_repaired=True,stroke_data_sha256=digest(stroke_data(new)))
    pending = path.parent/'待应用_右峰黑线'
    pending.mkdir(exist_ok=True)
    save(pending/'painting.json',new)
    result = audit(pending/'painting.json',path)
    save(pending/'右峰黑线校验.json',result)
    from scripts.source_wash_comparison import render
    roi = (1395,505,1595,635)
    board = Image.new('RGB',(1280,470),'#f7f7f0')
    draw = ImageDraw.Draw(board)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    for i,(bundle,label) in enumerate(((old,'修改前：内侧黑线缺段'),(new,'修改后：黑线补齐，移除贴边绿线'))):
        draw.text((20+i*640,12),label,font=font,fill='#304c3b')
        board.paste(render(bundle).crop(roi).resize((600,390)),(20+i*640,55))
    board.save(pending/'右峰黑线前后对照.png')
    save(pending/'应用记录.json',dict(baseline_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
         candidate_sha256=hashlib.sha256((pending/'painting.json').read_bytes()).hexdigest(),applied=False))
    print(json.dumps(result))

if __name__ == '__main__': main()
