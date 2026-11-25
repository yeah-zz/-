# -*- coding: utf-8 -*-
"""Publish candidate diagnostics, known limits and an all-join visual gallery."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,math,shutil
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont

ROOT=LAB/'long_stroke_recovery_20261105'
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
def read(p):return json.loads(p.read_text('utf8'))

def gallery(ident):
    folder=ROOT/ident;rgb=np.array(Image.open(folder/'source.png').convert('RGB'));diag=read(folder/'diagnostics.json')
    accepted=diag['accepted'];cols=4;board=Image.new('RGB',(1440,max(1,math.ceil(len(accepted)/cols))*208),'#f4f1e8');d=ImageDraw.Draw(board)
    for k,a in enumerate(accepted):
        bridge=np.array(a['bridge']);center=np.mean(bridge,axis=0);x,y=np.rint(center).astype(int)
        x0=max(0,x-12);y0=max(0,y-12);x1=min(rgb.shape[1],x0+25);y1=min(rgb.shape[0],y0+25)
        raw=Image.fromarray(rgb[y0:y1,x0:x1]).resize((145,145),Image.Resampling.NEAREST)
        overlay=np.array(raw).copy();p=(bridge-[x0,y0])*np.array([145/(x1-x0),145/(y1-y0)])
        cv2.polylines(overlay,[np.rint(p).astype(np.int32)],False,(15,190,70),2,cv2.LINE_AA)
        left=k%cols*360+8;top=k//cols*208
        board.paste(raw,(left,top+27));board.paste(Image.fromarray(overlay),(left+150,top+27))
        kind={'touching_endpoints':'端点续接','source_supported_gap':'沿暗线续接','curved_source_supported_gap':'弯折续接','same_band_overlap':'共享段拼接'}[a['kind']]
        d.text((left,top+3),f'{k+1:02}｜{kind}｜源曲线{a["i"]} / {a["j"]}',font=FONT,fill='#263e34')
        d.text((left,top+177),f'间隔{a["gap_px"]:.2f}px；证据{a["support"]:.2f}（非正确率）',font=FONT,fill='#263e34')
    board.save(folder/'join_gallery.jpg',quality=95)

def main():
    cat=read(ROOT/'catalog.json');fresh=read(ROOT/'fresh_synthetic_controls.json');control=read(ROOT/'synthetic_controls.json')
    for ident in ('G03_01','G05_03','T02'):gallery(ident)
    table=''.join(f'<tr><td>{r["id"]}</td><td>{r["stats"]["before"]["count"]}</td><td>{r["stats"]["after"]["count"]}</td><td>{r["stats"]["new_joins"]}</td><td>{r["stats"]["before"]["short_under_5px"]} → {r["stats"]["after"]["short_under_5px"]}</td><td>{r["stats"]["pending"]}</td></tr>' for r in cat['scenes'])
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>长笔触恢复 · 完整稿对照</title><style>body{font:17px/1.65 system-ui;background:#f4f1e8;color:#263e34;max-width:1800px;margin:auto;padding:25px}img{width:100%;background:white}table{border-collapse:collapse}td,th{border:1px solid #b8c5b5;padding:10px}.note{padding:16px;background:#e5ecde}a{color:#346943}</style><h1>长笔触恢复：整笔拼接，不按预算省略细节</h1><p><a href="/">进入笔序回放</a></p>'''
    page+='<p class="note">本轮修改的是完整提取稿的连接后处理，没有训练新的像素提取模型。新增多尺度端点方向、共享墨带重叠段拼接，以及沿原图暗脊的弯折连接。同一待定工作区需要更强墨迹证据；不同工作块仍不连接。无法确认的分支保留待定。</p>'
    page+='<table><tr><th>画作</th><th>之前</th><th>本轮</th><th>新增续接候选</th><th>不足5px短段</th><th>待定配对（非笔数）</th></tr>'+table+'</table>'
    page+='<p>所有原输入来源均保留，原输入采样点到新路径最大距离小于0.13像素。这是几何保留检查，不能说明真实主线已经完整，也不能证明每条续接正确。G03仍有414条短于5像素；本轮尚未显著解决整幅长笔触恢复。</p>'
    for title,path in [('亭子结构的分笔与续接','G03_01/pavilion.jpg'),('山石主线与内部皴线','G03_01/rock.jpg'),('整幅对照','G03_01/comparison.jpg')]:
        page+=f'<h2>{title}</h2><img src="{path}">'
    page+='<h2>合成控制检查与发现的问题</h2><p>初次245例控制检查发现2处Y形分支误连：一个竞争分支因抗锯齿未通过墨迹门槛，另一分支被误当作唯一续接。修正为先保留几何分支歧义，再检查图像证据。修正后的开发控制仍不能当盲测。</p>'
    t=fresh['totals'];page+=f'<p>随后新随机种子的{t["cases"]}个同类合成例中，错误合并组数为{t["wrong_merged_groups"]}；{t["expected_pairs"]}个已知可接关系恢复{t["recovered_pairs"]}个，漏接仍明显。样例只涵盖解析几何线条、淡墨、重叠、弯折和交汇，不代表真实山水画准确率。</p>'
    page+='<h2>逐处连接检查</h2><p>每张图左边为局部源图，右边标绿色的连接位置。全部连接仍是非专家实验候选，尚未实画验证。</p>'
    for ident in ('G03_01','G05_03','T02'):
        page+=f'<p><a href="{ident}/join_gallery.jpg">{ident} · 查看全部连接位置</a> · <a href="{ident}/comparison.jpg">整幅分笔对照</a></p>'
    page+='<p>保留了原分块分层顺序。未设置下压量、速度或补墨参数，未连接机械臂；颜色仅区分笔画，既不是皴法类别，也不是实际墨色和笔宽。</p></html>'
    (ROOT/'report.html').write_text(page,encoding='utf8')
    index=ROOT/'index.html';text=index.read_text('utf8')
    if '/report.html' not in text:text=text.replace('<footer>','<footer><p><a href="/report.html">查看长笔触恢复结果与逐处连接诊断</a></p>')
    index.write_text(text,encoding='utf8')
    snap=ROOT/'source_snapshot'
    for name in ('long_stroke_recovery_20261105.py','evaluate_long_stroke_recovery_20261105.py','finalize_long_stroke_recovery_20261105.py'):
        shutil.copy2(Path(__file__).with_name(name),snap/name)
    print('REPORT',ROOT/'report.html',flush=True)
if __name__=='__main__':main()
