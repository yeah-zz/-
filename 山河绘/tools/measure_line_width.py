import sys
import cv2
import numpy as np
img = cv2.imread(sys.argv[1])
g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
H, W = g.shape
bg = cv2.medianBlur(g.astype(np.uint8), 61).astype(np.float32)
soft = (bg - g > 22)
centres = [659, 761, 859, 952, 1042]
MPP = 10.0 / 60.0
print('scale 10 mm = 60 px on the cutting mat -> %.4f mm/px' % MPP)
print('')
print('line | y-extent(px) | length mm | n | FWHM px (p25/50/75) | core mm | +0.5px mm | min gray')
out = []
for i, c in enumerate(centres):
    band = soft[:, c - 4:c + 5].sum(axis=1)
    ys = np.nonzero(band > 0)[0]
    y0, y1 = int(ys[0]), int(ys[-1])
    ws, mins = [], []
    for y in range(y0 + (y1 - y0) // 4, y0 + 3 * (y1 - y0) // 4):
        seg = g[y, c - 10:c + 11]
        outer = np.concatenate([g[y, c - 32:c - 12], g[y, c + 13:c + 33]])
        b = float(np.percentile(outer, 75))
        m = float(seg.min())
        if b - m < 40:
            continue
        t = .5 * (b + m)
        idx = np.nonzero(seg <= t)[0]
        if len(idx) == 0:
            continue
        a, z = idx[0], idx[-1]
        def cross(i0, i1):
            if i1 < 0 or i1 >= len(seg) or seg[i1] == seg[i0]:
                return float(i0)
            return i0 + (t - seg[i0]) / (seg[i1] - seg[i0])
        x0 = cross(a - 1, a) if a > 0 else 0.
        x1 = cross(z + 1, z) if z + 1 < len(seg) else float(len(seg) - 1)
        ws.append(x1 - x0)
        mins.append(m)
    ws = np.array(ws)
    med = float(np.median(ws))
    out.append((c, y0, y1, (y1 - y0 + 1) * MPP, len(ws), med, np.median(mins)))
    print('  %d  | %4d..%4d | %6.1f | %4d | %5.2f/%5.2f/%5.2f | %5.2f mm | %5.2f mm | %3.0f'
          % (i + 1, y0, y1, (y1 - y0 + 1) * MPP, len(ws),
             np.percentile(ws, 25), med, np.percentile(ws, 75), med * MPP, (med + .5) * MPP, np.median(mins)))
print('')
print('x centres px:', centres)
print('spacing px:', np.diff(centres).tolist(), '-> mm:', (np.diff(centres) * MPP).round(1).tolist())
vis = img.copy()
for i, (c, y0, y1, L, n, med, mg) in enumerate(out):
    cv2.line(vis, (c, y0), (c, y1), (0, 0, 255), 2)
    cv2.putText(vis, '%.2fmm' % (med * MPP), (c - 30, max(30, y0 - 12)),
                cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
cv2.imwrite('_photo_widths.png', vis)
print('annotated -> _photo_widths.png')
