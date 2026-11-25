# -*- coding: utf-8 -*-
"""自研笔触场模型（StrokeField v1）：输入山石 patch，输出墨线/笔向/宽度场。

这是“image-to-stroke”的第一步：先学习可微的笔触场，再用确定性矢量化得到
笔触序列；比直接回归无序点列更稳定，也更容易加机器人约束。
运行：E:\Anaconda\python.exe -m master_studio.train_stroke_field
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
DATA = ROOT / "data"
S = 96


def target_from(path):
    im = Image.open(path).convert("L").resize((S, S), Image.BILINEAR)
    g = np.asarray(im, dtype=np.float32) / 255.0
    ink = (g < 0.62).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (7, 7), 0)
    jyy = cv2.GaussianBlur(gy * gy, (7, 7), 0)
    jxy = cv2.GaussianBlur(gx * gy, (7, 7), 0)
    den = jxx + jyy + 1e-6
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy * jxy) / den
    cos2 = np.where(coh > .12, (jxx - jyy) / den, 0)
    sin2 = np.where(coh > .12, 2 * jxy / den, 0)
    # 宽度：距离变换，归一化到 0..1
    dt = cv2.distanceTransform((ink * 255).astype(np.uint8), cv2.DIST_L2, 5)
    width = np.clip(dt / 8.0, 0, 1)
    return np.stack([ink, cos2, sin2, width], 0).astype(np.float32)


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(1, 24, 3, 2, 1), nn.BatchNorm2d(24), nn.ReLU(True))
        self.e2 = nn.Sequential(nn.Conv2d(24, 48, 3, 2, 1), nn.BatchNorm2d(48), nn.ReLU(True))
        self.e3 = nn.Sequential(nn.Conv2d(48, 96, 3, 2, 1), nn.BatchNorm2d(96), nn.ReLU(True))
        self.e4 = nn.Sequential(nn.Conv2d(96, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True))
        self.d3 = nn.Sequential(nn.Conv2d(128 + 96, 96, 3, 1, 1), nn.BatchNorm2d(96), nn.ReLU(True))
        self.d2 = nn.Sequential(nn.Conv2d(96 + 48, 48, 3, 1, 1), nn.BatchNorm2d(48), nn.ReLU(True))
        self.d1 = nn.Sequential(nn.Conv2d(48 + 24, 24, 3, 1, 1), nn.BatchNorm2d(24), nn.ReLU(True))
        self.out = nn.Conv2d(24, 4, 1)

    def forward(self, x):
        a = self.e1(x); b = self.e2(a); c = self.e3(b); d = self.e4(c)
        u = F.interpolate(d, size=c.shape[-2:], mode="bilinear", align_corners=False)
        u = self.d3(torch.cat([u, c], 1))
        u = F.interpolate(u, size=b.shape[-2:], mode="bilinear", align_corners=False)
        u = self.d2(torch.cat([u, b], 1))
        u = F.interpolate(u, size=a.shape[-2:], mode="bilinear", align_corners=False)
        u = self.d1(torch.cat([u, a], 1))
        u = F.interpolate(u, size=(S, S), mode="bilinear", align_corners=False)
        return self.out(u)


def collect():
    files = []
    for d in ("huang_gongwang", "ni_zan", "shi_tao", "gong_xian"):
        files += sorted((CORPUS / d / "patches").glob("*.png"))
    files += sorted((CORPUS / "bulk" / "patches").glob("*.png"))
    lr = CORPUS / "ltfc"
    if lr.exists():
        for d in lr.iterdir():
            pd = d / "patches"
            if pd.exists():
                files += sorted(pd.glob("*.png"))
    random.Random(5).shuffle(files)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--limit", type=int, default=400)
    args = ap.parse_args()
    files = collect()[:args.limit]
    if not files:
        raise SystemExit("没有 patch，请先提取。")
    print("patches", len(files), flush=True)
    data = [torch.from_numpy(target_from(p)) for p in files]
    device = "cuda" if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 12) else "cpu"
    model = TinyUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = len(data)
    idx = list(range(n))
    for ep in range(args.epochs):
        random.shuffle(idx)
        total = 0
        for i in range(0, n, 8):
            ids = idx[i:i + 8]
            x = torch.stack([data[j][0:1] for j in ids]).to(device)
            y = torch.stack([data[j] for j in ids]).to(device)
            pred = model(x)
            ink = torch.sigmoid(pred[:, 0])
            loss_ink = F.binary_cross_entropy(ink, y[:, 0])
            mask = y[:, 0:1]
            loss_dir = F.mse_loss(pred[:, 1:3] * mask, y[:, 1:3] * mask)
            loss_w = F.l1_loss(torch.sigmoid(pred[:, 3:4]) * mask, y[:, 3:4] * mask)
            # 平滑约束：机器人不喜欢高频抖动
            tv = (pred[:, 1:2, :, 1:] - pred[:, 1:2, :, :-1]).abs().mean() + (pred[:, 1:2, 1:, :] - pred[:, 1:2, :-1, :]).abs().mean()
            loss = loss_ink + 0.6 * loss_dir + 0.5 * loss_w + 0.02 * tv
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss) * len(ids)
        print("epoch", ep + 1, round(total / n, 4), flush=True)
    DATA.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "size": S}, DATA / "stroke_field.pt")
    # 样例预测
    with torch.no_grad():
        pred = model(torch.stack([data[0][0:1]]).to(device))[0].cpu()
    ink = torch.sigmoid(pred[0]).numpy()
    out = (255 * (1 - ink)).astype(np.uint8)
    Image.fromarray(out).resize((320, 320), Image.NEAREST).save(DATA / "stroke_field_sample.png")
    metrics = dict(format="stroke-field-v1", patches=n, epochs=args.epochs, device=device,
                   outputs=["ink", "cos2theta", "sin2theta", "width"],
                   note="山石 patch -> 墨线/笔向/宽度场；下一步矢量化成可执行笔触序列。")
    (DATA / "stroke_field_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()