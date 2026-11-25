# -*- coding: utf-8 -*-
"""训练一个轻量 CNN 编码器：先在大批中国山水 patch 上自监督重建，再用四类皴法微调。

运行环境：E:\Anaconda\envs\pytorch\python.exe（torch 2.8 + CUDA）。
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
DATA = ROOT / "data"
CLASSES = ["pima", "zhedai", "fupi", "midian"]
NAMES = {"pima": "披麻皴", "zhedai": "折带皴", "fupi": "斧劈皴", "midian": "米点皴"}
SIZE = 96


class Encoder(nn.Module):
    def __init__(self, z=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, z)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


class Decoder(nn.Module):
    def __init__(self, z=128):
        super().__init__()
        self.fc = nn.Linear(z, 6 * 6 * 128)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 96, 4, 2, 1), nn.BatchNorm2d(96), nn.ReLU(True),
            nn.ConvTranspose2d(96, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 128, 6, 6)
        return self.net(h)


def img_tensor(path):
    im = Image.open(path).convert("L").resize((SIZE, SIZE), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(a)[None, :, :]


def load_labeled():
    xs, ys = [], []
    for aid, cls in (("huang_gongwang", "pima"), ("ni_zan", "zhedai"),
                     ("shi_tao", "fupi"), ("gong_xian", "midian")):
        for p in sorted((CORPUS / aid / "patches").glob("*.png")):
            xs.append(img_tensor(p))
            ys.append(CLASSES.index(cls))
    ltfc = {"黄公望": "pima", "王蒙": "pima", "吴镇": "pima", "沈周": "pima", "文徵明": "pima",
            "倪瓒": "zhedai", "石涛": "fupi", "唐寅": "fupi", "仇英": "fupi", "龚贤": "midian"}
    for artist, cls in ltfc.items():
        for p in sorted((CORPUS / "ltfc" / artist / "patches").glob("*.png")):
            xs.append(img_tensor(p))
            ys.append(CLASSES.index(cls))
    return xs, ys


def load_bulk(limit=600):
    ps = sorted((CORPUS / "bulk" / "patches").glob("*.png"))
    random.Random(7).shuffle(ps)
    return [img_tensor(p) for p in ps[:limit]]


def augment(x):
    if torch.rand(1).item() < .5:
        x = torch.flip(x, dims=[2])
    if torch.rand(1).item() < .5:
        x = torch.flip(x, dims=[1])
    k = int(torch.randint(0, 4, (1,)))
    if k:
        x = torch.rot90(x, k, dims=[1, 2])
    x = x + torch.randn_like(x) * .015
    return x.clamp(0, 1)


def train(args):
    device = "cuda" if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 12) else "cpu"
    print("device", device, flush=True)
    bulk = load_bulk(args.bulk_limit)
    xs, ys = load_labeled()
    if not bulk:
        bulk = list(xs)
    print("bulk patches", len(bulk), "labeled", len(xs), flush=True)
    enc, dec = Encoder().to(device), Decoder().to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=2e-3)
    for epoch in range(args.pretrain_epochs):
        random.shuffle(bulk)
        total = 0.0
        for i in range(0, len(bulk), 32):
            x = torch.stack([augment(b) for b in bulk[i:i + 32]]).to(device)
            z = enc(x)
            y = dec(z)
            loss = F.mse_loss(y, x)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss) * len(x)
        print("pretrain", epoch + 1, round(total / max(1, len(bulk)), 5), flush=True)
    # 分类微调
    head = nn.Linear(128, len(CLASSES)).to(device)
    opt2 = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=4e-4)
    cnt = [max(1, ys.count(i)) for i in range(len(CLASSES))]
    cw = torch.tensor([len(ys) / (len(CLASSES) * c) for c in cnt], dtype=torch.float32).to(device)
    X = torch.stack(xs); Y = torch.tensor(ys, dtype=torch.long)
    idx = list(range(len(X))); random.Random(11).shuffle(idx)
    cut = max(1, int(len(idx) * .8))
    tr, va = idx[:cut], idx[cut:]
    best = 0.0
    for epoch in range(args.finetune_epochs):
        enc.train(); head.train()
        random.shuffle(tr)
        for i in range(0, len(tr), 16):
            ids = tr[i:i + 16]
            x = torch.stack([augment(X[j]) for j in ids]).to(device)
            y = Y[ids].to(device)
            logits = head(enc(x))
            loss = F.cross_entropy(logits, y, weight=cw)
            opt2.zero_grad(); loss.backward(); opt2.step()
        enc.eval(); head.eval()
        with torch.no_grad():
            logits = head(enc(torch.stack([X[j] for j in va]).to(device)))
            pred = logits.argmax(1).cpu()
            acc = float((pred == Y[va]).float().mean())
            recalls = []
            for c in range(len(CLASSES)):
                m = Y[va] == c
                if m.any():
                    recalls.append(float((pred[m] == c).float().mean()))
            macro = sum(recalls) / max(1, len(recalls))
        best = max(best, macro)
        if (epoch + 1) % 5 == 0:
            print("finetune", epoch + 1, "val", round(acc, 3), flush=True)
    # 全量再训 10 轮
    for epoch in range(10):
        enc.train(); head.train()
        random.shuffle(idx)
        for i in range(0, len(idx), 16):
            ids = idx[i:i + 16]
            x = torch.stack([augment(X[j]) for j in ids]).to(device)
            y = Y[ids].to(device)
            loss = F.cross_entropy(head(enc(x)), y, weight=cw)
            opt2.zero_grad(); loss.backward(); opt2.step()
    DATA.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": enc.state_dict(), "head": head.state_dict(),
                "classes": CLASSES, "size": SIZE}, DATA / "encoder.pt")
    metrics = dict(format="cnn-encoder-v1", classes=CLASSES, class_names=NAMES,
                   samples=len(xs), bulk_patches=len(bulk), val_accuracy=round(best, 3),
                   val_macro=round(best, 3),
                   pretrain_epochs=args.pretrain_epochs, finetune_epochs=args.finetune_epochs,
                   device=device, note="自监督重建 + 四类皴法微调；分类头仅用于标签校验，渲染仍用可解释参数。")
    (DATA / "encoder_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bulk-limit", type=int, default=600)
    ap.add_argument("--pretrain-epochs", type=int, default=15)
    ap.add_argument("--finetune-epochs", type=int, default=40)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()