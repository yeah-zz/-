"""Research-only ShanShuiSeg inference using the released tensors and PyTorch."""

from collections import Counter
import hashlib
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from professional_clpd import MAP, ROOT, write
from gongbi_training.runtime import configure


def forward(image, state, torch):
    """Evaluate the pinned B5 architecture from the publisher's model config."""
    from torch.nn import functional as f

    def conv(x, key, **kwargs):
        return f.conv2d(x,state[key+".weight"],state.get(key+".bias"),**kwargs)

    def ln(x,key):
        return f.layer_norm(x,(x.shape[-1],),state[key+".weight"],state[key+".bias"],eps=1e-6)

    def norm_conv(x,key):
        x=conv(x,key+".conv")
        bn=key+".bn"
        return f.batch_norm(x,state[bn+".running_mean"],state[bn+".running_var"],
                            state[bn+".weight"],state[bn+".bias"],training=False).relu()

    pyramid=[]
    for stage,(depth,heads,ratio) in enumerate(zip((3,6,40,3),(1,2,5,8),(8,4,2,1))):
        key=f"backbone.layers.{stage}"
        image=conv(image,key+".0.projection",stride=4 if stage==0 else 2,padding=3 if stage==0 else 1)
        b,c,h,w=image.shape
        def tokens(t):
            return t.flatten(2).transpose(1,2)
        def grid(t):
            return t.transpose(1,2).reshape(b,c,h,w)
        x=ln(tokens(image),key+".0.norm")
        for layer in range(depth):
            p=key+f".1.{layer}"
            y=ln(x,p+".norm1")
            reduce=p+".attn"
            small=ln(tokens(conv(grid(y),reduce+".sr",stride=ratio)),reduce+".norm") if ratio>1 else y
            attn=reduce+".attn"
            qw,kw,vw=state[attn+".in_proj_weight"].chunk(3)
            qb,kb,vb=state[attn+".in_proj_bias"].chunk(3)
            def proj(source,weight,bias):
                return f.linear(source,weight,bias).reshape(b,-1,heads,c//heads).transpose(1,2)
            q,k,v=proj(y,qw,qb),proj(small,kw,kb),proj(small,vw,vb)
            z=f.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(b,-1,c)
            x=x+f.linear(z,state[attn+".out_proj.weight"],state[attn+".out_proj.bias"])
            u=conv(grid(ln(x,p+".norm2")),p+".ffn.layers.0")
            u=f.gelu(conv(u,p+".ffn.layers.1",padding=1,groups=u.shape[1]))
            x=x+tokens(conv(u,p+".ffn.layers.4"))
        image=grid(ln(x,key+".2"))
        pyramid.append(image)
    head=[f.interpolate(norm_conv(im,f"decode_head.convs.{i}"),size=pyramid[0].shape[-2:],
                        mode="bilinear",align_corners=False) for i,im in enumerate(pyramid)]
    fused=norm_conv(torch.cat(head,1),"decode_head.fusion_conv")
    return conv(fused,"decode_head.conv_seg")


def main():
    configure()
    import torch
    from torch.nn import functional as f
    out=ROOT/"scene_s1"
    out.mkdir(exist_ok=True)
    source=ROOT/"source_audit"
    model=source/"s1_weights.pth"
    expected="3c8781f9a2d0b620c84a0d10455a1fbfe5f61d754497abecb044d8e9027dcd3d"
    with model.open("rb") as file:
        digest=hashlib.sha256()
        for block in iter(lambda:file.read(1<<20),b""):
            digest.update(block)
    assert digest.hexdigest()==expected
    device="cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    state=torch.load(model,map_location=device,weights_only=True)
    assert all(isinstance(v,torch.Tensor) for v in state.values())
    upstream=json.loads((source/"s1_labels.json").read_text("utf8"))
    assert upstream["raw_ids"]==list(range(1,15))
    assert [upstream["classes"][i-1] for i in (2,3,13)]==["mountain","trees","rock"]
    data=json.loads((ROOT/"clpd/manifest.json").read_text("utf8"))["rows"]
    rows=sorted([r for r in data if r["split"]=="evaluation"],key=lambda r:r["id"])
    lookup=np.full(256,255,np.uint8)
    for i,k in MAP.items():
        lookup[i]=k
    mean=np.array([.485,.456,.406],np.float32)
    std=np.array([.229,.224,.225],np.float32)
    palette=np.array([[227,224,214],[178,111,65],[48,112,86],[77,113,158]],np.uint8)
    board=Image.new("RGB",(660,38+200*min(6,len(rows))),"#f7f3e9")
    draw=ImageDraw.Draw(board)
    font=ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",15)
    draw.text((12,8),"原图 / CLPD 粗标 / S1 公开模型",font=font,fill="#243d31")
    total=np.zeros((4,4),int)
    group_results={}
    outputs=[]
    for i,row in enumerate(rows):
        with Image.open(row["image"]) as file:
            image=file.convert("RGB")
        with Image.open(row["mask"]) as file:
            target=lookup[np.asarray(file)]
        x=(np.asarray(image.resize((512,512),Image.Resampling.BILINEAR),np.float32)/255-mean)/std
        with torch.inference_mode():
            scores=forward(torch.from_numpy(x).permute(2,0,1)[None].to(device),state,torch)
            logits=f.interpolate(scores,image.size[::-1],mode="bilinear",align_corners=False)[0]
            assert torch.isfinite(logits).all()
            raw=(logits.argmax(0)+1).cpu().numpy().astype(np.uint8)
        pred=lookup[raw]
        keep=target!=255
        cf=np.bincount(4*target[keep].astype(int)+pred[keep],minlength=16).reshape(4,4)
        total+=cf
        group_results.setdefault(row["group"],np.zeros((4,4),int))[:] += cf
        outputs.append(dict(id=row["id"],group=row["group"],confusion=cf.tolist()))
        if i<6:
            for j,pic in enumerate((image,Image.fromarray(np.where((target==255)[...,None],150,palette[target.clip(0,3)])),
                                    Image.fromarray(palette[pred]))):
                pic.thumbnail((203,175))
                board.paste(pic,(12+215*j,40+i*200))
    board.save(out/"shanshui_comparison.png")
    scores={}
    for k,kind in ((1,"mountain_rock"),(2,"trees")):
        tp=total[k,k]
        scores[kind]={"precision":float(tp/max(total[:,k].sum(),1)),"recall":float(tp/max(total[k].sum(),1)),
                      "f1":float(2*tp/max(total[:,k].sum()+total[k].sum(),1)),
                      "iou":float(tp/max(total[:,k].sum()+total[k].sum()-tp,1))}
    write(out/"results.json",dict(weights_sha256=expected,source="ShanShuiSeg v1.0.0 research checkpoint",
          n=len(rows),groups=dict(Counter(r["group"] for r in rows)),metrics=scores,
          confusion=total.tolist(),group_confusion={k:v.tolist() for k,v in group_results.items()},
          outputs=outputs,device=device,input_size=512,tag="native PyTorch research port; upstream parity not measured",
          independent_source=False,overlap="class IDs and data lineage similar; external train image IDs unverified"))
    print(scores,flush=True)


if __name__=="__main__":
    main()
