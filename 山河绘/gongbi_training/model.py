"""MobileNetV3-small encoder + U-Net decoder, structure/detail logits."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_small,MobileNet_V3_Small_Weights


class ConvBlock(nn.Sequential):
    def __init__(self,cin,cout):
        super().__init__(nn.Conv2d(cin,cout,3,padding=1,bias=False),nn.GroupNorm(8,cout),nn.SiLU(inplace=True),
                         nn.Conv2d(cout,cout,3,padding=1,bias=False),nn.GroupNorm(8,cout),nn.SiLU(inplace=True))


class GongbiLineNet(nn.Module):
    def __init__(self,pretrained=False):
        super().__init__()
        weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder=mobilenet_v3_small(weights=weights).features
        self.bottleneck=nn.Sequential(nn.Conv2d(576,128,1),nn.SiLU(inplace=True))
        self.up16=ConvBlock(128+48,96)
        self.up8=ConvBlock(96+24,64)
        self.up4=ConvBlock(64+16,48)
        self.up2=ConvBlock(48+16,32)
        self.up1=ConvBlock(32+3,24)
        self.head=nn.Conv2d(24,2,1)
        nn.init.constant_(self.head.bias,-2.)
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))

    def forward(self,rgb):
        x=(rgb-self.mean)/self.std
        feats=[]
        for i,layer in enumerate(self.encoder):
            x=layer(x)
            if i in (0,1,3,8,12):feats.append(x)
        def up(x,skip,block):
            return block(torch.cat([F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip],1))
        x=self.bottleneck(feats[4])
        x=up(x,feats[3],self.up16)
        x=up(x,feats[2],self.up8)
        x=up(x,feats[1],self.up4)
        x=up(x,feats[0],self.up2)
        x=up(x,rgb,self.up1)
        return self.head(x)


def segmentation_loss(logits,target,confidence):
    positive=torch.tensor([9.,4.],device=logits.device).view(2,1,1)
    bce=F.binary_cross_entropy_with_logits(logits,target,pos_weight=positive,reduction='none')
    bce=(bce*confidence).sum()/confidence.sum().clamp_min(1.)
    prob=logits.sigmoid()
    intersection=(prob*target*confidence).sum((0,2,3))
    denom=((prob+target)*confidence).sum((0,2,3))
    dice=1-(2*intersection+1)/(denom+1)
    return bce+.65*(dice[0]+.35*dice[1]),dict(bce=float(bce.detach()),dice_structure=float(dice[0].detach()))


def ink_restoration_loss(logits,target,confidence):
    """Channel 0: ink coverage (outline AND cunfa); channel 1: soft ink density.

    No semantic outline/cunfa ground truth is claimed. Soft density retains dry
    brush holes and grayscale rather than converting every ink edge to a path.
    """
    logits=logits.float();target=target.float();confidence=confidence.float()
    cov=F.binary_cross_entropy_with_logits(logits[:,0],target[:,0],pos_weight=torch.tensor(2.5,device=logits.device),reduction='none')
    cov=(cov*confidence[:,0]).sum()/confidence[:,0].sum().clamp_min(1)
    prob=logits.sigmoid();a=confidence[:,0]
    dice=1-(2*(prob[:,0]*target[:,0]*a).sum()+1)/((prob[:,0]+target[:,0])*a).sum().add(1)
    density_weights=confidence[:,1]*(1+5*target[:,0])
    regression=(F.smooth_l1_loss(prob[:,1],target[:,1],beta=.1,reduction='none')*density_weights).sum()/density_weights.sum().clamp_min(1)
    return cov+.35*dice+2*regression,dict(coverage_bce=float(cov.detach()),ink_dice_loss=float(dice.detach()),density_loss=float(regression.detach()))
