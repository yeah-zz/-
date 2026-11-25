"""One learned per-pixel mixing weight shared by all four object classes."""
import torch
from torch import nn
class GlobalDetailGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(11,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,1,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):
        alpha=self.net(x).sigmoid()
        probability=alpha*x[:,3:7]+(1-alpha)*x[:,7:11]
        return probability.clamp_min(1e-8).log(),alpha
