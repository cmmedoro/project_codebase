import numpy as np
import main as gem
import torch
import torch.nn.functional as F
import self_modules as sm

def get_aggregator(agg_arch, agg_config={}):

    if 'gem' in agg_arch.lower():
        agg_config={'p': 3}
        return gem.GeM(**agg_config)
    
    elif 'mixvpr' in agg_arch.lower():
        agg_config={'in_channels' : 512,
                'in_h' : 7,
                'in_w' : 7,
                'out_channels' : 512,
                'mix_depth' : 4,
                'mlp_ratio' : 1,
                'out_rows' : 4}
        return sm.MixVPR(**agg_config)
    
class Flatten(torch.nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): assert x.shape[2] == x.shape[3] == 1; return x[:,:,0,0]

class L2Norm(torch.nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return F.normalize(x, p=2.0, dim=self.dim)