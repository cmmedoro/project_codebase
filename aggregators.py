import numpy as np
import torch
import torch.nn.functional as F
import self_modules as sm

def get_aggregator(agg_arch, agg_config={}):
    #return the aggregator and the corresponding parameters for the configuration
    if 'gem' in agg_arch.lower():
        agg_config={'p': 3}
        return sm.GeM(**agg_config)
    
    elif 'mixvpr' in agg_arch.lower():
        agg_config={'in_channels' : 512,
                'in_h' : 7,
                'in_w' : 7,
                'out_channels' : 256, #512
                'mix_depth' : 4,
                'mlp_ratio' : 1,
                'out_rows' : 4}
        return sm.MixVPR(**agg_config)
    
class Flatten(torch.nn.Module):
    def __init__(self): 
        super().__init__()

    def forward(self, x): 
        #assert is a debugging statement ---> checks if the condition presented is True or not
        #in this case: if the third and fourth dimensions of the descriptors tensor are equal and are equal to 1
        #this function returns a descriptor vector flattened, meaning that it will have a dimension of the kind
        #(num_batches, 512)
        assert x.shape[2] == x.shape[3] == 1;       
        return x[:,:,0,0]

class L2Norm(torch.nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        #consider transforming the vector along dimension dim
        self.dim = dim
    
    def forward(self, x):
        # Perform L2 normalization, meaning normalize the descriptors vector with respect to the
        # maximum between all the two-norms (sum of squared terms) of the vector --> so that
        # the sum of squared terms is always equal to 1
        return F.normalize(x, p=2.0, dim=self.dim)