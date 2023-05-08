import numpy as np
import main as aggregators

def get_aggregator(agg_arch, agg_config={}):

    if 'gem' in agg_arch.lower():
        agg_config={'p': 3}
        return aggregators.GeM(**agg_config)
    
    elif 'mixvpr' in agg_arch.lower():
        agg_config={'in_channels' : 1024,
                'in_h' : 20,
                'in_w' : 20,
                'out_channels' : 1024,
                'mix_depth' : 4,
                'mlp_ratio' : 1,
                'out_rows' : 4}
        return aggregators.MixVPR(**agg_config)