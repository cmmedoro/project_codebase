import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# class to implement the GeM pooling layer, to substitute to the current Average Pooling layer of ResNet-18
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p, requires_grad=True)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

#class to implement the MixVPR
class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1):
        super(FeatureMixerLayer, self).__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim), #49
            nn.Linear(in_dim, int(in_dim * mlp_ratio)), #(49, 49*1)
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim), #(49, 49)
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.mix(x)


class MixVPR(nn.Module):
    def __init__(self,
                 in_channels=512,
                 in_h=7,
                 in_w=7,
                 out_channels=512,
                 mix_depth=1,
                 mlp_ratio=1,
                 out_rows=4,
                 ) -> None:
        super(MixVPR,self).__init__()

        self.in_h = in_h # height of input feature maps
        self.in_w = in_w # width of input feature maps
        self.in_channels = in_channels # depth of input feature maps
        
        self.out_channels = out_channels # depth wise projection dimension
        self.out_rows = out_rows # row wise projection dimesion

        self.mix_depth = mix_depth # L the number of stacked FeatureMixers
        self.mlp_ratio = mlp_ratio # ratio of the mid projection layer in the mixer block

        hw = in_h*in_w #7*7 = 49
        self.mix = nn.Sequential(*[
            FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio)
            for _ in range(self.mix_depth)
        ])
        self.channel_proj = nn.Linear(in_channels, out_channels)  #(512, 512)
        self.row_proj = nn.Linear(hw, out_rows) #(49, 4)

    def forward(self, x):
        #Input: (num_batches, 512, 7, 7)
        x = x.flatten(2)
        #intermediate_output: (256, 512, 49)
        x = self.mix(x)
        #intermediate_output: (256, 512, 49)
        x = x.permute(0, 2, 1) #permute dimensions, invert second and third dimension
        #intermediate_output: (256, 49, 512)
        x = self.channel_proj(x)
        #intermediate_output: (256, 49, 512)
        x = x.permute(0, 2, 1)
        #intermediate_output: (256, 512, 49)
        x = self.row_proj(x)
        #intermediate_output: (256, 512, 4)
        x = F.normalize(x.flatten(1), p=2, dim=-1)
        #output: (256, 2048)
        return x
    """
#def print_nb_params(m):
 #   model_parameters = filter(lambda p: p.requires_grad, m.parameters())
  #  params = sum([np.prod(p.size()) for p in model_parameters])
   # print(f'Trainable parameters: {params/1e6:.3}M')


def main():
    x = torch.randn(1, 1024, 20, 20)
    agg = MixVPR(
        in_channels=1024,
        in_h=20,
        in_w=20,
        out_channels=1024,
        mix_depth=4,
        mlp_ratio=1,
        out_rows=4)

    print_nb_params(agg)
    output = agg(x)
    print(output.shape)

if __name__ == '__main__':
    main()"""