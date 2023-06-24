import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# class to implement the GeM pooling layer, to substitute to the current Average Pooling layer of ResNet-18
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        # Define a tensor to be considered as a parameter of class nn.Module, so it will be available in parameters()
        # tensor of size 1 filled with ones*p (overall value: 3)
        # requires_grad: specify if the parameter requires the gradient
        self.p = nn.Parameter(torch.ones(1)*p, requires_grad=True)
        self.eps = eps

    def forward(self, x):
        # call the gem function and apply it to the descriptors
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        # define the gem operation as follows:
        # apply a 2D average pooling operation on each feature map
        # input: descriptors limited in range [1e-6, inf) and we consider the values at the power of p
        # kernel size is a tuple considering size of the second to last dimension and the last one
        # ---> pooling region (kH, kW)
        # the result f the average pooling operation is elevated at the power of 1/p
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

#class to implement the MixVPR
class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1):
        super(FeatureMixerLayer, self).__init__()
        self.mix = nn.Sequential(
            # apply Layer Normalization considering an input dimension of 49
            nn.LayerNorm(in_dim), 
            # apply fc layer with input_features = 49 and output features 49*1
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            # apply ReLU activation function: does not change the output dimension with respect to the input one
            nn.ReLU(),
            # apply fc layer with input_features = 49*1 and output features
            nn.Linear(int(in_dim * mlp_ratio), in_dim), 
        )
        #in Figure 2 of the paper mixVPr we can see that the feature mixer(that will be repeated L times) is composed 
        # exactly by these stages: normalization, projection, activation and projection
        # self.modules() = iterator over all the modules of the network
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                # if we are considering a fc layer, initialize the parameters, so that they will not be considered by autograd.
                # Fill the tensor of the weights of the layer with values from a truncated Normal distribution with standard deviation
                # 0.02 and mean 0.0, truncated at bounds -2.0 and 2.0
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    # if the biases of the layer are not None, fill them with 0s
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # equation 2 of MixVPR paper
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
        # define a mix layer as a series of L = 4 FeatureMizerLayer
        self.mix = nn.Sequential(*[
            FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio)
            for _ in range(self.mix_depth)
        ])
        # define a channel projection operation ---> fc layer considering (512, 512)
        self.channel_proj = nn.Linear(in_channels, out_channels)
        # define a row projection operation ---> fc layer considering (49, 4)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x):
        #Input: (num_batches, 512, 7, 7)
        x = x.flatten(2)
        # for each image (size 0) sent to the backbone obtains c channels (size 1) of dimension n = h * w ---> flatten third and fourth dim
        #intermediate_output: (256, 512, 49) ---> (batch_size, c, n)
        x = self.mix(x)
        #intermediate_output: (256, 512, 49)
        # Now we want to reduce the dimensionality through channel wise and row wise projections with two fc layers
        x = x.permute(0, 2, 1) #permute dimensions, invert second and third dimension
        #intermediate_output: (256, 49, 512) ---> (batch_size, n, c)
        # Reduce number of channels from in_channels to out_channels
        x = self.channel_proj(x)
        #intermediate_output: (256, 49, 512) ---> (batch_size, n, out_c)
        x = x.permute(0, 2, 1)
        #intermediate_output: (256, 512, 49) ---> (batch_size, out_c, n)
        # Reduce number of rows from h*w to out_rows
        x = self.row_proj(x)
        #intermediate_output: (256, 512, 4)
        # x.flatten(1) ---> consider the descriptors flattened to (batch_size, out_c * out_rows), then apply on this L2 normalization
        x = F.normalize(x.flatten(1), p=2, dim=-1)
        #output: (256, 2048)
        return x
   