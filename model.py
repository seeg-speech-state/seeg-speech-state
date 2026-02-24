import torch
import numpy as np
from torch.nn import Conv1d,ReLU,LeakyReLU,InstanceNorm1d,BatchNorm1d,Flatten,Linear

class Mean(torch.nn.Module):
    def __init__(self,dim=2):
        super().__init__()
        self.dim = dim
    def __call__(self,x):
        return x.mean(dim=self.dim,keepdim=True)


class BatchNorm1d(torch.nn.BatchNorm1d):
    '''running stats on Nx1x1'''
    def __init__(self,num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
        super().__init__(num_features,eps=eps,
            momentum=momentum,affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype)


class VanillaCNN1d(torch.nn.Module):
    '''
    By design this network has no paddings, and thus performs with causality.
    block_filters: num of cnn filters for point-wise preprocessing.
    fc_units[0]: num of cnn filters for temproal fusing
    fc_units[1:]: num of resnet blocks for post-temporal processing.
                    each resnet block contains 3 conv layers and one adaptor cnn layer.
    output_layer: softmax or sigmoid
    '''
    def __init__(self, in_channels=117,in_length=200,cnn_ksizes=[16,16,16],block_filters=[256,128,32],fc_units=[512,256],drop_out=0.1,classes=50,act=ReLU,conv=Conv1d,norm=InstanceNorm1d,output_layer='softmax',dtype=torch.float64):
        super().__init__()
        
        # print(f"ConvnetPyramid1d: parameters ignored by this class:{args} {other_kws}")
        self.output_layer = output_layer
        layers = []
        input_features = in_channels
        stem_layer = conv(in_channels=input_features,out_channels=32,kernel_size=15,stride=4,padding=0,dtype=dtype)
        input_features = 32
        bn_i = norm(num_features=input_features,dtype=dtype)
        act_i = act()
        layers.extend([stem_layer, bn_i, act_i])
        #channel combination
        for ksize,filters in zip(cnn_ksizes,block_filters):
            c_i = conv(in_channels=input_features,out_channels=filters,kernel_size=ksize,stride=1,padding=ksize//2,dtype=dtype)
            maxpool_i = torch.nn.MaxPool1d(kernel_size=2,stride=2,padding=1)
            bn_i = norm(num_features=filters,dtype=dtype)
            act_i = act()
            drop_i = torch.nn.Dropout(p=drop_out)
            input_features = filters
            layers.extend([c_i,maxpool_i,bn_i,act_i,drop_i])
            # layers.extend([c_i,maxpool_i,drop_i])
        # feature is 1 x C x T

        layers.append(Mean(dim=-1))

        for linear_units in fc_units:
            c = conv(in_channels=input_features,out_channels=linear_units,kernel_size=1,stride=1,padding=0,bias=True,dtype=dtype)
            bn_i = norm(num_features=input_features,dtype=dtype)
            act_i = act()
            layers.extend([c,bn_i,act_i])
            input_features = linear_units

        c = conv(in_channels=input_features,out_channels=classes,kernel_size=1,stride=1,padding=0,bias=False,dtype=dtype)
        
        layers.append(c)

        self.model = torch.nn.Sequential(*layers)

    def forward(self,inputs):
        out = self.model(inputs) # batch x classes x frames
        if self.training: #inference mode: output probability
            return out 
        else:
            if self.output_layer == 'softmax':
                out = torch.softmax(out,dim=1)
            else:
                out = torch.sigmoid(out)
            return out

    def feature(self,inputs):
        if not hasattr(self,'feature_extractor'):
            self.feature_extractor = self.model[:-1]
        return self.feature_extractor(inputs)

    def channel_weight_based_on_preprocessing_layer(self):
        '''use the first layer as a band selection indicator'''
        first_cnn = self.model[0]
        cnn_kernel = first_cnn.weight.detach().clone() # outfeature x infeature x ksize
        w = cnn_kernel.abs().mean(dim=2).mean(dim=0) # (infeature,)
        return w
