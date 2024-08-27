import torch
import torch.nn as nn
import torch.nn.functional as F
from .efficientnet import EfficientNet
from .resnet import resnet18



model_dict = {
    'resnet18': [resnet18(pretrained=False), 512]
}

## efficient HAC loss
class LHACv1(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='efficientnet-b4', head='mlp', feat_dim=128, 
                 num_classes=10, bsz=16):
        super(LHACv1, self).__init__()
        self.bsz = bsz
        # EfficientNet output 1000d
        if 'efficientnet' in name:
            self.encoder, dim_in = EfficientNet.from_pretrained(name), 1000
        else:
            model_fun, dim_in = model_dict[name]
            self.encoder = model_fun        
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
            self.hiehead = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        
    def forward(self, x):
        # print("input x: ", x.size())
        _, feat = self.encoder(x) # for efficientnet
        # feat = self.encoder(x) # for resnet
        
        # h_feat = F.normalize(self.hiehead(x), dim=1)
        # feat = F.normalize(self.head(x), dim=1)
        
        # feat = self.head(feat) # for projection_MLP()
        # print("supconEFF feat:", feat.size())
        
        return feat
