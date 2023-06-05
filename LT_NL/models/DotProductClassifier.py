import torch.nn as nn
from utils import *

__all__ = ['DPC']

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out
    
class DotProduct_Classifier(nn.Module):
    
    def __init__(self, num_classes=1000, feat_dim=2048, use_norm=False):
        super(DotProduct_Classifier, self).__init__()
        if use_norm:
            self.fc = NormedLinear(feat_dim, num_classes)
        else:
            self.fc = nn.Linear(feat_dim, num_classes)
        
    def forward(self, x):
        x = self.fc(x)
        return x, None
    
def DPC(feat_dim, num_classes=1000, use_norm=False):
    return DotProduct_Classifier(num_classes, feat_dim, use_norm)