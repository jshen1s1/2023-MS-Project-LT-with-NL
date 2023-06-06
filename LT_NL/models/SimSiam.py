import torch
import torch.nn as nn
from resnet import ResNet18

__all__ = ['SimSiam_SSL']

class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512, freeze=True):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder
        self.freeze = freeze

        # build a 2-layer projector
        prev_dim = self.encoder.linear.weight.shape[1]
        self.encoder.linear = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        self.encoder.linear,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.linear[3].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

    def forward(self, x1, x2=None):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """
        if not self.freeze:
            # compute features for one view
            z1, _ = self.encoder(x1) # NxC
            z2, _ = self.encoder(x2) # NxC

            p1 = self.predictor(z1) # NxC
            p2 = self.predictor(z2) # NxC

            return p1, p2, z1.detach(), z2.detach()
        else:
            z1, feat = self.encoder(x1) # NxC
            return z1, feat
    
def SimSiam_SSL(base_encoder=None, num_classes=100, low_dim=False, pred_dim=512, freeze=True):
    if not base_encoder:
        base_encoder = ResNet18(num_classes, low_dim=low_dim)
    siamese = SimSiam(base_encoder, num_classes, pred_dim=pred_dim, freeze=freeze)
    return siamese