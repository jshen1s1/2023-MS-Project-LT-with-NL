import torch.nn as nn
from utils import *

__all__ = ['create_irm_classifer']

class ClassifierIRM(nn.Module):
    def __init__(self, label_freq_array,feat_dim,num_classes):
        super(ClassifierIRM, self).__init__()

        self.fc = nn.Linear(feat_dim, num_classes, bias=True)
        self.tro = nn.Parameter(torch.ones(num_classes).cuda(), requires_grad= True)
        self.adjustments = torch.log(label_freq_array.pow(self.tro) + 1e-12)

    def forward(self, x, add_inputs=None):

        y = self.fc(x)-self.adjustments
        return y

    def get_tro(self):
        return self.tro


def create_irm_classifer(train_loader, num_classes,feat_dim = 512):
    model = ClassifierIRM(train_loader,feat_dim=feat_dim,num_classes=num_classes)
    return model