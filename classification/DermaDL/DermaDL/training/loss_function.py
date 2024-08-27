import torch.nn as nn
import torch.nn.functional as F


class WSLoss(nn.Module):
    def __init__(self):  # pylint: disable=useless-super-delegation
        super(WSLoss, self).__init__()

    def forward(self, output, target, pre_output):  # pylint: disable=arguments-differ
        return F.cross_entropy(output, target) + ((pre_output - output) ** 2).mean()

    def __str__(self):
        return 'WSLoss'

    def __repr__(self):
        return 'WSLoss'


class WSPLoss(nn.Module):
    """
    Weekly supervised loss with additional value calculated outside
    mainly be used for supervising parameters in some layers of model
    """
    def __init__(self, gamma=10):
        super(WSPLoss, self).__init__()
        self.gamma = gamma

    def forward(self, output, target, value):   # pylint: disable=arguments-differ
        return F.cross_entropy(output, target) + value * self.gamma

    def __str__(self):
        return 'WSPLoss'

    def __repr__(self):
        return 'WSPLoss'
