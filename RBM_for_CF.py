import torch
from torch import nn

class CategoricalRBM(torch.nn.Module):

    def __init__(self, D, M, K):
        '''
        Categorical RBM has a visible part of D features, each feature is a K size one-hot tensor.
        Hidden part has size M, each is a Bernoulli variable.
        '''
        super(CategoricalRBM, self).__init__()
        self.v = torch.nn.Parameter(torch.randn(D, K, M))

        pass
    def forward(self, Visible, Mask):
        '''
        Visible is an N*D*K tensor, N: sample size; D:feature size(i.e., if Visible is user, then feature is movie or item);
        K: Category size.
        '''

        pass
