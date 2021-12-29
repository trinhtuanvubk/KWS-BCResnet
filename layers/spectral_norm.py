from torch import nn 
import torch 

class SubSpectralNorm(nn.Module):
    def __init__(self, C, S=5, eps=1e-5, affine=True):
        super().__init__()
        self.C = C 
        self.S = S 
        self.eps = eps 
        self.gamma = 1.0
        self.beta = 0 
        if affine:
            self.gamma = nn.Parameter(torch.FloatTensor(1, C*S, 1, 1))
            self.beta = nn.Parameter(torch.FloatTensor(1, C*S, 1, 1))
            nn.init.xavier_uniform_(self.gamma)
            nn.init.xavier_uniform_(self.beta)

    def forward(self, x):
        S, eps, gamma, beta = self.S, self.eps, self.gamma, self.beta
        N, C, F, T = x.size()
        x = x.view(N, C*S, F//S, T)
        mean = x.mean([0,2,3]).view([1, C*S, 1, 1])
        var = x.var([0,2,3]).view([1, C*S, 1, 1])
        x = gamma * (x - mean) / (var + eps).sqrt() + beta
        return x.view(N, C, F, T)