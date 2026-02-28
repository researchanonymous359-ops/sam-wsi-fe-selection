# model/dtfdmil/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
# [Modified] Import Classifier_1fc from network.py
from model.dtfdmil.network import Classifier_1fc

class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()
        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        ## x: N x L
        A_V = self.attention_V(x)  # N x D
        A_U = self.attention_U(x)  # N x D
        A = self.attention_weights(A_V * A_U) # N x K
        A = torch.transpose(A, 1, 0)  # K x N

        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N

        return A  # K x N


class Attention_with_Classifier(nn.Module):
    def __init__(self, L=512, D=128, K=1, num_cls=2, droprate=0, args=None):
        super(Attention_with_Classifier, self).__init__()
        self.L = L
        self.D = D
        self.K = K
        self.num_cls = num_cls

        self.attention = Attention_Gated(L, D, K)
        # Removed text/subsite logic -> Use input dimension L as is
        self.classifier = Classifier_1fc(L * K, num_cls, droprate)
    
    def forward(self, x): 
        ## x: N x L
        AA = self.attention(x)  # K x N
        
        # Attention Pooling
        afeat = torch.mm(AA, x) # K x L
        
        # Classification
        pred = self.classifier(afeat) # K x num_cls
        
        # 🔥 [Core Modification] Must return both pred and AA for DTFD logic to work
        return pred, AA
