# model/abmil.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    def __init__(self, feature_dim, mid_dim, out_dim=1, flatten=False, dropout=0.0):
        super(AttentionPooling, self).__init__()

        self.feature_dim = feature_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.flatten = flatten

        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.out_dim)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [1, N, D] or [N, D] -> [N, D]
        if x.dim() == 3:
            x = x.squeeze(0)

        H = x  # N x D
        
        A = self.attention(H)  # N x K
        A = torch.transpose(A, 1, 0)  # K x N
        A = F.softmax(A, dim=1)  # softmax over N
        A = self.dropout(A)

        M = torch.mm(A, H)  # K x D

        if self.flatten:
            M = M.flatten()
            return M.unsqueeze(0)
        
        return M  # K x D (or 1 x D if K=1)
        
    def get_attention_maps(self, x):
        '''Return Attention Map without Dropout'''
        if x.dim() == 3:
            x = x.squeeze(0)

        H = x # N x D
        
        A = self.attention(H)  # N x K
        A = torch.transpose(A, 1, 0)  # K x N
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # K x D

        if self.flatten:
            M = M.flatten()
            return M.unsqueeze(0), A
        
        return M, A


class ABMILClassifier(nn.Module):
    def __init__(self, pooling_layer, num_feats, num_classes):
        super().__init__()
        self.pooling_layer = pooling_layer
        self.out_layer = nn.Linear(num_feats, num_classes)

    def forward(self, x):
        # x: [1, N, D] -> Pooling -> [1, D] -> Linear -> [1, C]
        pooling = self.pooling_layer(x)
        out = self.out_layer(pooling)
        return out
    
    def get_attention_maps(self, x):
        pooling, A = self.pooling_layer.get_attention_maps(x)
        out = self.out_layer(pooling)
        return out, A