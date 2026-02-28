# model/gabmil.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedAttentionPooling(nn.Module):
    def __init__(self, feature_dim, mid_dim, out_dim=1, flatten=False, dropout=0.):
        super(GatedAttentionPooling, self).__init__()

        self.feature_dim = feature_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.flatten = flatten

        self.attention_V = nn.Sequential(
            nn.Linear(self.feature_dim, self.mid_dim),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.feature_dim, self.mid_dim),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.mid_dim, self.out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [1, N, D] -> squeeze -> [N, D]
        if x.dim() == 3:
            x = x.squeeze(0)

        H = x  # N x D
        
        A_V = self.attention_V(H)  # N x mid
        A_U = self.attention_U(H)  # N x mid
        A = self.attention_weights(A_V * A_U)  # N x K
        A = torch.transpose(A, 1, 0)  # K x N
        A = F.softmax(A, dim=1)  # softmax over N
        A = self.dropout(A)

        M = torch.mm(A, H)  # K x D

        if self.flatten:
            M = M.flatten()
            return M.unsqueeze(0) # 1 x (K*D)

        return M.unsqueeze(0) # 1 x K x D
    
    def get_attention_maps(self, x):
        if x.dim() == 3:
            x = x.squeeze(0)

        H = x
        A_V = self.attention_V(H)
        A_U = self.attention_U(H)
        A = self.attention_weights(A_V * A_U)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        # No dropout for visualization

        M = torch.mm(A, H)

        if self.flatten:
            M = M.flatten()
            return M.unsqueeze(0), A

        return M.unsqueeze(0), A


class GABMILClassifier(nn.Module):
    def __init__(self, pooling_layer, num_feats, num_classes):
        super(GABMILClassifier, self).__init__()
        self.pooling_layer = pooling_layer
        self.num_feats = num_feats
        self.num_classes = num_classes
        
        self.classifier = nn.Linear(num_feats, num_classes)

    def forward(self, x):
        # x: [1, N, D]
        pooled_features = self.pooling_layer(x) # [1, D] (if flatten=True)
        if pooled_features.dim() > 2:
            pooled_features = pooled_features.flatten(start_dim=1)
            
        logits = self.classifier(pooled_features)
        return logits
    
    def get_attention_maps(self, x):
        pooled_features, attention_maps = self.pooling_layer.get_attention_maps(x)
        if pooled_features.dim() > 2:
            pooled_features = pooled_features.flatten(start_dim=1)
            
        logits = self.classifier(pooled_features)
        return logits, attention_maps