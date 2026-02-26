# model/dsmil.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()
        self.feature_extractor = feature_extractor      
        self.fc = nn.Linear(feature_size, output_class)
        
    def forward(self, x):
        # x: N x K
        feats = self.feature_extractor(x) # N x K
        c = self.fc(feats.view(feats.shape[0], -1)) # N x C
        return feats.view(feats.shape[0], -1), c

class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0): 
        super(BClassifier, self).__init__()
        self.q = nn.Linear(input_size, 128)
        self.v = nn.Sequential(
            nn.Dropout(dropout_v),
            nn.Linear(input_size, input_size)
        )
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)
        
    def forward(self, feats, c): # N x K, N x C
        device = feats.device
        V = self.v(feats) # N x V
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q
        
        # handle multiple classes
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # critical instances
        q_max = self.q(m_feats) # queries of critical instances
        
        # Attention
        A = torch.mm(Q, q_max.transpose(0, 1)) 
        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) 
        
        B = torch.mm(A.transpose(0, 1), V) # C x V
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        
        C = self.fcc(B) # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B 

class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier        
            
    def forward(self, x):
        # feats: N x K, classes: N x C
        feats, classes = self.i_classifier(x)
        prediction_bag, A, B = self.b_classifier(feats, classes)
        
        return classes, prediction_bag, A, B