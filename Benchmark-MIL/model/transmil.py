# model/transmil.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations
            residual = True,         # extra residual
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x
    
    def get_attention_maps(self, x):
        x_attn, attn_matrix = self.attn(self.norm(x), return_attn=True)
        x = x + x_attn
        return x, attn_matrix


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, n_classes, input_size):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(input_size, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)

    def forward(self, feats):
        # feats: [1, N, D] or [N, D] -> [1, N, D]
        device = feats.device
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
        
        h = self._fc1(feats) #[B, n, 512]
        
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]], dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(device)
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]
        
        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]

        #---->cls_token
        h = self.norm(h)[:, 0]

        #---->predict
        logits = self._fc2(h) #[B, n_classes]

        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)

        return logits, Y_prob, Y_hat
    
    def get_attention_maps(self, feats):
        # feats: [1, N, D]
        device = feats.device
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
            
        h = self._fc1(feats) #[B, n, 512]
        
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]], dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(device)
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h, attn1 = self.layer1.get_attention_maps(h) 

        #---->PPEG
        h = self.pos_layer(h, _H, _W) 
        
        #---->Translayer x2
        h, attn2 = self.layer2.get_attention_maps(h) 

        #---->cls_token
        h = self.norm(h)[:, 0]

        #---->predict
        logits = self._fc2(h) #[B, n_classes]

        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)

        return logits, Y_prob, Y_hat, attn1, attn2

if __name__ == "__main__":
    # Test
    data = torch.randn((1, 6000, 1024)).cuda()
    model = TransMIL(n_classes=2, input_size=1024).cuda()
    logits, probs, preds = model(data)
    print(f"Logits: {logits.shape}, Probs: {probs.shape}, Preds: {preds.shape}")