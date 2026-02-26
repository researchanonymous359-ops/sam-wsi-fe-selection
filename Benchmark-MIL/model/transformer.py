# model/transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from nystrom_attention import NystromAttention

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, heads=8, dropout=0.1):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // heads,
            heads=heads,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=dropout,
        )

    def forward(self, x):
        # x: [B, N, C]
        x = x + self.attn(self.norm(x))
        return x

    def get_attention_maps(self, x):
        # returns: updated x and attention matrix
        x_attn, attn_matrix = self.attn(self.norm(x), return_attn=True)
        x = x + x_attn
        return x, attn_matrix


class TransformerMIL(nn.Module):
    """
    Standard Transformer MIL (CLS token based pooling)
    """
    def __init__(self, n_classes, input_size, hidden_dim=512, heads=8, dropout=0.1):
        super().__init__()
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim

        # feature -> hidden
        self._fc1 = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(inplace=True)
        )

        # learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # 2 transformer layers
        self.layer1 = TransLayer(dim=hidden_dim, heads=heads, dropout=dropout)
        self.layer2 = TransLayer(dim=hidden_dim, heads=heads, dropout=dropout)

        # final norm and classifier
        self.norm = nn.LayerNorm(hidden_dim)
        self._fc2 = nn.Linear(hidden_dim, self.n_classes)

    def forward(self, feats):
        """
        feats: [N, input_size] or [B, N, input_size]
        """
        # ensure [B, N, D]
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)

        device = feats.device

        # proj to hidden
        h = self._fc1(feats)                    # [B, N, hidden_dim]

        # prepend CLS token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(device)  # [B,1,C]
        h = torch.cat((cls_tokens, h), dim=1)   # [B, N+1, C]

        # transformer blocks
        h = self.layer1(h)                      # [B, N+1, C]
        h = self.layer2(h)                      # [B, N+1, C]

        # take CLS and classify
        h_cls = self.norm(h)[:, 0]              # [B, C]
        logits = self._fc2(h_cls)

        y_hat = torch.argmax(logits, dim=1)
        y_prob = F.softmax(logits, dim=1)
        return logits, y_prob, y_hat

    def get_attention_maps(self, feats):
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)

        device = feats.device

        # proj to hidden
        h = self._fc1(feats)

        # prepend CLS
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(device)
        h = torch.cat((cls_tokens, h), dim=1)

        # transformer blocks with attn maps
        h, attn1 = self.layer1.get_attention_maps(h)
        h, attn2 = self.layer2.get_attention_maps(h)

        # CLS
        h_cls = self.norm(h)[:, 0]
        logits = self._fc2(h_cls)

        y_hat = torch.argmax(logits, dim=1)
        y_prob = F.softmax(logits, dim=1)
        
        return logits, y_prob, y_hat, attn1, attn2