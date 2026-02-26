# model/ilra.py
"""
ILRA: Exploring Low-Rank Property in MIL for WSI Classification (ICLR 2023)
- 깃허브 프레임워크 스타일에 맞춰 경량/고정 하이퍼파라미터 버전
- forward(...) -> (logits, y_prob, y_hat)
- get_attention_maps(...) 지원: (logits, y_prob, y_hat, attn)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Multi-Head Attention block
# ---------------------------
class MultiHeadAttention(nn.Module):
    """
    Q: (B, SQ, Dq), K: (B, SK, Dk) -> output: (B, SQ, Dv), attn: (B, heads, SQ, SK) 평균 후 (B, SQ, SK)
    """
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln: bool = False, gated: bool = False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads

        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.mha = nn.MultiheadAttention(embed_dim=dim_V, num_heads=num_heads, batch_first=False)

        self.ln0 = nn.LayerNorm(dim_V) if ln else None
        self.ln1 = nn.LayerNorm(dim_V) if ln else None
        self.fc_o = nn.Linear(dim_V, dim_V)

        self.gate = None
        if gated:
            self.gate = nn.Sequential(nn.Linear(dim_Q, dim_V), nn.SiLU())

    def forward(self, Q, K, return_attention: bool = False):
        """
        Q: (B, SQ, DQ), K: (B, SK, DK)
        returns: O: (B, SQ, DV), attn: (B, SQ, SK) or None
        """
        Q0 = Q

        q = self.fc_q(Q)              # (B, SQ, DV)
        k = self.fc_k(K)              # (B, SK, DV)
        v = self.fc_v(K)              # (B, SK, DV)

        # MultiheadAttention(batch_first=False) → (SQ, B, DV)
        q_t = q.transpose(0, 1)
        k_t = k.transpose(0, 1)
        v_t = v.transpose(0, 1)

        out, attn = self.mha(q_t, k_t, v_t, need_weights=return_attention, average_attn_weights=True)
        # out: (SQ, B, DV) → (B, SQ, DV)
        out = out.transpose(0, 1)

        # residual + MLP + (optional) LN
        out = (q + out)
        if self.ln0 is not None:
            out = self.ln0(out)
        out = out + F.relu(self.fc_o(out))
        if self.ln1 is not None:
            out = self.ln1(out)

        # gated residual (원문 구현에 맞춤)
        if self.gate is not None:
            out = out * self.gate(Q0)

        attn_simple = None
        if return_attention and attn is not None:
            # attn: (B * heads?) 또는 (B, SQ, SK) 프레임워크 버전에 따라 다름
            # PyTorch 2.x의 average_attn_weights=True면 (B, SQ, SK) 반환
            attn_simple = attn  # (B, SQ, SK)

        return out, attn_simple


# ---------------------------
# GAB block (Eq.16)
# ---------------------------
class GAB(nn.Module):
    """
    Low-rank latent (num_inds x dim_out)로 X를 압축 후 복원 (proxy self-attention)
    """
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln: bool = True):
        super().__init__()
        self.latent = nn.Parameter(torch.empty(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.latent)

        self.project_forward = MultiHeadAttention(dim_out, dim_in, dim_out, num_heads, ln=ln, gated=True)
        self.project_backward = MultiHeadAttention(dim_in, dim_out, dim_out, num_heads, ln=ln, gated=True)

    def forward(self, X):
        """
        X: (B, N, dim_in)
        """
        B = X.size(0)
        latent_mat = self.latent.repeat(B, 1, 1)      # (B, num_inds, dim_out)
        H, _ = self.project_forward(latent_mat, X)    # (B, num_inds, dim_out)
        X_hat, _ = self.project_backward(X, H)        # (B, N, dim_out)
        return X_hat


# ---------------------------
# Non-Local Pooling for slide-level feature
# ---------------------------
class NLP(nn.Module):
    """
    Global token S (1,1,D)와 cross-attention → slide-level embedding
    """
    def __init__(self, dim, num_heads, ln: bool = True):
        super().__init__()
        self.S = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.xavier_uniform_(self.S)
        self.mha = MultiHeadAttention(dim, dim, dim, num_heads, ln=ln, gated=False)

    def forward(self, X, return_attention: bool = False):
        B = X.size(0)
        S = self.S.repeat(B, 1, 1)                           # (B, 1, D)
        out, attn = self.mha(S, X, return_attention=return_attention)  # out: (B,1,D), attn: (B,1,N)
        if return_attention and attn is not None:
            # 최종 어텐션 맵: (B, N)
            attn_vec = attn.sum(dim=1)  # 여기선 SQ=1 이므로 동일
        else:
            attn_vec = None
        return out, attn_vec


# ---------------------------
# ILRA (고정 하이퍼파라미터)
# ---------------------------
class ILRA(nn.Module):
    """
    입력: feats (N,C) or (B,N,C)
    출력: (logits, y_prob, y_hat)
    get_attention_maps: (logits, y_prob, y_hat, attn[N])
    """
    def __init__(
        self,
        n_classes: int,
        input_size: int,
        *,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        topk: int = 64,
        ln: bool = True
    ):
        super().__init__()
        self.n_classes = n_classes
        self.input_size = input_size

        # GAB blocks (첫 블록은 dim_in=input_size → embed_dim)
        blocks = []
        for i in range(num_layers):
            blocks.append(
                GAB(
                    dim_in=input_size if i == 0 else embed_dim,
                    dim_out=embed_dim,
                    num_heads=num_heads,
                    num_inds=topk,
                    ln=ln
                )
            )
        self.gabs = nn.ModuleList(blocks)

        # Non-Local Pooling
        self.pool = NLP(dim=embed_dim, num_heads=num_heads, ln=ln)

        # Classifier
        self.fc = nn.Linear(embed_dim, n_classes)

        # init
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def _encode(self, feats: torch.Tensor):
        """
        feats: (B,N,C)
        returns: enc (B,N,embed_dim)
        """
        x = feats
        for gab in self.gabs:
            x = gab(x)
        return x

    def forward(self, feats: torch.Tensor, caption=None, subsite=None):
        """
        (caption, subsite) 입력은 무시 (ILRA 원논문 구현에 없음)
        """
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)  # (1,N,C)

        enc = self._encode(feats)        # (B,N,E)
        slide_feat, _ = self.pool(enc)   # (B,1,E)
        slide_feat = slide_feat.squeeze(1)  # (B,E)

        logits = self.fc(slide_feat)     # (B, num_classes)
        y_prob = F.softmax(logits, dim=1)
        y_hat = torch.argmax(y_prob, dim=1)
        return logits, y_prob, y_hat

    @torch.no_grad()
    def get_attention_maps(self, feats: torch.Tensor, caption=None, subsite=None):
        """
        returns: logits, y_prob, y_hat, attn  with attn shape (N,) for single bag
        """
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)

        enc = self._encode(feats)              # (B,N,E)
        slide_feat, attn = self.pool(enc, return_attention=True)  # attn: (B,N)
        slide_feat = slide_feat.squeeze(1)     # (B,E)

        logits = self.fc(slide_feat)
        y_prob = F.softmax(logits, dim=1)
        y_hat = torch.argmax(y_prob, dim=1)

        # 배치 1 가정 (프레임워크가 bag 단위로 돌림)
        if attn is not None and attn.dim() == 2 and attn.size(0) == 1:
            attn = attn[0]
        return logits, y_prob, y_hat, attn
