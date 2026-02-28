# model/ilra.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Multi-Head Attention block
# ---------------------------
class MultiHeadAttention(nn.Module):
    """
    Q: (B, SQ, Dq), K: (B, SK, Dk) -> output: (B, SQ, Dv), attn: (B, heads, SQ, SK) after averaging (B, SQ, SK)
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

        # gated residual (matches the original paper implementation)
        if self.gate is not None:
            out = out * self.gate(Q0)

        attn_simple = None
        if return_attention and attn is not None:
            # attn: (B * heads?) or (B, SQ, SK) depending on the framework version
            # If PyTorch 2.x and average_attn_weights=True, returns (B, SQ, SK)
            attn_simple = attn  # (B, SQ, SK)

        return out, attn_simple


# ---------------------------
# GAB block (Eq.16)
# ---------------------------
class GAB(nn.Module):
    """
    Compresses X into a low-rank latent (num_inds x dim_out) and restores it (proxy self-attention)
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
    Cross-attention with global token S (1,1,D) → slide-level embedding
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
            # Final attention map: (B, N)
            attn_vec = attn.sum(dim=1)  # SQ=1 here, so it is the same
        else:
            attn_vec = None
        return out, attn_vec


# ---------------------------
# ILRA (Fixed hyperparameters)
# ---------------------------
class ILRA(nn.Module):
    """
    Input: feats (N,C) or (B,N,C)
    Output: (logits, y_prob, y_hat)
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

        # GAB blocks (First block dim_in=input_size → embed_dim)
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
        Ignores (caption, subsite) inputs (not in the original ILRA paper implementation)
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

        # Assumes batch size 1 (framework runs per bag)
        if attn is not None and attn.dim() == 2 and attn.size(0) == 1:
            attn = attn[0]
        return logits, y_prob, y_hat, attn