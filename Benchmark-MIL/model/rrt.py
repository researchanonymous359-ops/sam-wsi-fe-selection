# model/rrt.py
from typing import Optional
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce


# =========================
# Fixed hyperparameters
# =========================
RRT_CFG = dict(
    heads=8,
    region_attn="native",
    region_size=0,
    region_num=8,
    pool="attn",
    attn_act="tanh",
    use_epeg=False,
    epeg_k=5,
)


# =========================
# Basic Components
# =========================
class Attention(nn.Module):
    def __init__(self, input_dim=512, act='tanh', bias=False, dropout=False):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128, bias=bias)
        if act == 'gelu':
            self.act = nn.GELU()
        elif act == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.Tanh()
        self.drop = nn.Dropout(0.25) if dropout else nn.Identity()
        self.fc2 = nn.Linear(128, 1, bias=bias)

    def forward(self, x, no_norm: bool = False):
        A = self.fc2(self.drop(self.act(self.fc1(x))))
        A = A.transpose(1, 2)
        A_ori = A.clone()
        A = F.softmax(A, dim=-1)
        x = A @ x
        out = x.squeeze(1)
        return (out, A_ori.squeeze(1)) if no_norm else (out, A.squeeze(1))


class AttentionGated(nn.Module):
    def __init__(self, input_dim=512, act='tanh', bias=False, dropout=False):
        super().__init__()
        self.fc_a = nn.Linear(input_dim, 128, bias=bias)
        self.fc_b = nn.Linear(input_dim, 128, bias=bias)
        if act == 'gelu':
            self.act = nn.GELU()
        elif act == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.drop = nn.Dropout(0.25) if dropout else nn.Identity()
        self.fc_out = nn.Linear(128, 1, bias=bias)

    def forward(self, x, no_norm: bool = False):
        a = self.act(self.fc_a(x))
        b = self.sig(self.fc_b(x))
        A = self.fc_out(self.drop(a * b))
        A = A.transpose(1, 2)
        A_ori = A.clone()
        A = F.softmax(A, dim=-1)
        x = A @ x
        out = x.squeeze(1)
        return (out, A_ori.squeeze(1)) if no_norm else (out, A.squeeze(1))


class DAttention(nn.Module):
    def __init__(self, input_dim=512, act='tanh', gated: bool = False, bias=False, dropout=False):
        super().__init__()
        self.attn = AttentionGated(input_dim, act, bias, dropout) if gated else Attention(input_dim, act, bias, dropout)

    def forward(self, x, return_attn: bool = False, no_norm: bool = False, **kwargs):
        out, attn = self.attn(x, no_norm)
        return (out, attn) if return_attn else out


# =========================
# Nystrom Attention
# =========================
def exists(x):
    return x is not None

def moore_penrose_iter_pinv(x, iters: int = 6):
    device = x.device
    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row) + 1e-12)
    I = torch.eye(x.shape[-1], device=device)
    I = rearrange(I, 'i j -> () i j')
    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))
    return z

class NystromAttention(nn.Module):
    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8, num_landmarks: int = 256,
                 pinv_iterations: int = 6, residual: bool = True, residual_conv_kernel: int = 33,
                 eps: float = 1e-8, dropout: float = 0.0):
        super().__init__()
        self.eps = eps
        self.heads = heads
        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations
        inner_dim = heads * dim_head
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.residual = residual
        if residual:
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (residual_conv_kernel, 1),
                                      padding=(padding, 0), groups=heads, bias=False)

    def forward(self, x, mask: Optional[torch.Tensor] = None, return_attn: bool = False):
        b, n, _ = x.shape
        h, m, iters, eps = self.heads, self.num_landmarks, self.pinv_iterations, self.eps
        
        rem = n % m
        if rem > 0:
            pad = m - rem
            x = F.pad(x, (0, 0, pad, 0), value=0.0)
            if exists(mask):
                mask = F.pad(mask, (pad, 0), value=False)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        if exists(mask):
            mask_ = rearrange(mask, 'b n -> b () n')
            q, k, v = q * mask_[..., None], k * mask_[..., None], v * mask_[..., None]

        q = q * self.scale
        l = math.ceil(x.shape[1] / m)
        qL = reduce(q, 'b h (n l) d -> b h n d', 'sum', l=l)
        kL = reduce(k, 'b h (n l) d -> b h n d', 'sum', l=l)

        divisor = l
        if exists(mask):
            mask_l_sum = reduce(rearrange(mask, 'b (n l) -> b n l', l=l), 'b n l -> b n', 'sum')
            divisor = mask_l_sum[..., None].clamp_min(eps)
            maskL = rearrange(mask_l_sum > 0, 'b n -> b () n')

        qL, kL = qL / divisor, kL / divisor
        attn1 = torch.einsum('b h i d, b h j d -> b h i j', q,  kL)
        attn2 = torch.einsum('b h i d, b h j d -> b h i j', qL, kL)
        attn3 = torch.einsum('b h i d, b h j d -> b h i j', qL, k)

        if exists(mask):
            mask_full = rearrange(mask, 'b n -> b () n')
            mv = -torch.finfo(attn1.dtype).max
            attn1 = attn1.masked_fill(~(mask_full[..., None] * maskL[..., None, :]), mv)
            attn2 = attn2.masked_fill(~(maskL[..., None]     * maskL[..., None, :]), mv)
            attn3 = attn3.masked_fill(~(maskL[..., None]     * mask_full[..., None, :]), mv)

        attn1, attn2, attn3 = attn1.softmax(-1), attn2.softmax(-1), attn3.softmax(-1)
        attn2 = moore_penrose_iter_pinv(attn2, iters)
        out = (attn1 @ attn2) @ (attn3 @ v)

        if self.residual:
            out = out + self.res_conv(v)

        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        out = self.to_out(out)
        out = out[:, -n:]
        if return_attn:
            vis = (attn1 @ attn2) @ attn3
            return out, vis[..., -n:]
        return out


# =========================
# Region-wise Attention
# =========================
def _grid_pad(L: int, region_size: Optional[int], region_num: int):
    H = W = int(np.ceil(np.sqrt(L)))
    if region_size is not None and region_size > 0:
        add = -H % region_size
        H = W = H + add
        r_size = region_size
        r_num = H // r_size
    else:
        add = -H % region_num
        H = W = H + add
        r_num = region_num
        r_size = H // r_num
    add_len = H * W - L
    return H, W, add_len, r_num, r_size

def region_partition(x, region_size):
    B, H, W, C = x.shape
    x = x.view(B, H // region_size, region_size, W // region_size, region_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, region_size, region_size, C)

def region_reverse(regions, region_size, H, W):
    B = int(regions.shape[0] / (H * W / region_size / region_size))
    x = regions.view(B, H // region_size, W // region_size, region_size, region_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

class InnerAttention(nn.Module):
    def __init__(self, dim: int, head_dim: Optional[int] = None, num_heads: int = 8,
                 use_epeg: bool = False, epeg_k: int = 5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or max(16, dim // num_heads)
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, self.head_dim * num_heads * 3, bias=True)
        self.proj = nn.Linear(self.head_dim * num_heads, dim)
        
        self.use_epeg = use_epeg
        if self.use_epeg:
            padding = epeg_k // 2
            self.epeg_conv = nn.Conv2d(
                num_heads, num_heads, kernel_size=(epeg_k, 1), 
                padding=(padding, 0), groups=num_heads, bias=True
            )

    def forward(self, x: torch.Tensor):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_epeg:
            attn = attn + self.epeg_conv(attn)
            
        attn = attn.softmax(-1)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, self.num_heads * self.head_dim)
        return self.proj(out)

class RegionAttention(nn.Module):
    def __init__(self, dim: int, head_dim: Optional[int] = None, num_heads: int = 8,
                 region_attn: str = "native", region_size: int = 0, region_num: int = 8,
                 use_epeg: bool = False, epeg_k: int = 5):
        super().__init__()
        self.dim = dim
        self.region_size = region_size if region_size > 0 else None
        self.region_num = region_num

        if region_attn == "native":
            self.attn = InnerAttention(
                dim, head_dim=head_dim, num_heads=num_heads,
                use_epeg=use_epeg, epeg_k=epeg_k
            )
        else:
            self.attn = NystromAttention(
                dim=dim, dim_head=(head_dim or max(16, dim // num_heads)), heads=num_heads
            )

    def forward(self, x: torch.Tensor):
        B, L, C = x.shape
        H, W, add_len, _, r_size = _grid_pad(L, self.region_size, self.region_num)
        if add_len > 0:
            x = torch.cat([x, x.new_zeros(B, add_len, C)], dim=1)

        x = x.view(B, H, W, C)
        regions = region_partition(x, r_size).view(-1, r_size * r_size, C)
        regions = self.attn(regions).view(-1, r_size, r_size, C)
        x = region_reverse(regions, r_size, H, W).view(B, H * W, C)

        if add_len > 0:
            x = x[:, :-add_len]
        return x


# =========================
# RRTMIL (Refactored)
# =========================
class RRTMIL(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

        # --- 고정 설정 사용 ---
        heads       = RRT_CFG["heads"]
        region_attn = RRT_CFG["region_attn"]
        region_size = RRT_CFG["region_size"]
        region_num  = RRT_CFG["region_num"]
        pool_type   = RRT_CFG["pool"]
        attn_act    = RRT_CFG["attn_act"]
        use_epeg    = RRT_CFG["use_epeg"]
        epeg_k      = RRT_CFG["epeg_k"]

        self.backbone = RegionAttention(
            dim=in_dim,
            head_dim=None,
            num_heads=heads,
            region_attn=region_attn,
            region_size=region_size,
            region_num=region_num,
            use_epeg=use_epeg,
            epeg_k=epeg_k
        )

        self.pool = DAttention(input_dim=in_dim, act=attn_act, gated=False, dropout=False)
        self.head = nn.Linear(in_dim, num_classes)

    def forward(self, feats: torch.Tensor):
        if feats.dim() == 2:
            feats = feats.unsqueeze(0) # [1, N, D]

        h_all = self.backbone(feats)
        h = self.pool(h_all)
        logits = self.head(h)

        y_prob = F.softmax(logits, dim=1)
        y_hat  = torch.argmax(logits, dim=1)
        return logits, y_prob, y_hat