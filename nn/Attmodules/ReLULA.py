import torch
import torch.nn as nn
import torch.nn.functional as F

class Mul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k):
        return q @ k


class LinearAttention(nn.Module):
    """
    ReLU激活线性注意力模块（ReLULA）- 终极稳定版
    完全避免inplace操作，使用functional API
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * self.scale)

        # 使用Conv2d保持4D格式
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qk_mul = Mul()
        self.kv_mul = Mul()
        self.extra_mul = Mul()

        # 不使用nn.ReLU模块，改用functional API
        # 这样完全避免inplace问题

    def forward(self, x):
        B, C, H, W = x.shape
        
        # QKV生成
        qkv = self.qkv(x)  # [B, 3C, H, W]
        q, k, v = qkv.chunk(3, dim=1)  # 各 [B, C, H, W]
        
        # 重塑为 [B, heads, H*W, head_dim]
        q = q.reshape(B, self.num_heads, C // self.num_heads, H*W).permute(0, 1, 3, 2).contiguous()
        k = k.reshape(B, self.num_heads, C // self.num_heads, H*W).permute(0, 1, 3, 2).contiguous()
        v = v.reshape(B, self.num_heads, C // self.num_heads, H*W).permute(0, 1, 3, 2).contiguous()
        
        # 使用functional ReLU（非inplace）
        q = F.relu(q, inplace=False)
        k = F.relu(k, inplace=False)

        # 计算归一化分母 - 使用clone避免视图问题
        k_sum = k.transpose(-2, -1).sum(dim=-1, keepdim=True).clone()
        denom = torch.einsum('b h n d, b h n l -> b h n', q, k_sum).clone()
        denom = denom.unsqueeze(-1).clamp(min=1e2)

        # 线性注意力计算
        k_t = k.transpose(-2, -1).contiguous().clone()
        attn = torch.einsum('b h n d, b h n e -> b h d e', k_t, v).clone()
        attn = attn * self.temperature
        attn1 = torch.einsum('b h n d, b h d e -> b h n e', q, attn).clone()

        # 归一化
        attn = attn1 / denom

        # 重塑回 [B, C, H, W]
        x = attn.transpose(1, 2).reshape(B, C, H, W).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)

        return x