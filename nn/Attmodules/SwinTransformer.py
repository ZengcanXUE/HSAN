import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def to_2tuple(x):
    """Convert int to tuple of length 2"""
    return (x, x) if isinstance(x, int) else x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias."""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # 确保所有参数是float32
        self.qkv = self.qkv.float()
        self.proj = self.proj.float()
        self.relative_position_bias_table = self.relative_position_bias_table.float()
        
        # 确保输入是float32
        if x.dtype != torch.float32:
            x = x.float()
        
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            mask = mask.float()
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class SwinTransformerBlockStable(nn.Module):
    """
    稳定版Swin Transformer Block - 直接处理4D特征图 [B, C, H, W]
    自动处理维度问题，避免之前的错误
    """
    def __init__(self, dim, num_heads=4, window_size=7, mlp_ratio=4., 
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0., 
                 norm_layer=nn.LayerNorm, *args, **kwargs):
        super().__init__()
        
        # 参数验证和调整
        self.dim = dim
        self.num_heads = max(1, min(num_heads, dim // 4))  # 确保num_heads不超过dim/4
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        
        # 确保dim能被num_heads整除
        while dim % self.num_heads != 0:
            self.num_heads -= 1
            if self.num_heads < 1:
                self.num_heads = 1
                break
        
        print(f"SwinTransformerBlockStable初始化: dim={dim}, num_heads={self.num_heads}, window={window_size}")
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=self.num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)
        
        # 自适应层，用于处理维度不匹配
        self.adapt_conv = None

    def forward(self, x):
        # 获取当前设备
        device = x.device
        
        # 确保输入是float32
        if x.dtype != torch.float32:
            x = x.float()
        
        # 强制所有子模块的参数为float32
        self.norm1 = self.norm1.float()
        self.norm2 = self.norm2.float()
        self.mlp = self.mlp.float()
        self.attn = self.attn.float()
        
        # 如果有自适应卷积，也强制转换
        if self.adapt_conv is not None:
            self.adapt_conv = self.adapt_conv.float()
        
        B, C, H, W = x.shape
        
        # 确保输入通道与期望一致
        if C != self.dim:
            if self.adapt_conv is None:
                self.adapt_conv = nn.Conv2d(C, self.dim, 1).to(device).float()
                print(f"⚠️ SwinTransformer通道自适应: {C} -> {self.dim}")
            x = self.adapt_conv(x)
            B, C, H, W = x.shape
        
        # 保存原始输入用于残差连接
        identity = x
        
        # 转换为序列形式 [B, H*W, C]
        x = x.flatten(2).transpose(1, 2)
        
        shortcut = x
        x = self.norm1(x)
        
        # 重塑回空间形式用于窗口划分
        x = x.view(B, H, W, C)
        
        # 调整窗口大小使其能整除H和W
        window_size = min(self.window_size, H, W)
        while H % window_size != 0 or W % window_size != 0:
            window_size -= 1
            if window_size < 2:
                window_size = 2
                break
        
        # 如果窗口大小变化了，需要重新创建attention
        if window_size != self.window_size:
            self.attn = WindowAttention(
                C, window_size=to_2tuple(window_size), num_heads=self.num_heads,
                qkv_bias=True, attn_drop=0., proj_drop=0.
            ).to(device).float()
            self.window_size = window_size
        
        # partition windows
        x_windows = window_partition(x, window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, window_size * window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA
        attn_windows = self.attn(x_windows, mask=None)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, window_size, window_size, C)
        x = window_reverse(attn_windows, window_size, H, W)  # B H W C
        
        # 转回序列形式
        x = x.view(B, H * W, C)

        # 第一个残差连接
        x = shortcut + self.drop_path(x)
        
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # 重塑回4D [B, C, H, W]
        x = x.transpose(1, 2).view(B, C, H, W)
        
        return x


# 简化版 - 适合快速测试
class SwinTransformerSimple(nn.Module):
    """
    极简版Swin Transformer - 只做恒等映射
    用于快速验证集成是否成功
    """
    def __init__(self, dim, num_heads=4, window_size=7, *args, **kwargs):
        super().__init__()
        self.dim = dim
        print(f"SwinTransformerSimple初始化: dim={dim}")
        
    def forward(self, x):
        return x  # 恒等映射
    
class SwinTransformerHeadSimple(nn.Module):
    """
    简化的 Swin Transformer Head - 恒等映射版本
    """
    def __init__(self, dim, num_heads=4, window_size=7, mlp_ratio=2.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.adapt_conv = None
        print(f"SwinTransformerHeadSimple初始化: dim={dim} (使用恒等映射)")

    def forward(self, x):
        # 确保输入是浮点型
        x = x.float()
        
        # 通道自适应
        if x.shape[1] != self.dim:
            if self.adapt_conv is None:
                self.adapt_conv = nn.Conv2d(x.shape[1], self.dim, 1).to(x.device)
                self.adapt_conv.weight.data = self.adapt_conv.weight.data.float()
                if self.adapt_conv.bias is not None:
                    self.adapt_conv.bias.data = self.adapt_conv.bias.data.float()
            x = self.adapt_conv(x)
        
        return x  # 恒等映射

class SwinTransformerHeadIdentity(nn.Module):
    """
    恒等映射版本 - 不做任何变换
    """
    def __init__(self, dim, num_heads=4, window_size=7, mlp_ratio=2.):
        super().__init__()
        self.dim = dim
        print(f"SwinTransformerHeadIdentity初始化: dim={dim}")
        self.adapt_conv = None

    def forward(self, x):
        # 确保输入是float32
        if x.dtype == torch.float16:
            x = x.float()
        
        # 通道自适应
        if x.shape[1] != self.dim:
            if self.adapt_conv is None:
                self.adapt_conv = nn.Conv2d(x.shape[1], self.dim, 1).to(x.device)
                self.adapt_conv.weight.data = self.adapt_conv.weight.data.float()
                if self.adapt_conv.bias is not None:
                    self.adapt_conv.bias.data = self.adapt_conv.bias.data.float()
            x = self.adapt_conv(x)
        
        return x  # 恒等映射
