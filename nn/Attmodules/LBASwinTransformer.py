import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init  
from einops import rearrange

# ==================== 辅助函数 ====================
def to_2tuple(x):
    return (x, x)

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    
    # 确保 H 和 W 能被 window_size 整除
    if H % window_size != 0 or W % window_size != 0:
        # 如果不能整除，进行padding
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        B, H, W, C = x.shape
    
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W, original_H=None, original_W=None):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image (可能包含padding)
        W (int): Width of image (可能包含padding)
        original_H (int): Original height without padding
        original_W (int): Original width without padding

    Returns:
        x: (B, H, W, C)
    """
    if original_H is None:
        original_H = H
    if original_W is None:
        original_W = W
        
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    
    # 如果之前有padding，去掉padding
    if H != original_H or W != original_W:
        x = x[:, :original_H, :original_W, :]
    
    return x

class Mlp(nn.Module):
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

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

# ==================== 注意力模块 ====================
class FocusedLinearAttention(nn.Module):
    """
    Focused Linear Attention from SLBDetection-Net
    保留初始化信息，去掉运行时多余检查
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, 
                 attn_drop=0., proj_drop=0., focusing_factor=3, kernel_size=5):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.focusing_factor = focusing_factor
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        head_dim = dim // num_heads
        self.dwc = nn.Conv2d(head_dim, head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(1, 1, dim))
        
        # 位置编码（在forward中动态创建）
        self.positional_encoding = None
        
        # PReLU激活函数
        self.kernel_function = nn.PReLU()
        
        print(f'FocusedLinearAttention: window={window_size}, f={focusing_factor}, k={kernel_size}')

    def forward(self, x, mask=None):
        B, N, C = x.shape
        
        # 动态创建位置编码
        if self.positional_encoding is None or self.positional_encoding.shape[1] != N:
            self.positional_encoding = nn.Parameter(
                torch.zeros(1, N, C, device=x.device, dtype=x.dtype)
            )
            nn.init.trunc_normal_(self.positional_encoding, std=.02)

        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv.unbind(0)

        k = k + self.positional_encoding

        q = self.kernel_function(q) + 1e-6
        k = self.kernel_function(k) + 1e-6

        scale = F.softplus(self.scale)
        q = q / scale
        k = k / scale

        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)

        # Focused机制
        f = self.focusing_factor
        if f <= 6:
            q = q ** f
            k = k ** f
        else:
            q = (q / q.max(dim=-1, keepdim=True)[0]) ** f
            k = (k / k.max(dim=-1, keepdim=True)[0]) ** f

        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm

        # 重塑为多头 - 使用 reshape 替代 view
        q = rearrange(q, "b n (h c) -> (b h) n c", h=self.num_heads)
        k = rearrange(k, "b n (h c) -> (b h) n c", h=self.num_heads)
        v = rearrange(v, "b n (h c) -> (b h) n c", h=self.num_heads)

        # 线性注意力计算
        z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
        
        # 根据计算复杂度选择不同的计算路径
        if q.shape[1] * k.shape[1] * (k.shape[2] + v.shape[2]) > k.shape[2] * v.shape[2] * (q.shape[1] + k.shape[1]):
            kv = torch.einsum("b j c, b j d -> b c d", k, v)
            out = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
        else:
            qk = torch.einsum("b i c, b j c -> b i j", q, k)
            out = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)

        # Depthwise卷积增强
        num = int(v.shape[1] ** 0.5)
        feat_map = rearrange(v, "b (w h) c -> b c w h", w=num, h=num)
        feat_map = rearrange(self.dwc(feat_map), "b c w h -> b (w h) c")
        out = out + feat_map

        # 重塑回原始形状 - 使用 reshape 替代 view
        out = rearrange(out, "(b h) n c -> b n (h c)", h=self.num_heads)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out

class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module."""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if mask is not None:
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


# ==================== LBASwinTransformerBlock ====================
class LBASwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 focusing_factor=3, kernel_size=5, attn_type='L', *args, **kwargs):
        super().__init__()
        
        # 确保 num_heads 能整除 dim
        if dim % num_heads != 0:
            # 调整 num_heads 使其能整除 dim
            new_num_heads = dim
            while new_num_heads > 1 and dim % new_num_heads != 0:
                new_num_heads -= 1
            print(f"⚠️ 调整 num_heads: {num_heads} -> {new_num_heads} (确保整除 dim={dim})")
            num_heads = new_num_heads
        
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = norm_layer(dim)
        assert attn_type in ['L', 'S']
        
        if attn_type == 'L':
            self.attn = FocusedLinearAttention(
                dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                focusing_factor=focusing_factor, kernel_size=kernel_size)
        else:
            self.attn = WindowAttention(
                dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        # 注意力掩码
        self.attn_mask = None
        
        print(f"LBASwinTransformerBlock初始化: dim={dim}, num_heads={num_heads}, window={window_size}")

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.dim, f"输入通道数 {C} 应与初始化维度 {self.dim} 匹配"
        
        # 保存原始尺寸
        original_H, original_W = H, W
        
        # 调整窗口大小（如果输入太小）
        window_size = min(self.window_size, H, W)
        shift_size = self.shift_size if window_size == self.window_size else 0
        
        # 重塑为序列形式 [B, H*W, C] - 使用 reshape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        shortcut = x
        x = self.norm1(x)
        
        # 重塑回空间形式用于窗口划分 - 使用 reshape
        x = x.reshape(B, H, W, C)
        
        # cyclic shift
        if shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.reshape(-1, window_size * window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.reshape(-1, window_size, window_size, C)
        
        # 获取padding后的尺寸
        padded_H = ((original_H + window_size - 1) // window_size) * window_size
        padded_W = ((original_W + window_size - 1) // window_size) * window_size
        shifted_x = window_reverse(attn_windows, window_size, padded_H, padded_W, original_H, original_W)

        # reverse cyclic shift
        if shift_size > 0:
            x = torch.roll(shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
        else:
            x = shifted_x
            
        x = x.reshape(B, original_H * original_W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # 重塑回4D张量 [B, C, H, W] - 使用 reshape
        x = x.transpose(1, 2).reshape(B, C, original_H, original_W)
        
        return x

# ==================== SPDConv (可选) ====================
class SPDConv(nn.Module):
    """Space-to-Depth Convolution for downsampling without losing information"""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        # 输入通道放大4倍（因为space-to-depth）
        c1_in = c1 * 4
        self.conv = nn.Conv2d(c1_in, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        # Space-to-Depth: 将空间信息折叠到通道维度
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], 
                       x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        return self.act(self.bn(self.conv(x)))


def autopad(k, p=None, d=1):
    """自动计算padding"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p