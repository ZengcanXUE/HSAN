import torch
import torch.nn as nn
import torch.nn.functional as F


class MHSA(nn.Module):
    """
    Multi-Head Self-Attention (MHSA) - 稳定版
    特别适合区分相似行为（如不同姿势的举手、阅读等）
    
    特点：
    1. 使用1x1卷积代替全连接，保持空间结构
    2. 支持任意大小的特征图
    3. 输入输出维度一致 [B, C, H, W] -> [B, C, H, W]
    """
    def __init__(self, channels, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., *args, **kwargs):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert channels % num_heads == 0, f"channels {channels} 必须能被 num_heads {num_heads} 整除"
        
        # 使用1x1卷积生成Q、K、V
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=qkv_bias)
        
        # 注意力dropout
        self.attn_drop = nn.Dropout(attn_drop)
        
        # 输出投影
        self.proj = nn.Conv2d(channels, channels, 1)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 相对位置偏置（可选）
        self.relative_bias = nn.Parameter(torch.zeros(2 * 7 - 1, 2 * 7 - 1))  # 7x7窗口
        
        print(f"MHSA初始化: channels={channels}, num_heads={num_heads}")

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 生成Q、K、V [B, 3*C, H, W]
        qkv = self.qkv(x)
        
        # 重塑为多头格式 [B, 3, num_heads, head_dim, H*W]
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, B, num_heads, H*W, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # 每个都是 [B, num_heads, H*W, head_dim]
        
        # 计算注意力分数 [B, num_heads, H*W, H*W]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # 添加相对位置偏置（简化版）
        # 这里使用一个简单的窗口偏置
        pos_bias = self.relative_bias[:H, :W].reshape(-1)
        attn = attn + pos_bias[None, None, :, None]
        
        # softmax和dropout
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 应用注意力 [B, num_heads, H*W, head_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        
        # 重塑回 [B, C, H, W]
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        
        # 输出投影
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MHSABlock(nn.Module):
    """
    MHSA Block - 包含MHSA和前馈网络的完整模块
    带残差连接和LayerNorm
    """
    def __init__(self, channels, num_heads=8, mlp_ratio=4., qkv_bias=False, 
                 attn_drop=0., proj_drop=0., *args, **kwargs):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        # LayerNorm（使用GroupNorm模拟，因为BN在MHSA中效果不好）
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        
        # MHSA
        self.attn = MHSA(channels, num_heads, qkv_bias, attn_drop, proj_drop)
        
        # 前馈网络
        mlp_hidden_dim = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mlp_hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(mlp_hidden_dim, channels, 1),
            nn.Dropout(proj_drop)
        )
        
        print(f"MHSABlock初始化: channels={channels}, num_heads={num_heads}")

    def forward(self, x):
        # MHSA + 残差
        x = x + self.attn(self.norm1(x))
        
        # MLP + 残差
        x = x + self.mlp(self.norm2(x))
        
        return x


# 简化版MHSA - 只使用通道注意力，更轻量
class MHSALight(nn.Module):
    """
    轻量版MHSA - 只使用通道间的自注意力
    计算量更小，适合嵌入式设备
    """
    def __init__(self, channels, num_heads=4, *args, **kwargs):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert channels % num_heads == 0, f"channels {channels} 必须能被 num_heads {num_heads} 整除"
        
        # 全局平均池化后做自注意力
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels)
        
        print(f"MHSALight初始化: channels={channels}, num_heads={num_heads}")

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 全局特征 [B, C]
        x_pool = self.gap(x).view(B, C)
        
        # Q、K、V [B, C]
        qkv = self.qkv(x_pool).reshape(B, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(1, 0, 2, 3)  # [3, B, num_heads, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 注意力 [B, num_heads, head_dim, head_dim]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # 应用注意力
        x_out = (attn @ v).transpose(1, 2).reshape(B, C)
        
        # 投影
        x_out = self.proj(x_out)
        
        # 扩展回空间维度并残差连接
        return x + x_out.view(B, C, 1, 1)


# 窗口版MHSA - 在局部窗口内计算自注意力，效率更高
class WindowMHSA(nn.Module):
    """
    窗口版MHSA - 在7x7窗口内计算自注意力
    计算量 O(HW*49) vs O(H^2*W^2)
    """
    def __init__(self, channels, num_heads=8, window_size=7, qkv_bias=False, *args, **kwargs):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert channels % num_heads == 0, f"channels {channels} 必须能被 num_heads {num_heads} 整除"
        
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=qkv_bias)
        self.proj = nn.Conv2d(channels, channels, 1)
        
        # 相对位置偏置
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        
        # 生成相对位置索引
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # [2, window_size, window_size]
        coords_flatten = torch.flatten(coords, 1)  # [2, window_size^2]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, window_size^2, window_size^2]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [window_size^2, window_size^2, 2]
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # [window_size^2, window_size^2]
        self.register_buffer("relative_position_index", relative_position_index)
        
        print(f"WindowMHSA初始化: channels={channels}, num_heads={num_heads}, window_size={window_size}")

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.window_size == 0 and W % self.window_size == 0, \
            f"特征图大小 ({H}x{W}) 必须能被窗口大小 {self.window_size} 整除"
        
        # 生成QKV [B, 3*C, H, W]
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H, W)
        qkv = qkv.permute(1, 0, 2, 4, 5, 3)  # [3, B, num_heads, H, W, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 分窗口
        q_windows = q.reshape(B, self.num_heads, H // self.window_size, self.window_size,
                              W // self.window_size, self.window_size, self.head_dim)
        q_windows = q_windows.permute(0, 1, 2, 4, 3, 5, 6).reshape(
            B, self.num_heads, -1, self.window_size * self.window_size, self.head_dim)
        
        k_windows = k.reshape(B, self.num_heads, H // self.window_size, self.window_size,
                              W // self.window_size, self.window_size, self.head_dim)
        k_windows = k_windows.permute(0, 1, 2, 4, 3, 5, 6).reshape(
            B, self.num_heads, -1, self.window_size * self.window_size, self.head_dim)
        
        v_windows = v.reshape(B, self.num_heads, H // self.window_size, self.window_size,
                              W // self.window_size, self.window_size, self.head_dim)
        v_windows = v_windows.permute(0, 1, 2, 4, 3, 5, 6).reshape(
            B, self.num_heads, -1, self.window_size * self.window_size, self.head_dim)
        
        # 计算注意力
        attn = (q_windows @ k_windows.transpose(-2, -1)) * self.scale
        
        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [num_heads, window_size^2, window_size^2]
        attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = attn.softmax(dim=-1)
        
        # 应用注意力
        x_windows = (attn @ v_windows).transpose(-2, -1).reshape(
            B, self.num_heads, H // self.window_size, W // self.window_size,
            self.window_size, self.window_size, self.head_dim)
        x_windows = x_windows.permute(0, 1, 2, 4, 3, 5, 6).reshape(B, C, H, W)
        
        # 输出投影
        x = self.proj(x_windows)
        
        return x