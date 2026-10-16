import torch
import torch.nn as nn
import torch.nn.functional as F

class LSKA(nn.Module):
    """Large Separable Kernel Attention"""
    def __init__(self, dim, k_size=7):
        super().__init__()
        
        # 确保k_size是奇数
        if k_size % 2 == 0:
            k_size = k_size + 1
            print(f"调整k_size为奇数: {k_size}")
        
        # 分解为水平和垂直1D卷积
        self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, k_size), padding=(0, k_size//2), groups=dim)
        self.conv0v = nn.Conv2d(dim, dim, kernel_size=(k_size, 1), padding=(k_size//2, 0), groups=dim)
        
        # 第二级卷积
        k_size2 = 2*k_size - 1
        self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, k_size2), padding=(0, k_size2//2), groups=dim)
        self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(k_size2, 1), padding=(k_size2//2, 0), groups=dim)
        
        # 注意力生成
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        u = x.clone()
        
        # 第一级分解卷积
        attn = self.conv0h(x)
        attn = self.conv0v(attn)
        
        # 第二级分解卷积
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)
        
        # 注意力生成
        attn = self.conv1(attn)
        attn = self.activation(attn)
        attn = self.conv2(attn)
        attn = self.sigmoid(attn)
        
        # 确保注意力权重与输入维度匹配
        if attn.shape != u.shape:
            attn = F.interpolate(attn, size=(u.shape[2], u.shape[3]), mode='bilinear', align_corners=False)
        
        return u * attn


class LSKA_Attention(nn.Module):
    """LSKA注意力模块（含残差连接）- 确保输出通道与输入相同"""
    def __init__(self, d_model, k_size=7, *args, **kwargs):
        super().__init__()
        print(f"LSKA_Attention初始化: d_model={d_model}, k_size={k_size}")
        
        # 确保所有卷积都保持相同的通道数
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating = LSKA(d_model, k_size)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)
        
        # 添加一个shortcut连接，确保梯度流动
        self.shortcut = nn.Identity()

    def forward(self, x):
        identity = x  # 保存输入
        
        # 主路径
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating(x)
        x = self.proj_2(x)
        
        # 残差连接
        out = x + identity
        
        return out