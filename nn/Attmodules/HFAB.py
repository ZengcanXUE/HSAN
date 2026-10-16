import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """
    空间注意力模块 - 用于HFAB的第一阶段
    引导模型关注目标区域
    """
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.channels = channels
        
        # 通道压缩
        self.conv1 = nn.Conv2d(channels, channels // reduction_ratio, 1)
        self.bn1 = nn.BatchNorm2d(channels // reduction_ratio)
        self.conv2 = nn.Conv2d(channels // reduction_ratio, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # 生成空间注意力图
        attn = self.conv1(x)
        attn = self.bn1(attn)
        attn = self.relu(attn)
        attn = self.conv2(attn)
        attn = self.sigmoid(attn)
        
        return x * attn


class HighFrequencyEnhancement(nn.Module):
    """
    高频增强模块 - 用于HFAB的第二阶段
    通过高通滤波增强边缘和细节
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # 拉普拉斯高通滤波器 (固定参数，不训练)
        self.laplacian = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False, groups=channels)
        
        # 初始化拉普拉斯核 (depthwise conv)
        laplacian_kernel = torch.tensor([[[[-1, -1, -1],
                                            [-1,  8, -1],
                                            [-1, -1, -1]]]], dtype=torch.float32)
        # 扩展到所有通道
        laplacian_kernel = laplacian_kernel.repeat(channels, 1, 1, 1)
        self.laplacian.weight.data = laplacian_kernel
        self.laplacian.weight.requires_grad = False  # 固定参数，不训练
        
        # 可学习的融合权重
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # 1x1卷积调整
        self.conv = nn.Conv2d(channels * 2, channels, 1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # 提取高频分量
        high_freq = self.laplacian(x)
        
        # 自适应融合
        enhanced = x + self.gamma * high_freq
        
        # 拼接并调整
        out = torch.cat([x, enhanced], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        
        return out


class HFAB(nn.Module):
    """
    High-Frequency Attention Block (HFAB)
    两阶段结构：
    1. 空间注意力 (Spatial-wise Attention) - 定位目标区域
    2. 高频增强 (High-frequency Enhancement) - 增强边缘细节
    """
    def __init__(self, channels, reduction_ratio=16, *args, **kwargs):
        super().__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        
        # 阶段1：空间注意力
        self.spatial_att = SpatialAttention(channels, reduction_ratio)
        
        # 阶段2：高频增强
        self.hf_enhance = HighFrequencyEnhancement(channels)
        
        # 残差连接
        self.residual = nn.Identity()
        
        print(f"HFAB初始化: channels={channels}, reduction={reduction_ratio}")

    def forward(self, x):
        identity = x  # 保存输入用于残差连接
        
        # 阶段1：空间注意力
        x = self.spatial_att(x)
        
        # 阶段2：高频增强
        x = self.hf_enhance(x)
        
        # 残差连接（保持输入输出维度一致）
        return x + identity


# 轻量版HFAB - 减少参数量，适合嵌入式设备
class HFABLight(nn.Module):
    """
    轻量版HFAB - 参数量更少，速度更快
    """
    def __init__(self, channels, *args, **kwargs):
        super().__init__()
        self.channels = channels
        
        # 简化版空间注意力 (直接用1x1卷积生成)
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # 简化版高频增强 (只用拉普拉斯)
        self.laplacian = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False, groups=channels)
        laplacian_kernel = torch.tensor([[[[-1, -1, -1],
                                            [-1,  8, -1],
                                            [-1, -1, -1]]]], dtype=torch.float32)
        laplacian_kernel = laplacian_kernel.repeat(channels, 1, 1, 1)
        self.laplacian.weight.data = laplacian_kernel
        self.laplacian.weight.requires_grad = False
        
        # 可学习融合参数
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
        print(f"HFABLight初始化: channels={channels}")

    def forward(self, x):
        # 空间注意力
        attn = self.spatial_att(x)
        x_att = x * attn
        
        # 高频增强
        high_freq = self.laplacian(x_att)
        
        # 融合
        out = x_att + self.alpha * high_freq
        
        return out + x  # 残差连接