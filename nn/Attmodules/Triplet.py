import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletAttentionSimple(nn.Module):
    """
    简化版Triplet Attention - 超稳定版本
    避免复杂的维度变换，使用分离的注意力分支
    """
    def __init__(self, channels, reduction_ratio=16, *args, **kwargs):
        super().__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        
        # 通道注意力 (C方向) - 标准实现
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 空间注意力 - 简化版，避免维度变换问题
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1),
            nn.BatchNorm2d(channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, 1, 1),
            nn.Sigmoid()
        )
        
        print(f"TripletAttentionSimple初始化: channels={channels}")

    def forward(self, x):
        # 通道注意力
        att_c = self.channel_attention(x)
        
        # 空间注意力
        att_s = self.spatial_attention(x)
        
        # 融合两种注意力
        att = (att_c + att_s) / 2.0
        
        return x * att


# 或者用这个更简单的版本（推荐先测试这个）
class TripletAttentionMinimal(nn.Module):
    """
    极简版Triplet Attention - 只使用通道注意力和简单空间注意力
    完全避免维度变换问题
    """
    def __init__(self, channels, reduction_ratio=16, *args, **kwargs):
        super().__init__()
        self.channels = channels
        
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力 - 使用1x1卷积直接生成
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )
        
        print(f"TripletAttentionMinimal初始化: channels={channels}")

    def forward(self, x):
        # 通道注意力
        att_c = self.channel_att(x)
        
        # 空间注意力
        att_s = self.spatial_att(x)
        
        # 融合
        return x * att_c * att_s