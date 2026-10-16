import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialExpansion(nn.Module):
    """
    空间扩展模块 - 在降采样前增强空间细节
    通过不同膨胀率的卷积捕获多尺度上下文
    """
    def __init__(self, channels, dilation_rates=[1, 2, 3], *args, **kwargs):
        super().__init__()
        self.channels = channels
        self.dilation_rates = dilation_rates
        
        # 多分支膨胀卷积
        self.convs = nn.ModuleList()
        for rate in dilation_rates:
            self.convs.append(
                nn.Conv2d(channels, channels, kernel_size=3, padding=rate, 
                         dilation=rate, groups=channels, bias=False)
            )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * len(dilation_rates), channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
        print(f"SpatialExpansion初始化: channels={channels}, dilation_rates={dilation_rates}")

    def forward(self, x):
        identity = x
        
        # 多分支膨胀卷积
        branches = []
        for conv in self.convs:
            branches.append(conv(x))
        
        # 拼接并融合
        fused = torch.cat(branches, dim=1)
        fused = self.fusion(fused)
        
        # 门控选择
        gate = self.gate(fused)
        
        return identity + fused * gate


class ChannelExpansion(nn.Module):
    """
    通道扩展模块 - 在通道维度上进行注意力增强
    捕获通道间的依赖关系
    """
    def __init__(self, channels, reduction=16, *args, **kwargs):
        super().__init__()
        self.channels = channels
        
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # 局部通道交互（1D卷积）
        self.local_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        
        print(f"ChannelExpansion初始化: channels={channels}, reduction={reduction}")

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 全局通道注意力
        att_global = self.channel_att(x)
        
        # 局部通道交互
        x_pool = x.mean(dim=[2, 3])  # [B, C]
        x_pool = x_pool.unsqueeze(1)  # [B, 1, C]
        att_local = self.local_conv(x_pool).squeeze(1)  # [B, C]
        att_local = att_local.view(B, C, 1, 1)
        att_local = torch.sigmoid(att_local)
        
        # 融合两种注意力
        att = (att_global + att_local) / 2.0
        
        return x * att


class ExpansionAttention(nn.Module):
    """
    Expansion Attention - 专门为小目标设计的注意力机制
    
    特点：
    1. 空间扩展：多尺度膨胀卷积捕获上下文
    2. 通道扩展：全局+局部通道注意力
    3. 适合放在降采样层前，保护小目标细节
    """
    def __init__(self, channels, dilation_rates=[1, 2, 3], reduction=16, *args, **kwargs):
        super().__init__()
        self.channels = channels
        
        # 空间扩展分支
        self.spatial_exp = SpatialExpansion(channels, dilation_rates)
        
        # 通道扩展分支
        self.channel_exp = ChannelExpansion(channels, reduction)
        
        # 自适应融合
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        print(f"ExpansionAttention初始化: channels={channels}, dilation_rates={dilation_rates}")

    def forward(self, x):
        # 两个分支
        spatial_out = self.spatial_exp(x)
        channel_out = self.channel_exp(x)
        
        # 拼接融合
        out = torch.cat([spatial_out, channel_out], dim=1)
        out = self.fusion(out)
        
        return out


# 轻量版Expansion Attention - 只保留关键组件
class ExpansionAttentionLight(nn.Module):
    """
    轻量版Expansion Attention - 更稳定的版本
    """
    def __init__(self, channels, *args, **kwargs):
        super().__init__()
        self.channels = channels
        
        # 先创建通道调整层（可选）
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
        
        # 简化的空间增强
        self.spatial_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        print(f"ExpansionAttentionLight初始化: channels={channels}")

    def forward(self, x):
        identity = x
        
        # 通道注意力
        att = self.channel_att(x)
        x = x * att
        
        # 空间增强
        x = self.spatial_enhance(x)
        
        return identity + x


# 降采样前保护模块 - 专门放在Conv层前
class DownsampleProtection(nn.Module):
    """
    降采样保护模块 - 在步长为2的卷积前使用
    """
    def __init__(self, channels, *args, **kwargs):
        super().__init__()
        self.channels = channels
        self.adaptive = False
        
        print(f"DownsampleProtection初始化: channels={channels}")

    def forward(self, x):
        # 如果输入通道数不匹配，动态创建适配层
        if x.shape[1] != self.channels:
            if not hasattr(self, 'adapt_conv'):
                self.adapt_conv = nn.Conv2d(x.shape[1], self.channels, 1).to(x.device)
                print(f"DownsampleProtection自适应: {x.shape[1]}->{self.channels}")
            x = self.adapt_conv(x)
        
        # 多尺度特征提取（动态创建）
        if not hasattr(self, 'conv1'):
            self.conv1 = nn.Conv2d(self.channels, self.channels, 3, padding=1, groups=self.channels).to(x.device)
            self.conv2 = nn.Conv2d(self.channels, self.channels, 3, padding=2, dilation=2, groups=self.channels).to(x.device)
            self.conv3 = nn.Conv2d(self.channels, self.channels, 3, padding=3, dilation=3, groups=self.channels).to(x.device)
            self.fusion = nn.Sequential(
                nn.Conv2d(self.channels * 3, self.channels, 1),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(inplace=True)
            ).to(x.device)
        
        # 多尺度特征
        f1 = self.conv1(x)
        f2 = self.conv2(x)
        f3 = self.conv3(x)
        
        # 融合
        out = torch.cat([f1, f2, f3], dim=1)
        out = self.fusion(out)
        
        return x + out