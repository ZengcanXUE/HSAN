import torch
import torch.nn as nn
import torch.nn.functional as F

class AFCA(nn.Module):
    """
    Adaptive Fine-Grained Channel Attention (AFCA) - 优化稳定版
    
    特点：
    1. 联合建模全局与局部通道依赖关系
    2. 自适应融合因子动态平衡通道权重
    3. 适合小目标、低对比目标检测
    4. 输入输出维度一致 [B, C, H, W] -> [B, C, H, W]
    """
    def __init__(self, channels, k=3, *args, **kwargs):
        super().__init__()
        self.channels = channels
        self.k = k
        
        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # 局部信息提取的卷积（1D卷积实现band matrix效果）
        # 使用padding保证维度不变
        self.local_conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        
        # 可学习的自适应融合因子
        self.theta = nn.Parameter(torch.tensor(0.5))
        
        # 全局通道依赖建模（使用1x1卷积代替全连接，更稳定）
        self.global_conv = nn.Conv2d(channels, channels, 1, bias=False)
        
        # 用于生成最终权重的sigmoid
        self.sigmoid = nn.Sigmoid()
        
        print(f"AFCA初始化: channels={channels}, k={k}")

    def forward(self, x):
        """
        x: 输入特征图 [B, C, H, W]
        return: 注意力增强后的特征图 [B, C, H, W]
        """
        batch_size, channels, height, width = x.shape
        
        # 如果输入通道数与初始化不匹配，自适应调整（防止scale缩放导致的问题）
        if channels != self.channels:
            if not hasattr(self, 'channel_adapt'):
                self.channel_adapt = nn.Conv2d(channels, self.channels, 1).to(x.device)
                print(f"AFCA通道自适应: {channels}->{self.channels}")
            x = self.channel_adapt(x)
            batch_size, channels, height, width = x.shape
        
        # Step 1: 全局平均池化获取通道描述符 U ∈ [B, C]
        U = self.gap(x).view(batch_size, channels)  # [B, C]
        
        # Step 2: 获取局部通道信息 U_lc（1D卷积捕获局部依赖）
        U_reshaped = U.unsqueeze(1)  # [B, 1, C]
        U_lc = self.local_conv(U_reshaped).squeeze(1)  # [B, C]
        
        # Step 3: 获取全局通道信息 U_gc
        U_gc = self.global_conv(x).mean(dim=[2, 3])  # [B, C]
        
        # Step 4: 计算相关性矩阵 M = U_gc · U_lc^T
        # M ∈ [B, C, C]
        M = torch.bmm(U_gc.unsqueeze(2), U_lc.unsqueeze(1))  # [B, C, 1] x [B, 1, C] = [B, C, C]
        
        # Step 5: 从相关性矩阵中提取权重
        U_gc_w = M.sum(dim=2)  # [B, C] 全局权重
        U_lc_w = M.transpose(1, 2).sum(dim=2)  # [B, C] 局部权重
        
        # Step 6: 自适应融合
        theta_sig = self.sigmoid(self.theta)
        U_gc_w_sig = self.sigmoid(U_gc_w)
        U_lc_w_sig = self.sigmoid(U_lc_w)
        
        W = self.sigmoid(theta_sig * U_gc_w_sig + (1 - theta_sig) * U_lc_w_sig)
        
        # Step 7: 应用权重
        W = W.view(batch_size, channels, 1, 1)
        output = x * W
        
        return output


# 轻量版AFCA - 参数量更少，速度更快
class AFCALight(nn.Module):
    """
    轻量版AFCA - 计算效率更高，适合嵌入式设备
    """
    def __init__(self, channels, *args, **kwargs):
        super().__init__()
        self.channels = channels
        
        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # 简化版：使用1x1卷积同时建模全局和局部
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
        print(f"AFCALight初始化: channels={channels}")

    def forward(self, x):
        # 检查输入通道
        if x.shape[1] != self.channels:
            if not hasattr(self, 'channel_adapt'):
                self.channel_adapt = nn.Conv2d(x.shape[1], self.channels, 1).to(x.device)
                print(f"AFCALight通道自适应: {x.shape[1]}->{self.channels}")
            x = self.channel_adapt(x)
        
        # 通道注意力
        att = self.fc(self.gap(x))
        return x * att