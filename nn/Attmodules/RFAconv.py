import torch
import torch.nn as nn
from einops import rearrange
from ..modules.conv import Conv

__all__ = ['RFAConv', 'C3k2_RFAConv', 'BottleneckRFAConv']

class RFAConv(nn.Module):
    """Receptive Field Attention Convolution"""
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        self.get_weight = nn.Sequential(
            nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=1, groups=in_channel, bias=False)
        )
        self.generate_feature = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=kernel_size,
                     padding=kernel_size//2, stride=stride, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU()
        )
        self.conv = Conv(in_channel, out_channel, k=kernel_size, s=kernel_size, p=0)

    def forward(self, x):
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h, w)
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', 
                            n1=self.kernel_size, n2=self.kernel_size)
        return self.conv(conv_data)


class BottleneckRFAConv(nn.Module):
    """Bottleneck with RFAConv"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = RFAConv(c_, c2, k[1], 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k2_RFAConv(nn.Module):
    """C3k2 with RFAConv module - 带通道自适应"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        # 注意：c1 是从配置传入的预期输入通道，但实际可能不匹配
        self.expected_c1 = c1
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            BottleneckRFAConv(self.c, self.c, shortcut, g) for _ in range(n)
        )
        # 自适应卷积，用于处理输入通道不匹配的情况
        self.adapt_conv = None

    def forward(self, x):
        # 检查输入通道是否匹配
        if x.shape[1] != self.expected_c1:
            if self.adapt_conv is None:
                # 动态创建自适应卷积
                self.adapt_conv = Conv(x.shape[1], self.expected_c1, 1, 1).to(x.device)
                print(f"⚠️ C3k2_RFAConv 通道自适应: {x.shape[1]} -> {self.expected_c1}")
            x = self.adapt_conv(x)
        
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))