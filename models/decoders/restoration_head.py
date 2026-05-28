import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """把 backbone 的二维特征图投影到统一的 embedding 维度。"""
    def __init__(self, input_dim=2048, embed_dim=768):
        super(MLP, self).__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        # 输入:  [B, C, H, W]
        # 输出:  [B, H*W, embed_dim]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class RestorationHead(nn.Module):
    """RGB-Polar 图像恢复头。

    前因：
        原来的 MLPDecoder 来自语义分割任务，最后输出的是类别 logits。
        当前任务需要的是 3 通道恢复图，因此这里保留多尺度特征融合思想，
        但把最后的 prediction head 改成图像重建头。

    后果：
        输入仍然是 backbone + CM-FRM/FFM 得到的 4 层融合特征；
        输出变为 [B, 3, H/4, W/4] 的 RGB 图像预测，再由 builder 上采样回原图尺寸。
    """
    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 out_channels=3,
                 norm_layer=nn.BatchNorm2d,
                 embed_dim=512,
                 align_corners=False):
        super(RestorationHead, self).__init__()
        self.out_channels = out_channels
        self.align_corners = align_corners
        self.in_channels = in_channels

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        # 四个尺度先分别投影到同一通道数，便于在空间尺寸对齐后 concat。
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embed_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embed_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embed_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embed_dim)

        # 多尺度融合后的重建头。
        # 使用 3x3 conv 比单个 1x1 分类层更适合恢复局部纹理和颜色细节。
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1),
            norm_layer(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            norm_layer(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.pred = nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, inputs):
        # inputs 是 backbone 输出的四层融合特征：
        #   c1: 1/4 resolution
        #   c2: 1/8 resolution
        #   c3: 1/16 resolution
        #   c4: 1/32 resolution
        c1, c2, c3, c4 = inputs
        n = c4.shape[0]

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(
            n, -1, c4.shape[2], c4.shape[3]
        )
        _c4 = F.interpolate(
            _c4, size=c1.size()[2:], mode='bilinear',
            align_corners=self.align_corners
        )

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(
            n, -1, c3.shape[2], c3.shape[3]
        )
        _c3 = F.interpolate(
            _c3, size=c1.size()[2:], mode='bilinear',
            align_corners=self.align_corners
        )

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(
            n, -1, c2.shape[2], c2.shape[3]
        )
        _c2 = F.interpolate(
            _c2, size=c1.size()[2:], mode='bilinear',
            align_corners=self.align_corners
        )

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(
            n, -1, c1.shape[2], c1.shape[3]
        )

        fused = self.fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        restored = self.pred(fused)

        # clean_target 在 dataloader 中已经归一到 [0, 1]。
        # 这里用 sigmoid 约束预测图范围，使 L1Loss 的数值含义直接对应像素差。
        restored = torch.sigmoid(restored)
        return restored
