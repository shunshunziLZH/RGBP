import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGELU(nn.Module):
    """基础卷积单元：Conv 3x3 + Norm + GELU。"""
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(ConvGELU, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)


class RestoreBlock(nn.Module):
    """轻量恢复残差块。

    update.md 的要求是第一版不要引入复杂 attention，先用：
        Conv 3x3 -> GELU/ReLU -> Conv 3x3 + residual

    这里用 GELU，并在输入通道和输出通道不一致时用 1x1 Conv 做 shortcut 对齐。
    这样它既能处理单尺度特征，也能处理 concat 后的跨尺度融合特征。
    """
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(RestoreBlock, self).__init__()
        self.conv1 = ConvGELU(in_channels, out_channels, norm_layer)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels)
        )
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.act(out + residual)
        return out


class RestorationHead(nn.Module):
    """U-Net 风格 RGB-Polar 图像恢复 decoder。

    前因：
        旧的 MLPDecoder / 当前早期 RestorationHead 更接近“分类解码器”：
        把四层特征统一投影后一次性 concat，再输出预测。
        update.md 明确要求 decoder 不要继续做像素分类，而要做多尺度特征重建。

    当前结构：
        1. 通道对齐：F1/F2/F3/F4 分别用 1x1 Conv 对齐到 decoder_dim；
        2. 自顶向下融合：G4 -> G3 -> G2 -> G1 逐级上采样、concat、RestoreBlock；
        3. 原图恢复：D1 上采样到 RGB 输入大小；
        4. 残差输出：预测 residual，加回输入 RGB，再 clamp 到 [0, 1]。

    注意：
        dataloader 送进模型的 image_rgb 已经做过 ImageNet mean/std 标准化。
        clean_target 是 [0, 1]。因此 residual learning 前必须先把 rgb 反归一化回 [0, 1]，
        否则 RGB_input + residual 的数值空间会错。
    """
    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 out_channels=3,
                 norm_layer=nn.BatchNorm2d,
                 embed_dim=96,
                 align_corners=False,
                 rgb_mean=None,
                 rgb_std=None):
        super(RestorationHead, self).__init__()
        self.out_channels = out_channels
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.decoder_dim = embed_dim

        if rgb_mean is None:
            rgb_mean = [0.485, 0.456, 0.406]
        if rgb_std is None:
            rgb_std = [0.229, 0.224, 0.225]

        self.register_buffer(
            'rgb_mean',
            torch.tensor(rgb_mean, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'rgb_std',
            torch.tensor(rgb_std, dtype=torch.float32).view(1, 3, 1, 1)
        )

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        # 第一部分：通道对齐。
        # 不直接拼接 [64, 128, 320, 512]，先统一到 decoder_dim，降低计算量并稳定融合。
        self.align_c1 = nn.Conv2d(c1_in_channels, embed_dim, kernel_size=1)
        self.align_c2 = nn.Conv2d(c2_in_channels, embed_dim, kernel_size=1)
        self.align_c3 = nn.Conv2d(c3_in_channels, embed_dim, kernel_size=1)
        self.align_c4 = nn.Conv2d(c4_in_channels, embed_dim, kernel_size=1)

        # 第二部分：自顶向下逐级上采样融合。
        self.restore4 = RestoreBlock(embed_dim, embed_dim, norm_layer)
        self.restore3 = RestoreBlock(embed_dim * 2, embed_dim, norm_layer)
        self.restore2 = RestoreBlock(embed_dim * 2, embed_dim, norm_layer)
        self.restore1 = RestoreBlock(embed_dim * 2, embed_dim, norm_layer)

        # 第三部分：恢复到原图分辨率后预测 RGB residual。
        self.final = nn.Sequential(
            ConvGELU(embed_dim, embed_dim, norm_layer),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1)
        )

    def _upsample_like(self, x, target):
        return F.interpolate(
            x,
            size=target.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners
        )

    def _denormalize_rgb(self, rgb):
        # image_rgb 进入模型前做了 (rgb - mean) / std。
        # 这里恢复到 [0, 1]，让 residual learning 和 clean_target 处在同一数值空间。
        return torch.clamp(rgb * self.rgb_std + self.rgb_mean, 0.0, 1.0)

    def forward(self, inputs, rgb):
        # inputs 是 backbone 输出的四层 FFM 融合特征：
        #   c1: 1/4  resolution，细节多；
        #   c2: 1/8  resolution；
        #   c3: 1/16 resolution；
        #   c4: 1/32 resolution，全局语义和退化信息更强。
        c1, c2, c3, c4 = inputs

        g1 = self.align_c1(c1)
        g2 = self.align_c2(c2)
        g3 = self.align_c3(c3)
        g4 = self.align_c4(c4)

        # F4 -> 上采样 -> 与 F3 融合 -> 恢复块
        d4 = self.restore4(g4)
        d3 = self.restore3(torch.cat([self._upsample_like(d4, g3), g3], dim=1))

        # -> 上采样 -> 与 F2 融合 -> 恢复块
        d2 = self.restore2(torch.cat([self._upsample_like(d3, g2), g2], dim=1))

        # -> 上采样 -> 与 F1 融合 -> 恢复块
        d1 = self.restore1(torch.cat([self._upsample_like(d2, g1), g1], dim=1))

        # F1 通常是原图 1/4 分辨率，继续上采样到 RGB 输入尺寸。
        d0 = F.interpolate(
            d1,
            size=rgb.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners
        )

        residual = self.final(d0)
        base_rgb = self._denormalize_rgb(rgb)
        restored = torch.clamp(base_rgb + residual, 0.0, 1.0)
        return restored
