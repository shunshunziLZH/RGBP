import torch.nn as nn
import torch.nn.functional as F

from utils.init_func import init_weight

from engine.logger import get_logger

logger = get_logger()

class EncoderDecoder(nn.Module):
    def __init__(self, cfg=None, norm_layer=nn.BatchNorm2d):
        super(EncoderDecoder, self).__init__()
        # 当前项目已经固定为 RGB + polarization 的图像恢复模型，
        # 不再保留原 RGB-X 项目里根据 cfg.backbone 选择不同 backbone 的分支。
        #
        # 固定结构：
        #   backbone: dual SegFormer-B2
        #   head:     RestorationHead
        #   output:   3-channel restored RGB
        self.channels = [64, 128, 320, 512]
        self.norm_layer = norm_layer
        logger.info('Using fixed backbone: dual SegFormer-B2')
        from .encoders.dual_segformer import mit_b2 as backbone
        self.backbone = backbone(norm_fuse=norm_layer)

        # 外部接口必须保持原项目风格：model(rgb, x)。
        # 这里的 x 是完整 polarization_input，不拆成 i0/i60/i120 三个参数。
        #
        # 当前 dataset 读取到的 x 是 9 通道：
        #   Polar/0.jpg   的 RGB 三通道
        #   Polar/60.jpg  的 RGB 三通道
        #   Polar/120.jpg 的 RGB 三通道
        #
        # 但继承来的 RGB-X backbone 第二分支第一层仍然期望 3 通道输入。
        # 因此这里在 model 内部放一个很小的 1x1 Conv 做通道适配：
        #   [B, 9, H, W] -> [B, 3, H, W]
        # 这样外部调用方式不变，backbone 结构也不需要在这一步大改。
        self.x_input_channels = getattr(cfg, 'x_input_channels', 9)
        self.x_backbone_channels = 3
        if self.x_input_channels == self.x_backbone_channels:
            self.x_input_adapter = nn.Identity()
        else:
            self.x_input_adapter = nn.Conv2d(
                self.x_input_channels,
                self.x_backbone_channels,
                kernel_size=1
            )

        # 输出头固定为 RestorationHead。
        # 它接收 backbone 中 CM-FRM / FFM 融合后的 4 层特征，
        # 输出 3 通道恢复图，不再输出语义类别 logits。
        logger.info('Using fixed head: RestorationHead')
        from .decoders.restoration_head import RestorationHead
        self.restoration_head = RestorationHead(
            in_channels=self.channels,
            out_channels=cfg.output_channels,
            norm_layer=norm_layer,
            embed_dim=cfg.restoration_head_embed_dim
        )

        self.init_weights(cfg, pretrained=cfg.pretrained_model)

    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            logger.info('Loading pretrained model: {}'.format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)
        logger.info('Initing weights ...')
        init_weight(self.restoration_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    def encode_decode(self, rgb, x):
        """编码 RGB 与偏振输入，并解码得到恢复后的 RGB 图像。

        模型保持原 RGB-X 项目的调用风格：
            model(rgb, x)

        这里的 x 是完整的 polarization_input tensor，不是三个独立参数。
        当前数据集中，x 的期望 shape 是 [B, 9, H, W]，
        由 Polar/0.jpg、Polar/60.jpg、Polar/120.jpg 的 RGB 通道拼接得到。
        """
        orisize = rgb.shape
        # 在进入 backbone 前先检查通道数。
        # 如果 dataloader 或 config 配置错了，这里会直接报出清楚的错误。
        if x.shape[1] != self.x_input_channels:
            raise ValueError(
                'Expected x with {} channels, but got {}'.format(
                    self.x_input_channels, x.shape[1]
                )
            )
        # 将 9 通道偏振输入适配为继承 backbone 当前能接收的 3 通道输入。
        x = self.x_input_adapter(x)
        features = self.backbone(rgb, x)
        out = self.restoration_head(features)
        # RestorationHead 输出为 1/4 尺度，这里上采样回输入尺寸，
        # 保证 pred 和 clean_target 可以逐像素计算 L1Loss。
        out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)
        return out

    def forward(self, rgb, x):
        """图像恢复任务的 forward 接口。

        输入：
            rgb: 退化 RGB 图像，[B, 3, H, W]
            x:   偏振输入图像，[B, 9, H, W]

        输出：
            restored RGB prediction，[B, 3, H, W]

        重要约束：
            x 始终作为一个整体输入，保持双分支调用形式 model(rgb, x)。
            不要把接口改成 model(i0, i60, i120)。
        """
        return self.encode_decode(rgb, x)
