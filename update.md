decoder 的核心改法：**把“分类解码器”改成“重建解码器”**。


恢复 decoder 应该采用 **U-Net 风格的逐级上采样结构**，

数据流可以写成：

`F4 → 上采样 → 与 F3 融合 → 恢复块`

`→ 上采样 → 与 F2 融合 → 恢复块`

`→ 上采样 → 与 F1 融合 → 恢复块`

`→ 上采样到原图大小 → final conv → 输出 3 通道 RGB`

更具体地说，decoder 可以分成 4 个部分。

第一部分：**通道对齐**。

CMX 的四层特征通道数通常不一样，比如 MiT-B2 风格可能是 `[64, 128, 320, 512]`。恢复 decoder 不应该直接拼接这些特征，先用 `1×1 Conv` 把它们统一到一个较小通道数，比如 64 或 96。

例如：

`F1 → Conv1x1 → G1`

`F2 → Conv1x1 → G2`

`F3 → Conv1x1 → G3`

`F4 → Conv1x1 → G4`

第二部分：**自顶向下上采样融合**。

从 `G4` 开始：

`D4 = RestoreBlock(G4)`

然后上采样到 `G3` 的尺寸：

`D3 = RestoreBlock(concat(upsample(D4), G3))`

再上采样到 `G2` 的尺寸：

`D2 = RestoreBlock(concat(upsample(D3), G2))`

再上采样到 `G1` 的尺寸：

`D1 = RestoreBlock(concat(upsample(D2), G1))`

这里的 `RestoreBlock` 不需要一开始太复杂。第一版建议用：

`Conv 3×3 → GELU/ReLU → Conv 3×3 + residual`

也就是轻量残差卷积块。不要一开始就上复杂 attention，否则项目会迅速臃肿。

第三部分：**恢复到原图分辨率**。

如果 `F1` 仍然是原图的 1/4 分辨率，那么 `D1` 还需要继续上采样到输入大小：

`D0 = upsample(D1, size=RGB_input_size)`

然后：

`residual = Conv3x3(D0) → Conv3x3 → 3 channels`

最后输出：

`J_hat = RGB_input + residual`

这就是残差恢复。它比直接输出清晰图更稳，因为水下恢复不是从零生成图像，而是在原图基础上去散射、校色、增强细节。

第四部分：**输出约束**。

输出可以有两种方式。

第一种：

`J_hat = clamp(RGB_input + residual, 0, 1)`
可以明确要求：

**新 decoder 输入 FFM 的多尺度融合特征，输出 3 通道 RGB 恢复图。**

参考配置：

`decoder_dim = 64` 或 `96`；

每个尺度用 1 个轻量残差块；

上采样用 bilinear interpolation；

融合方式用 `concat + 3×3 conv`；

输出用 residual learning。

不要一开始用 PixelShuffle、复杂 Transformer decoder、扩散模块、物理参数头。第一版目标是验证：**CM-FRM + FFM 的双分支交互融合是否能让恢复效果超过 RGB-only 和简单拼接。**

一句话总结：**decoder 不要再做“像素分类”，而要做“多尺度特征重建”；用 F4 判断全局退化，用 F1/F2 补细节，最后预测 RGB 残差加回输入图。**
