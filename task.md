推荐这样改：**不要直接沿用语义分割配置。** 这几个参数在 CMX 原项目里主要服务于分割训练，尤其 `train_scale_array=[0.5, ..., 1.75]` 是分割常用的随机尺度增强；原论文也说明训练时用了随机翻转和 `[0.5, 1.75]` 的随机缩放。 但你现在做 RGB 图像恢复，目标是重建连续图像，不是预测类别图，增强策略要更保守。

建议第一版这样写：

```python
C.image_height = 256
C.image_width = 256

# 第一版基准：先不要做大范围随机缩放
C.train_scale_array = [1.0]

# RGB encoder 如果继续用 ImageNet 预训练 MiT，可以保留这个归一化
C.rgb_norm_mean = np.array([0.485, 0.456, 0.406])
C.rgb_norm_std = np.array([0.229, 0.224, 0.225])

# 偏振分支不要用 ImageNet mean/std
# 如果 polar 输入是 [I, DoP, AoP] 且都已经映射到 [0, 1]
C.polar_norm_mean = np.array([0.5, 0.5, 0.5])
C.polar_norm_std = np.array([0.5, 0.5, 0.5])

# GT 清晰图不要做 ImageNet norm，loss 在 [0, 1] 空间计算
C.target_norm_mean = np.array([0.0, 0.0, 0.0])
C.target_norm_std = np.array([1.0, 1.0, 1.0])
```

如果你当前代码只有这一组：

```python
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])
```

那我建议先不要删除，而是**拆成 RGB 分支、偏振分支、GT 三套归一化**。这是最关键的修改点。

具体解释如下。

`C.image_height = 256` 和 `C.image_width = 256` 可以保留。对 400 对左右的数据，256 patch 是比较稳的起点。太小，比如 128，恢复任务容易只学局部纹理，看不到大范围浑浊和颜色偏移；太大，比如 384，一开始显存压力大，也更容易训练慢。第一版用 256 合理。后面稳定后可以改成 320 或 384 微调。

`C.train_scale_array` 第一版建议改成：

```python
C.train_scale_array = [1.0]
```

或者稍微增强：

```python
C.train_scale_array = [0.75, 1.0, 1.25]
```

不建议第一版继续使用：

```python
[0.5, 0.75, 1, 1.25, 1.5, 1.75]
```

原因是图像恢复对纹理、边缘、颜色、退化强度很敏感。大范围缩放会引入插值变化，改变局部频率和模糊程度，可能让网络学到不稳定的恢复映射。分割任务只要类别轮廓对，尺度增强通常有利；恢复任务要像素级对齐，增强要保守。

更推荐你的训练增强以这些为主：

```python
C.train_scale_array = [1.0]
C.random_crop = True
C.random_horizontal_flip = True
C.random_vertical_flip = True  # 如果你的水下场景不依赖重力方向，可开
C.random_rotate_90 = True      # 如果 AoP 处理正确才建议开
```

但如果偏振输入包含 AoP，翻转和旋转要谨慎。DoP 是程度量，普通翻转问题较小；AoP 是方向量，水平翻转、旋转后角度物理含义会变。如果你的代码只是把 AoP 当普通灰度图 flip，严格说不完全物理。第一版最稳妥是：**RGB、偏振、GT 同步 crop；水平翻转可以先开；旋转先别开。**

关于 `norm_mean` 和 `norm_std`，要看你是否使用 ImageNet 预训练 encoder。

如果你保留 CMX 的 MiT encoder，并加载 ImageNet 预训练权重，那么 RGB 分支输入可以继续用 ImageNet 归一化：

```python
rgb_norm = (rgb - rgb_mean) / rgb_std
```

但注意：**GT 清晰图不要用 ImageNet 归一化后计算 L1/SSIM。** 图像恢复的 loss 应该在 `[0, 1]` 图像空间计算。否则 PSNR、SSIM、残差输出都会变得混乱。

模型里最好保留两份 RGB：

```python
rgb_raw   # [0, 1]，用于残差相加和 loss
rgb_norm  # ImageNet norm 后，送入 RGB encoder
```

输出时：

```python
residual = decoder(features)
restored = torch.clamp(rgb_raw + residual, 0, 1)
loss = L1(restored, clean_raw) + SSIM_loss(restored, clean_raw)
```

偏振分支不要用 ImageNet 的 mean/std。ImageNet 均值方差只适合自然 RGB 图像，不适合 AoP、DoP、I 这种物理量图。第一版如果你输入 `[I, DoP, AoP]`，可以简单用：

```python
C.polar_norm_mean = np.array([0.5, 0.5, 0.5])
C.polar_norm_std = np.array([0.5, 0.5, 0.5])
```

把 `[0, 1]` 映射到 `[-1, 1]`。更严谨的做法是统计你训练集的 polar mean/std，但第一版不必复杂化。

所以最终我建议你的配置改成这个版本：

```python
# patch size for restoration training
C.image_height = 256
C.image_width = 256

# conservative scale augmentation for first baseline
C.train_scale_array = [1.0]
# optional after baseline is stable:
# C.train_scale_array = [0.75, 1.0, 1.25]

# RGB branch normalization, only for pretrained encoder input
C.rgb_norm_mean = np.array([0.485, 0.456, 0.406])
C.rgb_norm_std = np.array([0.229, 0.224, 0.225])

# Polar branch normalization, do not use ImageNet stats
C.polar_norm_mean = np.array([0.5, 0.5, 0.5])
C.polar_norm_std = np.array([0.5, 0.5, 0.5])

# Restoration target remains in [0, 1]
C.target_norm_mean = np.array([0.0, 0.0, 0.0])
C.target_norm_std = np.array([1.0, 1.0, 1.0])
```

最重要的一句话：**RGB encoder 可以继续 ImageNet norm；偏振输入不要 ImageNet norm；清晰 GT 和最终输出必须保持 `[0,1]`，loss 在 `[0,1]` 上算。**
