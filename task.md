提到了。CMX 论文在实验设置里明确写到：训练时采用 **ImageNet 预训练的 Mix Transformer encoder，也就是 MiT，作为 backbone**；decoder 使用 SegFormer 里的 MLP decoder；优化器是 AdamW，初始学习率是 `6e-5`，损失是 cross-entropy。

所以答案是：**原论文确实用了 ImageNet 预训练 encoder。**

你需不需要这样做？我的建议是：

**需要，至少 RGB encoder 要用 ImageNet 预训练。**

原因很直接：你的数据量只有几百对左右，恢复任务又需要学颜色、纹理、边缘、结构。如果 RGB encoder 从零训练，很容易出现收敛慢、过拟合、纹理表达弱的问题。ImageNet 预训练虽然不是水下图像恢复任务，但它给 encoder 提供了基础视觉能力：边缘、纹理、局部结构、物体区域、层级特征。这些对恢复任务仍然有用。

但要分清楚 **RGB encoder** 和 **偏振 encoder**。

RGB encoder：强烈建议加载 ImageNet 预训练 MiT。
偏振 encoder：可以加载，但要谨慎解释。

如果你的偏振输入是 3 通道，例如：

```python
polar_input = [I, DoP, AoP]
```

那偏振分支的第一层输入通道数也是 3，可以直接复用 ImageNet 预训练权重。虽然 `[I, DoP, AoP]` 不是自然 RGB 图像，但预训练权重仍然可以提供低层边缘、纹理、区域响应的初始化。后续训练会把它调整到偏振域。

如果你的偏振输入是 2 通道，例如：

```python
polar_input = [DoP, AoP]
```

那第一层 patch embedding 的输入通道数和 ImageNet 预训练权重不匹配。这时有三种选择：

第一种，也是我最推荐的：把偏振输入设计成 3 通道 `[I, DoP, AoP]`，这样最少改代码，也最容易加载预训练。

第二种：修改第一层输入为 2 通道，其他层加载预训练权重，第一层随机初始化或用 RGB 权重平均后适配。

第三种：偏振 encoder 完全从零训练。这不推荐作为第一版，因为你的数据量不大。

所以你的项目里最稳的配置是：

```text
RGB encoder：ImageNet 预训练 MiT
Polar encoder：同样加载 ImageNet 预训练 MiT，但输入用 [I, DoP, AoP] 三通道
CM-FRM / FFM：随机初始化，从任务中学习
Restoration decoder：随机初始化，从任务中学习
```

训练时不建议长期冻结 encoder。可以有两种策略：

第一版简单策略：全部参与训练，但 encoder 学习率小一点，decoder 和新模块学习率大一点。

例如：

```text
encoder lr = 1e-5 或 2e-5
CM-FRM / FFM / decoder lr = 1e-4
```

更稳的策略：前几个 epoch 冻结 encoder，只训练新 decoder 和融合模块；然后解冻全网微调。但这会增加训练脚本复杂度，第一版不一定需要。

还要注意归一化。既然 RGB encoder 用 ImageNet 预训练，那么 RGB 输入送进 encoder 前可以继续用 ImageNet mean/std：

```python
rgb_norm_mean = [0.485, 0.456, 0.406]
rgb_norm_std  = [0.229, 0.224, 0.225]
```

但偏振输入不要照搬 ImageNet 归一化。`I/DoP/AoP` 可以先映射到 `[0,1]`，再用 `[0.5,0.5,0.5] / [0.5,0.5,0.5]` 映射到 `[-1,1]`，或者统计训练集自己的 mean/std。

最关键的一点：**GT 清晰图和最终恢复图不要用 ImageNet norm 计算 loss。**
恢复 loss 应该在 `[0,1]` 图像空间算：

```text
encoder 输入：可以 norm
最终输出 restored：保持 [0,1]
clean GT：保持 [0,1]
loss：L1 / SSIM / PSNR 都在 [0,1] 上计算
```

总结成一句话：

**论文用了 ImageNet 预训练 MiT。你也应该用，RGB encoder 必须用；偏振 encoder 第一版最好通过 `[I, DoP, AoP]` 做成 3 通道，也加载 ImageNet 预训练，但不要冻结太久，让它微调到偏振恢复任务。**
