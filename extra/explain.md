# RGB-Polar 图像恢复项目说明书

这份文档是项目的**总控说明书**，用于帮助读者从宏观上理解和调控当前工程。它不参与训练，也不是运行入口。

**一句话总览**：当前项目保留原 RGB-X 跨模态融合骨架，把任务从语义分割改成了图像恢复：

```text
metadata.csv 组织样本
seed 随机划分 scene
dataloader 同步增强 RGB、偏振、清晰图
model(rgb, x) 用 U-Net 风格 decoder 输出 3 通道恢复图
train.py 用 L1Loss 和 clean_target 形成训练闭环
eval.py 用 256 x 256 滑窗推理计算 L1 / PSNR
```

## 1. 项目定位

原项目是 **RGB-X Semantic Segmentation**，核心价值在于双分支 backbone 和跨模态融合模块。现在项目已经改成 **RGB-Polar 图像恢复任务**。

当前训练闭环：

```text
image_rgb [B, 3, H, W]
    +
polarization_input [B, 9, H, W]
    ↓
model(rgb, x)
    ↓
restored_rgb [B, 3, H, W]
    ↓
L1Loss(restored_rgb, clean_target)
```

**重点约束**：模型接口保持原项目风格：

```python
model(rgb, x)
```

这里的 `x` 是完整的 `polarization_input`，不要改成 `model(i0, i60, i120)`。

## 2. 数据格式

### 数据入口

| 项目 | 当前设置 |
| --- | --- |
| 数据目录 | `../../DATA/RGBP` |
| metadata | `../../DATA/RGBP/metadata.csv` |
| 路径含义 | 项目目录的上级的上级目录下的 `DATA/RGBP` |
| 示例 | 项目在 `D:/CODE/RGBP_restoration` 时，数据在 `D:/DATA/RGBP` |

只要保持“项目目录/../../DATA/RGBP”这个相对关系，项目移动后不需要修改绝对路径。

### 单条样本

| 字段 | 路径 | shape | 含义 |
| --- | --- | --- | --- |
| `image_rgb` | `scene/sample/RGB/I.jpg` | `H x W x 3` | 退化 RGB 输入 |
| `polarization_input` | `scene/sample/Polar/0.jpg`、`60.jpg`、`120.jpg` | `H x W x 9` | 三个偏振角 RGB 图拼接 |
| `clean_target` | `scene/GT/RGB/I.jpg` | `H x W x 3` | 清晰 RGB 监督目标 |

`polarization_input` 通道顺序：

```text
[0_R, 0_G, 0_B, 60_R, 60_G, 60_B, 120_R, 120_G, 120_B]
```

对应代码：

```text
dataloader/RGBXDataset.py
```

它负责读取 `metadata.csv`、过滤 `sample == GT`、加载三路图像，并按 `train / val / test` 返回数据。

## 3. 数据集划分

当前使用 **scene 级划分**，不是 sample 级随机划分。

**原因**：同一个 scene 下的多个 sample 共享同一张 `clean_target`。如果同一个 scene 同时出现在 train 和 val/test，验证或测试会间接见过同一个清晰目标，结果会偏乐观。

当前流程：

```text
收集全部 scene
→ 自然排序
→ 用 config.split_seed 确定性随机打乱
→ 按 config.split_ratios 切分 train / val / test
```

关键配置：

```python
C.split_strategy = 'scene'
C.split_ratios = {'train': 0.8, 'val': 0.1, 'test': 0.1}
C.split_seed = 12345
```

调控建议：

| 目标 | 修改 |
| --- | --- |
| 换一套随机划分 | 改 `C.split_seed` |
| 改 train/val/test 比例 | 改 `C.split_ratios`，三者总和必须是 `1.0` |
| 避免数据泄漏 | 保持 scene 级划分，不要改回 sample 级随机划分 |

## 4. 数据增强与预处理

对应代码：

```text
dataloader/dataloader.py
```

训练阶段使用 `TrainPre`，同时处理：

```text
degraded_input
polarization_input
clean_target
```

**重点**：三路图像必须使用完全一致的几何变换。当前 loss 是逐像素 `L1Loss`，任何一路单独 crop/flip/resize 都会造成像素错位。

训练预处理流程：

| 步骤 | 操作 | 说明 |
| --- | --- | --- |
| 1 | `random_mirror` | 三路同步水平翻转 |
| 2 | `random_scale` | 三路同步缩放，图像任务统一用双线性插值 |
| 3 | `normalize` | 输入用 `0.5 / 0.5` 标准化到约 `[-1, 1]`，target 只缩放到 `[0, 1]` |
| 4 | `random crop/pad` | 三路共用同一个 `crop_pos` |
| 5 | `HWC -> CHW` | 适配 PyTorch Conv2d |

关键配置：

```python
C.image_height = 256
C.image_width = 256
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.norm_mean = np.array([0.5, 0.5, 0.5])
C.norm_std = np.array([0.5, 0.5, 0.5])
```

调控建议：

| 目标 | 修改 |
| --- | --- |
| 降低训练显存 | 降低 `C.image_height / C.image_width` |
| 小样本过拟合调试 | 把 `C.train_scale_array` 改成 `[1]` 或 `None` |
| 修改输入归一化 | 改 `C.norm_mean / C.norm_std`，当前默认是 `[0.5, 0.5, 0.5]`；偏振 9 通道会自动重复 RGB 统计量 |

## 5. 模型结构

对应代码：

```text
models/builder.py
models/decoders/restoration_head.py
```

整体结构：

```text
RGB branch + Polar branch
    ↓
dual SegFormer-B2 backbone
    ↓
CM-FRM / FFM 跨模态融合
    ↓
U-Net 风格 RestorationHead
    ↓
3 通道恢复图
```

### 输入接口

```python
model(rgb, x)
```

| 参数 | shape | 含义 |
| --- | --- | --- |
| `rgb` | `[B, 3, H, W]` | 退化 RGB 输入 |
| `x` | `[B, 9, H, W]` | 偏振输入 |

当前 `builder.py` 内部有一个 `x_input_adapter`：

```text
[B, 9, H, W] -> [B, 3, H, W]
```

原因是继承来的 RGB-X backbone 第二分支第一层仍期望 3 通道输入。偏振输入已在数据层做成 9 通道，所以模型内部先用 `1x1 Conv` 适配到 3 通道。

### RestorationHead

当前 decoder 是 **U-Net 风格逐级上采样恢复头**。

流程：

```text
F1/F2/F3/F4
    ↓ 1x1 Conv 对齐通道
G1/G2/G3/G4
    ↓
G4 -> upsample -> concat G3 -> RestoreBlock
    ↓
D3 -> upsample -> concat G2 -> RestoreBlock
    ↓
D2 -> upsample -> concat G1 -> RestoreBlock
    ↓
D1 -> upsample 到输入大小
    ↓
final conv 预测 residual
    ↓
clamp(input_rgb + residual, 0, 1)
```

**前因**：`update.md` 要求 decoder 不再做“像素分类”，而要做“多尺度特征重建”。

**后果**：模型学习的是在退化 RGB 基础上的残差恢复，更适合去散射、校色和细节增强；同时保持第一版 decoder 足够轻量，不引入复杂 attention、PixelShuffle 或扩散模块。

关键配置：

```python
C.model_name = 'RGBPolarRestoration_MiT-B2_RestorationHead'
C.pretrained_model = C.root_dir + '/pretrained/segformer/mit_b2.pth'
C.x_input_channels = 9
C.restoration_head_embed_dim = 96
C.output_channels = 3
```

## 6. Loss 与训练闭环

对应代码：

```text
train.py
```

当前 loss：

```python
criterion = nn.L1Loss(reduction='mean')
```

训练逻辑：

```python
restored_rgb = model(image_rgb, polarization_input)
loss = criterion(restored_rgb, clean_target)
```

为什么不用 `CrossEntropyLoss`：

`clean_target` 是 RGB 清晰图，不是类别 mask。`CrossEntropyLoss` 是分类损失，不适合当前恢复任务。

为什么 target 和 pred 都在 `[0, 1]`：

| 张量 | 数值处理 |
| --- | --- |
| `clean_target` | `clean_target / 255.0` |
| `restored_rgb` | 输入 RGB 按 `C.norm_mean / C.norm_std` 反归一化到 `[0, 1]`，加 residual，再 clamp 到 `[0, 1]` |

这样 `L1Loss` 的含义就是平均像素绝对误差。

## 7. 训练入口与日志

训练入口：

```bash
python train.py
```

日志与权重目录由 `config.py` 控制：

```python
C.log_dir
C.tb_dir
C.checkpoint_dir
C.checkpoint_start_epoch
C.checkpoint_step
```

TensorBoard 当前记录：

```text
train_loss
```

调控建议：

| 目标 | 修改 |
| --- | --- |
| 更早保存模型 | 降低 `C.checkpoint_start_epoch` |
| 更频繁保存 | 降低 `C.checkpoint_step` |
| 缩短实验 | 降低 `C.nepochs` |
| 调学习率 | 改 `C.lr` |
| 改 batch size | 改 `C.batch_size` |
| 调数据读取并行度 | 改 `C.num_workers` |

`C.num_workers` 说明：

| 场景 | 建议 |
| --- | --- |
| 读取慢，CPU 和内存足够 | 适当增大 |
| Windows 上卡住、启动慢 | 降到 `0 / 2 / 4` |
| 小样本过拟合调试 | 建议 `0` 或 `2`，更容易排查问题 |

## 8. 验证与测试

当前 `eval.py` 已经是恢复任务评估入口，不再使用旧语义分割的 `mIoU / num_classes / sliding_eval_rgbX`。

当前 eval 流程：

```text
读取 val 或 test split
→ normalize image_rgb / polarization_input
→ clean_target 缩放到 [0, 1]
→ 单尺度滑窗推理
→ patch 结果累加平均并拼回整图
→ 计算 L1 和 PSNR
→ 可选保存 restored_rgb
```

滑窗配置：

```python
C.eval_crop_size = [256, 256]
C.eval_stride_rate = 2 / 3
```

为什么只保留滑窗：整图推理在大图上更吃显存，滑窗可以降低单次推理显存压力。

为什么删除多尺度和 flip test：当前第一目标是让恢复评估入口清楚、轻量、可排查；多尺度和 flip test 会让逻辑变复杂，也会增加评估时间。

常用命令：

```bash
python eval.py -e last --split val
python eval.py -e last --split test
python eval.py -e last --split val -p eval_results/val_last
```

## 9. 最常用的调控旋钮

| 类别 | 配置/文件 | 作用 |
| --- | --- | --- |
| 数据划分 | `C.split_seed` | 换一套可复现随机划分 |
| 数据划分 | `C.split_ratios` | 调整 train/val/test 比例 |
| 输入尺寸 | `C.image_height`, `C.image_width` | 控制训练 crop 大小，目前为 `256 x 256` |
| 增强强度 | `C.train_scale_array` | 控制训练随机缩放 |
| 训练长度 | `C.nepochs` | 控制训练 epoch 数 |
| Batch | `C.batch_size` | 控制每步样本数 |
| 数据读取 | `C.num_workers` | 控制 DataLoader 并行进程数 |
| 学习率 | `C.lr`, `C.lr_power`, `C.warm_up_epoch` | 控制优化节奏 |
| 模型容量 | `C.restoration_head_embed_dim` | 控制 decoder 统一通道数 |
| 偏振输入 | `C.x_input_channels = 9` | 控制模型期望偏振输入通道 |
| 评估滑窗 | `C.eval_crop_size` | 控制 eval patch 大小，目前为 `256 x 256` |
| 评估滑窗 | `C.eval_stride_rate` | 控制滑窗重叠比例 |
| Loss | `train.py` 中 `nn.L1Loss` | 控制监督目标 |
| Decoder | `models/decoders/restoration_head.py` | 控制恢复头结构 |

## 10. 修改优先级建议

### 第一优先级：先保证闭环正确

| 检查项 | 期望 |
| --- | --- |
| `image_rgb` | `[B, 3, H, W]` |
| `polarization_input` | `[B, 9, H, W]` |
| `clean_target` | `[B, 3, H, W]` |
| `restored_rgb` | `[B, 3, H, W]` |
| target 范围 | `[0, 1]` |
| pred 范围 | `[0, 1]` |
| loss | `L1Loss(restored_rgb, clean_target)` |

### 第二优先级：做小样本过拟合

如果小样本都无法过拟合，优先排查：

1. target 是否读错；
2. RGB/BGR 是否错；
3. polarization 9 通道顺序是否错；
4. crop/flip/resize 是否同步；
5. pred 和 target 尺寸是否一致；
6. pred/target 归一化范围是否一致。

### 第三优先级：再扩大训练

小样本闭环通了之后，再考虑更强增强、更长训练、更复杂 loss、更大模型和正式 val/test 指标。

## 11. 快速定位问题

| 问题 | 优先检查 |
| --- | --- |
| loss 不下降 | 关闭随机 scale，只保留固定 resize/crop，再做极小样本过拟合 |
| 输出颜色奇怪 | 检查 OpenCV 读图后是否 BGR -> RGB |
| 输出尺寸不一致 | 检查 `RestorationHead` 是否上采样到 `rgb.shape[2:]`，eval 滑窗是否裁回原始尺寸 |
| x 通道数报错 | 检查 `C.x_input_channels = 9` 和 `Polar/0,60,120` 是否按 RGB 三通道 concat |
| 验证结果异常乐观 | 检查是否仍然按 scene 划分，避免同 scene 同时出现在 train 和 val/test |
| 训练显存不够 | 降低 `batch_size / image_height / image_width` |
| eval 显存不够 | 降低 `eval_crop_size` |
| 数据读取卡住 | 降低 `C.num_workers` 到 `0 / 2 / 4` |

## 12. 后续建议

在当前闭环稳定后，可以小步增加：

1. `SSIM` 指标；
2. 保存 `image_rgb / restored_rgb / clean_target` 拼接图；
3. 单独的可视化脚本；
4. 小样本过拟合调试脚本；
5. RGB-only、简单拼接、RGB-Polar 融合三组对照实验。