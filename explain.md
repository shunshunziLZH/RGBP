# 项目结构说明与图像恢复改造指南

这份说明面向“刚下载项目、想把 RGB-X 语义分割项目改造成图像恢复项目”的研究者。它不会假设你已经熟悉深度学习工程代码，而是先讲这个项目整体在做什么，再逐个解释文件职责，最后给出改造方向。

## 1. 这个项目原本在做什么

这个仓库是论文 **CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation with Transformers** 的实现。它处理的是 **RGB 图像 + 另一种模态 X** 的语义分割任务。

这里的 `X` 可以是深度图、HHA、热红外、偏振图、事件图等。项目的核心思想是：

1. 同时读入 RGB 图像和 X 模态图像。
2. 用两条相似的 Transformer 编码分支分别提取 RGB 和 X 的特征。
3. 在每个尺度上用跨模态融合模块把两个模态的信息合起来。
4. 用分割解码器把融合特征还原成每个像素的类别预测。
5. 用语义标签图做监督，训练目标是交叉熵损失。

原项目的数据流可以直观理解为：

```text
RGB 图像 + X 模态图像
        |
        v
dataloader 读取并增强
        |
        v
双分支 Transformer backbone
        |
        v
FRM / FFM 跨模态融合
        |
        v
MLPDecoder / UPerNet / DeepLabV3+ / FCNHead
        |
        v
num_classes 通道的分割 logits
        |
        v
CrossEntropyLoss + mIoU 评估
```

如果改成图像恢复，目标会变成：

```text
退化图像 + 可选辅助模态 X
        |
        v
模型
        |
        v
恢复后的 RGB 图像
        |
        v
L1 / L2 / Charbonnier / SSIM / perceptual loss
        |
        v
PSNR / SSIM / LPIPS 等指标
```

最重要的区别是：**语义分割输出的是类别图，图像恢复输出的是连续 RGB 图像**。所以数据、模型输出头、损失函数、评估指标都要换。

## 2. 顶层文件

### `README.md`

项目说明文档。包含论文信息、环境依赖、数据组织方式、训练命令、评估命令、预训练权重和引用格式。

对你最有用的是数据格式部分。原项目希望数据长这样：

```text
datasets/DatasetName/
    RGB/
    HHA 或其他 X 模态文件夹/
    Label/
    train.txt
    test.txt
```

对语义分割来说，`Label` 是类别标签图。对图像恢复来说，你通常要把 `Label` 替换成 `GT`、`Clean`、`Target` 之类的清晰图像文件夹。

### `config.py`

全局配置文件。这个项目大量参数都从这里读取。

主要内容：

- 数据集路径：`dataset_path`、`rgb_root_folder`、`gt_root_folder`、`x_root_folder`。
- 文件后缀：`rgb_format`、`gt_format`、`x_format`。
- 标签相关：`num_classes`、`class_names`、`background`、`gt_transform`。
- 输入尺寸：`image_height`、`image_width`。
- 归一化参数：`norm_mean`、`norm_std`。
- 模型配置：`backbone`、`pretrained_model`、`decoder`、`decoder_embed_dim`。
- 训练配置：学习率、batch size、epoch 数、warmup、数据增强尺度等。
- 评估配置：滑窗大小、多尺度、翻转测试等。
- 日志和 checkpoint 路径。

如果改图像恢复，这个文件是必改的：

- `gt_root_folder` 应改成清晰目标图像路径。
- `gt_format` 应改成目标图像后缀。
- `num_classes` 对恢复任务没有意义，可以换成 `out_channels = 3`。
- `class_names`、`background`、`gt_transform` 基本可以删除或不使用。
- `eval` 相关应从 mIoU 配置改成 PSNR/SSIM 配置。
- `decoder` 应换成恢复解码器，比如 U-Net 风格上采样头、Restormer 风格 head，或者简单的多尺度融合重建头。

注意：`README.md` 里写的是 `configs.py`，但仓库里实际文件叫 `config.py`。

### `train.py`

训练入口。运行训练时从这里开始。

它做的事情按顺序是：

1. 创建 `Engine`，处理命令行参数和分布式训练。
2. 设置随机种子和 cudnn。
3. 调用 `get_train_loader` 构建训练数据。
4. 创建 TensorBoard 日志目录。
5. 定义分割损失：`nn.CrossEntropyLoss`。
6. 根据配置创建 `EncoderDecoder` 模型。
7. 用 `group_weight` 把参数分成有 weight decay 和无 weight decay 两组。
8. 创建优化器，支持 `AdamW` 和 `SGDM`。
9. 创建 warmup + poly 学习率策略。
10. 如果是分布式训练，用 `DistributedDataParallel` 包住模型。
11. 如果指定了 checkpoint，恢复训练。
12. 进入 epoch/iteration 循环。
13. 每个 batch 取出 `data`、`label`、`modal_x`。
14. 调用 `model(imgs, modal_xs, gts)` 得到 loss。
15. 反向传播、更新参数、调整学习率。
16. 写 TensorBoard、保存 checkpoint。

对图像恢复来说，这里要重点改：

- `criterion = nn.CrossEntropyLoss(...)` 要换成恢复损失，例如 `nn.L1Loss()`。
- `gts` 原本是类别标签，改造后应该是清晰 RGB 图像张量。
- `loss = model(imgs, modal_xs, gts)` 可以保留接口，但模型内部要从分类损失改成图像重建损失。
- 训练日志可以增加 `PSNR` 或 `L1`。

还有一个小兼容点：代码里用了 `dataloader.next()`，更通用的 Python 写法是 `next(dataloader)`。

### `eval.py`

评估入口。它继承 `engine.evaluator.Evaluator`，实现 RGB-X 分割任务的单张图评估和总体指标计算。

核心类是 `SegEvaluator`：

- `func_per_iteration`：对一张图做推理，得到预测类别图。
- `compute_metric`：汇总混淆矩阵，计算 mIoU、pixel accuracy 等分割指标。

它使用 `sliding_eval_rgbX` 做滑窗推理。输出是 `pred`，形状是二维类别图，每个像素是一个类别 id。

对图像恢复来说，这个文件基本要重写：

- 推理结果应该是 RGB 图像，而不是类别 id。
- 不需要 `hist_info`、`compute_score`、`print_iou`。
- 需要实现 PSNR、SSIM、LPIPS 等指标。
- 保存结果时不需要 palette 彩色标签图，而是保存恢复图像。

注意：当前 `eval.py` 里保存彩色图时用了 `Image.fromarray` 和 `get_class_colors()`，但文件顶部没有导入 `PIL.Image`，也没有定义或导入 `get_class_colors`。如果直接跑保存预测图，可能会报错。可以改成 `RGBXDataset.get_class_colors()` 并导入 `from PIL import Image`。

### `requirements.txt`

Python 依赖列表。包含：

- `easydict`：让配置可以用 `config.xxx` 的方式访问。
- `opencv-python`：读写图像和做增强。
- `timm`：Transformer 相关层和工具。
- `scipy`：可视化和部分工具函数依赖。
- `tqdm`：进度条。
- `tensorboardX`：TensorBoard 日志。

这里没有固定 PyTorch 版本，只在注释里写了推荐版本。

### `.gitignore`

Git 忽略规则。通常会忽略临时文件、缓存、日志、模型权重等。

### `LICENSE`

开源许可证文件，说明代码使用权限。

### `segmentation.jpg`

README 中展示的分割效果图，只是说明性图片，不参与训练。

## 3. 数据加载部分

### `dataloader/RGBXDataset.py`

这是数据集类，继承自 `torch.utils.data.Dataset`。

它负责从硬盘读取：

- RGB 图像：返回为 `data`。
- 语义分割标签图：返回为 `label`。
- X 模态图像：返回为 `modal_x`。
- 文件名：返回为 `fn`。

关键方法：

- `__init__`：保存路径、后缀、split、预处理函数等配置。
- `__len__`：返回数据集长度。
- `__getitem__`：按 index 读取一组 RGB、label、X。
- `_get_file_names`：从 `train.txt` 或 `test.txt` 读取样本名。
- `_construct_new_file_names`：当指定 `file_length` 时，把数据列表重复到固定长度。
- `_open_image`：用 OpenCV 读图。
- `_gt_transform`：把标签 `gt - 1`，因为很多分割数据集标签从 1 开始，而 PyTorch 分类通常从 0 开始。
- `get_class_colors`：生成类别颜色表，用于可视化分割结果。

读取流程大致是：

```text
item_name
  -> RGB/item_name.jpg
  -> Label/item_name.png
  -> HHA/item_name.jpg
  -> preprocess(rgb, gt, x)
  -> torch tensor
  -> dict(data=rgb, label=gt, modal_x=x)
```

对图像恢复来说，这个文件需要重点改：

- `gt` 不应再用灰度方式读取，而应读取 RGB 清晰图像。
- `_gt_transform(gt - 1)` 要删除。
- `gt` 的 dtype 不应是 `long`，而应是 `float`。
- `label` 可以改名为 `target`，当然为了少改训练代码也可以继续叫 `label`。
- 如果恢复任务是单输入，比如低照度增强，可以让 `data` 是低质图，`label` 是清晰图，`modal_x` 可选。
- 如果恢复任务是多模态，比如 RGB+偏振恢复清晰 RGB，则可以保留 `modal_x`。

### `dataloader/dataloader.py`

这个文件定义训练和验证前处理，以及创建 DataLoader。

主要函数和类：

- `random_mirror`：随机水平翻转 RGB、标签、X。
- `random_scale`：随机缩放 RGB、标签、X。
- `TrainPre`：训练前处理。
- `ValPre`：验证前处理，当前几乎不处理，原样返回。
- `get_train_loader`：构建训练集和 DataLoader。

`TrainPre.__call__` 的流程：

```text
随机翻转
随机尺度缩放
RGB 和 X 归一化
随机裁剪到 config.image_height x config.image_width
RGB 和 X 转成 C x H x W
返回 rgb, gt, modal_x
```

分割任务里，标签图要用最近邻插值，避免类别 id 被插值成小数。图像恢复任务里，目标图是连续图像，通常应该用双线性或双三次插值。

对图像恢复来说，这里要改：

- `gt` 也要做图像归一化，或者至少转成 `[0, 1]`。
- `gt` 缩放时应使用 `cv2.INTER_LINEAR` 或 `cv2.INTER_CUBIC`，不要用 `INTER_NEAREST`。
- 裁剪 padding 的标签填充值 `255` 不再适合，恢复任务可用 `0` 或反射 padding。
- 数据增强可以加入退化相关增强，比如噪声、模糊、JPEG 压缩、低照度模拟等，但如果数据已经配对，增强必须对输入、辅助模态、目标保持空间对齐。

## 4. 模型构建部分

### `models/builder.py`

这是模型总装配文件。核心类是 `EncoderDecoder`。

它负责：

1. 根据 `cfg.backbone` 选择编码器：
   - `mit_b0` 到 `mit_b5`：SegFormer/MiT 双分支编码器。
   - `swin_s`、`swin_b`：Swin Transformer 双分支编码器。
2. 根据 `cfg.decoder` 选择解码器：
   - `MLPDecoder`
   - `UPernet`
   - `deeplabv3+`
   - 默认 `FCNHead`
3. 初始化预训练权重。
4. 定义 `encode_decode`：backbone 提特征，decoder 输出分割 logits，再插值回原图大小。
5. 定义 `forward`：训练时返回 loss，推理时返回预测 logits。

当前输出是：

```text
B x num_classes x H x W
```

其中每个通道代表一个语义类别。

对图像恢复来说，这个文件是核心改造点：

- 输出应改为 `B x 3 x H x W`。
- `criterion` 不应是 `CrossEntropyLoss`，而应是 `L1Loss`、`CharbonnierLoss` 等。
- `label.long()` 要删除，因为目标图像是 float。
- decoder 最后一层不能是 `num_classes`，而是 `out_channels=3`。
- 可以保留双分支 backbone 和 FRM/FFM 融合模块，把分割 head 换成恢复 head。

一个最小改法是：新增 `RestorationDecoder`，输入四个尺度特征，逐级上采样融合，最后输出 3 通道图像。

### `models/net_utils.py`

这是 CMX 的跨模态融合核心模块。

主要模块：

- `ChannelWeights`：根据 RGB 和 X 特征的全局平均池化、最大池化结果，预测通道级权重。
- `SpatialWeights`：根据 RGB 和 X 特征，预测空间位置上的权重。
- `FeatureRectifyModule`，简称 FRM：用通道权重和空间权重让两个模态互相修正。
- `CrossAttention`：跨模态注意力的基础实现。
- `CrossPath`：将两个模态特征投影后做交叉注意力。
- `ChannelEmbed`：把拼接后的 token 特征变回卷积特征图。
- `FeatureFusionModule`，简称 FFM：用 cross attention 融合 RGB 和 X，输出单个融合特征图。

直观理解：

```text
FRM: 先让 RGB 和 X 互相“校正”
FFM: 再把校正后的 RGB 和 X “合成一个融合特征”
```

这部分对图像恢复很有价值，因为它不依赖“分割”这个任务本身。你完全可以保留 FRM/FFM，用它来做 RGB-X 图像恢复。

## 5. 编码器部分

### `models/encoders/dual_segformer.py`

这是基于 SegFormer/MiT 的双分支编码器。

主要结构：

- 一条 RGB 分支：`patch_embed1-4`、`block1-4`、`norm1-4`。
- 一条 X 模态分支：`extra_patch_embed1-4`、`extra_block1-4`、`extra_norm1-4`。
- 每个 stage 后做：
  - FRM：两个模态互相修正。
  - FFM：融合成一个特征。
- 最后返回 4 个尺度的融合特征。

关键类：

- `DWConv`：深度卷积，用在 MLP 中引入局部空间信息。
- `Mlp`：Transformer block 里的前馈网络。
- `Attention`：MiT 的高效 self-attention，支持 `sr_ratio` 降低计算量。
- `Block`：一个 Transformer block。
- `OverlapPatchEmbed`：重叠 patch embedding，把图像变成 token。
- `RGBXTransformer`：完整双分支 MiT 编码器。
- `mit_b0` 到 `mit_b5`：不同规模的 MiT 配置。
- `load_dualpath_model`：把单模态预训练权重复制到 RGB 分支和 X 分支。

输出特征大致是：

```text
c1: 1/4  分辨率，浅层纹理和边缘多
c2: 1/8  分辨率，中层结构
c3: 1/16 分辨率，语义更强
c4: 1/32 分辨率，全局上下文更强
```

对图像恢复来说，浅层高分辨率特征很重要，因为恢复需要纹理、边缘和细节。这个编码器可以保留，但解码器最好比分割 head 更重一些，更像图像重建网络。

### `models/encoders/dual_swin.py`

这是基于 Swin Transformer 的双分支编码器。

主要结构与 `dual_segformer.py` 类似，也是 RGB 分支 + X 分支 + 多尺度融合。

关键类：

- `Mlp`：Swin block 里的前馈网络。
- `window_partition` / `window_reverse`：把特征图切成窗口，再拼回去。
- `WindowAttention`：窗口内 self-attention。
- `SwinTransformerBlock`：Swin 的窗口注意力 block，支持 shifted window。
- `PatchMerging`：下采样，空间尺寸减半、通道增加。
- `BasicLayer`：多个 Swin block 组成一个 stage。
- `PatchEmbed`：把图像切成 patch。
- `DualSwinTransformer`：完整双分支 Swin 编码器。
- `swin_s`、`swin_b`：small/base 两个配置。
- `load_dualpath_model`：把单模态 Swin 权重复制到双分支。

对图像恢复来说，Swin 编码器也可以保留。但需要注意，恢复任务通常更敏感于空间细节，过多下采样会损失细节，所以 decoder 设计要能补回高频信息。

## 6. 解码器部分

### `models/decoders/MLPDecoder.py`

SegFormer 风格的轻量分割解码器。

流程：

1. 对四个尺度特征分别用 MLP 投影到同一维度。
2. 把低分辨率特征上采样到最高分辨率特征 `c1` 的大小。
3. 拼接四个尺度。
4. 用 `1x1 conv + BN + ReLU` 融合。
5. 用 `linear_pred` 输出 `num_classes` 通道。

它很适合分割，因为只需要输出类别 logits。但对图像恢复来说可能偏轻，输出细节能力不足。

如果想快速改恢复，可以把 `linear_pred` 输出从 `num_classes` 改成 `3`。但更推荐新增恢复 decoder，使用卷积、残差块、跳连和逐级上采样。

### `models/decoders/UPernet.py`

UPerNet 分割头。

它结合了：

- PSP/PPM：在最高层特征上做多尺度池化，获取全局上下文。
- FPN：自顶向下融合多尺度特征。
- `conv_seg`：输出类别 logits。

优点是多尺度能力比 MLPDecoder 强。对恢复任务可以借鉴 FPN 的多尺度融合思想，但最后输出要改成图像，并且通常要增加更细致的重建模块。

### `models/decoders/fcnhead.py`

最简单的 FCN 分割头。

结构是：

```text
Conv -> BN -> ReLU -> 1x1 Conv 输出 num_classes
```

它也被 `UPernet` 和 `deeplabv3+` 当作辅助监督头使用。

对恢复任务来说，这个文件可参考但不够用。恢复输出需要 3 通道连续值，不是类别 logits。

### `models/decoders/deeplabv3plus.py`

DeepLabV3+ 分割头。

主要模块：

- `ASPP`：多尺度空洞卷积，扩大感受野。
- `low_level`：处理浅层特征。
- `block`：拼接 ASPP 高层特征和浅层特征后分类。

它适合语义分割，不是专门为图像恢复设计的。对恢复任务可以借鉴 ASPP 的多尺度上下文，但最后仍需替换成重建输出。

## 7. 训练引擎部分

### `engine/engine.py`

训练过程管理器。核心类：

- `State`：保存 epoch、iteration、dataloader、model、optimizer。
- `Engine`：处理参数、设备、分布式、checkpoint 保存和恢复。

主要功能：

- 自动注入命令行参数：
  - `--devices`
  - `--continue`
  - `--local_rank`
  - `--port`
- 判断是否分布式训练。
- 保存 checkpoint。
- 恢复 checkpoint。
- 创建 TensorBoard 软链接。

对图像恢复来说，这个文件大概率不用大改。它和具体任务关系不大。

注意：`link_file` 底层使用 Linux 的 `rm -rf` 和 `ln -s`，在 Windows PowerShell 环境下可能不工作。你当前工作目录在 Windows，如果以后运行训练日志链接出问题，这里是优先检查点。

### `engine/evaluator.py`

评估基类。它提供：

- 单进程评估。
- 多进程多 GPU 评估。
- 单 RGB 分割滑窗评估。
- RGB-X 分割滑窗评估。
- 图像归一化和 padding。
- flip test。

里面和 RGB-X 相关的关键方法：

- `sliding_eval_rgbX`
- `scale_process_rgbX`
- `val_func_process_rgbX`
- `process_image_rgbX`

这些方法默认模型输出是 `class_num` 个类别通道，然后用 `argmax` 得到类别图。

对图像恢复来说，应重写一套恢复评估逻辑：

- 不要 `argmax`。
- 滑窗拼接时要对重叠区域做平均，而不是简单相加。
- 输出通道是 3。
- 需要反归一化并 clamp 到 `[0, 1]` 或 `[0, 255]`。
- 指标用 PSNR/SSIM。

这里还有一个小 typo：`process_image_rgbX` 里有 `amodal_xis=2`，应该是 `axis=2`。这段只有在 RGB 输入通道少于 3 时会触发。

### `engine/dist_test.py`

另一个评估基类，像是早期单 RGB 分割评估代码。它没有 RGB-X 的完整逻辑，当前主评估入口用的是 `engine/evaluator.py`。

如果你改图像恢复，可以优先忽略这个文件，除非你想整理历史代码。

### `engine/logger.py`

日志工具。提供 `get_logger`，用于在终端和文件中输出日志。

它会设置日志格式、颜色和等级。模型、训练、评估代码里都通过它打印信息。

### `engine/__init__.py`

空文件，用来让 `engine` 成为 Python package。

## 8. 工具函数部分

### `utils/transforms.py`

图像预处理和增强工具。

常用函数：

- `get_2dshape`：把输入尺寸规范成 `(h, w)`。
- `random_crop_pad_to_shape`：随机裁剪后不足尺寸则 padding。
- `generate_random_crop_pos`：生成随机裁剪坐标。
- `pad_image_to_shape`：padding 到指定尺寸。
- `pad_image_size_to_multiples_of`：padding 到某个倍数。
- `resize_ensure_shortest_edge`：保证短边长度。
- `random_scale`：随机缩放图像和标签。
- `random_scale_rgbx`：随机缩放 RGB、标签、X。
- `random_mirror`：随机水平翻转。
- `random_rotation`：随机旋转。
- `random_gaussian_blur`：随机高斯模糊。
- `center_crop`：中心裁剪。
- `random_crop`：随机裁剪。
- `normalize`：把图像从 `[0,255]` 转到 `[0,1]`，再做 mean/std 归一化。

对图像恢复来说，这个文件可以继续用，但要注意：

- 标签不再是类别图，而是图像，所以 resize 插值方式要改。
- padding 方式可能要从 constant 改成 reflect，更适合图像恢复。
- `normalize` 对输入和目标是否都使用，需要统一设计。很多恢复任务直接用 `[0,1]`，不使用 ImageNet mean/std。

### `utils/pyt_utils.py`

PyTorch 工程工具集合。

主要功能：

- `reduce_tensor` / `all_reduce_tensor`：分布式训练中同步 loss。
- `load_restore_model` / `load_model`：读取 checkpoint。
- `parse_devices`：解析 GPU 参数。
- `extant_file`：argparse 检查文件是否存在。
- `link_file`：创建软链接。
- `ensure_dir`：确保目录存在。
- `get_logger`：这里也有一份日志函数，和 `engine/logger.py` 有重复。

对图像恢复来说，一般不用改。需要注意的是 `link_file` 使用 Linux 命令，在 Windows 上可能失败。

### `utils/load_utils.py`

加载预训练权重的工具。

主要函数：

- `get_dist_info`：获取分布式 rank/world size。
- `load_state_dict`：更宽松地加载权重，并打印 missing/unexpected keys。
- `load_pretrain`：从 checkpoint 中取出 `state_dict` 或 `model`，去掉 `module.` 前缀后加载。

对图像恢复来说，如果你保留原 backbone 并换 decoder，这个文件很有用。你可以只加载 backbone 预训练权重，让新 decoder 随机初始化。

### `utils/init_func.py`

权重初始化和优化器参数分组。

主要函数：

- `init_weight`：初始化卷积和归一化层。
- `group_weight`：把模型参数分成两组：
  - 卷积/线性权重使用 weight decay。
  - bias 和 norm 参数不使用 weight decay。

对图像恢复来说可以继续用。

### `utils/lr_policy.py`

学习率调度策略。

包含：

- `PolyLR`：多项式衰减。
- `WarmUpPolyLR`：先 warmup，再多项式衰减。
- `MultiStageLR`：分阶段学习率。
- `LinearIncreaseLR`：线性增大学习率。

当前训练使用 `WarmUpPolyLR`。图像恢复任务也能用，但很多恢复论文会使用 cosine decay、step decay 或固定学习率，可按实验习惯调整。

### `utils/loss_opr.py`

分割和其他任务的损失函数集合。

包含：

- `FocalLoss2d`
- `RCELoss`
- `BalanceLoss`
- `berHuLoss`
- `SigmoidFocalLoss`
- `ProbOhemCrossEntropy2d`

大多数是分类/分割损失。`berHuLoss` 更像深度估计回归损失，和图像恢复更接近一点，但普通恢复任务更常用：

- `L1Loss`
- `MSELoss`
- `CharbonnierLoss`
- `SSIMLoss`
- `PerceptualLoss`
- `EdgeLoss`

如果改恢复，可以在这个文件里新增恢复损失。

### `utils/metric.py`

分割指标。

主要函数：

- `hist_info`：构建混淆矩阵。
- `compute_score`：计算 IoU、mean IoU、frequency weighted IoU、mean pixel accuracy、pixel accuracy。

对图像恢复来说，这个文件要新增或替换成：

- `calc_psnr`
- `calc_ssim`
- `calc_lpips`
- `calc_mae`

### `utils/visualize.py`

分割可视化工具。

主要功能：

- 给类别图上色。
- 把原图、预测图、GT 拼在一起显示。
- 打印每个类别 IoU。

对图像恢复来说，应改成：

- 保存恢复图。
- 拼接 degraded / restored / target。
- 可选保存 error map。

### `utils/__init__.py`

空文件，用来让 `utils` 成为 Python package。

## 9. 这个项目中真正“任务相关”的部分

如果只看“什么地方绑定了语义分割”，主要是这些：

```text
config.py
  num_classes, class_names, background, gt_transform

dataloader/RGBXDataset.py
  标签按灰度类别图读取，gt - 1，label 转 long

dataloader/dataloader.py
  gt 用最近邻缩放，padding 填 255

models/builder.py
  decoder 输出 num_classes，criterion 使用 CrossEntropyLoss

models/decoders/*
  最后一层都是分类头

train.py
  CrossEntropyLoss，gts.long()

eval.py
engine/evaluator.py
utils/metric.py
utils/visualize.py
  mIoU、argmax、类别图、palette、分割可视化
```

而相对“可复用”的部分是：

```text
models/encoders/dual_segformer.py
models/encoders/dual_swin.py
models/net_utils.py
engine/engine.py
utils/init_func.py
utils/load_utils.py
utils/lr_policy.py
部分 transforms 工具
```

也就是说，CMX 的双分支跨模态特征融合思想可以保留，但分割任务外壳需要换掉。

## 10. 改造成图像恢复项目的建议路线

### 路线 A：最小改造，先跑通

目标是尽快把代码从分割训练改成恢复训练。

需要做：

1. 改 `RGBXDataset`，让 `label` 读取清晰 RGB 图像，返回 float tensor。
2. 改 `TrainPre`，让目标图像和输入图像做同样裁剪、翻转、缩放。
3. 改 `MLPDecoder` 最后一层输出 3 通道。
4. 改 `EncoderDecoder.forward`，使用 `L1Loss(out, target)`。
5. 改 `train.py`，criterion 换成 `nn.L1Loss()`。
6. 暂时不跑原 `eval.py`，先写一个简单验证脚本保存恢复图。

优点是最快能跑。缺点是恢复效果可能一般，因为 MLPDecoder 偏分割，不擅长重建细节。

### 路线 B：保留 CMX backbone，新增恢复 decoder

这是更推荐的研究路线。

保留：

- 双分支 MiT/Swin encoder。
- FRM/FFM 跨模态融合。
- 预训练权重加载。
- 训练 engine。

替换：

- 数据集读取方式。
- decoder。
- loss。
- eval。

新的恢复 decoder 可以像这样：

```text
c4 -> 上采样 -> 融合 c3
   -> 上采样 -> 融合 c2
   -> 上采样 -> 融合 c1
   -> 卷积细化
   -> 上采样到原图
   -> 输出 3 通道 restored image
```

可以考虑加残差学习：

```text
restored = degraded_rgb + residual
```

这对去噪、去雨、去模糊、增强等任务常常更稳定。

### 路线 C：把它改成完整多模态恢复框架

如果你的研究目标是 RGB-Polar 图像恢复，可以设计成：

```text
RGB degraded image
Polar / AoLP / DoLP / S0-S1-S2 image
        |
        v
CMX-style fusion encoder
        |
        v
restoration decoder
        |
        v
clean RGB image
```

可研究点包括：

- 不同偏振表示作为 X 模态的效果。
- FRM/FFM 在恢复任务中的作用。
- 在浅层还是深层融合更重要。
- 是否需要保留 RGB 分支与 X 分支对称结构。
- 是否用 residual output。
- 是否加入边缘损失或频域损失。

## 11. 最小改造时的关键代码方向

### 数据输出应变成这样

当前：

```python
output_dict = dict(data=rgb, label=gt, modal_x=x, fn=str(item_name), n=len(self._file_names))
```

图像恢复可以继续用这个结构，但语义变成：

```text
data: 退化 RGB 图
label: 清晰 RGB 目标图
modal_x: 辅助模态
```

这样 `train.py` 改动少。

### 模型输出应变成这样

当前：

```text
out: B x num_classes x H x W
label: B x H x W
loss: CrossEntropyLoss(out, label.long())
```

恢复：

```text
out: B x 3 x H x W
target: B x 3 x H x W
loss: L1Loss(out, target)
```

### 评估应变成这样

当前：

```text
pred = logits.argmax(channel)
mIoU(pred, label)
```

恢复：

```text
restored = model(input, modal_x)
PSNR(restored, target)
SSIM(restored, target)
保存 restored.png
```

## 12. 当前代码中几个值得注意的小问题

这些不一定影响你理解项目，但以后运行时可能遇到：

- `README.md` 写 `configs.py`，实际是 `config.py`。
- `eval.py` 保存彩色图时缺少 `PIL.Image` 和 `get_class_colors` 的导入。
- `engine/evaluator.py` 中 `amodal_xis=2` 应该是 `axis=2`。
- `utils/pyt_utils.py` 的 `link_file` 用 `rm -rf` 和 `ln -s`，Windows 下可能失败。
- `train.py` 使用 `dataloader.next()`，更稳妥写法是 `next(dataloader)`。
- `engine/evaluator.py` 的滑窗融合里有 `count_scale`，但最后没有除以它，当前是累加 score。分割时影响可能有限，恢复任务滑窗必须认真做重叠平均。

## 13. 一句话总结

这个项目的精华不是“语义分割 head”，而是 **RGB-X 双分支 Transformer + FRM/FFM 跨模态融合**。如果要改成图像恢复，建议保留 `models/encoders` 和 `models/net_utils` 的跨模态特征提取能力，重点替换 `dataloader`、`decoder`、`loss`、`eval metric` 和 `visualize`。

