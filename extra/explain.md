"""RGB-Polar 图像恢复项目说明书。

这份文件不是训练入口，而是项目的“总控说明书”。
读者可以从这里快速理解：
    1. 当前项目要解决什么任务；
    2. 数据如何进入模型；
    3. 模型如何把 RGB + 偏振信息变成恢复图；
    4. loss 如何形成训练闭环；
    5. 想调控实验时，应该优先改哪些配置和文件。

运行方式：
    python explain.py

运行后会把这份说明打印到终端。直接打开本文件阅读也可以。
"""

PROJECT_MANUAL = """
一、当前项目的定位
==================

这个仓库原本是 RGB-X Semantic Segmentation 项目，核心价值在于双分支 backbone
和跨模态融合模块。现在项目已经被改成 RGB-Polar 图像恢复任务：

    degraded RGB input + polarization input -> restored RGB image -> L1Loss(clean target)

也就是说，当前目标不是输出语义类别 mask，而是输出一张 3 通道 RGB 恢复图。

当前训练闭环是：

    image_rgb [B, 3, H, W]
        +
    polarization_input [B, 9, H, W]
        |
        v
    model(rgb, x)
        |
        v
    restored_rgb [B, 3, H, W]
        |
        v
    L1Loss(restored_rgb, clean_target)

最重要的一点：
    模型接口保持原项目风格 model(rgb, x)。
    x 是完整的 polarization_input，不要改成 model(i0, i60, i120)。


二、数据格式
============

当前数据集入口：

    datasets/MyRGBP_by_scene/metadata.csv

每个有效训练样本来自一个 scene 下的一个 sample。
metadata.csv 中 sample == GT 的行只表示清晰图位置，不作为退化输入样本。

一条样本包含三部分：

    1. image_rgb
       路径：
           scene/sample/RGB/I.jpg
       含义：
           当前退化 RGB 输入。
       shape:
           H x W x 3

    2. polarization_input
       路径：
           scene/sample/Polar/0.jpg
           scene/sample/Polar/60.jpg
           scene/sample/Polar/120.jpg
       含义：
           三个偏振角的 RGB 图像拼接而成。
       shape:
           H x W x 9
       通道顺序：
           [0_R, 0_G, 0_B, 60_R, 60_G, 60_B, 120_R, 120_G, 120_B]

    3. clean_target
       路径：
           scene/GT/RGB/I.jpg
       含义：
           同一个 scene 的清晰 RGB 监督目标。
       shape:
           H x W x 3

对应代码：

    dataloader/RGBXDataset.py

这个文件负责：
    1. 读取 metadata.csv；
    2. 过滤 sample == GT 的行；
    3. 读取 image_rgb、polarization_input、clean_target；
    4. 按 split_name 返回 train、val、test 对应的数据。


三、训练集、验证集、测试集划分
==============================

当前使用 scene 级划分，而不是 sample 级划分。

原因：
    同一个 scene 下的多个 sample 共享同一张 clean_target。
    如果同一个 scene 的 sample 同时出现在 train 和 val/test，
    验证或测试就会间接见过同一个清晰目标，结果会偏乐观。

当前流程：

    1. 从 metadata.csv 中收集全部 scene；
    2. 按 scene_1、scene_2、scene_10 这种自然顺序排序；
    3. 使用 config.split_seed 做确定性随机打乱；
    4. 按 config.split_ratios 切分为 train/val/test。

默认配置：

    config.py

    C.split_strategy = 'scene'
    C.split_ratios = {'train': 0.8, 'val': 0.1, 'test': 0.1}
    C.split_seed = 12345

调控方式：

    想换一套随机划分：
        改 C.split_seed。

    想改比例：
        改 C.split_ratios。
        三个比例必须加起来等于 1.0。

    想避免数据泄漏：
        不要改回 sample 级随机划分。
        保持 scene 级划分。

注意：
    37 个 scene 不能被 80/10/10 精确整除。
    代码使用最大余数法分配 scene 数，使比例尽量接近配置值。


四、数据增强与预处理
====================

对应代码：

    dataloader/dataloader.py

训练阶段使用 TrainPre。
它同时处理三路数据：

    degraded_input
    polarization_input
    clean_target

必须保证三路图像使用完全一致的几何变换：

    1. 同一个 flip 决策；
    2. 同一个 scale；
    3. 同一个 crop 位置；
    4. 同一个 resize/crop 后尺寸。

原因：
    当前 loss 是逐像素 L1Loss。
    只要 input 和 clean_target 在几何位置上错开，loss 就会拿错位像素相减。

当前训练预处理流程：

    1. random_mirror
       同步水平翻转 degraded RGB、polarization input、clean target。

    2. random_scale
       同步缩放三路图像。
       clean_target 是 RGB 图像，不是类别 mask，因此三路都用双线性插值。

    3. normalize
       image_rgb:
           使用 config.norm_mean / config.norm_std 标准化。
       polarization_input:
           9 通道，使用 RGB mean/std 重复 3 次后标准化。
       clean_target:
           只缩放到 [0, 1]，不做 mean/std 标准化。

    4. random crop/pad
       三路数据共用同一个 crop_pos。

    5. HWC -> CHW
       适配 PyTorch Conv2d 输入格式。

关键配置：

    C.image_height = 480
    C.image_width = 640
    C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
    C.norm_mean = np.array([0.485, 0.456, 0.406])
    C.norm_std = np.array([0.229, 0.224, 0.225])

调控方式：

    想减小显存：
        降低 C.image_height / C.image_width。

    想关闭复杂增强做小样本过拟合：
        把 C.train_scale_array 改成 [1] 或 None。
        保留 resize/crop 即可。

    想改变输入标准化：
        改 C.norm_mean / C.norm_std。
        注意 polarization_input 会自动重复这三个 RGB 统计量。


五、模型结构
============

对应代码：

    models/builder.py
    models/decoders/restoration_head.py

当前模型已经固定为 RGB-Polar restoration，不再保留原项目根据任务选择分支的结构。

整体结构：

    RGB branch + Polar branch
        |
        v
    dual SegFormer-B2 backbone
        |
        v
    CM-FRM / FFM 跨模态融合
        |
        v
    RestorationHead
        |
        v
    3 通道恢复图

输入接口：

    model(rgb, x)

其中：

    rgb:
        image_rgb，[B, 3, H, W]

    x:
        polarization_input，[B, 9, H, W]

当前 builder.py 内部有一个 x_input_adapter：

    [B, 9, H, W] -> [B, 3, H, W]

原因：
    继承来的 RGB-X backbone 第二分支第一层目前仍期望 3 通道输入。
    数据层已经把偏振输入做成 9 通道，所以模型内部用 1x1 Conv 先适配到 3 通道。

如果以后想让 backbone 原生吃 9 通道：
    应该改 encoders/dual_segformer.py 里第二分支的 patch embedding 输入通道，
    并重新检查预训练权重加载逻辑。
    那时可以移除或改造 x_input_adapter。

当前 RestorationHead：

    1. 接收 backbone 的四层多尺度融合特征；
    2. 每层投影到统一 embed_dim；
    3. 上采样到 c1 的 1/4 尺度；
    4. concat 后用卷积融合；
    5. 输出 3 通道图像；
    6. sigmoid 约束到 [0, 1]；
    7. builder.py 再上采样回输入图尺寸。

关键配置：

    C.model_name = 'RGBPolarRestoration_MiT-B2_RestorationHead'
    C.pretrained_model = C.root_dir + '/pretrained/segformer/mit_b2.pth'
    C.x_input_channels = 9
    C.restoration_head_embed_dim = 512
    C.output_channels = 3

调控方式：

    想增大恢复头容量：
        增大 C.restoration_head_embed_dim。

    想减小显存和计算：
        降低 C.restoration_head_embed_dim。

    想换 backbone：
        当前 builder.py 已经固定为 dual SegFormer-B2。
        如果要换，需要主动修改 builder.py，而不是只改 config。


六、loss 与训练闭环
===================

对应代码：

    train.py

当前 loss：

    criterion = nn.L1Loss(reduction='mean')

训练逻辑：

    restored_rgb = model(image_rgb, polarization_input)
    loss = criterion(restored_rgb, clean_target)

为什么不用 CrossEntropyLoss：
    clean_target 是 RGB 清晰图，不是类别 mask。
    CrossEntropyLoss 是分类损失，不适合当前恢复任务。

为什么 clean_target 是 [0, 1]：
    dataloader 中 clean_target = clean_target / 255.0。
    RestorationHead 输出经过 sigmoid，也在 [0, 1]。
    两者数值范围一致，L1Loss 的含义就是平均像素绝对误差。

调控方式：

    想让图像更平滑：
        可以在 L1Loss 外再加入 SSIM loss、Charbonnier loss 或 perceptual loss。

    想先确认闭环是否有效：
        用小样本过拟合测试。
        例如只取 5 到 10 张图，关闭复杂增强，训练几百到几千 iteration。

    小样本过拟合时观察：
        1. loss 是否稳定下降；
        2. pred 是否逐渐接近 clean_target；
        3. 输出图是否有通道错乱、尺寸错位、颜色反转、归一化错误。


七、训练入口与日志
==================

训练入口：

    train.py

常规启动方式仍沿用原项目 Engine 参数。
如果只是单机单卡，一般从下面形式开始：

    python train.py

如果原项目环境要求指定 GPU 或分布式参数，以 engine/engine.py 支持的参数为准。

日志和权重目录由 config.py 控制：

    C.log_dir
    C.tb_dir
    C.checkpoint_dir
    C.checkpoint_start_epoch
    C.checkpoint_step

TensorBoard 中目前记录：

    train_loss

当前 checkpoint 逻辑：

    epoch >= C.checkpoint_start_epoch 且 epoch % C.checkpoint_step == 0 时保存；
    最后一个 epoch 也会保存。

调控方式：

    想更早保存模型：
        降低 C.checkpoint_start_epoch。

    想更频繁保存：
        降低 C.checkpoint_step。

    想缩短实验：
        降低 C.nepochs。

    想调学习率：
        改 C.lr。

    想改 batch size：
        改 C.batch_size。
        注意显存不够时优先降低 image_height/image_width 或 batch_size。


八、验证与测试现状
==================

重要现状：
    当前 eval.py 仍然偏向原语义分割流程。
    数据集已经支持 val/test split，但完整的恢复任务评估脚本还没有彻底重写。

这意味着：
    训练闭环已经是恢复任务；
    但如果要正式报告 PSNR、SSIM 或保存恢复图，需要新增或重写恢复评估代码。

建议后续增加：

    1. restoration_eval.py
       读取 val/test split；
       调用 model(image_rgb, polarization_input)；
       保存 restored_rgb；
       计算 L1、PSNR、SSIM。

    2. visualize_restore.py
       拼接 image_rgb、restored_rgb、clean_target；
       用于人工检查颜色、通道、尺寸、纹理恢复情况。

    3. overfit_debug.py
       固定极小样本集合；
       关闭复杂增强；
       每隔一定 iteration 保存 pred；
       用于确认训练闭环真的能拟合。


九、最常用的调控旋钮
====================

数据划分：

    config.py
        C.split_seed
        C.split_ratios

输入尺寸：

    config.py
        C.image_height
        C.image_width

增强强度：

    config.py
        C.train_scale_array

训练长度：

    config.py
        C.nepochs
        C.batch_size

学习率：

    config.py
        C.lr
        C.lr_power
        C.warm_up_epoch

模型容量：

    config.py
        C.restoration_head_embed_dim

偏振输入通道：

    config.py
        C.x_input_channels = 9

    dataloader/RGBXDataset.py
        _open_polarization_input

    models/builder.py
        x_input_adapter

损失函数：

    train.py
        criterion = nn.L1Loss(reduction='mean')

输出头：

    models/decoders/restoration_head.py


十、修改项目时的优先级建议
==========================

第一优先级：先保证闭环正确。

    输入 shape 正确：
        image_rgb          [B, 3, H, W]
        polarization_input [B, 9, H, W]
        clean_target       [B, 3, H, W]

    输出 shape 正确：
        restored_rgb       [B, 3, H, W]

    数值范围正确：
        clean_target       [0, 1]
        restored_rgb       [0, 1]

    loss 正确：
        L1Loss(restored_rgb, clean_target)

第二优先级：做小样本过拟合。

    如果小样本都无法过拟合，优先排查：
        1. target 是否读错；
        2. RGB/BGR 是否错；
        3. polarization 9 通道顺序是否错；
        4. crop/flip/resize 是否不同步；
        5. pred 和 target 尺寸是否一致；
        6. pred/target 归一化范围是否一致。

第三优先级：再扩大训练。

    小样本闭环通了之后，再讨论：
        1. 更强增强；
        2. 更长训练；
        3. 更复杂 loss；
        4. 更大模型；
        5. 正式 val/test 指标。


十一、快速定位问题
==================

如果 loss 不下降：
    先关掉随机 scale，只保留固定 resize/crop。
    再用极小样本过拟合。

如果输出颜色奇怪：
    检查 cv2 读图后是否 BGR -> RGB。
    当前 RGBXDataset._open_image 已经显式转换 RGB。

如果输出尺寸不一致：
    检查 builder.py 中 F.interpolate 是否上采样回 rgb.shape[2:]。

如果报 x 通道数错误：
    检查 config.x_input_channels 是否为 9。
    检查 Polar/0.jpg、60.jpg、120.jpg 是否都按 RGB 三通道读取并 concat。

如果验证结果异常乐观：
    检查是否仍然按 scene 划分。
    不要让同一个 scene 同时出现在 train 和 val/test。

如果显存不够：
    优先降低 batch_size、image_height、image_width。
    其次降低 restoration_head_embed_dim。


十二、当前项目的一句话总览
==========================

当前项目保留原 RGB-X 跨模态融合骨架，把任务从语义分割改成了图像恢复：

    metadata.csv 组织样本
    seed 随机划分 scene
    dataloader 同步增强 RGB、偏振、清晰图
    model(rgb, x) 输出 3 通道恢复图
    train.py 用 L1Loss 和 clean_target 形成训练闭环

后续所有修改都应该围绕这个闭环展开。
"""


def main():
    print(PROJECT_MANUAL.strip())


if __name__ == "__main__":
    main()
