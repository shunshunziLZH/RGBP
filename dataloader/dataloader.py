import cv2
import torch
import numpy as np
from torch.utils import data
import random
from config import config
from utils.transforms import generate_random_crop_pos, random_crop_pad_to_shape, normalize

def random_mirror(degraded_input, polarization_input, clean_target):
    """同步随机水平翻转三路图像。

    训练目标是逐像素恢复，所以 degraded_input、polarization_input、clean_target
    必须保持几何位置完全一致。只要其中一路单独翻转，L1Loss 就会拿错位像素相减。
    """
    if random.random() >= 0.5:
        degraded_input = cv2.flip(degraded_input, 1)
        polarization_input = cv2.flip(polarization_input, 1)
        clean_target = cv2.flip(clean_target, 1)

    return degraded_input, polarization_input, clean_target

def normalize_polarization_input(polarization_input, mean, std):
    """归一化 9 通道偏振输入。

    config.norm_mean / norm_std 是 RGB 的 3 通道统计量。
    polarization_input 是 0/60/120 三张 RGB 图拼接而成的 9 通道图，
    因此需要把 RGB mean/std 重复 3 次后再做 normalize。
    """
    # 9 通道排列约定：
    #   [I0_R, I0_G, I0_B, I60_R, I60_G, I60_B, I120_R, I120_G, I120_B]
    if len(polarization_input.shape) == 3 and polarization_input.shape[2] != len(mean):
        repeat = polarization_input.shape[2] // len(mean)
        mean = np.tile(mean, repeat)
        std = np.tile(std, repeat)
    return normalize(polarization_input, mean, std)

def random_scale(degraded_input, polarization_input, clean_target, scales):
    """同步随机缩放三路图像。

    clean_target 是 RGB 图像，不是离散类别图，所以三路都使用双线性插值。
    """
    scale = random.choice(scales)
    sh = int(degraded_input.shape[0] * scale)
    sw = int(degraded_input.shape[1] * scale)
    degraded_input = cv2.resize(degraded_input, (sw, sh), interpolation=cv2.INTER_LINEAR)
    polarization_input = cv2.resize(polarization_input, (sw, sh), interpolation=cv2.INTER_LINEAR)
    clean_target = cv2.resize(clean_target, (sw, sh), interpolation=cv2.INTER_LINEAR)

    return degraded_input, polarization_input, clean_target, scale

class TrainPre(object):
    """训练阶段的数据预处理。

    输入来自 RGBXDataset，均为 numpy 格式 H x W x C：
        degraded_input: H x W x 3
        clean_target: H x W x 3
        polarization_input: H x W x 9

    输出给 DataLoader / train.py，均为 C x H x W：
        p_degraded_input: [3, image_height, image_width]
        p_clean_target: [3, image_height, image_width]
        p_polarization_input: [9, image_height, image_width]
    """
    def __init__(self, norm_mean, norm_std):
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def __call__(self, degraded_input, clean_target, polarization_input):
        # 1. 同步随机翻转。
        # 注意 random_mirror 返回顺序是 degraded, polarization, clean；
        # 本函数最后返回给 dataset 的顺序仍是 degraded, clean, polarization。
        degraded_input, polarization_input, clean_target = random_mirror(
            degraded_input, polarization_input, clean_target
        )

        # 2. 同步随机缩放。
        # config.train_scale_array 为 None 时跳过缩放。
        if config.train_scale_array is not None:
            degraded_input, polarization_input, clean_target, scale = random_scale(
                degraded_input, polarization_input, clean_target, config.train_scale_array
            )

        # 3. 数值归一化。
        # 网络输入 image_rgb / polarization_input 使用 mean/std 标准化；
        # 监督目标 clean_target 只缩放到 [0, 1]，方便和 sigmoid 后的 pred 做 L1Loss。
        degraded_input = normalize(degraded_input, self.norm_mean, self.norm_std)
        polarization_input = normalize_polarization_input(
            polarization_input, self.norm_mean, self.norm_std
        )
        clean_target = clean_target.astype(np.float64) / 255.0

        crop_size = (config.image_height, config.image_width)
        # 4. 同步随机 crop/pad。
        # 只生成一次 crop_pos，三路数据共用同一个左上角坐标。
        crop_pos = generate_random_crop_pos(degraded_input.shape[:2], crop_size)

        # crop 区域不足时用 0 padding。三路 padding 策略一致，避免边界错位。
        p_degraded_input, _ = random_crop_pad_to_shape(
            degraded_input, crop_pos, crop_size, 0
        )
        p_polarization_input, _ = random_crop_pad_to_shape(
            polarization_input, crop_pos, crop_size, 0
        )
        p_clean_target, _ = random_crop_pad_to_shape(
            clean_target, crop_pos, crop_size, 0
        )

        # 5. HWC -> CHW，适配 PyTorch Conv2d 输入格式。
        p_degraded_input = p_degraded_input.transpose(2, 0, 1)
        p_polarization_input = p_polarization_input.transpose(2, 0, 1)
        p_clean_target = p_clean_target.transpose(2, 0, 1)

        return p_degraded_input, p_clean_target, p_polarization_input

class ValPre(object):
    def __call__(self, degraded_input, clean_target, polarization_input):
        # 验证/测试阶段不做随机增强，保持原始图像尺寸和内容。
        # 当前 eval.py 会在评估入口中完成 normalize、HWC->CHW 和 tensor 转换。
        return degraded_input, clean_target, polarization_input

def get_train_loader(engine, dataset):
    """构建训练 DataLoader。

    当前 dataset 需要三个配置：
        config.metadata_source -> datasets/MyRGBP_by_scene/metadata.csv
        config.split_seed      -> scene 随机打乱 seed
        config.split_ratios    -> train/val/test 划分比例

    DataLoader 输出 batch 后，train.py 会读取：
        minibatch['image_rgb']
        minibatch['polarization_input']
        minibatch['clean_target']
    """
    data_setting = {
        'metadata_source': getattr(config, 'metadata_source', None),
        # 前因：同一个 scene 的多个退化样本共享同一个 clean_target。
        # 后果：训练 loader 必须先用 seed 生成 scene 级 split，
        # 再只读取 train scene，不能默认读取全部 metadata。
        'split_seed': getattr(config, 'split_seed', None),
        'split_ratios': getattr(config, 'split_ratios', None)
    }
    train_preprocess = TrainPre(config.norm_mean, config.norm_std)

    # 先构造一次真实 train/val/test split，用长度回填 config。
    # 前因：split_seed 改变后，每个 split 中的 sample 数可能变化。
    # 后果：num_train_imgs / num_eval_imgs / num_test_imgs 不再硬编码，
    # 始终跟真实数据划分一致。
    train_dataset = dataset(data_setting, "train", train_preprocess)
    val_dataset = dataset(data_setting, "val")
    test_dataset = dataset(data_setting, "test")
    config.num_train_imgs = len(train_dataset)
    config.num_eval_imgs = len(val_dataset)
    config.num_test_imgs = len(test_dataset)
    config.niters_per_epoch = config.num_train_imgs // config.batch_size + 1

    # file_length 用来把真实样本重复扩展到一个 epoch 所需的 item 数。
    # 这样每个 epoch 的 iteration 数仍由 config.niters_per_epoch 稳定控制。
    train_dataset = dataset(
        data_setting,
        "train",
        train_preprocess,
        config.batch_size * config.niters_per_epoch
    )

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        # 分布式训练时，每个进程拿到数据集的不同切片。
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler)

    return train_loader, train_sampler
