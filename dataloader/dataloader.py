import cv2
import torch
import numpy as np
from torch.utils import data
import random
from config import config
from utils.transforms import generate_random_crop_pos, random_crop_pad_to_shape, normalize

def random_mirror(degraded_input, polarization_input, clean_target):
    # 三路数据必须使用同一个随机决定。
    # 如果只翻转其中一路，输入和 clean target 的像素位置会错位。
    if random.random() >= 0.5:
        degraded_input = cv2.flip(degraded_input, 1)
        polarization_input = cv2.flip(polarization_input, 1)
        clean_target = cv2.flip(clean_target, 1)

    return degraded_input, polarization_input, clean_target

def normalize_polarization_input(polarization_input, mean, std):
    # RGB mean/std 只有 3 个值；新的 polarization_input 是 9 通道：
    #   [I0_R, I0_G, I0_B, I60_R, I60_G, I60_B, I120_R, I120_G, I120_B]
    # 因此把 RGB 的 mean/std 重复 3 次，保证 9 个通道都能按对应 RGB 统计归一化。
    if len(polarization_input.shape) == 3 and polarization_input.shape[2] != len(mean):
        repeat = polarization_input.shape[2] // len(mean)
        mean = np.tile(mean, repeat)
        std = np.tile(std, repeat)
    return normalize(polarization_input, mean, std)

def random_scale(degraded_input, polarization_input, clean_target, scales):
    # 同一个 scale 同时作用在 degraded input / polarization input / clean target 上。
    # restoration target 是图像，所以三路都使用双线性插值。
    scale = random.choice(scales)
    sh = int(degraded_input.shape[0] * scale)
    sw = int(degraded_input.shape[1] * scale)
    degraded_input = cv2.resize(degraded_input, (sw, sh), interpolation=cv2.INTER_LINEAR)
    polarization_input = cv2.resize(polarization_input, (sw, sh), interpolation=cv2.INTER_LINEAR)
    clean_target = cv2.resize(clean_target, (sw, sh), interpolation=cv2.INTER_LINEAR)

    return degraded_input, polarization_input, clean_target, scale

class TrainPre(object):
    def __init__(self, norm_mean, norm_std):
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def __call__(self, degraded_input, clean_target, polarization_input):
        # 输入顺序来自 RGBXDataset：
        #   degraded_input     -> 当前退化 RGB 图像
        #   clean_target       -> 同 scene 的 GT/RGB/I.jpg
        #   polarization_input -> 0/60/120 三张 RGB 偏振图拼接成的 9 通道图像
        #
        # 注意 random_mirror/random_scale 的内部参数顺序把 polarization_input
        # 放在 clean_target 前面，只是为了强调两个 input 分支先同步处理；
        # 返回给 dataset 时仍保持 degraded, clean, polarization 的顺序。
        degraded_input, polarization_input, clean_target = random_mirror(
            degraded_input, polarization_input, clean_target
        )
        if config.train_scale_array is not None:
            degraded_input, polarization_input, clean_target, scale = random_scale(
                degraded_input, polarization_input, clean_target, config.train_scale_array
            )

        # degraded input 和 polarization input 作为网络输入，使用 ImageNet 风格归一化。
        # clean target 是恢复目标图像，只缩放到 [0, 1]，不做 mean/std 标准化。
        degraded_input = normalize(degraded_input, self.norm_mean, self.norm_std)
        polarization_input = normalize_polarization_input(
            polarization_input, self.norm_mean, self.norm_std
        )
        clean_target = clean_target.astype(np.float64) / 255.0

        crop_size = (config.image_height, config.image_width)
        # 只生成一次 crop_pos，三路数据共用同一个左上角坐标。
        # 这是保证监督信号和输入像素严格对齐的关键。
        crop_pos = generate_random_crop_pos(degraded_input.shape[:2], crop_size)

        # 三路数据使用同一个 crop_size 和 crop_pos。
        # padding value 全部为 0，保持图像空白区域的默认值一致。
        p_degraded_input, _ = random_crop_pad_to_shape(
            degraded_input, crop_pos, crop_size, 0
        )
        p_polarization_input, _ = random_crop_pad_to_shape(
            polarization_input, crop_pos, crop_size, 0
        )
        p_clean_target, _ = random_crop_pad_to_shape(
            clean_target, crop_pos, crop_size, 0
        )

        # OpenCV / numpy 阶段使用 H x W x C；
        # PyTorch 网络输入使用 C x H x W。
        p_degraded_input = p_degraded_input.transpose(2, 0, 1)
        p_polarization_input = p_polarization_input.transpose(2, 0, 1)
        p_clean_target = p_clean_target.transpose(2, 0, 1)

        return p_degraded_input, p_clean_target, p_polarization_input

class ValPre(object):
    def __call__(self, degraded_input, clean_target, polarization_input):
        # 验证阶段不做随机增强，保持原始图像尺寸和内容。
        return degraded_input, clean_target, polarization_input

def get_train_loader(engine, dataset):
    # 当前 dataset 只依赖 metadata.csv。
    # 如果 config 里没有 metadata_source，RGBXDataset 会自动扫描 datasets 目录。
    data_setting = {'metadata_source': getattr(config, 'metadata_source', None)}
    train_preprocess = TrainPre(config.norm_mean, config.norm_std)

    train_dataset = dataset(data_setting, "train", train_preprocess, config.batch_size * config.niters_per_epoch)

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
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
