import os
import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config
from dataloader.RGBXDataset import RGBXDataset
from dataloader.dataloader import ValPre, normalize_polarization_input
from models.builder import EncoderDecoder as restoration_model
from utils.transforms import normalize
from utils.pyt_utils import ensure_dir
from engine.logger import get_logger

logger = get_logger()


def build_data_setting():
    """构建当前恢复任务的数据配置。

    前因：
        旧 eval.py 继承自语义分割项目，需要 rgb_root、gt_root、x_root、class_names 等字段。
        当前项目已经改成 metadata.csv 驱动的 RGB-Polar 恢复任务。

    后果：
        eval 只保留当前 dataset 真正需要的三个入口：
            metadata_source: 数据索引；
            split_seed:      scene 随机划分 seed；
            split_ratios:    train/val/test 比例。
    """
    return {
        'metadata_source': getattr(config, 'metadata_source', None),
        'split_seed': getattr(config, 'split_seed', None),
        'split_ratios': getattr(config, 'split_ratios', None)
    }


def resolve_checkpoint(epoch_arg):
    """把命令行传入的 epoch 标识解析成 checkpoint 路径。"""
    if os.path.isfile(epoch_arg):
        return epoch_arg

    if epoch_arg == 'last':
        ckpt_name = 'epoch-last.pth'
    elif epoch_arg.isdigit():
        ckpt_name = 'epoch-{}.pth'.format(epoch_arg)
    else:
        ckpt_name = epoch_arg

    ckpt_path = os.path.join(config.checkpoint_dir, ckpt_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError('Checkpoint not found: {}'.format(ckpt_path))
    return ckpt_path


def load_checkpoint(model, checkpoint_path, device):
    """加载训练保存的恢复模型权重。"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # 兼容 DataParallel / DistributedDataParallel 保存出的 module. 前缀。
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[7:]
        cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict, strict=True)
    logger.info('Loaded checkpoint: {}'.format(checkpoint_path))
    return model


def get_device(devices_arg):
    """选择评估设备。

    轻量 eval 不走原项目复杂的多卡 Evaluator。
    单卡或 CPU 足够完成 val/test 恢复评估，也更容易排查问题。
    """
    if torch.cuda.is_available() and devices_arg.lower() != 'cpu':
        first_device = devices_arg.split(',')[0]
        if first_device == '':
            first_device = '0'
        return torch.device('cuda:{}'.format(int(first_device)))
    return torch.device('cpu')


def to_chw_tensor(image):
    """HWC numpy 图像转成 [1, C, H, W] tensor。"""
    image = image.transpose(2, 0, 1)
    image = np.ascontiguousarray(image)
    return torch.from_numpy(image).unsqueeze(0).float()


def prepare_sample(sample, device):
    """把 RGBXDataset 返回的原始 numpy 样本转成模型输入。

    前因：
        train.py 中 TrainPre 会做 normalize、crop、HWC->CHW，并把结果转 tensor。
        eval 阶段不做随机增强，因此 RGBXDataset 返回的是原始 RGB numpy 图。

    后果：
        eval.py 必须在这里手动完成与训练一致的数值处理：
            image_rgb / polarization_input 使用 mean/std 标准化；
            clean_target 只缩放到 [0, 1]；
            三路都转为 [1, C, H, W]。
    """
    image_rgb = sample['image_rgb']
    polarization_input = sample['polarization_input']
    clean_target = sample['clean_target']

    if image_rgb.shape[:2] != clean_target.shape[:2]:
        raise ValueError(
            'image_rgb and clean_target size mismatch for {}: {} vs {}'.format(
                sample['fn'], image_rgb.shape[:2], clean_target.shape[:2]
            )
        )

    image_rgb = normalize(image_rgb, config.norm_mean, config.norm_std)
    polarization_input = normalize_polarization_input(
        polarization_input, config.norm_mean, config.norm_std
    )
    clean_target = clean_target.astype(np.float64) / 255.0

    image_rgb = to_chw_tensor(image_rgb).to(device, non_blocking=True)
    polarization_input = to_chw_tensor(polarization_input).to(device, non_blocking=True)
    clean_target = to_chw_tensor(clean_target).to(device, non_blocking=True)

    return image_rgb, polarization_input, clean_target


def tensor_to_rgb_uint8(tensor):
    """把 [1, 3, H, W] 或 [3, H, W] 的 [0, 1] tensor 转成 RGB uint8。"""
    if tensor.dim() == 4:
        tensor = tensor[0]
    image = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    image = (image * 255.0 + 0.5).astype(np.uint8)
    return image


def safe_image_name(name):
    """scene/sample 形式的名字转成安全文件名。"""
    return str(name).replace('\\', '_').replace('/', '_') + '.png'


def compute_psnr(pred, target):
    """计算单张图像 PSNR，输入范围为 [0, 1]。"""
    mse = F.mse_loss(pred, target).item()
    if mse <= 1e-12:
        return float('inf')
    return 10.0 * np.log10(1.0 / mse)


def get_window_starts(length, crop_size, stride):
    """生成一维滑窗起点，保证最后一个窗口覆盖到图像边界。"""
    if length <= crop_size:
        return [0]

    starts = list(range(0, length - crop_size + 1, stride))
    last_start = length - crop_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def sliding_window_inference(model, image_rgb, polarization_input):
    """单尺度滑窗恢复推理。

    前因：
        整图推理在测试图较大时容易占用过多显存。
        但当前项目还不需要多尺度和 flip test，这些增强会让评估逻辑变复杂。

    后果：
        eval 只保留一个清晰的滑窗路径：
            1. 按 eval_crop_size 裁 patch；
            2. patch 级调用 model(rgb, x)；
            3. 重叠区域累加后除以 count；
            4. 裁回原图大小。

        这样可以降低单次推理显存，同时保持实现轻量、可排查。
    """
    _, _, height, width = image_rgb.shape
    crop_h, crop_w = config.eval_crop_size
    stride_h = max(1, int(crop_h * config.eval_stride_rate))
    stride_w = max(1, int(crop_w * config.eval_stride_rate))

    pad_h = max(crop_h - height, 0)
    pad_w = max(crop_w - width, 0)
    if pad_h > 0 or pad_w > 0:
        # 只在右侧和下侧补零，避免改变原图左上角坐标系。
        image_rgb = F.pad(image_rgb, (0, pad_w, 0, pad_h), mode='constant', value=0)
        polarization_input = F.pad(
            polarization_input, (0, pad_w, 0, pad_h), mode='constant', value=0
        )

    _, _, padded_h, padded_w = image_rgb.shape
    y_starts = get_window_starts(padded_h, crop_h, stride_h)
    x_starts = get_window_starts(padded_w, crop_w, stride_w)

    pred_sum = image_rgb.new_zeros((1, config.output_channels, padded_h, padded_w))
    count_map = image_rgb.new_zeros((1, 1, padded_h, padded_w))

    for y in y_starts:
        for x in x_starts:
            rgb_crop = image_rgb[:, :, y:y + crop_h, x:x + crop_w]
            polar_crop = polarization_input[:, :, y:y + crop_h, x:x + crop_w]
            pred_crop = model(rgb_crop, polar_crop)
            pred_sum[:, :, y:y + crop_h, x:x + crop_w] += pred_crop
            count_map[:, :, y:y + crop_h, x:x + crop_w] += 1

    restored = pred_sum / count_map.clamp_min(1.0)
    restored = restored[:, :, :height, :width]
    return restored.clamp(0.0, 1.0)


def evaluate(model, dataset, device, save_path=None):
    """执行轻量滑窗恢复评估。

    当前 eval 的设计取舍：
        1. 使用单尺度滑窗，降低大图推理显存；
        2. 删除多尺度和 flip test，避免评估入口臃肿；
        3. 只计算 L1 和 PSNR 两个基础指标；
        4. save_path 不为空时保存 restored_rgb，便于人工检查颜色、尺寸和通道。
    """
    if save_path is not None:
        ensure_dir(save_path)

    model.eval()
    l1_values = []
    psnr_values = []

    for index in range(len(dataset)):
        sample = dataset[index]
        image_rgb, polarization_input, clean_target = prepare_sample(sample, device)

        with torch.no_grad():
            restored_rgb = sliding_window_inference(model, image_rgb, polarization_input)

        if restored_rgb.shape != clean_target.shape:
            raise ValueError(
                'Prediction and target size mismatch for {}: {} vs {}'.format(
                    sample['fn'], tuple(restored_rgb.shape), tuple(clean_target.shape)
                )
            )

        l1_value = F.l1_loss(restored_rgb, clean_target).item()
        psnr_value = compute_psnr(restored_rgb, clean_target)
        l1_values.append(l1_value)
        psnr_values.append(psnr_value)

        if save_path is not None:
            restored = tensor_to_rgb_uint8(restored_rgb)
            restored_bgr = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_path, safe_image_name(sample['fn'])), restored_bgr)

        logger.info(
            '[{}/{}] {}  L1={:.6f}  PSNR={:.3f}'.format(
                index + 1, len(dataset), sample['fn'], l1_value, psnr_value
            )
        )

    mean_l1 = float(np.mean(l1_values)) if l1_values else 0.0
    finite_psnr = [value for value in psnr_values if np.isfinite(value)]
    mean_psnr = float(np.mean(finite_psnr)) if finite_psnr else float('inf')

    return mean_l1, mean_psnr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str,
                        help='checkpoint epoch, "last", or a checkpoint path')
    parser.add_argument('-d', '--devices', default='0', type=str,
                        help='GPU id such as 0, or cpu')
    parser.add_argument('--split', default='val', choices=['val', 'test'],
                        help='evaluate val or test split')
    parser.add_argument('--save_path', '-p', default=None,
                        help='optional directory for saving restored RGB images')
    args = parser.parse_args()

    # 前因：
    #   原 eval.py 是分割评估入口，依赖 num_classes、mIoU、sliding_eval_rgbX 等旧逻辑。
    #   当前项目的核心任务已经是 RGB+偏振 -> RGB 恢复图。
    #
    # 后果：
    #   这里改为轻量滑窗恢复评估：加载 checkpoint，遍历 val/test，滑窗推理，计算 L1/PSNR，按需保存恢复图。
    #   当前只保留 eval_crop_size 和 eval_stride_rate；多尺度和 flip test 已删除，避免评估入口臃肿。
    device = get_device(args.devices)
    checkpoint_path = resolve_checkpoint(args.epochs)

    data_setting = build_data_setting()
    dataset = RGBXDataset(data_setting, args.split, ValPre())

    model = restoration_model(cfg=config, norm_layer=nn.BatchNorm2d)
    model = load_checkpoint(model, checkpoint_path, device)
    model.to(device)

    mean_l1, mean_psnr = evaluate(model, dataset, device, args.save_path)
    logger.info(
        'Eval split={} checkpoint={}  mean_L1={:.6f}  mean_PSNR={:.3f}'.format(
            args.split, checkpoint_path, mean_l1, mean_psnr
        )
    )