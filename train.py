import os.path as osp
import os
import sys
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel

from config import config
from dataloader.dataloader import get_train_loader
from models.builder import EncoderDecoder as restoration_model
from dataloader.RGBXDataset import RGBXDataset
from utils.init_func import group_weight
from utils.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
logger = get_logger()

os.environ['MASTER_PORT'] = '169710'

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    # 固定随机种子，保证同一份配置下训练过程尽量可复现。
    # 分布式训练时每个 rank 使用不同 seed，避免所有进程采样完全一致。
    cudnn.benchmark = True
    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # data loader
    # 当前 DataLoader 输出三个核心张量：
    #   image_rgb:          退化 RGB 输入，[B, 3, H, W]
    #   polarization_input: 偏振输入，[B, 9, H, W]
    #   clean_target:       清晰监督图，[B, 3, H, W]
    train_loader, train_sampler = get_train_loader(engine, RGBXDataset)

    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        # TensorBoard 日志目录由 config.log_dir 派生。
        # engine.link_tb 会维护一个 tb 软链接/快捷入口，方便查看最新日志。
        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + '/tb'
        tb = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)

    # 当前训练闭环：
    #   image_rgb + polarization_input -> model -> restored_rgb
    #   restored_rgb 和 clean_target 直接计算 L1Loss。
    # 这里不再使用 CrossEntropyLoss，因为 clean_target 是 RGB 图像，不是类别 mask。
    criterion = nn.L1Loss(reduction='mean')

    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d

    # 模型接口固定为 model(rgb, x)：
    #   rgb -> image_rgb
    #   x   -> polarization_input
    # 模型只输出 restored_rgb，不在模型内部计算 loss。
    model = restoration_model(cfg=config, norm_layer=BatchNorm2d)

    # 按照原项目的参数分组方式设置 optimizer：
    #   Conv/Linear 权重使用 weight_decay
    #   Norm/bias 不使用 weight_decay
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr

    params_list = []
    params_list = group_weight(params_list, model, BatchNorm2d, base_lr)

    if config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
    elif config.optimizer == 'SGDM':
        optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError

    # 学习率策略沿用原项目 WarmUpPolyLR。
    # total_iteration = 总 epoch 数 * 每个 epoch 的 iteration 数。
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(
        base_lr,
        config.lr_power,
        total_iteration,
        config.niters_per_epoch * config.warm_up_epoch
    )

    # 设备放置：
    #   分布式：模型包进 DistributedDataParallel
    #   单进程：放到 cuda 或 cpu
    if engine.distributed:
        logger.info('.............distributed training.............')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model, device_ids=[engine.local_rank],
                                            output_device=engine.local_rank, find_unused_parameters=False)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    # 开始训练。当前任务的训练目标非常直接：
    #   pred = model(image_rgb, polarization_input)
    #   loss = L1Loss(pred, clean_target)
    optimizer.zero_grad()
    model.train()
    logger.info('begin trainning:')

    for epoch in range(engine.state.epoch, config.nepochs+1):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)

        sum_loss = 0

        for idx in pbar:
            engine.update_iteration(epoch, idx)

            minibatch = next(dataloader)
            image_rgb = minibatch['image_rgb']
            polarization_input = minibatch['polarization_input']
            clean_target = minibatch['clean_target']

            # 将三路张量移动到当前训练设备。
            # clean_target 已经是 float 图像张量，不能再做 long()。
            if engine.distributed:
                image_rgb = image_rgb.cuda(non_blocking=True)
                polarization_input = polarization_input.cuda(non_blocking=True)
                clean_target = clean_target.cuda(non_blocking=True)
            else:
                image_rgb = image_rgb.to(device, non_blocking=True)
                polarization_input = polarization_input.to(device, non_blocking=True)
                clean_target = clean_target.to(device, non_blocking=True)

            # model 只负责完成 RGB+偏振 -> 3 通道恢复图；
            # loss 在 train.py 中显式计算，训练目标更清晰。
            restored_rgb = model(image_rgb, polarization_input)
            loss = criterion(restored_rgb, clean_target)

            # 多 GPU 时，把各进程 loss 做一次 all-reduce，用于日志显示。
            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_idx = (epoch- 1) * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            # 每个 iteration 后更新 optimizer 的学习率。
            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

            if engine.distributed:
                sum_loss += reduce_loss.item()
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                        + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.4e' % lr \
                        + ' loss=%.4f total_loss=%.4f' % (reduce_loss.item(), (sum_loss / (idx + 1)))
            else:
                loss_value = loss.item()
                sum_loss += loss_value
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                        + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.4e' % lr \
                        + ' loss=%.4f total_loss=%.4f' % (loss_value, (sum_loss / (idx + 1)))

            del loss
            del restored_rgb
            pbar.set_description(print_str, refresh=False)

        if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
            # TensorBoard 中记录每个 epoch 的平均 L1Loss。
            tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)

        if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (epoch == config.nepochs):
            # 按 config 中的 checkpoint 策略保存模型和 optimizer 状态。
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                config.log_dir,
                                                config.log_dir_link)
