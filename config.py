import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C

C.seed = 12345

remoteip = os.popen('pwd').read()
C.root_dir = os.path.abspath(os.path.join(os.getcwd(), './'))
C.abs_dir = osp.realpath(".")

# Dataset config
"""Dataset Path"""
# 前因：
# 原项目是 RGB-X semantic segmentation，config 里保存的是
# NYUDepthv2/RGB/HHA/Label/train.txt 这一套旧路径。
# 当前任务已经改成 underwater RGB-polar restoration：
#   image_rgb          -> degraded RGB input
#   polarization_input -> Polar/0.jpg, 60.jpg, 120.jpg 拼成 9 通道输入
#   clean_target       -> GT/RGB/I.jpg 作为清晰 RGB 监督目标
#
# 后果：
# dataloader 不再读取旧的 rgb_root / x_root / gt_root / label mask；
# 这里只保留当前数据集根目录和 metadata.csv 入口。
#
# 数据目录约定：
#   项目目录的上级的上级目录下有 DATA/RGBP。
#   例如当前项目在 D:/CODE/RGBP_restoration，则数据在 D:/DATA/RGBP。
#
# 使用相对路径的原因：
#   1. 项目移动到别的机器或别的盘符时，只要仍保持 “项目/../../DATA/RGBP”
#      这个相对位置关系，就不需要改绝对路径；
#   2. 训练代码和 dataloader 都从 config.metadata_source 获取入口，
#      因此全项目只需要维护这一处数据路径。
C.dataset_name = 'RGBP'
C.dataset_path = osp.normpath(osp.join(C.root_dir, '..', '..', 'DATA', C.dataset_name))
C.metadata_source = osp.join(C.dataset_path, 'metadata.csv')

# 数据集划分策略：
#   1. 仍然按 scene 划分，而不是按单张 sample 随机划分。
#      前因：同一个 scene 下的多个退化样本共享同一张 GT/RGB/I.jpg。
#      如果把同一个 scene 的不同 sample 同时放进 train 和 val/test，
#      验证或测试时就会“见过”同一个清晰目标，结果会偏乐观。
#      后果：一个 scene 只会出现在 train/val/test 其中一个集合里。
#
#   2. 不再手写固定 scene 列表，而是用 split_seed 对 scene 名称做确定性随机打乱。
#      前因：手工维护 split_scenes 容易写错、漏写，也不方便换一组随机划分做对照。
#      后果：只要 metadata.csv、split_seed 和 split_ratios 不变，每次生成的划分都完全一致；
#      如果想换一套划分，只需要改 split_seed。
#
#   3. 当前比例按 scene 数量近似 80/10/10。
#      前因：37 个 scene 不能被 80/10/10 精确整除。
#      后果：RGBXDataset 会用最大余数法计算各集合 scene 数，
#      并把剩余 scene 分配给小数部分更大的集合，使比例尽量接近配置值。
C.split_strategy = 'scene'
C.split_ratios = {'train': 0.8, 'val': 0.1, 'test': 0.1}
C.split_seed = 12345

# metadata.csv 中 sample == GT 的行只提供 clean target，不作为退化输入样本。
# 下面三个数量由 dataloader 按 seed 划分后自动回填。
# 前因：换 split_seed 后，每个 split 中的 sample 数可能变化。
# 后果：训练入口不再依赖硬编码样本数，niters_per_epoch 会跟真实 train split 对齐。
C.is_test = False
C.num_train_imgs = None
C.num_eval_imgs = None
C.num_test_imgs = None

# 当前任务输出的是 RGB 恢复图像，输出通道固定为 3。
C.output_channels = 3

"""Image Config"""
C.image_height = 256
C.image_width = 256
# image_rgb 使用 3 通道 RGB mean/std；
# polarization_input 是 9 通道，dataloader 会把这里的 mean/std 重复 3 次后使用。
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

"""Model Config"""
# 当前 builder.py 不再根据 config 选择 backbone / decoder。
# 模型结构已经固定为：
#   encoder: dual SegFormer-B2
#   head:    RestorationHead
#   input:   image_rgb [3ch] + polarization_input [9ch]
#   output:  clean RGB [3ch]
C.model_name = 'RGBPolarRestoration_MiT-B2_RestorationHead'
C.pretrained_model = C.root_dir + '/pretrained/segformer/mit_b2.pth'
C.x_input_channels = 9
C.restoration_head_embed_dim = 256
C.optimizer = 'AdamW'

"""Train Config"""
C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 8
C.nepochs = 500
# 由 get_train_loader 根据随机划分后的 train split 自动设置。
C.niters_per_epoch = None
C.num_workers = 16
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.warm_up_epoch = 10

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

"""Eval Config"""
C.eval_iter = 25
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1] # [0.75, 1, 1.25] #
C.eval_flip = False # True #
C.eval_crop_size = [480, 640] # [height weight]

"""Store Config"""
C.checkpoint_start_epoch = 250
C.checkpoint_step = 25

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(C.root_dir))

C.log_dir = osp.abspath('log_' + C.dataset_name + '_' + C.model_name)
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

if __name__ == '__main__':
    print(config.nepochs)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-tb', '--tensorboard', default=False, action='store_true')
    args = parser.parse_args()

    if args.tensorboard:
        open_tensorboard()
