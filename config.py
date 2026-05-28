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
C.dataset_name = 'MyRGBP_by_scene'
C.dataset_path = osp.join(C.root_dir, 'datasets', C.dataset_name)
C.metadata_source = osp.join(C.dataset_path, 'metadata.csv')

# metadata.csv 中 sample == GT 的行只提供 clean target，不作为退化输入样本。
# 当前数据集中非 GT 样本数为 392，用于计算每个 epoch 的 iteration 数。
# 目前 RGBXDataset 对 train/val 使用同一个 metadata 样本列表；
# 如果之后要严格划分训练集和验证集，需要再增加 split 文件或 split 字段。
C.is_test = False
C.num_train_imgs = 392
C.num_eval_imgs = 392

# 当前任务输出的是 RGB 恢复图像，输出通道固定为 3。
C.output_channels = 3

"""Image Config"""
C.image_height = 480
C.image_width = 640
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
C.restoration_head_embed_dim = 512
C.optimizer = 'AdamW'

"""Train Config"""
C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 8
C.nepochs = 500
C.niters_per_epoch = C.num_train_imgs // C.batch_size  + 1
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
