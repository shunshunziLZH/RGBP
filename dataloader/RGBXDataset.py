import os
import csv
import cv2
import torch
import random
import numpy as np

import torch.utils.data as data


class RGBXDataset(data.Dataset):
    """当前项目专用的数据集类。

    这个类只服务于 RGB-Polar 图像恢复任务，不再兼容旧的语义分割格式。

    一条训练样本包含三部分：
        image_rgb:
            退化 RGB 输入图，路径为 scene/sample/RGB/I.jpg，shape 为 H x W x 3。
        polarization_input:
            偏振输入图，由 scene/sample/Polar/0.jpg、60.jpg、120.jpg
            三张 RGB 图在通道维拼接得到，shape 为 H x W x 9。
        clean_target:
            清晰监督图，路径为 scene/GT/RGB/I.jpg，shape 为 H x W x 3。

    数据集划分方式：
        从 metadata.csv 收集全部 scene 后，先按自然顺序排序，再使用 config.split_seed
        做确定性随机打乱，最后按 config.split_ratios 切成 train/val/test。
        同一个 seed 会得到同一套 split；换 seed 才会换 split。

    返回给 train.py 的 key 固定为：
        image_rgb, polarization_input, clean_target
    """
    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        super(RGBXDataset, self).__init__()
        self._split_name = split_name
        # 数据集不再依赖 train.txt/test.txt，而是以 metadata.csv 为唯一索引。
        # metadata.csv 每行给出 scene 和 sample，从而能拼出三路图像路径：
        #   degraded input     -> scene/sample/RGB/I.jpg
        #   polarization input -> scene/sample/Polar/{0,60,120}.jpg
        #   clean target       -> scene/GT/RGB/I.jpg
        self._metadata_path = self._find_metadata_path(setting)
        # scene 级划分参数来自 config.py。
        # 不再手写 split_scenes，而是用 split_seed 随机打乱 scene 后自动切分。
        self._split_ratios = setting.get('split_ratios')
        self._split_seed = setting.get('split_seed')
        self._split_scenes = None
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length
        self.preprocess = preprocess

    def __len__(self):
        # file_length 用于把有限样本重复扩展成一个 epoch 需要的长度。
        # 例如 train split 真实样本有限，但训练代码希望一个 epoch
        # 有 batch_size * niters_per_epoch 个 item。
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        # 当 file_length 大于真实样本数时，用 _construct_new_file_names 生成重复后的索引列表。
        # 这样 DataLoader 可以按固定 epoch 长度取样。
        if self._file_length is not None:
            item_name = self._construct_new_file_names(self._file_length)[index]
        else:
            item_name = self._file_names[index]

        # 从磁盘读取当前 restoration 任务需要的三元组。
        # 注意这里读出来仍是 numpy 图像，通道顺序为 RGB。
        #   image_rgb: degraded input, H x W x 3
        #   clean_target: clean target, H x W x 3
        #   polarization_input: polarization input, H x W x 9
        image_rgb, clean_target, polarization_input = self._load_polar_clean_sample(item_name)

        if self.preprocess is not None:
            # preprocess 必须同时处理三路数据，并保持几何增强完全一致：
            # 同一个 flip、scale、crop 位置必须同时作用到 input 和 target，
            # 否则 restored_rgb 和 clean_target 无法逐像素计算 L1Loss。
            image_rgb, clean_target, polarization_input = self.preprocess(
                image_rgb, clean_target, polarization_input
            )

        if self._split_name == 'train':
            # 训练阶段转为 torch tensor。
            # 这里不再有类别 mask，所以不需要 long；三路都是图像，统一 float。
            # preprocess 已经把 shape 从 H x W x C 转成 C x H x W。
            image_rgb = torch.from_numpy(np.ascontiguousarray(image_rgb)).float()
            clean_target = torch.from_numpy(np.ascontiguousarray(clean_target)).float()
            polarization_input = torch.from_numpy(
                np.ascontiguousarray(polarization_input)
            ).float()

        # 只返回当前任务需要的三个核心字段。
        # train.py 会直接取这三个 key：
        #   restored_rgb = model(image_rgb, polarization_input)
        #   loss = L1Loss(restored_rgb, clean_target)
        output_dict = dict(
            image_rgb=image_rgb,
            polarization_input=polarization_input,
            clean_target=clean_target,
            fn=self._format_item_name(item_name),
            n=len(self._file_names)
        )

        return output_dict

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val', 'test']
        # sample == GT 的行只表示清晰目标，不是退化输入，所以必须跳过。
        all_samples = self._get_polar_clean_samples()
        available_scenes = sorted(
            set([item['scene'] for item in all_samples]),
            key=self._scene_sort_key
        )
        self._split_scenes = self._build_seeded_scene_split(available_scenes)

        split_scene_names = self._get_split_scene_names(split_name)
        split_samples = [
            item for item in all_samples
            if item['scene'] in split_scene_names
        ]

        if len(split_samples) == 0:
            raise RuntimeError(
                'No samples found for split "{}". Please check split_seed and split_ratios.'.format(
                    split_name
                )
            )

        return split_samples

    def _build_seeded_scene_split(self, available_scenes):
        # 前因：
        # 当前数据集只有 metadata.csv，没有现成 train/val/test txt；
        # 同时同一个 scene 的所有 sample 共用 GT，必须以 scene 为单位切分。
        #
        # 后果：
        # 这里先按自然顺序排序，再用 split_seed 打乱，避免不同文件系统返回顺序不同；
        # 然后用最大余数法把 scene 数近似分配为 80/10/10。
        # 这样既避免数据泄漏，又能通过改 seed 得到另一套可复现实验划分。
        self._validate_split_setting()

        shuffled_scenes = list(available_scenes)
        random.Random(self._split_seed).shuffle(shuffled_scenes)

        split_names = ['train', 'val', 'test']
        split_counts = self._get_split_counts(len(shuffled_scenes), split_names)

        split_scenes = {}
        start = 0
        for split_name in split_names:
            end = start + split_counts[split_name]
            split_scenes[split_name] = shuffled_scenes[start:end]
            start = end

        return split_scenes

    def _validate_split_setting(self):
        if self._split_seed is None:
            raise RuntimeError('config.split_seed is required for seeded scene split.')

        if self._split_ratios is None:
            raise RuntimeError('config.split_ratios is required for seeded scene split.')

        required_splits = ['train', 'val', 'test']
        for split_name in required_splits:
            if split_name not in self._split_ratios:
                raise RuntimeError(
                    'Split ratio "{}" is missing in config.split_ratios.'.format(split_name)
                )

        ratio_sum = sum([self._split_ratios[name] for name in required_splits])
        if abs(ratio_sum - 1.0) > 1e-6:
            raise RuntimeError(
                'config.split_ratios must sum to 1.0, got {:.6f}.'.format(ratio_sum)
            )

    def _get_split_counts(self, scene_count, split_names):
        # 最大余数法：
        #   1. 先取每个 split 的 floor(scene_count * ratio)；
        #   2. 还没分完的 scene，按小数余数从大到小补给对应 split；
        #   3. 余数相同则按 train/val/test 的顺序稳定处理。
        raw_counts = {
            name: scene_count * self._split_ratios[name]
            for name in split_names
        }
        split_counts = {
            name: int(raw_counts[name])
            for name in split_names
        }

        remaining = scene_count - sum(split_counts.values())
        remainder_order = sorted(
            split_names,
            key=lambda name: (raw_counts[name] - split_counts[name]),
            reverse=True
        )
        for name in remainder_order[:remaining]:
            split_counts[name] += 1

        return split_counts

    def _get_split_scene_names(self, split_name):
        # 前因：
        # 旧项目依赖 train.txt / test.txt；当前数据集只有 metadata.csv，
        # 而且同一个 scene 下的所有 sample 共用 GT。
        #
        # 后果：
        # RGBXDataset 不再自己随机划分，也不读取旧 split txt；
        # 只使用 split_seed + split_ratios 生成 scene 级边界。
        if self._split_scenes is None:
            raise RuntimeError('Seeded scene split has not been built yet.')

        if split_name not in self._split_scenes:
            raise RuntimeError(
                'Split "{}" is missing in seeded scene split.'.format(split_name)
            )

        return set(self._split_scenes[split_name])

    @staticmethod
    def _scene_sort_key(scene_name):
        # scene_2 应该排在 scene_10 前面。
        # 这里提取下划线后的数字做自然排序，保证 seed 打乱前的输入顺序稳定。
        prefix, number = scene_name.rsplit('_', 1)
        if number.isdigit():
            return prefix, int(number)
        return scene_name, 0

    def _find_metadata_path(self, setting):
        # 优先使用 config.py 显式传入的 metadata_source。
        # 这样数据集目录改变时，只需要改 config，不需要改 dataset 代码。
        metadata_path = setting.get('metadata_source')
        if metadata_path and os.path.isfile(metadata_path):
            return metadata_path

        # 如果没有显式传入，则先找默认目录，再扫描 datasets 下的一级子目录。
        # 这只是为了提高路径容错，不表示兼容旧数据格式。
        candidates = [
            os.path.join(os.getcwd(), 'datasets', 'MyRGBP_by_scene', 'metadata.csv')
        ]

        datasets_dir = os.path.join(os.getcwd(), 'datasets')
        if os.path.isdir(datasets_dir):
            for name in os.listdir(datasets_dir):
                candidates.append(os.path.join(datasets_dir, name, 'metadata.csv'))

        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate

        raise FileNotFoundError('metadata.csv not found in datasets directory')

    def _get_polar_clean_samples(self):
        # metadata.csv 每行对应一个 scene/sample。
        # 返回 dict 而不是拼好的字符串，后续构造路径时更清楚：
        #   item['scene']  -> scene_1
        #   item['sample'] -> 1, 2, 3, ...
        samples = []
        with open(self._metadata_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('sample') == 'GT':
                    continue
                samples.append({'scene': row['scene'], 'sample': row['sample']})
        return samples

    def _load_polar_clean_sample(self, item):
        root = os.path.dirname(self._metadata_path)
        scene = item['scene']
        sample = item['sample']

        # 路径对应关系：
        #   image_rgb:
        #       当前退化样本的 RGB/I.jpg
        #   clean_target:
        #       同一个 scene 下 GT/RGB/I.jpg
        #   polarization_input:
        #       当前退化样本的 Polar/0.jpg、60.jpg、120.jpg
        rgb_path = os.path.join(root, scene, sample, 'RGB', 'I.jpg')
        clean_target_path = os.path.join(root, scene, 'GT', 'RGB', 'I.jpg')
        polar_paths = [
            os.path.join(root, scene, sample, 'Polar', '0.jpg'),
            os.path.join(root, scene, sample, 'Polar', '60.jpg'),
            os.path.join(root, scene, sample, 'Polar', '120.jpg')
        ]

        image_rgb = self._open_image(rgb_path, cv2.COLOR_BGR2RGB)
        clean_target = self._open_image(clean_target_path, cv2.COLOR_BGR2RGB)
        polarization_input = self._open_polarization_input(polar_paths)

        return image_rgb, clean_target, polarization_input

    def _open_polarization_input(self, polar_paths):
        # 每个偏振角文件本身是 RGB 三通道：
        #   0.jpg   -> H x W x 3
        #   60.jpg  -> H x W x 3
        #   120.jpg -> H x W x 3
        # 三张图在 channel 维拼接后得到 H x W x 9。
        # 经过 preprocess transpose 后，进入模型的 shape 是 [9, H, W]。
        channels = [
            self._open_image(path, cv2.COLOR_BGR2RGB)
            for path in polar_paths
        ]
        return np.concatenate(channels, axis=2)

    def _format_item_name(self, item_name):
        if isinstance(item_name, dict):
            return os.path.join(item_name['scene'], item_name['sample'])
        return str(item_name)

    def _construct_new_file_names(self, length):
        # 把真实样本列表重复到指定长度。
        # 用途：保持每个 epoch 的 iteration 数由 config.niters_per_epoch 控制。
        assert isinstance(length, int)
        files_len = len(self._file_names)
        new_file_names = self._file_names * (length // files_len)

        rand_indices = torch.randperm(files_len).tolist()
        new_indices = rand_indices[:length % files_len]

        new_file_names += [self._file_names[i] for i in new_indices]

        return new_file_names

    def get_length(self):
        return self.__len__()

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        # OpenCV 默认按 BGR 读取彩色图。
        # 本项目后续按 RGB 处理，因此当 mode == cv2.COLOR_BGR2RGB 时，
        # 这里先用 cv2.imread 读图，再显式转换到 RGB。
        if mode == cv2.COLOR_BGR2RGB:
            img = cv2.imread(filepath, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imread(filepath, mode)

        if img is None:
            # 直接暴露具体文件路径，方便定位 metadata 或目录结构问题。
            raise FileNotFoundError('Failed to read image: {}'.format(filepath))

        img = np.array(img, dtype=dtype)
        return img
