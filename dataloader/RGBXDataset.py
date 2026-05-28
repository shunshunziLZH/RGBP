import os
import csv
import cv2
import torch
import numpy as np

import torch.utils.data as data


class RGBXDataset(data.Dataset):
    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        super(RGBXDataset, self).__init__()
        self._split_name = split_name
        # 新数据集以 metadata.csv 为索引：
        #   degraded input     -> scene/sample/RGB/I.jpg
        #   polarization input -> scene/sample/Polar/{0,60,120}.jpg
        #   clean target       -> scene/GT/RGB/I.jpg
        self._metadata_path = self._find_metadata_path(setting)
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length
        self.preprocess = preprocess

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        if self._file_length is not None:
            item_name = self._construct_new_file_names(self._file_length)[index]
        else:
            item_name = self._file_names[index]

        # 当前 restoration 数据格式：
        #   image_rgb: degraded input, H x W x 3
        #   clean_target: clean target, H x W x 3
        #   polarization_input: polarization input, H x W x 9
        image_rgb, clean_target, polarization_input = self._load_polar_clean_sample(item_name)

        if self.preprocess is not None:
            # preprocess 必须同时处理三路数据，并保持几何增强完全一致。
            image_rgb, clean_target, polarization_input = self.preprocess(
                image_rgb, clean_target, polarization_input
            )

        if self._split_name == 'train':
            # 训练阶段转为 tensor。三路数据都是图像张量，统一使用 float。
            image_rgb = torch.from_numpy(np.ascontiguousarray(image_rgb)).float()
            clean_target = torch.from_numpy(np.ascontiguousarray(clean_target)).float()
            polarization_input = torch.from_numpy(
                np.ascontiguousarray(polarization_input)
            ).float()

        # 只返回当前任务需要的三个核心字段。
        output_dict = dict(
            image_rgb=image_rgb,
            polarization_input=polarization_input,
            clean_target=clean_target,
            fn=self._format_item_name(item_name),
            n=len(self._file_names)
        )

        return output_dict

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val']
        # 直接从 metadata.csv 生成样本列表。
        # 这里跳过 sample == GT，因为 GT 只作为 clean target，不作为 degraded input。
        return self._get_polar_clean_samples()

    def _find_metadata_path(self, setting):
        # 优先使用外部显式传入的 metadata_source，方便以后在 config 中指定。
        metadata_path = setting.get('metadata_source')
        if metadata_path and os.path.isfile(metadata_path):
            return metadata_path

        # 当前项目的数据已经放在 datasets/MyRGBP_by_scene 下；
        # 同时保留一个通用扫描，便于数据集目录名变化时仍能找到 metadata.csv。
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
        # 返回 dict 而不是拼好的字符串，是为了后面构造路径时更直观：
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

        # degraded input 来自当前 sample 的 RGB/I.jpg；
        # clean target 来自同一 scene 的 GT/RGB/I.jpg；
        # polarization input 来自当前 sample 的三张偏振角图像。
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
        # 每张偏振图都是 RGB 三通道：
        #   0.jpg   -> H x W x 3
        #   60.jpg  -> H x W x 3
        #   120.jpg -> H x W x 3
        # 在 channel 维拼接后得到 H x W x 9，
        # 对应后续 tensor 的 [9, H, W]。
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
        # OpenCV 默认读取顺序是 BGR；当调用方传入 cv2.COLOR_BGR2RGB 时，
        # 这里显式先 imread，再做颜色空间转换，保证返回的是 RGB。
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
