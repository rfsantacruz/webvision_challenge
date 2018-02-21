import webvision.config as wvc_config
import torch.utils.data as data
from PIL import Image
import logging, os
import numpy as np

_logger = logging.getLogger(__name__)


class WebVision(data.Dataset):
    def __init__(self, split='train', transform=None):

        # Load data
        db_info = wvc_config.LoadInfo()
        self.split = split
        self.img_ids = db_info[db_info.type == split].image_id.values.astype(np.str)
        self.img_files = db_info[db_info.type == split].image_path.values.astype(np.str)
        self.img_labels = db_info[db_info.type == split].label.values.astype(np.long)
        self.transform = transform
        assert len(self.img_ids) == len(self.img_files)
        assert len(self.img_ids) == len(self.img_labels)

        # Compute class frequency
        self.class_freq = np.bincount(self.img_labels)
        assert self.class_freq.size == 1000
        self.sample_weight = self.img_labels.size / (self.class_freq[self.img_labels] + 1e-3)

        _logger.info("Webvision {} dataset read with {} images".format(split, len(self.img_ids)))

    def __getitem__(self, index):
        img = Image.open(self.img_files[index])
        label = self.img_labels[index]
        if img.mode != 'RGB':
            img = img.convert(mode='RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.img_ids)
