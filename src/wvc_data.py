import torch.utils.data as data
from PIL import Image
import logging, os
import numpy as np
import lmdb
import cv2
from torchvision import transforms


_logger = logging.getLogger(__name__)


class WebVision(data.Dataset):
    def __init__(self, db_info, split='train', transform=None):

        # Load data
        self.split = split
        self.img_ids = db_info[db_info.type == split].image_id.values.astype(np.str)
        self.img_files = db_info[db_info.type == split].image_path.values.astype(np.str)
        self.img_labels = db_info[db_info.type == split].label.values.astype(np.long)
        self.transform = transform
        assert len(self.img_ids) == len(self.img_files)
        assert len(self.img_ids) == len(self.img_labels)

        # Adapt filenames to jpg
        for i in range(self.img_files.size):
            self.img_files[i] = os.path.splitext(self.img_files[i])[0] + ".jpg"

        # Compute class frequency
        self.class_freq = np.bincount(self.img_labels if split != 'test' else np.ones_like(self.img_labels))
        # assert self.class_freq.size == 5000
        self.sample_weight = self.img_labels.size / (self.class_freq[self.img_labels] + 1e-6)

        _logger.info("Webvision {} dataset read with {} images".format(split, len(self.img_ids)))

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        label = self.img_labels[index]
        img = Image.open(self.img_files[index])
        if img.mode != 'RGB':
            img = img.convert(mode='RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img_id, img, label

    def __len__(self):
        return len(self.img_ids)


class LMDBDataset(data.Dataset):
    def __init__(self, lmdb_path, transform):
        # read parameters
        self.lmdb_path = lmdb_path
        self.transform = transform

        # open and read general info of the dataset
        self.lmdb_env = lmdb.open(self.lmdb_path, readonly=True)
        with self.lmdb_env.begin() as lmdb_txn:
            self.num_samples = np.fromstring(lmdb_txn.get('num_samples'.encode()), dtype=np.int)[0]
            self.class_feq = np.fromstring(lmdb_txn.get('class_feq'.encode()), dtype=np.float)
            self.sample_weight = np.fromstring(lmdb_txn.get('sample_weight'.encode()), dtype=np.float)

    def __getitem__(self, index):
        id_key, img_key, lbl_key = "id_{:09d}".format(index), "img_{:09d}".format(index), "lbl_{:09d}".format(index)
        with self.lmdb_env.begin() as lmdb_txn:
            id_b, img_b, lbl_b = lmdb_txn.get(id_key.encode()), lmdb_txn.get(img_key.encode()), lmdb_txn.get(lbl_key.encode())
        img_id = id_b.decode()
        img = Image.fromarray(cv2.imdecode(np.fromstring(img_b, dtype=np.uint8), cv2.IMREAD_COLOR), mode='RGB')
        label = np.fromstring(lbl_b, dtype=np.float).astype(np.long)[0]

        if self.transform is not None:
            img = self.transform(img)

        return img_id, img, label

    def __len__(self):
        return self.num_samples
