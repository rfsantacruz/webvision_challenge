import torch
import torch.utils.data as data
from PIL import Image
import logging, os
import numpy as np
import lmdb
import cv2
from torchvision import transforms
import random
import itertools
from webvision import config as wv_config

_logger = logging.getLogger(__name__)


class WebVision(data.Dataset):
    def __init__(self, db_info, split='train', transform=None, jigsaw=False):

        # Load data
        self.split = split
        self.img_ids = db_info[db_info.type == split].image_id.values.astype(np.str)[:200]
        self.img_files = db_info[db_info.type == split].image_path.values.astype(np.str)[:200]
        self.img_labels = db_info[db_info.type == split].label.values.astype(np.long)[:200]
        self.transform = transform
        self.jigsaw = jigsaw
        assert len(self.img_ids) == len(self.img_files)
        assert len(self.img_ids) == len(self.img_labels)

        # Adapt filenames to jpg
        for i in range(self.img_files.size):
            self.img_files[i] = os.path.splitext(self.img_files[i])[0] + ".jpg"

        # Compute class frequency
        if self.split != 'test':
            self.class_freq = np.bincount(self.img_labels)
            # assert self.class_freq.size == 5000
            self.sample_weight = self.img_labels.size / (self.class_freq[self.img_labels] + 1e-6)
        else:
            self.class_freq = -1*np.ones(5000)
            self.sample_weight = np.ones(self.img_labels.size, np.float)

        _logger.info("Webvision {} dataset read with {} images".format(split, len(self.img_ids)))

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        label = self.img_labels[index]
        img = Image.open(self.img_files[index])
        if img.mode != 'RGB':
            img = img.convert(mode='RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.jigsaw:
            assert img.dim() > 3
            label = torch.randperm(img.size(0)).long()
            img = img[label]
            label = torch.eye(img.size(0), img.size(0))[label]
            label = label.view(-1)
        return img_id, img, label

    def __len__(self):
        return len(self.img_ids)


class LMDBDataset(data.Dataset):
    def __init__(self, lmdb_path, image_transform=None, jigsaw=False):
        # read parameters
        self.lmdb_path = lmdb_path
        self.transform = image_transform
        self.jigsaw = jigsaw

        # open and read general info of the dataset
        self.lmdb_env = lmdb.open(self.lmdb_path, max_readers=1, readonly=True, lock=False, readahead=False,
                                  meminit=False)
        with self.lmdb_env.begin() as lmdb_txn:
            self.num_samples = np.fromstring(lmdb_txn.get('num_samples'.encode()), dtype=np.int)[0]
            self.class_feq = np.fromstring(lmdb_txn.get('class_feq'.encode()), dtype=np.float)
            self.sample_weight = np.fromstring(lmdb_txn.get('sample_weight'.encode()), dtype=np.float)
        _logger.info("Webvision dataset loaded from LMDB {} with {} images".format(lmdb_path, self.num_samples))

    def __getitem__(self, index):
        id_key, img_key, lbl_key = "id_{:09d}".format(index), "img_{:09d}".format(index), "lbl_{:09d}".format(index)
        with self.lmdb_env.begin() as lmdb_txn:
            id_b, img_b, lbl_b = lmdb_txn.get(id_key.encode()), lmdb_txn.get(img_key.encode()), lmdb_txn.get(
                lbl_key.encode())
        img_id = id_b.decode()
        img = Image.fromarray(cv2.imdecode(np.fromstring(img_b, dtype=np.uint8), cv2.IMREAD_COLOR), mode='RGB')
        label = np.fromstring(lbl_b, dtype=np.float).astype(np.long)[0]

        if self.transform is not None:
            img = self.transform(img)

        if self.jigsaw:
            assert img.dim() > 3
            label = torch.randperm(img.size(0)).long()
            img = img[label]
            label = torch.eye(img.size(0), img.size(0))[label]
            label = label.view(-1)

        return img_id, img, label

    def __len__(self):
        return self.num_samples


class JigsawTransform:
    def __init__(self, grid_size=3, patch_size=64):
        self.grid_size = grid_size
        self.patch_size = patch_size

    def __call__(self, img):
        w, h = img.size
        crops = []
        for c_i, c_j in itertools.product(range(self.grid_size), range(self.grid_size)):
            # find patch coordinates
            tile_h, tile_w = (h / self.grid_size), (w / self.grid_size)
            pad_h, pad_w = (tile_h - self.patch_size) / 2.0, (tile_w - self.patch_size) / 2.0
            l, u, r, b = c_j * tile_w + pad_w, c_i * tile_h + pad_h, (c_j + 1) * tile_w - pad_w, (
                    c_i + 1) * tile_h - pad_h
            l, u, r, b = int(np.floor(l)), int(np.floor(u)), int(np.floor(r)), int(np.floor(b))
            # jitter
            l = l + random.randint(0, r - self.patch_size - l)
            u = u + random.randint(0, b - self.patch_size - u)
            r = l + self.patch_size
            b = u + self.patch_size
            # crop
            crops.append(img.crop((l, u, r, b)))
        return crops


def get_datasets(pre_train, is_lmdb):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if pre_train:
        image_transform = {'train': transforms.Compose([transforms.RandomCrop(224), JigsawTransform(3, 64),
                                                        transforms.Lambda(lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops]))]),
                           'val': transforms.Compose([transforms.CenterCrop(224),
                                                      JigsawTransform(3, 64), transforms.Lambda(lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops]))]),
                           'test': transforms.Compose([transforms.CenterCrop(224),
                                                       JigsawTransform(3, 64), transforms.Lambda(lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops]))])
                           }

    else:
        image_transform = {'train': transforms.Compose([transforms.RandomCrop(224), transforms.ToTensor(), normalize]),
                           'val': transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), normalize]),
                           'test': transforms.Compose([transforms.TenCrop(224), transforms.Lambda(
                               lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops]))])
                           }

    if is_lmdb:
        train_db = LMDBDataset('/data/home/rfsc/wvc/lmdb/train/', jigsaw=pre_train, image_transform=image_transform['train'])
        val_db = LMDBDataset('/data/home/rfsc/wvc/lmdb/val/', jigsaw=pre_train, image_transform=image_transform['val'])
        test_db = LMDBDataset('/data/home/rfsc/wvc/lmdb/test/', jigsaw=pre_train, image_transform=image_transform['test'])
    else:
        db_info = wv_config.LoadInfo()
        train_db = WebVision(db_info, 'train', jigsaw=pre_train, transform=image_transform['train'])
        val_db = WebVision(db_info, 'val', jigsaw=pre_train, transform=image_transform['val'])
        test_db = WebVision(db_info, 'test', jigsaw=pre_train, transform=image_transform['test'])

    return train_db, val_db, test_db
