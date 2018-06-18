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
from torchvision.transforms import functional as trans_func


_logger = logging.getLogger(__name__)


class WebVision(data.Dataset):
    def __init__(self, db_info, split='train', transform=None, jigsaw=False, frac=None, subset=None):

        # Load data
        self.split = split
        self.img_ids = db_info[db_info.type == split].image_id.values.astype(np.str)
        self.img_files = db_info[db_info.type == split].image_path.values.astype(np.str)
        self.img_labels = db_info[db_info.type == split].label.values.astype(np.long)
        self.transform = transform
        self.jigsaw = jigsaw
        assert len(self.img_ids) == len(self.img_files)
        assert len(self.img_ids) == len(self.img_labels)

        if subset is not None:
            subset_idx = np.isin(self.img_ids, subset)
            self.img_ids = self.img_ids[subset_idx]
            self.img_files = self.img_files[subset_idx]
            self.img_labels = self.img_labels[subset_idx]
            assert np.unique(self.img_labels).size == 5000
            _logger.info("Selecting subset of {} images".format(len(subset)))

        # compute fraction of the dataset
        if frac is not None:
            sampled_idxs, frac = [], 0.01
            for k in np.unique(self.img_labels):
                idxs = np.where(self.img_labels == k)[0]
                idxs = np.random.choice(idxs, int(np.ceil(len(idxs)*frac)), replace=False)
                sampled_idxs.extend(idxs.tolist())
            self.img_ids, self.img_files, self.img_labels = self.img_ids[sampled_idxs], self.img_files[sampled_idxs], self.img_labels[sampled_idxs]

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


def get_datasets(pre_train, is_lmdb, subset=None):
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
                           # 'val': transforms.Compose([DenseCropTransform(), transforms.Lambda(
                           #     lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops]))]),
                           # 'val': transforms.Compose([transforms.TenCrop(224), transforms.Lambda(
                           #     lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops]))]),
                           'test': transforms.Compose([transforms.TenCrop(224), transforms.Lambda(
                               lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops]))])
                           }

    if is_lmdb:
        train_db = LMDBDataset('/data/home/rfsc/wvc/lmdb/train/', jigsaw=pre_train, image_transform=image_transform['train'])
        val_db = LMDBDataset('/data/home/rfsc/wvc/lmdb/val/', jigsaw=pre_train, image_transform=image_transform['val'])
        test_db = LMDBDataset('/data/home/rfsc/wvc/lmdb/test/', jigsaw=pre_train, image_transform=image_transform['test'])
    else:
        db_info = wv_config.LoadInfo()
        train_db = WebVision(db_info, 'train', jigsaw=pre_train, transform=image_transform['train'], subset=subset)
        val_db = WebVision(db_info, 'val', jigsaw=pre_train, transform=image_transform['val'])
        test_db = WebVision(db_info, 'test', jigsaw=pre_train, transform=image_transform['test'])

    return train_db, val_db, test_db


class DenseCropTransform:
    def __init__(self, scales=(256, 288, 320, 352), crop_size=224):
        self.scales = scales
        self.crop_size = crop_size

    def __call__(self, img):
        crops = []
        for scale in self.scales:
            if min(img.size) != scale:
                r_img = trans_func.resize(img, scale)
            else:
                r_img = img.copy()
            w, h = r_img.size
            square_crops_coord = [(0, 0, scale, scale),
                                  (int(round((h - scale) / 2.)), int(round((w - scale) / 2.)), scale, scale),
                                  (h-scale, w-scale, scale, scale)]
            for upper, left, height, width in square_crops_coord:
                square = trans_func.crop(r_img, upper, left, height, width)
                sq_ten_crops = trans_func.ten_crop(square, self.crop_size)
                sq_crop = trans_func.resize(square, self.crop_size)
                sq_crop_mirror = trans_func.hflip(sq_crop)
                crops.extend((sq_crop, sq_crop_mirror) + sq_ten_crops)
        return crops


# import wvc_utils
# num_models = 5
# frac_samples = 0.25
# output_dir = '/home/rfsc/Projects/webvision_challenge/outputs/esemble/'
#
# db_info = wv_config.LoadInfo()
# img_ids = db_info[db_info.type == 'train'].image_id.values.astype(np.str)
# img_labels = db_info[db_info.type == 'train'].label.values.astype(np.long)
# sampler_dict = {label: wvc_utils.CycleIterator(img_ids[np.equal(img_labels, label)].tolist(), shuffle=True)
#                 for label in np.unique(img_labels)}
#
# # compute subsets
# for num in range(num_models):
#     subset = []
#     for label, it in sampler_dict.items():
#         print("Computing model {}, label {}".format(num+1, label))
#         num_samples = int(np.ceil(frac_samples * len(it._items)))
#         samples = [next(it) for _ in range(num_samples)]
#         subset.extend(samples)
#     subset = np.array(subset, dtype=np.str)
#     print("Model {} has found {} samples".format(num+1, len(subset)))
#     np.save(os.path.join(output_dir, 'subset_{}'.format(num+1)), subset)

# files = ['/data/home/rfsc/wvc/outputs/baseline/sub_file_val_10crops_NEW.txt.prob.txt',
#         '/data/home/rfsc/wvc/outputs/esemble/1/sub_file_val_10crops_new_e1.txt.prob.txt',
#          '/data/home/rfsc/wvc/outputs/esemble/2/sub_file_val_10crops_new_e2.txt.prob.txt',
#          '/data/home/rfsc/wvc/outputs/esemble/3/sub_file_val_10crops_new_e3.txt.prob.txt',
#          '/data/home/rfsc/wvc/outputs/sub_val_esemble_base10crop.txt']
#
#
# with open(files[0], 'r') as p1, open(files[1], 'r') as p2, open(files[2], 'r') as p3, open(files[3], 'r') as p4, open(files[4], 'w') as r :
#     for lines in zip(p1, p2, p3, p4):
#         lines = np.stack([np.array(line.split('\t'), dtype=np.str) for line in lines])
#         id = [lines[0, 0]]
#         vals = lines[:, 1:].astype(np.float).mean(axis=0)
#         vals = np.argsort(vals)[-5:][::-1].astype(np.str).tolist()
#         r.write("{}\n".format("\t".join(id + vals)))

# probs = np.zeros((294099, 5000), dtype=np.float)
# ids = np.zeros((294099, 1), dtype=np.str)
# for file in files:
#     b = np.loadtxt(file, dtype=np.str)
#     ids = b[:, 0]
#     probs += b[:, 1:].astype(np.float)
# probs = probs * 0.25
#
# sc = np.argsort(probs, axis=1)[:, -5:][:, ::-1].astype(np.str)
# sc = np.concatenate([ids, sc], axis=1).tolist()
# with open('/data/home/rfsc/wvc/outputs/sub_val_esemble.txt', 'w') as f:
#     for line in sc:
#         f.write("{}\n".format("\t".join(line)))

