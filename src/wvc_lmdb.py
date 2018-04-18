import sys, os, lmdb, cv2, logging, tqdm
import numpy as np
import webvision.config as wvc_config
import wvc_utils

_logger = logging.getLogger(__name__)


def write_to_lmdb(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)


def create_lmdb(output_path, wvc_split):
    _logger.info("Creating a lmdb for split {} at {}".format(wvc_split, output_path))

    # get images and labels
    info_df = wvc_config.LoadInfo()
    img_ids = info_df[info_df.type == wvc_split].image_id.values.astype(np.str)
    img_files = info_df[info_df.type == wvc_split].image_path.values.astype(np.str)
    img_labels = info_df[info_df.type == wvc_split].label.values.astype(np.long)
    num_samples = img_files.size
    _logger.info("Loaded {} images".format(num_samples))

    # Compute class frequency
    if wvc_split != 'test':
        class_freq = np.bincount(img_labels).astype(np.float)
        # assert self.class_freq.size == 5000
        sample_weight = (img_labels.size / (class_freq[img_labels] + 1e-6)).astype(np.float)
    else:
        class_freq = -1*np.ones(np.unique(img_labels).size, np.float)
        sample_weight = -1*np.ones(img_labels.size, np.float)

    # Adapt filenames to jpg
    for i in range(num_samples):
        img_files[i] = os.path.splitext(img_files[i])[0] + ".jpg"

    # open lmdb
    env = lmdb.open(output_path, map_size=1099511627776)
    write_to_lmdb(env, dict({'num_samples': np.array([num_samples], np.int), 'class_feq': class_freq,
                             'sample_weight': sample_weight}))
    cache = dict()
    for i, (image_id, image_path, label) in enumerate(tqdm.tqdm(zip(img_ids, img_files, img_labels),
                                                                desc='Writing images', total=num_samples)):
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        cache["id_{:09d}".format(i)] = image_id.encode()
        cache["img_{:09d}".format(i)] = cv2.imencode('.jpg', image)[1]
        cache["lbl_{:09d}".format(i)] = np.array([label], np.float)
        if len(cache) % 1e3 == 0 or i == num_samples-1:
            write_to_lmdb(env, cache)
            cache = dict()
    _logger.info("LMDB dataset created at {} with {} samples".format(output_path, num_samples))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create LMDB dataset')
    parser.add_argument('wvc_split', type=str, help='Dataset Split')
    parser.add_argument('output_path', type=str, help='output_path')
    args = parser.parse_args()
    wvc_utils.init_logging()
    create_lmdb(args.output_path, args.wvc_split)