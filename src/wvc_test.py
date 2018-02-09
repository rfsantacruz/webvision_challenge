# Test script
# Author: Rodrigo Santa Cruz
# Date: 8/02/18
import logging, os
import wvc_data, wvc_model, wvc_utils
from webvision import config as db_webv
import tensorflow as tf
import numpy as np

_logger = logging.getLogger(__name__)


def _test(data_split, model_name, model_kwargs_str, model_file):
    # Setup dataset and input pipeline
    _logger.info("Reading daataset...")
    db_info = db_webv.LoadInfo()
    img_files = db_info[db_info.type == data_split].image_path.tolist()
    labels = db_info[db_info.type == data_split].label.tolist() if data_split != 'test' else None
    num_classes, num_imgs = np.unique(labels).size, len(img_files)
    _logger.info("Dataset: split {} has {} classes and {} images".format(data_split, num_classes, num_imgs))

    # Setup input pipeline
    _logger.info("Setting up data input pipeline...")
    input_fn = lambda: wvc_data.input_fn_from_files(img_files, labels=labels, shuffle=False, repeats=1, batch_size=1)

    # Build Keras Model
    _logger.info("Building keras models...")
    model_kwargs_dict = wvc_utils.get_kwargs_dic(model_kwargs_str)
    keras_model = wvc_model.cnn_factory(model_name, num_classes, **model_kwargs_dict)
    keras_model.summary(print_fn=_logger.info)

    # Setup and run estimator's test
    _logger.info("Setting up tensorflow estimator and predict labels...")
    tf_estimator = tf.keras.estimator.model_to_estimator(keras_model=keras_model)
    preds_prob = tf_estimator.predict(input_fn=input_fn, predict_keys=['predictions'], checkpoint_path=model_file)
    _logger.info("Predictions: {}".format(preds_prob))


if __name__ == '__main__':
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Test Tool",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_split', type=str, help="Data split.")
    parser.add_argument('model_name', type=str, help="CNN model name.")
    parser.add_argument('model_kwargs_str', type=str, help="CNN model parameters (ex: k1=v1;k2=v2;...).")
    parser.add_argument('model_file', type=str, help="Model checkpoint path.")
    parser.add_argument('-gpu_str', type=str, default='0', help="CUDA_VISIBLE_DEVICES string.")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_str
    log_file = os.path.join(args.mode_file, "test_{}.log".format(datetime.now().strftime("%Y%m%d-%H%M%S")))
    wvc_utils.init_logging(log_file)
    _logger.info("Test tool: {}".format(args))
    _test(args.data_split, args.model_name, args.model_kwargs_str, args.model_file)
