# Training and Validation script
# Author: Rodrigo Santa Cruz
# Date: 8/02/18

import tensorflow as tf
import numpy as np
import wvc_model, wvc_data, wvc_utils
from webvision import config as db_webv
import logging, os, math

_logger = logging.getLogger(__name__)


def _train_val(model_name, model_kwargs_str, output_dir, batch_size, num_epochs, validation_interval):
    # Setup dataset
    _logger.info("Reading daataset...")
    db_info = db_webv.LoadInfo()
    train_img_files, train_labels = db_info[db_info.type == 'train'].image_path.tolist(), \
                                    db_info[db_info.type == 'train'].label.tolist()
    val_img_files, val_labels = db_info[db_info.type == 'val'].image_path.tolist(), \
                                db_info[db_info.type == 'val'].label.tolist()
    num_classes, num_train_imgs, num_val_imgs = np.unique(train_labels).size, len(train_img_files), len(val_img_files)
    _logger.info("Dataset: {} classes, {} training images and {} validation images".format(
        num_classes, num_train_imgs, num_val_imgs))

    # Build keras model
    _logger.info("Building keras models...")
    model_kwargs_dict = wvc_utils.get_kwargs_dic(model_kwargs_str)
    keras_model = wvc_model.cnn_factory(model_name, num_classes, wvc_data.IMAGE_INPUT_SHAPE, **model_kwargs_dict)
    keras_model.summary(print_fn=_logger.info)
    input_name = keras_model.layers[0].name

    # Setup input pipeline
    _logger.info("Setting up data input pipeline...")
    train_input_fn = lambda: wvc_data.input_fn_from_files(
        input_name, train_img_files, labels=train_labels, mode='train', shuffle=True, repeats=num_epochs, batch_size=batch_size)
    val_input_fn = lambda: wvc_data.input_fn_from_files(
        input_name, val_img_files, labels=val_labels, mode='val', shuffle=False, repeats=1, batch_size=batch_size)

    # Setup and run estimator's train and validation
    _logger.info("Setting up tensorflow estimator...")
    tf_estimator = tf.keras.estimator.model_to_estimator(keras_model=keras_model, model_dir=output_dir)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_epochs*int(math.ceil(num_train_imgs/batch_size)))
    val_spec = tf.estimator.EvalSpec(input_fn=val_input_fn)
    _logger.info("Training...")
    tf.estimator.train_and_evaluate(tf_estimator, train_spec, val_spec)


if __name__ == '__main__':
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Training and Validation Tool",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_name', type=str, help="CNN model name.")
    parser.add_argument('model_kwargs_str', type=str, help="CNN model parameters (ex: k1=v1;k2=v2;...).")
    parser.add_argument('output_dir', type=str, help="Output directory.")
    parser.add_argument('-batch_size', type=int, default=128, help="Number of samples per batch.")
    parser.add_argument('-num_epochs', type=int, default=100, help="Number of training epochs.")
    parser.add_argument('-val_int', type=int, default=25, help="Evaluation interval in epochs.")
    parser.add_argument('-gpu_str', type=str, default='0', help="CUDA_VISIBLE_DEVICES string.")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_str
    log_file = os.path.join(args.output_dir, "train_{}.log".format(datetime.now().strftime("%Y%m%d-%H%M%S")))
    wvc_utils.init_logging(log_file)
    _logger.info("Train and Validation tool: {}".format(args))
    _train_val(args.model_name, args.model_kwargs_str, args.output_dir, args.batch_size, args.num_epochs, args.val_int)
