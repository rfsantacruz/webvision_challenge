# Input data pipeline
# Author: Rodrigo Santa Cruz
# Date: 8/02/18
import numpy as np
import tensorflow as tf

# Constants
DATASET_MEAN = np.array([100, 100, 100], np.float32)
IMAGE_INPUT_SHAPE = [224, 224, 3]


# Input function
def input_fn_from_files(input_name, filenames, labels=None, mode='train', shuffle=False, repeats=1, batch_size=1):
    # Set up tensorflow Dataset
    labels = np.expand_dims(-1*np.ones(len(filenames)), axis=1) if labels is None \
        else np.expand_dims(np.array(labels), axis=1)
    labels = tf.cast(tf.constant(labels), tf.float32)
    filenames = tf.constant(filenames)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    # Processing images and labels
    prep_funcs = {'train': _train_input_parse, 'val': _val_input_parse, 'test': _test_input_parse}
    dataset = dataset.map(prep_funcs[mode], num_parallel_calls=10)
    dataset = dataset.map(lambda image, label: (dict(zip([input_name], [image])), label))
    dataset = dataset.prefetch(4*batch_size)

    # Configure batches
    if shuffle:
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeats)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_imgs, batch_labels = iterator.get_next()
    return batch_imgs, batch_labels


# Parse functions
# Training parsing function
def _train_input_parse(filename, label):
    image_string = tf.read_file(filename)
    image = tf.image.decode_image(image_string, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.random_crop(image, IMAGE_INPUT_SHAPE)
    image.set_shape(IMAGE_INPUT_SHAPE)
    return image, label


# Validation parsing function
def _val_input_parse(filename, label):
    image_string = tf.read_file(filename)
    image = tf.image.decode_image(image_string, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_image_with_crop_or_pad(image, IMAGE_INPUT_SHAPE[0], IMAGE_INPUT_SHAPE[1])
    image.set_shape(IMAGE_INPUT_SHAPE)
    return image, label


# Test parsing function
def _test_input_parse(filename, label):
    return _val_input_parse(filename, label)
