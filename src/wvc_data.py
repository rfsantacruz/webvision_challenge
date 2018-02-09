# Input data pipeline
# Author: Rodrigo Santa Cruz
# Date: 8/02/18
import numpy as np
import tensorflow as tf


# Input function
def input_fn_from_files(filenames, labels=None, prep_func=None, shuffle=False, repeats=1, batch_size=1):

    # Set up tensorflow Dataset
    labels = np.expand_dims(-1*np.ones(len(filenames)), axis=1) if labels is None \
        else np.expand_dims(np.array(labels), axis=1)
    labels = tf.cast(tf.constant(labels), tf.float32)
    filenames = tf.constant(filenames)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    # Preprocessing and augmentation
    if prep_func is None:
        dataset = dataset.map(prep_func)
    else:
        dataset = dataset.map(_default_img_label_prep)

    # Configure batches
    if shuffle:
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeats)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_imgs, batch_labels = iterator.get_next()
    return batch_imgs, batch_labels


def _default_img_label_prep(filename, label):
    image_string = tf.read_file(filename)
    image = tf.image.decode_image(image_string, channels=3)
    image.set_shape([None, None, None])
    image = tf.image.resize_images(image, [150, 150])
    image = tf.subtract(image, 116.779)  # Zero-center by mean pixel
    image.set_shape([150, 150, 3])
    image = tf.reverse(image, axis=[2])  # 'RGB'->'BGR'
    d = dict(zip(['input_img'], [image])), label
    return d
