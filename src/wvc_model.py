# Model definition
# Author: Rodrigo Santa Cruz
# Date: 8/02/18

from keras.applications import VGG16, InceptionResNetV2, DenseNet201
import keras as k
import tensorflow as tf
import logging

_logger = logging.getLogger(__name__)
MODEL_NAME = ['vgg16', 'resnet', 'dense']


def cnn_factory(model_name, num_classes, **kwargs):
    if model_name == MODEL_NAME[0]:
        return vgg16(num_classes, **kwargs)
    elif model_name == MODEL_NAME[1]:
        return resnet(num_classes, **kwargs)
    elif model_name == MODEL_NAME[2]:
        return densenet(num_classes, **kwargs)
    else:
        raise ValueError("Model {} does not supported".format(model_name))


def vgg16(num_classes, **kwargs):
    _logger.info("Building VGG16")
    lr = float(kwargs.get('lr', 1e-3))
    num_gpus = int(kwargs.get('num_gpus', 1))
    _logger.info("Model parameters VGG16: num_classes={}, lr={}, num_gpus={}".format(num_classes, lr, num_gpus))

    if num_gpus >= 2:
        with tf.device('/cpu:0'):
            model_vgg16 = VGG16(include_top=True, classes=num_classes, weights=None)
        model_vgg16 = k.utils.multi_gpu_model(model_vgg16, num_gpus)
    else:
        model_vgg16 = VGG16(include_top=True, classes=num_classes, weights=None)
    model_vgg16.compile(loss=k.losses.sparse_categorical_crossentropy,
                        optimizer=k.optimizers.adam(lr=lr),
                        metrics=k.metrics.sparse_top_k_categorical_accuracy)
    return model_vgg16


def resnet(num_classes, **kwargs):
    _logger.info("Building InceptionResNetV2")
    lr = float(kwargs.get('lr', 1e-3))
    num_gpus = int(kwargs.get('num_gpus', 1))
    _logger.info("Model parameters for InceptionResNetV2: num_classes={}, lr={}, num_gpus={}".format(num_classes, lr, num_gpus))

    if num_gpus >= 2:
        with tf.device('/cpu:0'):
            model_resnet = InceptionResNetV2(include_top=True, classes=num_classes, weights=None)
        model_resnet = k.utils.multi_gpu_model(model_resnet, num_gpus)
    else:
        model_resnet = InceptionResNetV2(include_top=True, classes=num_classes, weights=None)
    model_resnet.compile(loss=k.losses.sparse_categorical_crossentropy,
                         optimizer=k.optimizers.adam(lr=lr),
                         metrics=k.metrics.sparse_top_k_categorical_accuracy)
    return model_resnet


def densenet(num_classes, **kwargs):
    _logger.info("Building DenseNet")
    lr = float(kwargs.get('lr', 1e-3))
    num_gpus = int(kwargs.get('num_gpus', 1))
    _logger.info("Model parameters for DenseNet: num_classes={}, lr={}, num_gpus={}".format(num_classes, lr, num_gpus))

    if num_gpus >= 2:
        with tf.device('/cpu:0'):
            model_densenet = DenseNet201(include_top=True, classes=num_classes, weights=None)
        model_densenet = k.utils.multi_gpu_model(model_densenet, num_gpus)
    else:
        model_densenet = DenseNet201(include_top=True, classes=num_classes, weights=None)
    model_densenet.compile(loss=k.losses.sparse_categorical_crossentropy,
                           optimizer=k.optimizers.adam(lr=lr),
                           metrics=k.metrics.sparse_top_k_categorical_accuracy)
    return model_densenet
