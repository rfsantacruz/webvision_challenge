import shutil

import os
import torch
import torchvision
import wvc_model
import logging

_logger = logging.getLogger(__name__)
MODELS = ['vgg16', 'inception', 'resnet', 'densenet']


def train(train_loader, model, criterion, optimizer, epoch):
    # metrics
    c_acc1, c_acc5, c_loss = 0.0, 0.0, 0.0
    total_batches = int(len(train_loader.dataset) / train_loader.batch_size)

    # switch to train mode
    model.train()
    for i, (images, target) in enumerate(train_loader):
        target = target.cuda(async=True)
        image_var = torch.autograd.Variable(images)
        label_var = torch.autograd.Variable(target)

        # forward
        y_pred = model(image_var)
        loss = criterion(y_pred, label_var)

        # compute gradients and backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure loss and performance
        c_acc1 += wvc_model.top_k_acc(target, y_pred.data, top_k=1)
        c_acc5 += wvc_model.top_k_acc(target, y_pred.data, top_k=5)
        c_loss += loss.data[0]

        if i % 100 == 0:
            _logger.info("Train epoch {}({}/{}): Loss={}, ACC_1={}, ACC_={}".format(
                epoch, i, total_batches, c_loss / i, c_acc1 / i, c_acc5 / i))


def validate(val_loader, model, criterion, epoch):
    c_loss, c_acc1, c_acc5 = 0.0, 0.0, 0.0
    total_batches = int(len(val_loader.dataset) / val_loader.batch_size)

    # switch to evaluate mode
    model.eval()

    for i, (images, labels) in enumerate(val_loader):
        labels = labels.cuda(async=True)
        image_var = torch.autograd.Variable(images, volatile=True)
        label_var = torch.autograd.Variable(labels, volatile=True)

        # compute prediction and loss
        y_pred = model(image_var)
        loss = criterion(y_pred, label_var)

        # measure loss and performance
        c_acc1 += wvc_model.top_k_acc(labels, y_pred.data, top_k=1)
        c_acc5 += wvc_model.top_k_acc(labels, y_pred.data, top_k=5)
        c_loss += loss.data[0]
        if i % 100 == 0:
            _logger.info("Validation {} ({}/{}): Loss={}, ACC_1={}, ACC_={}".format(
                epoch, i, total_batches, c_loss / i, c_acc1 / i, c_acc5 / i))

    return c_loss / total_batches, c_acc1 / total_batches, c_acc5 / total_batches


def model_factory(model_name, model_kwargs_dict):
    if model_name == MODELS[0]:
        return torchvision.models.vgg16(pretrained=False)
    if model_name == MODELS[1]:
        return torchvision.models.inception_v3(pretrained=False)
    if model_name == MODELS[2]:
        return torchvision.models.resnet152(pretrained=False)
    if model_name == MODELS[3]:
        return torchvision.models.densenet201(pretrained=False)
    else:
        raise ValueError("Model {} is not supported".format(model_name))


def top_k_acc(y_true, y_pred, top_k=5):
    pred_labels = torch.topk(y_pred, top_k)[1].int()
    true_labels = y_true.int()
    acc = torch.eq(true_labels, pred_labels).int()
    acc = torch.sum(acc, 1)
    acc = torch.gt(acc, 0).int()
    acc = torch.mean(acc.float())
    return acc


def save_checkpoint(state, is_best, ckpt_dir, ckpt_name):
    torch.save(state, os.path.join(ckpt_dir, ckpt_name))
    if is_best:
        shutil.copyfile(os.path.join(ckpt_dir, ckpt_name), os.path.join(ckpt_dir, 'model_best.pth.tar'))
