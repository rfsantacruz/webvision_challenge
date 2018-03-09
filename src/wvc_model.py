import torch
import torchvision
from tqdm import tqdm
import shutil, os, logging, math
import wvc_utils

_logger = logging.getLogger(__name__)
MODELS = ['vgg16', 'inception', 'resnet', 'densenet']


def train(train_loader, model, criterion, optimizer, epoch):
    # metrics
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()

    # switch to train mode
    model.train()
    pbar = tqdm(train_loader, leave=False)
    for i, (images, target) in enumerate(pbar):
        target = target.cuda(async=True)
        images = images.cuda(async=True)
        image_var = torch.autograd.Variable(images)
        label_var = torch.autograd.Variable(target)

        # forward
        y_pred = model(image_var)
        loss = criterion(y_pred, label_var)

        # measure loss and performance
        accs = top_k_acc(target, y_pred.data, top_k=(1, 5))
        losses.update(loss.data[0], images.size(0))
        top1.update(accs[0], images.size(0))
        top5.update(accs[1], images.size(0))

        # compute gradients and backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # epoch progress bar
        pbar.set_description("Train epoch {}: Loss={:.3f}, ACC_1={:.3f}, ACC_5={:.3f}".format(
                epoch, losses.avg, top1.avg, top5.avg))

    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, epoch):
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()

    # switch to evaluate mode
    model.eval()
    pbar = tqdm(val_loader, leave=False)
    for i, (images, labels) in enumerate(pbar):
        labels = labels.cuda(async=True)
        images = images.cuda(async=True)
        image_var = torch.autograd.Variable(images, volatile=True)
        label_var = torch.autograd.Variable(labels, volatile=True)

        # compute prediction and loss
        y_pred = model(image_var)
        loss = criterion(y_pred, label_var)

        # measure loss and performance
        accs = top_k_acc(labels, y_pred.data, top_k=(1, 5))
        losses.update(loss.data[0], images.size(0))
        top1.update(accs[0], images.size(0))
        top5.update(accs[1], images.size(0))

        # validation progress
        pbar.set_description("Validation {}: Loss={:.3f}, ACC_1={:.3f}, ACC_5={:.3f}".format(
                epoch, losses.avg, top1.avg, top5.avg))

    return losses.avg, top1.avg, top5.avg


def model_factory(model_name, model_kwargs_dict):
    if model_name == MODELS[0]:
        return torchvision.models.vgg16(pretrained=False, num_classes=1000)
    if model_name == MODELS[1]:
        return torchvision.models.inception_v3(pretrained=False, num_classes=1000)
    if model_name == MODELS[2]:
        return torchvision.models.resnet50(pretrained=False, num_classes=1000)
    if model_name == MODELS[3]:
        return torchvision.models.densenet121(pretrained=False, num_classes=1000)
    else:
        raise ValueError("Model {} is not supported".format(model_name))


def top_k_acc(y_true, y_pred, top_k=(1,)):
    pred_labels = torch.topk(y_pred, max(top_k))[1].int()
    true_labels = y_true.view(-1, 1).int()
    tps = torch.eq(true_labels, pred_labels).int()

    res = []
    for k in top_k:
        acc = torch.sum(tps[:, :k], 1)
        acc = torch.mean(acc.float())
        res.append(acc)
    return res


def save_checkpoint(state, is_best, ckpt_dir, ckpt_name):
    torch.save(state, os.path.join(ckpt_dir, ckpt_name))
    if is_best:
        shutil.copyfile(os.path.join(ckpt_dir, ckpt_name), os.path.join(ckpt_dir, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, initial_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
