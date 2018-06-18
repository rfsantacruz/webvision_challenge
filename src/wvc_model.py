import itertools
import torch
from torch.nn import functional as t_func
import torchvision
from tqdm import tqdm
import shutil, os, logging
import math

_logger = logging.getLogger(__name__)
MODELS = ['vgg16', 'inception', 'resnet', 'densenet']


def train(train_loader, model, criterion, optimizer, metric_func, epoch):
    # metrics
    losses, metrics = AverageMeter(), dict()

    # switch to train mode
    model.train()
    # epoch_samp_frac = int(len(train_loader) * epoch_size)
    # pbar = tqdm(itertools.islice(train_loader, epoch_samp_frac), total=epoch_samp_frac, leave=False)
    pbar = tqdm(train_loader, leave=False)
    for i, (_, images, target) in enumerate(pbar):
        target = target.cuda(async=True)
        images = images.cuda(async=True)
        image_var = torch.autograd.Variable(images)
        label_var = torch.autograd.Variable(target)

        # forward
        y_pred = model(image_var)
        loss = criterion(y_pred, label_var)

        # measure loss and performance
        losses.update(loss.data[0], images.size(0))
        for m_key, m_value in metric_func(label_var.data, y_pred.data).items():
            if m_key not in metrics:
                metrics[m_key] = AverageMeter()
            metrics[m_key].update(m_value, images.size(0))

        # compute gradients and backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # epoch progress bar
        pbar.set_description("Train epoch {}: Loss={:.3f}, {}".format(
                epoch, losses.avg, ", ".join(["{}={:.3f}".format(k, v.avg) for k, v in metrics.items()])))

    return losses.avg, {k: v.avg for k, v in metrics.items()}


def validate(val_loader, model, criterion, metric_func, epoch):
    losses, metrics = AverageMeter(), dict()

    # switch to evaluate mode
    model.eval()
    pbar = tqdm(val_loader, leave=False)
    for i, (_, images, labels) in enumerate(pbar):
        labels = labels.cuda(async=True)
        images = images.cuda(async=True)
        image_var = torch.autograd.Variable(images, volatile=True)
        label_var = torch.autograd.Variable(labels, volatile=True)

        # compute prediction and loss
        y_pred = model(image_var)
        loss = criterion(y_pred, label_var)

        # measure loss and performance
        losses.update(loss.data[0], images.size(0))
        for m_key, m_value in metric_func(label_var.data, y_pred.data).items():
            if m_key not in metrics:
                metrics[m_key] = AverageMeter()
            metrics[m_key].update(m_value, images.size(0))

        # validation progress
        pbar.set_description("Validation {}: Loss={:.3f}, {}".format(
                epoch, losses.avg, ", ".join(["{}={:.3f}".format(k, v.avg) for k, v in metrics.items()])))

    return losses.avg, {k: v.avg for k, v in metrics.items()}


class PermLearning(torch.nn.Module):

    def __init__(self, cnn_module, num_ite=2, seq_len=9, lamb=1e-3, epi=1e-9):
        super(PermLearning, self).__init__()
        self.features = cnn_module.features
        self.cls = torch.nn.Linear(seq_len*1024*4, seq_len ** 2)
        self.seq_len = int(seq_len)
        self.pre_func = torch.nn.Softmax(-1)
        self.num_ite, self.lamb, self.epi = num_ite, lamb, epi

    def forward(self, x):
        # x[b,p,c,h,w]
        batch_size, parts, others = x.size(0), x.size(1), x.size()[2:]
        assert parts == self.seq_len

        # compute cnn representation
        out = x.contiguous().view((batch_size*parts,) + others)
        out = self.features(out)

        # compute outputs
        out = out.contiguous().view(batch_size, -1)
        out = self.pre_func(self.cls(out))

        # sinkhorn normalization
        out = out.contiguous().view(batch_size, self.seq_len, self.seq_len).add(self.lamb)
        for i in range(self.num_ite):
            # row normalization
            out = out.contiguous().view(batch_size*self.seq_len, self.seq_len)
            out = out.div(out.sum(-1, True) + self.epi)
            out = out.contiguous().view(batch_size, self.seq_len, self.seq_len)
            # col normalization
            out = out.permute(0, 2, 1).contiguous().view(batch_size*self.seq_len, self.seq_len)
            out = out.div(out.sum(-1, True) + self.epi)
            out = out.contiguous().view(batch_size, self.seq_len, self.seq_len).permute(0, 2, 1)
        out = out.contiguous().view(batch_size, -1)
        return out


class WeightedMultiLabelBinaryCrossEntropy(torch.nn.Module):

    def __init__(self, size_average=True):
        super(WeightedMultiLabelBinaryCrossEntropy, self).__init__()
        self.size_average = size_average

    def forward(self, input, target):
        weights = target.sum(0, True).expand_as(target)
        weights = (weights * target) + ((target.size(0) - weights) * (1-target))
        weights = 1 / (weights + 1e-6)
        return t_func.binary_cross_entropy_with_logits(input, target, weights, self.size_average)


def model_factory(model_name, model_kwargs_dict):
    if model_name == MODELS[0]:
        return torchvision.models.vgg16(pretrained=False, num_classes=5000)
    if model_name == MODELS[1]:
        return torchvision.models.inception_v3(pretrained=False, num_classes=5000)
    if model_name == MODELS[2]:
        return torchvision.models.resnet50(pretrained=False, num_classes=5000)
    if model_name == MODELS[3]:
        return torchvision.models.densenet121(pretrained=False, num_classes=5000)
    else:
        raise ValueError("Model {} is not supported".format(model_name))


def top_k_acc(y_true, y_pred, top_k=(1, 5)):
    pred_labels = torch.topk(y_pred, max(top_k))[1].int()
    true_labels = y_true.view(-1, 1).int()
    tps = torch.eq(true_labels, pred_labels).int()

    res = dict()
    for k in top_k:
        acc = torch.sum(tps[:, :k], 1)
        acc = torch.mean(acc.float())
        res["acc{}".format(k)] = acc
    return res


def multilabel_metrics(y_true, y_prob, pos_th=0.5):
    """
    Multilabel metrics computation.
    :param y_true: Ground truth tensor in the format (Number of samples x Number of classes).
    :param y_prob: Predicted probs tensor in the format (Number of samples x Number of classes) with values in [0,1].
    :param pos_th: Threshold to assign class membership.
    :return: metric values in a dictionary
    """
    # Check dimension
    assert [y_true.dim(), y_prob.dim()] == [2, 2]
    assert y_true.size() == y_prob.size()

    # Threshold
    y_true_th = y_true.gt(0.0)
    y_pred_th = y_prob.gt(pos_th)

    # Compute quantities
    inter = y_pred_th.mul(y_true_th).sum(1).float()
    union = y_pred_th.add(y_true_th).gt(0).sum(1).float()
    pred = y_pred_th.sum(1).float()
    actual = y_true_th.sum(1).float()

    # Accuracy = proportion of corrected labels in all predicted and actual labels averaged over samples
    acc = inter.div(union.add(1e-6)).mean()
    # Precision = proportion of predicted correct labels to the number of predicted labels averaged over samples
    prec = inter.div(pred.add(1e-6)).mean()
    # Recall = proportion of predicted correct labels to the number of actual labels averaged over samples
    rec = inter.div(actual.add(1e-6)).mean()
    # Fscore = harmonic mean between precision and recall
    fsc = inter.mul(2).div(actual.add(pred).add(1e-6)).mean()
    return {'ml_acc': acc, 'ml_prec': prec, 'ml_rec': rec, 'ml_fsc': fsc}


def save_checkpoint(state, is_best, ckpt_dir, ckpt_name):
    torch.save(state, os.path.join(ckpt_dir, ckpt_name))
    if is_best:
        shutil.copyfile(os.path.join(ckpt_dir, ckpt_name), os.path.join(ckpt_dir, 'model_best.pth.tar'))


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
