import torch
import torchvision
from tqdm import tqdm
import shutil, os, logging, math
import wvc_utils

_logger = logging.getLogger(__name__)
MODELS = ['vgg16', 'inception', 'resnet', 'densenet']


def train(train_loader, model, criterion, optimizer, epoch):
    # metrics
    c_acc1, c_acc5, c_loss = 0.0, 0.0, 0.0
    total_batches = math.ceil(len(train_loader.dataset) / train_loader.batch_size)

    # switch to train mode
    model.train()
    pbar = tqdm(train_loader)
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
        c_acc1 += accs[0]; c_acc5 += accs[1]; c_loss += loss.data[0]

        # compute gradients and backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # epoch progress bar
        pbar.set_description("Train epoch {}: Loss={:.3f}, ACC_1={:.3f}, ACC_5={:.3f}".format(
                epoch+1, c_loss / (i+1), c_acc1 / (i+1), c_acc5 / (i+1)))

    return c_loss / total_batches, c_acc1 / total_batches, c_acc5 / total_batches


def validate(val_loader, model, criterion, epoch):
    c_loss, c_acc1, c_acc5 = 0.0, 0.0, 0.0
    total_batches = math.ceil(len(val_loader.dataset) / val_loader.batch_size)

    # switch to evaluate mode
    model.eval()
    pbar = tqdm(val_loader)
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
        c_acc1 += accs[0]; c_acc5 += accs[1]; c_loss += loss.data[0]

        # validation progress
        pbar.set_description("Validation {}: Loss={:.3f}, ACC_1={:.3f}, ACC_5={:.3f}".format(
                epoch+1, c_loss / (i+1), c_acc1 / (i+1), c_acc5 / (i+1)))

    return c_loss / total_batches, c_acc1 / total_batches, c_acc5 / total_batches


def model_factory(model_name, model_kwargs_dict):
    if model_name == MODELS[0]:
        return torchvision.models.vgg16(pretrained=False, num_classes=1000)
    if model_name == MODELS[1]:
        return torchvision.models.inception_v3(pretrained=False, num_classes=1000)
    if model_name == MODELS[2]:
        return torchvision.models.resnet152(pretrained=False, num_classes=1000)
    if model_name == MODELS[3]:
        return torchvision.models.densenet201(pretrained=False, num_classes=1000)
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
