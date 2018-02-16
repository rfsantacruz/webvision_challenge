import torch
import wvc_model
import logging
import torch.utils.data as d


_logger = logging.getLogger(__name__)


def train(train_loader, model, criterion, optimizer, epoch):

    # metrics
    acc1, acc5, loss, steps = 0.0, 0.0, 0.0, 0.0
    total_batches = int(len(train_loader.dataset)/train_loader.batch_size)

    # switch to train mode
    model.train()
    for i, (images, target) in enumerate(train_loader):
        target = target.cuda(async=True)
        image_var = torch.autograd.Variable(images)
        label_var = torch.autograd.Variable(target)

        # compute prediction
        y_pred = model(image_var)
        loss = criterion(y_pred, label_var)

        # compute gradients and backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure loss and performance
        acc1 += wvc_model.top_k_acc(target, y_pred.data, top_k=1)
        acc5 += wvc_model.top_k_acc(target, y_pred.data, top_k=5)
        loss += wvc_model.top_k_acc(target, y_pred.data, top_k=5)

        if i > 100:
            _logger.info("Batch {}/{}: Loss={}, ACC_1={}, ACC_={}".format(i, total_batches, loss.data[0], acc1, acc5))

