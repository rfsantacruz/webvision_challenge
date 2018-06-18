from torch.utils.data import dataloader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.cuda import device_count
import torch
import wvc_data, wvc_model, wvc_utils
import logging, os, warnings
from tensorboard_logger import tensorboard_logger as tb_log
import numpy as np

warnings.filterwarnings("ignore")
_logger = logging.getLogger(__name__)


def main(model_name, output_dir, batch_size=320, num_epochs=15, valid_int=1, checkpoint=None, init_weights=None,
         num_workers=5, pre_train=False, train_subset_path=None, kwargs_str=None):
    # Data loading
    _logger.info("Reading WebVision Dataset")
    subset = None if train_subset_path is None else np.load(train_subset_path)
    train_db, val_db, _ = wvc_data.get_datasets(pre_train, is_lmdb=False, subset=subset)
    balanced_sampler = WeightedRandomSampler(train_db.sample_weight, train_db.sample_weight.size, replacement=True)
    train_data_loader = dataloader.DataLoader(train_db, batch_size=batch_size, sampler=balanced_sampler,
                                              num_workers=num_workers, pin_memory=True)
    val_data_loader = dataloader.DataLoader(val_db, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    # Model building
    _logger.info("Building Model: {}".format(model_name))
    kwargs_dic = wvc_utils.get_kwargs_dic(kwargs_str)
    _logger.info("Arguments: {}".format(kwargs_dic))
    model = wvc_model.model_factory(model_name, kwargs_dic)
    if pre_train:
        model = wvc_model.PermLearning(model)
    _logger.info("Running model with {} GPUS and {} data workers".format(device_count(), num_workers))
    model = torch.nn.DataParallel(model).cuda()

    # Optionally load weights
    if init_weights is not None:
        _logger.info("Loading weights from {}".format(init_weights))
        init_weights = torch.load(init_weights)
        model.load_state_dict(init_weights['state_dict'])

    # Objective and Optimizer
    _logger.info("Setting up loss function and optimizer")
    criterion = torch.nn.CrossEntropyLoss() if not pre_train else wvc_model.WeightedMultiLabelBinaryCrossEntropy(False)
    criterion = criterion.cuda()
    init_lr = float(kwargs_dic.get("lr", 1e-1))
    optimizer = torch.optim.SGD([{'params': model.module.features.parameters(), 'lr': init_lr},
                                 {'params': model.module.classifier.parameters(), 'lr': init_lr}], lr=init_lr,
                                momentum=float(kwargs_dic.get("momentum", 0.9)),
                                weight_decay=float(kwargs_dic.get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(kwargs_dic.get('lr_step', 4)),
                                                gamma=float(kwargs_dic.get('lr_decay', 0.1)))

    # Optionally resume from a checkpoint
    if checkpoint is not None:
        _logger.info("Resume training from {}".format(checkpoint))
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        best_acc5 = checkpoint['best_acc5']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.step(start_epoch-1)

    else:
        start_epoch, best_acc5 = 0, 0.0

    # Training and Validation loop
    _logger.info("Training...")
    tb_logger = tb_log.Logger(output_dir)
    metric_func = wvc_model.multilabel_metrics if pre_train else wvc_model.top_k_acc
    best_metric_name = 'ml_acc' if pre_train else "acc5"
    for epoch in range(start_epoch, num_epochs):
        # update learning rate for this epoch
        scheduler.step()

        # train for one epoch
        tr_loss, metrics = wvc_model.train(train_data_loader, model, criterion, optimizer, metric_func, epoch)
        _logger.info("Epoch Train {}/{}: tr_loss={:.3f}, {}"
                     .format(epoch, num_epochs, tr_loss, ", ".join(["tr_{}={:.3f}".format(k, v) for k, v in metrics.items()])))
        tb_logger.log_value('tr_loss', tr_loss, epoch)
        for k, v in metrics.items():
            tb_logger.log_value('tr_{}'.format(k), v, epoch)

        # Validation
        if (epoch + 1) % valid_int == 0:
            _logger.info("Validating...")
            val_loss, metrics = wvc_model.validate(val_data_loader, model, criterion, metric_func, epoch)
            _logger.info("Epoch Validation {}/{}: val_loss={:.3f}, {}"
                         .format(epoch, num_epochs, val_loss, ", ".join(["val_{}={:.3f}".format(k,  v) for k, v in metrics.items()])))
            tb_logger.log_value('val_loss', val_loss, epoch)
            for k, v in metrics.items():
                tb_logger.log_value('val_{}'.format(k), v, epoch)
            curr_metric_val = metrics[best_metric_name]

            # save checkpoint
            model_ckpt_name = 'checkpoint.pth.tar'
            _logger.info("Save model checkpoint to {}".format(os.path.join(output_dir, model_ckpt_name)))
            is_best = curr_metric_val >= best_acc5
            best_acc5 = max(curr_metric_val, best_acc5)
            wvc_model.save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc5': best_acc5}, is_best, output_dir, model_ckpt_name)


if __name__ == '__main__':
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Training and Validation CNN Model.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_name', type=str, choices=wvc_model.MODELS, help='Name of the CNN architecture.')
    parser.add_argument('output_dir', type=str, help='Path to the output directory.')
    parser.add_argument('-batch_size', type=int, default=320, help='Batch size.')
    parser.add_argument('-num_epochs', type=int, default=15, help='Number of epochs.')
    parser.add_argument('-valid_int', type=int, default=1, help='Number epochs between evaluations.')
    parser.add_argument('-ckp_file', type=str, default=None, help='Resume from checkpoint file.')
    parser.add_argument('-init_weights', type=str, default=None, help='Start from pretrained weights.')
    parser.add_argument('-gpu_str', type=str, default="0", help='Set CUDA_VISIBLE_DEVICES variable.')
    parser.add_argument('-num_workers', type=int, default=5, help='Number of preprocessing workers.')
    parser.add_argument('-pre_train', default=False, action='store_true', help='Pretrain the model.')
    parser.add_argument('-train_subset', default=None, help='Path to the array with a subset of training images ids.')
    parser.add_argument('-kwargs_str', type=str, default=None,
                        help="Hyper parameters as string of key value, e.g., k1=v1; k2=v2; ...")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_str
    log_file = os.path.join(args.output_dir, "train_val_{}.log".format(datetime.now().strftime("%Y%m%d-%H%M%S")))
    wvc_utils.init_logging(log_file)
    _logger.info("Training and Validation CNN Model Tool: {}".format(args))
    main(args.model_name, args.output_dir, args.batch_size, args.num_epochs, args.valid_int, args.ckp_file,
         args.init_weights, args.num_workers, args.pre_train, args.train_subset, args.kwargs_str)
