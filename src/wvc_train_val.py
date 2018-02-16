from torchvision import transforms
from torch.utils.data import dataloader
from torch.cuda import device_count
import torch
import wvc_data, wvc_model, wvc_utils
import logging, os



_logger = logging.getLogger(__name__)


def main(model_name, output_dir, batch_size=128, num_epochs=100, valid_int=25, checkpoint=None, num_workers=5, kwargs_str=None):
    # Data loading
    _logger.info("Reading WebVision Dataset")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_db = wvc_data.WebVision('train', transform=transforms.Compose([transforms.RandomResizedCrop(224),
                                                                         transforms.RandomHorizontalFlip(),
                                                                         transforms.ToTensor(), normalize]))
    train_data_loader = dataloader.DataLoader(train_db, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    val_db = wvc_data.WebVision('val', transform=transforms.Compose([transforms.CenterCrop(224),
                                                                     transforms.ToTensor(), normalize]))
    val_data_loader = dataloader.DataLoader(val_db, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    # Model building
    _logger.info("Building Model: {}".format(model_name))
    kwargs_dic = wvc_utils.get_kwargs_dic(kwargs_str)
    _logger.info("Arguments: {}".format(kwargs_dic))
    model = wvc_model.model_factory(model_name, kwargs_dic)
    _logger.info("Parallelizing model in {} GPUS".format(device_count()))
    model = torch.nn.DataParallel(model).cuda()
    if checkpoint is not None:
        _logger.info("Resume training from {}".format(checkpoint))
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] - 1
        best_acc5 = checkpoint['best_acc5']
        model.load_state_dict(checkpoint['state_dict'])
    else:
        start_epoch, best_acc5 = 0, 0.0

    # Objective and Optimizer
    _logger.info("Setting up loss function and optimizer")
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(kwargs_dic.get("lr", 1e-3)),
                                 weight_decay=float(kwargs_dic.get("weight_decay", 1e-4)))

    # Training and Validation loop
    _logger.info("Training...")
    for epoch in range(start_epoch, num_epochs):
        # train for one epoch
        wvc_model.train(train_data_loader, model, criterion, optimizer, epoch)

        # Validation
        if (epoch + 1) % valid_int == 0:
            _logger.info("Validating...")
            val_loss, val_acc1, val_acc5 = wvc_model.validate(val_data_loader, model, criterion, epoch)
            _logger.info("Epoch {}/{}: val_loss={:.3f}, val_acc1={:.3f}, val_acc5={:.3f}"
                         .format(epoch+1, num_epochs, val_loss, val_acc1, val_acc5))

            # save checkpoint
            model_ckp_name = "M{}_E{}_L{:.3f}_ACC1{:.3f}_ACC5_{:.3f}.pth.tar".format(model_name, epoch+1, val_loss, val_acc1, val_acc5)
            _logger.info("Save model checkpoint to {}".format(os.path.join(output_dir, model_ckp_name)))
            is_best = val_acc5 > best_acc5
            best_acc5 = max(val_acc5, best_acc5)
            wvc_model.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc5': best_acc5}, is_best, output_dir, model_ckp_name)


if __name__ == '__main__':
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Training and Validation CNN Model.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_name', type=str, choices=wvc_model.MODELS, help='Name of the CNN architecture.')
    parser.add_argument('output_dir', type=str, help='Path to the output directory.')
    parser.add_argument('-batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('-num_epochs', type=int, default=100, help='Number of epochs.')
    parser.add_argument('-valid_int', type=int, default=25, help='Number epochs between evaluations.')
    parser.add_argument('-ckp_file', type=str, default=None, help='Resume from checkpoint file.')
    parser.add_argument('-gpu_str', type=str, default="0", help='Set CUDA_VISIBLE_DEVICES variable.')
    parser.add_argument('-num_workers', type=int, default=5, help='Number of preprocessing workers.')
    parser.add_argument('-kwargs_str', type=str, default=None, help="Hyper parameters as string of key value, e.g., k1=v1; k2=v2; ...")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_str
    log_file = os.path.join(args.output_dir, "train_val_{}.log".format(datetime.now().strftime("%Y%m%d-%H%M%S")))
    wvc_utils.init_logging(log_file)
    _logger.info("Training and Validation CNN Model Tool: {}".format(args))
    main(args.model_name, args.output_dir, args.batch_size, args.num_epochs, args.valid_int, args.ckp_file,
         args.num_workers, args.kwargs_str)
