from torch.utils.data import dataloader
from torch.nn import functional as t_func
from torch.cuda import device_count
import torch
import wvc_data, wvc_model, wvc_utils
import logging, os, warnings
from tqdm import tqdm
import numpy as np
warnings.filterwarnings("ignore")
_logger = logging.getLogger(__name__)


def main(model_name, model_ckp, split_name, submission_file, batch_size=320, num_workers=5, kwargs_str=None):
    # Data loading
    _logger.info("Reading WebVision Dataset")
    train_db, val_db, test_db = wvc_data.get_datasets(pre_train=False, is_lmdb=False)
    test_db = {'train': train_db, 'val': val_db, 'test': test_db}[split_name]
    test_data_loader = dataloader.DataLoader(test_db, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    # Model building
    _logger.info("Building Model: {}".format(model_name))
    kwargs_dic = wvc_utils.get_kwargs_dic(kwargs_str)
    _logger.info("Arguments: {}".format(kwargs_dic))
    model = wvc_model.model_factory(model_name, kwargs_dic)
    _logger.info("Running model with {} GPUS and {} data workers".format(device_count(), num_workers))
    model = torch.nn.DataParallel(model).cuda()

    # Loading weights
    _logger.info("Loading model state from {}".format(model_ckp))
    checkpoint = torch.load(model_ckp)
    model.load_state_dict(checkpoint['state_dict'])

    # Test Model
    _logger.info("Testing model")
    model.eval()
    submission_values = []
    for i, (img_idxs, images, _) in enumerate(tqdm(test_data_loader, desc='Test iterations')):
        image_var = torch.autograd.Variable(images.cuda(async=True), volatile=True)
        img_idxs = np.array(img_idxs, np.str)

        # compute prediction and loss
        bs, ncrops, c, h, w = image_var.size()
        image_var = image_var.view(-1, c, h, w)
        y_prob = t_func.softmax(model(image_var), -1)
        y_prob = y_prob.view(bs, ncrops, -1).mean(1)
        y_pred = torch.topk(y_prob.data, 5, -1)[1].int()
        for file, preds in zip(img_idxs, y_pred.cpu().numpy()):
            submission_values.append([file] + preds.astype(np.str).tolist())
    _logger.info("Predicted top 5 labels for {} images.".format(len(submission_values)))

    # Save submission file
    _logger.info("Saving submission file to {}".format(submission_file))
    with open(submission_file, 'w') as f:
        for line in submission_values:
            f.write("{}\n".format("\t".join(line)))
    _logger.info("Test model has finished!")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Training and Validation CNN Model.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_name', type=str, choices=wvc_model.MODELS, help='Name of the CNN architecture.')
    parser.add_argument('model_ckp', type=str, help='Model checkpoint file.')
    parser.add_argument('split_name', type=str, help='Webvision split name')
    parser.add_argument('submission_file', type=str, help='Path to output submission file.')
    parser.add_argument('-batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('-gpu_str', type=str, default="0", help='Set CUDA_VISIBLE_DEVICES variable.')
    parser.add_argument('-num_workers', type=int, default=5, help='Number of preprocessing workers.')
    parser.add_argument('-kwargs_str', type=str, default=None,
                        help="Hyper parameters as string of key value, e.g., k1=v1; k2=v2; ...")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_str
    wvc_utils.init_logging()
    _logger.info("Test CNN Model Tool: {}".format(args))
    main(args.model_name, args.model_ckp, args.split_name, args.submission_file, args.batch_size, args.num_workers,
         args.kwargs_str)
