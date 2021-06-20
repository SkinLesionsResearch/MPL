import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
import network
from torch.utils.data import DataLoader
import torch.nn.functional as F
from object.data_list import ImageList
import json
import random
from evaluation.metrics import get_metrics, get_test_data
from object.transforms import image_test, image_train
from object.loss import CrossEntropyLabelSmooth
from object.imbalanced import ImbalancedDatasetSampler
from object import utils

import warnings
warnings.filterwarnings("ignore")

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_txt = open(osp.join(args.src_dset_path, 'train.txt')).readlines()
    test_txt = open(osp.join(args.src_dset_path, 'test.txt')).readlines()
    dsets["train"] = ImageList(train_txt, args, transform=image_train())
    if args.imb == True:
        dset_loaders["train"] = DataLoader(dsets["train"], batch_size=args.batch_size,
                                           sampler=ImbalancedDatasetSampler(dsets["train"]),
                                           num_workers=args.worker, drop_last=True)
    else:
        dset_loaders["train"] = DataLoader(dsets["train"], batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=args.worker, drop_last=True)
    dsets["test"] = ImageList(test_txt, args, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.worker, drop_last=True)
    print('Training Data Distribution:')
    df_train = pd.DataFrame(dsets["train"].imgs, columns=['img_path', 'grade'])
    print(df_train.groupby('grade').count())
    print('\nTest Data Distribution:')
    df_train = pd.DataFrame(dsets["test"].imgs, columns=['img_path', 'grade'])
    print(df_train.groupby('grade').count())
    return dset_loaders

def train_source(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    dset_loaders = data_load(args)
    ## set base network
    net = utils.get_model(args.net, args.num_classes)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    acc_init = 0
    iter_per_epoch = len(dset_loaders["train"])
    max_iter = args.max_epoch * len(dset_loaders["train"])
    interval_iter = max_iter // 10
    iter_num = 0

    net.train()

    losses = []
    while iter_num < max_iter:
        epoch = int(iter_num / iter_per_epoch)
        try:
            inputs_src, lables_src = iter_source.next()
        except:
            iter_source = iter(dset_loaders["train"])
            inputs_src, lables_src = iter_source.next()
        if inputs_src.size(0) == 1:
            continue
        iter_num += 1

        inputs_src, lables_src = inputs_src.cuda(), lables_src.cuda()
        features_src, logits_src = net(inputs_src)

        if args.smooth > 0:
            loss = CrossEntropyLabelSmooth(num_classes=args.num_classes, epsilon=args.smooth)(logits_src, lables_src)
        else:
            loss = F.cross_entropy(logits_src, lables_src)
        losses.append(loss.item())

        print('epoch:{}/{}, iter:{}/{}, loss: {:.2f}'
              .format(epoch + 1, args.max_epoch, iter_num, max_iter, loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(epoch)

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            net.eval()
            features, logits, y_true, y_predict = get_test_data(dset_loaders['test'], net)
            accuracy, kappa, report, sensitivity, specificity, roc_auc = get_metrics(logits, y_true, y_predict)

            if args.num_classes == 2:
                log_str = 'Epoch:{}/{}, Iter:{}/{}; Accuracy = {:.2f}%, Kappa = {:.4f},' \
                          ' Sensitivity = {:.4f}, Specificity = {:.4f}, AUROC = {:.4f}' \
                    .format(epoch + 1, args.max_epoch, iter_num, max_iter, accuracy, kappa, sensitivity, specificity, roc_auc)
            else:
                log_str = 'Epoch:{}/{}, Iter:{}/{}; Accuracy = {:.2f}%, Kappa = {:.4f},'.format(epoch + 1, args.max_epoch, iter_num, max_iter, accuracy, kappa)

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')

            if accuracy >= acc_init:
                acc_init = accuracy
                torch.save(net.state_dict(), osp.join(args.output_dir_train, "best_params.pt"))

            net.train()

    with open(osp.join(args.output_dir_train, 'losses.txt'), "w") as fp:
        json.dump(losses, fp)

    return net

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='My Classification')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--num_classes', type=int, default=2, help="number of classes")
    parser.add_argument('--max_epoch', type=int, default=60, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--step_size', type=int, default=15, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--lr', type=float, default=5e-5, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50')
    parser.add_argument('--seed', type=int, default=2021, help="random seed")
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--smooth', type=float, default=0)
    parser.add_argument('--imb', type=bool, default=False, help="imbalanced sampler")
    parser.add_argument('--suffix', type=str, default='')
    args = parser.parse_args()

    args.src_dset_path = './data'

    args.output_dir_train = os.path.join('./ckps/', args.net + args.suffix)
    if not osp.exists(args.output_dir_train):
        os.system('mkdir -p ' + args.output_dir_train)
    if not osp.exists(args.output_dir_train):
        os.makedirs(args.output_dir_train)

    args.out_file = open(osp.join(args.output_dir_train, 'log.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    train_source(args)

