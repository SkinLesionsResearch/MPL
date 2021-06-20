import argparse
import logging
from object.data_list import ImageList
from object.transforms import image_test, image_train
import os.path as osp
from torch.utils.data import DataLoader
from object.imbalanced import ImbalancedDatasetSampler
import pandas as pd

logger = logging.getLogger(__name__)
# --name cifar10-4K.5 --expand-labels --dataset cifar10 --num-classes 10 --num-labeled 4000 --total-steps 300000 --eval-step 1000 --randaug 2 16 --batch-size 128 --teacher_lr 0.05 --student_lr 0.05 --weight-decay 5e-4 --ema 0.995 --nesterov --mu 7 --label-smoothing 0.15 --temperature 0.7 --threshold 0.6 --lambda-u 8 --warmup-steps 5000 --uda-steps 5000 --student-wait-steps 3000 --teacher-dropout 0.2 --student-dropout 0.2
parser = argparse.ArgumentParser(description='My Classification')

parser.add_argument('--name', type=str, default="cifar10-4K.5", help='experiment name')
parser.add_argument('--data-path', default='./data', type=str, help='data path')
parser.add_argument('--save-path', default='./checkpoint', type=str, help='save path')
parser.add_argument('--batch-size', default=32, type=int, help='batch size')
parser.add_argument('--worker', default=32, type=int, help='batch size')
parser.add_argument('--max_epoch', type=int, default=60, help="max iterations")
parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
parser.add_argument('--step_size', type=int, default=15, help="batch_size")
parser.add_argument('--epsilon', type=float, default=1e-5)
parser.add_argument('--smooth', type=float, default=0)
parser.add_argument('--imb', type=bool, default=False, help="imbalanced sampler")
parser.add_argument('--suffix', type=str, default='')


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_txt = open(osp.join(args.src_dset_path, 'train.txt')).readlines()
    test_txt = open(osp.join(args.src_dset_path, 'test.txt')).readlines()
    dsets["train"] = ImageList(train_txt, args, transform=image_train())
    if args.imb == True:
        dset_loaders["train"] = DataLoader(dsets["train"], batch_size=args.batch_size,
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


def main():
    args = parser.parse_args()
    args.src_dset_path = './data_ham10000'
    dset_loaders = data_load(args)


if __name__ == '__main__':
    main()
