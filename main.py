import argparse
import logging
import math
import os
import random
import time

import numpy as np
import torch
from torch.cuda import amp
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import DATASET_GETTERS
from data import x_u_split, TransformMPL, cifar10_mean, cifar10_std
import pandas as pd
from models import WideResNet, ModelEMA
from utils import (AverageMeter, accuracy, create_loss_fn,
                   save_checkpoint, reduce_tensor, model_load_state_dict)
import os.path as osp
from object.data_list import ImageList
from object.transforms import image_test, image_train

logger = logging.getLogger(__name__)
# --name cifar10-4K.5 --expand-labels --dataset cifar10 --num-classes 10 --num-labeled 4000 --total-steps 300000 --eval-step 1000 --randaug 2 16 --batch-size 128 --teacher_lr 0.05 --student_lr 0.05 --weight-decay 5e-4 --ema 0.995 --nesterov --mu 7 --label-smoothing 0.15 --temperature 0.7 --threshold 0.6 --lambda-u 8 --warmup-steps 5000 --uda-steps 5000 --student-wait-steps 3000 --teacher-dropout 0.2 --student-dropout 0.2
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default="cifar10-4K.5", help='experiment name')
parser.add_argument('--data-path', default='./data', type=str, help='data path')
parser.add_argument('--save-path', default='./checkpoint', type=str, help='save path')
parser.add_argument('--dataset', default='cifar10', type=str,
                    choices=['cifar10', 'cifar100'], help='dataset name')
parser.add_argument('--num-labeled', type=int, default=4000, help='number of labeled data')
parser.add_argument("--expand-labels", action="store_true", help="expand labels to fit eval steps")
parser.add_argument('--total-steps', default=300000, type=int, help='number of total steps to run')
parser.add_argument('--eval-step', default=500, type=int, help='number of eval steps to run')
parser.add_argument('--start-step', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--workers', default=6, type=int, help='number of workers')
parser.add_argument('--num-classes', default=7, type=int, help='number of classes')
parser.add_argument('--perc-labeled', default=0.8, type=float, help='number of classes')

parser.add_argument('--resize', default=32, type=int, help='resize image')
parser.add_argument('--batch-size', default=32, type=int, help='train batch size')
parser.add_argument('--teacher-dropout', default=0, type=float, help='dropout on last dense layer')
parser.add_argument('--student-dropout', default=0, type=float, help='dropout on last dense layer')
parser.add_argument('--teacher_lr', default=0.01, type=float, help='train learning late')
parser.add_argument('--student_lr', default=0.01, type=float, help='train learning late')
parser.add_argument('--momentum', default=0.9, type=float, help='SGD Momentum')
parser.add_argument('--nesterov', action='store_true', help='use nesterov')
parser.add_argument('--weight-decay', default=0, type=float, help='train weight decay')
parser.add_argument('--ema', default=0, type=float, help='EMA decay rate')
parser.add_argument('--warmup-steps', default=0, type=int, help='warmup steps')
parser.add_argument('--student-wait-steps', default=0, type=int, help='warmup steps')
parser.add_argument('--grad-clip', default=0., type=float, help='gradient norm clipping')  # 梯度截断
parser.add_argument('--resume', default='', type=str, help='path to checkpoint')    # 恢复
parser.add_argument('--evaluate', action='store_true', help='only evaluate model on validation set')
parser.add_argument('--finetune', action='store_true',
                    help='only finetune model on labeled dataset')
parser.add_argument('--finetune-epochs', default=125, type=int, help='finetune epochs')
parser.add_argument('--finetune-batch-size', default=512, type=int, help='finetune batch size')
parser.add_argument('--finetune-lr', default=1e-5, type=float, help='finetune learning late')
parser.add_argument('--finetune-weight-decay', default=0, type=float, help='finetune weight decay')
parser.add_argument('--finetune-momentum', default=0, type=float, help='finetune SGD Momentum')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--label-smoothing', default=0, type=float, help='label smoothing alpha')
parser.add_argument('--mu', default=7, type=int, help='coefficient of unlabeled batch size')
parser.add_argument('--threshold', default=0.95, type=float, help='pseudo label threshold')
parser.add_argument('--temperature', default=1, type=float, help='pseudo label temperature')
parser.add_argument('--lambda-u', default=1, type=float, help='coefficient of unlabeled loss')
parser.add_argument('--uda-steps', default=1, type=float, help='warmup steps of lambda-u')
parser.add_argument("--randaug", nargs="+", type=int, help="use it like this. --randaug 2 10")
parser.add_argument("--amp", action="store_true", help="use 16-bit (mixed) precision")
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument("--local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_wait_steps=0,
                                    num_cycles=0.5,
                                    last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_wait_steps:
            return 0.0

        if current_step < num_warmup_steps + num_wait_steps:
            return float(current_step) / float(max(1, num_warmup_steps + num_wait_steps))

        progress = float(current_step - num_warmup_steps - num_wait_steps) / \
            float(max(1, num_training_steps - num_warmup_steps - num_wait_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def train_loop(args, labeled_loader, unlabeled_loader, test_loader,
               teacher_model, student_model, avg_student_model, criterion,
               t_optimizer, s_optimizer, t_scheduler, s_scheduler, t_scaler, s_scaler):
    logger.info("***** Running Training *****")
    logger.info(f"   Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"   Total steps = {args.total_steps}")

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_loader.sampler.set_epoch(labeled_epoch)
        unlabeled_loader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    for step in range(args.start_step, args.total_steps):
        if step % args.eval_step == 0:
            pbar = tqdm(range(args.eval_step), disable=args.local_rank not in [-1, 0])
            batch_time = AverageMeter()
            data_time = AverageMeter()
            s_losses = AverageMeter()
            t_losses = AverageMeter()
            t_losses_l = AverageMeter()
            t_losses_u = AverageMeter()
            t_losses_mpl = AverageMeter()
            mean_mask = AverageMeter()

        teacher_model.train()
        student_model.train()
        end = time.time()

        try:
            images_l, targets = labeled_iter.next()
        except:
            if args.world_size > 1:
                labeled_epoch += 1
                labeled_loader.sampler.set_epoch(labeled_epoch)
            labeled_iter = iter(labeled_loader)
            images_l, targets = labeled_iter.next()

        try:
            (images_uw, images_us), _ = unlabeled_iter.next()
        except:
            if args.world_size > 1:
                unlabeled_epoch += 1
                unlabeled_loader.sampler.set_epoch(unlabeled_epoch)
            unlabeled_iter = iter(unlabeled_loader)
            (images_uw, images_us), _ = unlabeled_iter.next()

        data_time.update(time.time() - end)

        images_l = images_l.to(args.gpu)
        images_uw = images_uw.to(args.gpu)
        images_us = images_us.to(args.gpu)
        targets = targets.to(args.gpu)
        with amp.autocast(enabled=args.amp):
            batch_size = images_l.shape[0]
            t_images = torch.cat((images_l, images_uw, images_us))
            t_logits = teacher_model(t_images)
            t_logits_l = t_logits[:batch_size]
            t_logits_uw, t_logits_us = t_logits[batch_size:].chunk(2)
            del t_logits

            targets = targets.long()
            t_loss_l = criterion(t_logits_l, targets)

            soft_pseudo_label = torch.softmax(t_logits_uw.detach() / args.temperature, dim=-1)
            max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()
            t_loss_u = torch.mean(
                -(soft_pseudo_label * torch.log_softmax(t_logits_us, dim=-1)).sum(dim=-1) * mask
            )
            weight_u = args.lambda_u * min(1., (step + 1) / args.uda_steps)
            t_loss_uda = t_loss_l + weight_u * t_loss_u

            s_images = torch.cat((images_l, images_us))
            s_logits = student_model(s_images)
            s_logits_l = s_logits[:batch_size]
            s_logits_us = s_logits[batch_size:]
            del s_logits

            s_loss_l_old = F.cross_entropy(s_logits_l.detach(), targets.type(torch.int64))
            s_loss = criterion(s_logits_us, hard_pseudo_label)

        s_scaler.scale(s_loss).backward()
        if args.grad_clip > 0:
            s_scaler.unscale_(s_optimizer)
            nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)
        s_scaler.step(s_optimizer)
        s_scaler.update()
        s_scheduler.step()
        if args.ema > 0:
            avg_student_model.update_parameters(student_model)

        with amp.autocast(enabled=args.amp):
            with torch.no_grad():
                s_logits_l = student_model(images_l)
            s_loss_l_new = F.cross_entropy(s_logits_l.detach(), targets.type(torch.int64))
#             dot_product = s_loss_l_new - s_loss_l_old
            # test
            dot_product = s_loss_l_old - s_loss_l_new
#             moving_dot_product = moving_dot_product * 0.99 + dot_product * 0.01
#             dot_product = dot_product - moving_dot_product
            _, hard_pseudo_label = torch.max(t_logits_us.detach(), dim=-1)
            a = F.cross_entropy(t_logits_us, hard_pseudo_label)
            print(f"{a}____{dot_product}")
            t_loss_mpl = dot_product * a
            t_loss = t_loss_uda + t_loss_mpl

        t_scaler.scale(t_loss).backward()
        if args.grad_clip > 0:
            t_scaler.unscale_(t_optimizer)
            nn.utils.clip_grad_norm_(teacher_model.parameters(), args.grad_clip)
        t_scaler.step(t_optimizer)
        t_scaler.update()
        t_scheduler.step()

        teacher_model.zero_grad()
        student_model.zero_grad()

        if args.world_size > 1:
            s_loss = reduce_tensor(s_loss.detach(), args.world_size)
            t_loss = reduce_tensor(t_loss.detach(), args.world_size)
            t_loss_l = reduce_tensor(t_loss_l.detach(), args.world_size)
            t_loss_u = reduce_tensor(t_loss_u.detach(), args.world_size)
            t_loss_mpl = reduce_tensor(t_loss_mpl.detach(), args.world_size)
            mask = reduce_tensor(mask, args.world_size)

        s_losses.update(s_loss.item())
        t_losses.update(t_loss.item())
        t_losses_l.update(t_loss_l.item())
        t_losses_u.update(t_loss_u.item())
        t_losses_mpl.update(t_loss_mpl.item())
        mean_mask.update(mask.mean().item())

        batch_time.update(time.time() - end)
        pbar.set_description(
            f"Train Iter: {step+1:3}/{args.total_steps:3}. "
            f"LR: {get_lr(s_optimizer):.4f}. Data: {data_time.avg:.2f}s. "
            f"Batch: {batch_time.avg:.2f}s. S_Loss: {s_losses.avg:.4f}. "
            f"T_Loss: {t_losses.avg:.4f}. Mask: {mean_mask.avg:.4f}. ")
        pbar.update()
        if args.local_rank in [-1, 0]:
            args.writer.add_scalar("lr", get_lr(s_optimizer), step)   # 写入文件

        args.num_eval = step // args.eval_step
        if (step + 1) % args.eval_step == 0:
            pbar.close()
            if args.local_rank in [-1, 0]:
                args.writer.add_scalar("train/1.s_loss", s_losses.avg, args.num_eval)
                args.writer.add_scalar("train/2.t_loss", t_losses.avg, args.num_eval)
                args.writer.add_scalar("train/3.t_labeled", t_losses_l.avg, args.num_eval)
                args.writer.add_scalar("train/4.t_unlabeled", t_losses_u.avg, args.num_eval)
                args.writer.add_scalar("train/5.t_mpl", t_losses_mpl.avg, args.num_eval)
                args.writer.add_scalar("train/6.mask", mean_mask.avg, args.num_eval)

                test_model = avg_student_model if avg_student_model is not None else student_model
                test_loss, top1, top5 = evaluate(args, test_loader, test_model, criterion)

                args.writer.add_scalar("test/loss", test_loss, args.num_eval)
                args.writer.add_scalar("test/acc@1", top1, args.num_eval)
                args.writer.add_scalar("test/acc@5", top5, args.num_eval)

                is_best = top1 > args.best_top1
                if is_best:
                    args.best_top1 = top1
                    args.best_top5 = top5

                logger.info(f"top-1 acc: {top1:.2f}")
                logger.info(f"Best top-1 acc: {args.best_top1:.2f}")

                save_checkpoint(args, {
                    'step': step + 1,
                    'teacher_state_dict': teacher_model.state_dict(),
                    'student_state_dict': student_model.state_dict(),
                    'avg_state_dict': avg_student_model.state_dict() if avg_student_model is not None else None,
                    'best_top1': args.best_top1,
                    'best_top5': args.best_top5,
                    'teacher_optimizer': t_optimizer.state_dict(),
                    'student_optimizer': s_optimizer.state_dict(),
                    'teacher_scheduler': t_scheduler.state_dict(),
                    'student_scheduler': s_scheduler.state_dict(),
                    'teacher_scaler': t_scaler.state_dict(),
                    'student_scaler': s_scaler.state_dict(),
                }, is_best)
    # finetune

    del t_scaler, t_scheduler, t_optimizer, teacher_model, unlabeled_loader
    del s_scaler, s_scheduler, s_optimizer
    ckpt_name = f'{args.save_path}/{args.name}_best.pth.tar'
    loc = f'cuda:{args.gpu}'
    checkpoint = torch.load(ckpt_name, map_location=loc)
    logger.info(f"=> loading checkpoint '{ckpt_name}'")
    if checkpoint['avg_state_dict'] is not None:
        model_load_state_dict(student_model, checkpoint['avg_state_dict'])
    else:
        model_load_state_dict(student_model, checkpoint['student_state_dict'])
    finetune(args, labeled_loader, test_loader, student_model, criterion)
    return


def evaluate(args, test_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    test_iter = tqdm(test_loader, disable=args.local_rank not in [-1, 0])
    with torch.no_grad():
        end = time.time()
        for step, (images, targets) in enumerate(test_iter):
            data_time.update(time.time() - end)
            batch_size = targets.shape[0]
            images = images.to(args.device)
            targets = targets.to(args.device)
            with amp.autocast(enabled=args.amp):
                outputs = model(images)
                loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs, targets, (1, 5))
            losses.update(loss.item(), batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)
            batch_time.update(time.time() - end)
            end = time.time()
            test_iter.set_description(
                f"Test Iter: {step + 1:3}/{len(test_loader):3}. Data: {data_time.avg:.2f}s. "
                f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}. "
                f"top1: {top1.avg:.2f}. top5: {top5.avg:.2f}. ")

        test_iter.close()
        return losses.avg, top1.avg, top5.avg


def finetune(args, train_loader, test_loader, model, criterion):
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    labeled_loader = DataLoader(
        train_loader.dataset,
        sampler=train_sampler(train_loader.dataset),
        batch_size=args.finetune_batch_size,
        num_workers=args.workers,
        pin_memory=True)
    optimizer = optim.SGD(model.parameters(),
                          lr=args.finetune_lr,
                          momentum=args.finetune_momentum,
                          weight_decay=args.finetune_weight_decay)
    scaler = amp.GradScaler(enabled=args.amp)

    logger.info("***** Running Finetuning *****")
    logger.info(f"   Finetuning steps = {len(labeled_loader) * args.finetune_epochs}")

    for epoch in range(args.finetune_epochs):
        if args.world_size > 1:
            labeled_loader.sampler.set_epoch(epoch + 624)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        model.train()
        end = time.time()
        labeled_iter = tqdm(labeled_loader, disable=args.local_rank not in [-1, 0])
        for step, (images, targets) in enumerate(labeled_iter):
            data_time.update(time.time() - end)
            batch_size = targets.shape[0]
            images = images.to(args.device)
            targets = targets.to(args.device)
            with amp.autocast(enabled=args.amp):
                model.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if args.world_size > 1:
                loss = reduce_tensor(loss.detach(), args.world_size)
            losses.update(loss.item(), batch_size)
            batch_time.update(time.time() - end)
            labeled_iter.set_description(
                f"Finetune Epoch: {epoch + 1:2}/{args.finetune_epochs:2}. Data: {data_time.avg:.2f}s. "
                f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}. ")
        labeled_iter.close()
        if args.local_rank in [-1, 0]:
            args.writer.add_scalar("finetune/train_loss", losses.avg, epoch)
            test_loss, top1, top5 = evaluate(args, test_loader, model, criterion)
            args.writer.add_scalar("finetune/test_loss", test_loss, epoch)
            args.writer.add_scalar("finetune/acc@1", top1, epoch)
            args.writer.add_scalar("finetune/acc@5", top5, epoch)
            is_best = top1 > args.best_top1
            if is_best:
                args.best_top1 = top1
                args.best_top5 = top5

            logger.info(f"top-1 acc: {top1:.2f}")
            logger.info(f"Best top-1 acc: {args.best_top1:.2f}")

            save_checkpoint(args, {
                'step': step + 1,
                'best_top1': args.best_top1,
                'best_top5': args.best_top5,
                'student_state_dict': model.state_dict(),
                'avg_state_dict': None,
                'student_optimizer': optimizer.state_dict(),
            }, is_best, finetune=True)
    return


def read_img_list(args):
    # labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](args)
    src_dset_path = './data_ham10000'
    train_labeled_txt = open(osp.join(src_dset_path, 'train.txt')).readlines()
    test_txt = open(osp.join(src_dset_path, 'test.txt')).readlines()
    train_df = pd.read_csv(osp.join(src_dset_path, 'train.txt'),
                           delimiter=" ",
                           index_col=None, header=None)
    train_df.columns = ["path", "labels"]
    labeled_idx, unlabeled_idx = x_u_split(args, train_df['labels'])
    train_labeled_df = train_df.iloc[labeled_idx]
    train_unlabeld_df = train_df.iloc[unlabeled_idx]
    train_labeled_list = ImageList(
                            image_list=list(train_labeled_df["path"]),
                            labels=list(train_labeled_df["labels"]),
                            args=args, transform=image_train(args.resize))

    train_unlabeled_list = ImageList(
                            image_list=list(train_unlabeld_df["path"]),
                            args=args, labels=list(train_unlabeld_df["labels"]),
                            transform=TransformMPL(args, mean=cifar10_mean,std=cifar10_std))
    test_image_list = ImageList(image_list=test_txt,
                                args=args, transform=image_test())
    return train_labeled_list, train_unlabeled_list, test_image_list


def main():
    args = parser.parse_args()
    args.best_top1 = 0.
    args.best_top5 = 0.

    if args.local_rank != -1:
        args.gpu = args.local_rank
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
    else:
        args.gpu = 0
        args.world_size = 1

    args.device = torch.device('cuda:0')  # torch.device('cuda', args.gpu)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARNING)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}")

    logger.info(dict(args._get_kwargs()))

    if args.local_rank in [-1, 0]:
        args.writer = SummaryWriter(f"results/{args.name}")

    if args.seed is not None:
        set_seed(args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    train_labeled_list, train_unlabeled_list,  test_image_list = read_img_list(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_loader = DataLoader(
        train_labeled_list,
        sampler=train_sampler(train_labeled_list),  # 采样，类似于打乱顺序
        batch_size=args.batch_size,
        num_workers=args.workers,  # 线程
        drop_last=True)  # batch_size最后剩下的要不要
    # for step, (x, y) in enumerate(labeled_loader):
    #     print(step, ", ", x.shape, ", ", y.shape)
    #     break
    # print("check for train_labeled_list")
    # for img,target in train_labeled_list:
    #     print(img.shape, ", ", target)
    #     break

    unlabeled_loader = DataLoader(
        train_unlabeled_list,
        sampler=train_sampler(train_unlabeled_list),
        batch_size=args.batch_size * args.mu,
        num_workers=args.workers,
        drop_last=True)
    # for step, (x, y) in enumerate(unlabeled_loader):
    #     print(step, ", ", x.shape, ", ", y.shape)
    #     break

    test_loader = DataLoader(test_image_list,
                             sampler=SequentialSampler(test_image_list),
                             batch_size=args.batch_size,
                             num_workers=args.workers)
    # for step, (x, y) in enumerate(test_loader):
    #     print(step, ", ", x.shape, ", ", y.shape)
    #     break

    if args.dataset == "cifar10":
        depth, widen_factor = 28, 2
    elif args.dataset == 'cifar100':
        depth, widen_factor = 28, 8

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    teacher_model = WideResNet(num_classes=args.num_classes,
                               depth=depth,
                               widen_factor=widen_factor,
                               dropout=0,
                               dense_dropout=args.teacher_dropout)
    student_model = WideResNet(num_classes=args.num_classes,
                               depth=depth,
                               widen_factor=widen_factor,
                               dropout=0,
                               dense_dropout=args.student_dropout)

    if args.local_rank == 0:
        torch.distributed.barrier()

    logger.info(f"Model: WideResNet {depth}x{widen_factor}")
    logger.info(f"Params: {sum(p.numel() for p in teacher_model.parameters()) / 1e6:.2f}M")

    teacher_model.to(args.device)
    student_model.to(args.device)
    avg_student_model = None
    if args.ema > 0:
        avg_student_model = ModelEMA(student_model, args.ema)

    criterion = create_loss_fn(args)

    no_decay = ['bn']
    teacher_parameters = [
        {'params': [p for n, p in teacher_model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in teacher_model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    student_parameters = [
        {'params': [p for n, p in student_model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in student_model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    t_optimizer = optim.SGD(teacher_parameters,
                            lr=args.teacher_lr,
                            momentum=args.momentum,
                            # weight_decay=args.weight_decay,
                            nesterov=args.nesterov)
    s_optimizer = optim.SGD(student_parameters,
                            lr=args.student_lr,
                            momentum=args.momentum,
                            # weight_decay=args.weight_decay,
                            nesterov=args.nesterov)

    t_scheduler = get_cosine_schedule_with_warmup(t_optimizer,
                                                  args.warmup_steps,
                                                  args.total_steps)
    s_scheduler = get_cosine_schedule_with_warmup(s_optimizer,
                                                  args.warmup_steps,
                                                  args.total_steps,
                                                  args.student_wait_steps)

    t_scaler = amp.GradScaler(enabled=args.amp)
    s_scaler = amp.GradScaler(enabled=args.amp)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"=> loading checkpoint '{args.resume}'")
            loc = f'cuda:{args.gpu}'
            checkpoint = torch.load(args.resume, map_location=loc)
            args.best_top1 = checkpoint['best_top1'].to(torch.device('cpu'))
            args.best_top5 = checkpoint['best_top5'].to(torch.device('cpu'))
            if not (args.evaluate or args.finetune):
                args.start_step = checkpoint['step']
                t_optimizer.load_state_dict(checkpoint['teacher_optimizer'])
                s_optimizer.load_state_dict(checkpoint['student_optimizer'])
                t_scheduler.load_state_dict(checkpoint['teacher_scheduler'])
                s_scheduler.load_state_dict(checkpoint['student_scheduler'])
                t_scaler.load_state_dict(checkpoint['teacher_scaler'])
                s_scaler.load_state_dict(checkpoint['student_scaler'])
                model_load_state_dict(teacher_model, checkpoint['teacher_state_dict'])
                if avg_student_model is not None:
                    model_load_state_dict(avg_student_model, checkpoint['avg_state_dict'])

            else:
                if checkpoint['avg_state_dict'] is not None:
                    model_load_state_dict(student_model, checkpoint['avg_state_dict'])
                else:
                    model_load_state_dict(student_model, checkpoint['student_state_dict'])

            logger.info(f"=> loaded checkpoint '{args.resume}' (step {checkpoint['step']})")
        else:
            logger.info(f"=> no checkpoint found at '{args.resume}'")

    if args.local_rank != -1:
        teacher_model = nn.parallel.DistributedDataParallel(
            teacher_model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)
        student_model = nn.parallel.DistributedDataParallel(
            student_model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    if args.finetune:
        del t_scaler, t_scheduler, t_optimizer, teacher_model, unlabeled_loader
        del s_scaler, s_scheduler, s_optimizer
        finetune(args, labeled_loader, test_loader, student_model, criterion)
        return

    if args.evaluate:
        del t_scaler, t_scheduler, t_optimizer, teacher_model, unlabeled_loader, labeled_loader
        del s_scaler, s_scheduler, s_optimizer
        evaluate(args, test_loader, student_model, criterion)
        return

    teacher_model.zero_grad()
    student_model.zero_grad()
    train_loop(args, labeled_loader, unlabeled_loader, test_loader,
               teacher_model, student_model, avg_student_model, criterion,
               t_optimizer, s_optimizer, t_scheduler, s_scheduler, t_scaler, s_scaler)
    return


if __name__ == '__main__':
    main()
