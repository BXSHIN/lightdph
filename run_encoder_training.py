from __future__ import print_function

import os
import sys
import argparse
import time
import math

from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from tqdm import tqdm, trange
from datasets import DERMADataset
from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConEfficientNet, LHACv1
from losses_hac import SupConLoss
# from losses_hmlc import HMLC

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    exp_num = 'exp1'
    group_name = f"{exp_num}-LHAC-res18"

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=20,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=70,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=4, # 15 6-gpu 96%-mem resnet
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=70,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='50,80,120',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='efficientnet-b0')
    parser.add_argument('--dataset', type=str, default='path',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='HMLC',
                        choices=['SupCon', 'SimCLR', 'HIL', 'LHAC', 'HMLC'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--w_super', type=int, default=0,
                        help='weight for superclass')


    opt = parser.parse_args()

    opt.w_super = 0.05
    opt.data_folder = f'/mnt/nfs_bdalab4_data/bennyhsu/mlhc/subset_multi' # ISIC_2019 train subset
    
    ## skin dataset
    # opt.data_folder = f'/mnt/nfs_bdalab4_data/bennyhsu/mlhc/skin_3/five_fold/{exp_num}/train'
    opt.mean = (0.6678, 0.5300, 0.5245)
    opt.std = (0.1320, 0.1462, 0.1573)
    opt.hierarchy = [[[2, 3, 5, 7]], [[0], [1, 6], [4]]] # isic 2019 LHAC
    # opt.hierarchy = [[[3, 5]], [[0], [1, 4], [2]]] # pad-ufes LHAC
    opt.weights = [0, 1, 0.67]
    
    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    
    if 'skin' in opt.data_folder:
        ## for skin
        opt.model_path = './model_save/encoder/{}_models'.format(group_name)   

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_{}_{}_lr_{}_bsz_{}'.\
        format(opt.w_super, exp_num, opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate
    

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):

    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    # use skin dataset
    elif opt.dataset == 'path':
        # mean = eval(opt.mean)
        # std = eval(opt.std)
        mean = opt.mean
        std = opt.std
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    
    normalize = transforms.Normalize(mean=mean, std=std)


    train_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        normalize,
    ])
    
    train_dataset = ''
    print('Setting training dataset...')
    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
    elif opt.dataset == 'path':
        train_dataset = DERMADataset(data_location=opt.data_folder, transform=train_transform)        
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader


def set_model(opt):
    if opt.method == 'LHAC':
        model = LHACv1(name=opt.model)
        criterion = SupConLoss(temperature=opt.temp, contrast_mode='hil')
    else:
        model = SupConEfficientNet(name=opt.model)
        # model = SupConResNet(name='resnet18')
        if opt.method == 'HIL' or 'HAC':
            criterion = SupConLoss(temperature=opt.temp, contrast_mode='hil')
        else:
            criterion = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    ## "_" shouhld be the path
    for idx, (images, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        if opt.method == 'LHAC':
            hie_features, features = model(images)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            f3, f4 = torch.split(hie_features, [bsz, bsz], dim=0)
            hie_features = torch.cat([f3.unsqueeze(1), f4.unsqueeze(1)], dim=1)            
            loss1 = criterion(features, labels, w_super=0)
            loss2 = criterion(hie_features, labels, w_super=opt.w_super, weight=opt.weights, hierarchy=opt.hierarchy)            
            loss = (loss1+loss2)/2.
        else:    
            features = model(images)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            # (add the 'n_views' dimension here)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
 
            if opt.method == 'SupCon' or opt.method == 'HIL':
                loss = criterion(features, labels, w_super=opt.w_super)
            elif opt.method == 'SimCLR':
                loss = criterion(features)
            else:
                raise ValueError('contrastive method not supported: {}'.
                                format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

  
    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':    
    
    main()
