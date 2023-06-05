import argparse
import sys
import builtins
import os
import random
import shutil
import time
import warnings
from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torch.optim as optim
from dataset import input_dataset
import numpy as np
import models
from loss import *
from sampler import *
from utils import *
np.random.seed(0)

model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Cross Entropy')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='ResNet34',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: ResNet34)')
parser.add_argument('--batch_size', type=int, default=64, help='Number of images in each mini-batch')
parser.add_argument('--lr', type = float, default=0.1)
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=200, help='Number of sweeps over the dataset to train')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.3)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric,instance]', default='symmetric')
parser.add_argument('--dataset', type = str, help='[cifar10,cifar100]', default='cifar10')
parser.add_argument('--lt_type', type = str, help='[None, exp, step]', default='exp')
parser.add_argument('--lt_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.02)
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('--loss', type = str, default='cross_entropy')
parser.add_argument('--random_state', type=int, default=0, help='random state')
parser.add_argument('--WVN_RS', action='store_true', help = 'whether to use WVN and re-scale weight vector or not')
parser.add_argument('--model_dir', type=str, default=None, help = 'teacher model path')
parser.add_argument('--save_dir', type=str, default=None, help='save directory path')
parser.add_argument('--train_rule', default='None', type=str, help='data sampling strategy for train loader')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')

best_acc1 = 0

def main():
    args = parser.parse_args()

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.dataset == 'cifar10':
        args.scheduler_steps = [60,120]
    elif args.dataset == 'cifar100':
        args.scheduler_steps = [60,120]
    use_norm = True if args.loss == 'LDAM' else False
    torch.cuda.set_device(args.gpu)
    model = models.__dict__[args.arch](num_classes=args.num_classes, use_norm=use_norm, WVN=args.WVN_RS)
    model.to(args.gpu)
    if 'KD' in args.loss:
        model2 = models.__dict__[args.arch](num_classes=args.num_classes, use_norm=use_norm, WVN=args.WVN_RS)
        model2 = load_network(model2, args)
        model2.to(args.gpu)
    else:
        model2 = None

    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.scheduler_steps)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    train_dataset,test_dataset = input_dataset(args.dataset, args.noise_type, args.noise_rate,args.lt_type,args.lt_rate,args.random_state)
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                    batch_size = args.batch_size, 
                                    num_workers=12,
                                    shuffle=(train_sampler is None),
                                    pin_memory=True,
                                    sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                    batch_size = args.batch_size, 
                                    num_workers=12,
                                    shuffle=False,
                                    pin_memory=True)


    img_num_per_cls = np.array(train_dataset.get_cls_num_list())
    args.cls_num_list = img_num_per_cls
    num_training_samples = sum(img_num_per_cls)
    loss_all = np.zeros((num_training_samples,int(args.epochs/5)))


    # init log for training
    if not args.save_dir:
        save_dir = 'results'+ '/' + args.dataset + '/' + args.loss
        args.save_dir = save_dir
    else:
        save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.system('mkdir -p %s' % save_dir)


    txtfile=save_dir + '/' +  args.noise_type + str(args.noise_rate) + args.lt_type + str(args.lt_rate) + ('WVN_RS' if args.WVN_RS else '') + '.txt' 
    if os.path.exists(txtfile):
        os.system('rm %s' % txtfile)
    with open(txtfile, "a") as myfile:
        myfile.write('epoch: train_acc last_acc best_acc\n')

    # training epoches
    for epoch in range(args.start_epoch, args.epochs):
        print('=> current epoch',epoch)
        if args.train_rule == 'None':
            train_sampler = None
            per_cls_weights = None
        elif args.train_rule == 'Resample':
            train_sampler = ImbalancedDatasetSampler(train_dataset)
            per_cls_weights = None
        elif args.train_rule == 'Reweight':
            train_sampler = None
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, img_num_per_cls)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(img_num_per_cls)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        elif args.train_rule == 'IBReweight':
            train_sampler = None
            per_cls_weights = 1.0 / np.array(img_num_per_cls)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(img_num_per_cls)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        elif args.train_rule == 'DRW':
            train_sampler = None
            idx = epoch // 160
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], img_num_per_cls)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(img_num_per_cls)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        else:
            warnings.warn('Sample rule is not listed')


        if args.loss == 'CE':
            criterion = cross_entropy
        elif args.loss == 'LDAM':
            criterion = ldam
        elif args.loss == 'LADE':
            criterion = lade
        elif args.loss == 'logits_adjustment':
            criterion = logits_adjustment
        elif args.loss == 'CB_CE':
            criterion = cb_ce
        elif args.loss == 'CB_Focal':
            criterion = cb_focal
        elif args.loss == 'Focal':
            criterion = focal_loss
        elif args.loss == 'KD':
            criterion = KDLoss
        elif args.loss == 'BKD':
            criterion = BKDLoss
        elif args.loss == 'IB':
            if epoch < 100:
                criterion = cross_entropy
            else:
                criterion = IB_Loss
        elif args.loss == 'IBFocal':
            if epoch < 100:
                criterion = cross_entropy
            else:
                criterion = IB_FocalLoss
        elif args.loss == 'VS':
            if epoch < 5:
                criterion = cross_entropy
            else:
                criterion = vs_loss
        else:
            warnings.warn('Loss type is not listed')
            return
        

        train_acc = train(train_loader, model, criterion, per_cls_weights, optimizer, lr_scheduler, epoch, args, loss_all, model2)

        acc1 = validate(test_loader, model, criterion, epoch, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1>best_acc1
        best_acc1 = max(acc1,best_acc1)
        path = save_dir + '/' + args.noise_type + str(args.noise_rate) + args.lt_type + str(args.lt_rate) + ('WVN_RS' if args.WVN_RS else '') + '.pth'
        save_checkpoint(path, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best)
        print('\ntrain acc',train_acc)
        print('best acc',best_acc1)
        print('last acc', acc1)

        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ': '  + str(train_acc) +' ' + str(acc1) + ' ' + str(best_acc1) + "\n")



def train(train_loader, model, criterion, per_cls_weights, optimizer, lr_scheduler, epoch, args, loss_all, model2=None):
    # switch to train mode
    model.train()

    img_num_per_cls = args.cls_num_list
    correct = 0
    total = 0
    num_iter = (len(train_loader.dataset)//train_loader.batch_size)+1
    for i, (images, labels, true_labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
        batch_num = len(indexes)
        images = Variable(images).cuda(args.gpu)
        labels = Variable(labels).cuda(args.gpu)
        output, features = model(images)

        if 'KD' in args.loss:
            model2.eval()
            with torch.no_grad():
                output2, _ = model2(images)
            loss = criterion(epoch,output,output2,labels,ind,img_num_per_cls,per_cls_weights,loss_all)
        elif 'IB' in args.loss and epoch >= 100:
            loss = criterion(epoch,output,labels,features,ind,img_num_per_cls,per_cls_weights,loss_all)
        else:
            loss = criterion(epoch,output,labels,ind,img_num_per_cls,per_cls_weights,loss_all)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicted = output.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()     

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s+%.3f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Loss: %.4f'
                %(args.dataset, args.noise_rate, args.noise_type, args.lt_rate, args.lt_type, epoch, args.epochs, i+1, num_iter, loss.item()))
        sys.stdout.flush()       

    lr_scheduler.step()

    acc = 100.*float(correct)/float(total) 
    np.save(args.save_dir + '/' + args.noise_type + str(args.noise_rate) + args.lt_type + str(args.lt_rate)  + ('WVN_RS' if args.WVN_RS else '') +'_loss_all.npy',loss_all)
    return acc



def validate(val_loader, model, criterion, epoch, args):
    
    if args.WVN_RS and epoch == args.epochs-1:
        print("Re-scale weight vectors")
        current_state = model.state_dict()
        W = current_state['linear.weight']

        num_sample = args.cls_num_list

        gama = [0.3, 0.4]
        idx = 0 if args.dataset == 'cifar10' else 1
        ns = [ float(n) / max(num_sample) for n in num_sample ]
        ns = [ n**gama[idx] for n in ns ]
        ns = torch.FloatTensor(ns).unsqueeze(-1).cuda(args.gpu)
        new_W = W / ns

        current_state['linear.weight'] = new_W
        model.load_state_dict(current_state)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = Variable(images).cuda(args.gpu)
            # compute output
            logits, features = model(images)
            outputs = F.softmax(logits, dim=1)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred.cpu() == labels).sum()
    acc = 100*float(correct)/float(total) 
    return acc



def load_network(network, args):
    if args.dataset == 'cifar10':
        save_path = 'results/cifar10/CE/symmetric0.3exp0.02.best.pth'
    else:
        save_path = 'results/cifar100/CE/symmetric0.3exp0.02.best.pth'
    if args.model_dir:
        save_path = os.path.join(args.model_dir, args.noise_type + str(args.noise_rate) + args.lt_type + str(args.lt_rate) +'.best.pth')
    network.load_state_dict(torch.load(save_path)['state_dict'])
    return network



if __name__ == '__main__':
    main()