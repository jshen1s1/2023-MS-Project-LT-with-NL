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
from metrics import *
from utils import *
from trainer import *
np.random.seed(0)

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
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
parser.add_argument('--lr_decay', type=int, default=50, help='learning rate decay')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=150, help='Number of sweeps over the dataset to train')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.3)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric,instance]', default='symmetric')
parser.add_argument('--num_gradual', type = int, default = 10,
                    help='how many epochs for linear drop rate. This parameter is equal to Ek for lambda(E) in the paper.')
parser.add_argument('--dataset', type = str, help='[cifar10,cifar100]', default='cifar10')
parser.add_argument('--lt_type', type = str, help='[None, exp, step]', default='exp')
parser.add_argument('--lt_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.02)
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--loss', type = str, default='cross_entropy')
parser.add_argument('--random_state', type=int, default=0, help='random state')
parser.add_argument('--WVN_RS', action='store_true', help = 'whether to use WVN and re-scale weight vector or not')
parser.add_argument('--model_dir', type=str, default=None, help = 'teacher model path')
parser.add_argument('--save_dir', type=str, default=None, help='save directory path')
parser.add_argument('--train_rule', default='None', type=str, help='data sampling strategy for train loader')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')

args = parser.parse_args()

best_acc1 = 0

def main():
    global best_acc1

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.dataset == 'cifar10':
        args.scheduler_steps = [60,120]
        init_epoch = 20
    elif args.dataset == 'cifar100':
        args.scheduler_steps = [60,120]
        init_epoch = 5

    # Data loading code
    train_dataset,test_dataset = input_dataset(args.dataset, args.noise_type, args.noise_rate,args.lt_type,args.lt_rate,args.random_state)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                  batch_size = args.batch_size, 
                                  num_workers=12,
                                  shuffle=True,
                                  pin_memory=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                  batch_size = args.batch_size, 
                                  num_workers=12,
                                  shuffle=False,
                                  pin_memory=True)

    img_num_per_cls = np.array(train_dataset.get_cls_num_list())
    args.cls_num_list = img_num_per_cls
    num_training_samples = sum(img_num_per_cls)
    loss_all = np.zeros((num_training_samples,int(args.epochs/5)))

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=args.num_classes)
    model.cuda()

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

    # select train rule
    if args.train_rule == 'None':
        dual_T_estimation = np.eye(args.num_classes)
    elif args.train_rule == 'Dual_t':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.1)

        est_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                    batch_size = args.batch_size, 
                                    num_workers=12,
                                    shuffle=False,pin_memory=True)
        true_t_matrix = train_loader.dataset.t_matrix

        model_warmup = models.__dict__[args.arch](num_classes=args.num_classes)
        model_warmup = load_network(model_warmup, args)
        model_warmup.cuda()

        T_spadesuit, T_clubsuit = run_est_T_matrices(est_loader, model_warmup, args.num_classes)
        # the emprical noisy class posterior is equal to the intermediate class posterior, therefore T_estimation and T_clubsuit are identical
        T_estimation = T_clubsuit
        T_estimator_err = l1_error_calculator(target = true_t_matrix, target_hat = T_clubsuit)
        print("T-estimator error", T_estimator_err)

        dual_T_estimation = compose_T_matrices(T_spadesuit=T_spadesuit, T_clubsuit = T_clubsuit)
        dual_T_estimator_err = l1_error_calculator(target = true_t_matrix, target_hat = dual_T_estimation)  
        print("DT-estimator error", dual_T_estimator_err)

        T_noise_rate = get_noise_rate(T_estimation)
        DT_noise_rate = get_noise_rate(dual_T_estimation)

        dual_T_estimation = torch.Tensor(dual_T_estimation).cuda()
    elif args.train_rule == 'CORES':
        noise_prior = img_num_per_cls/num_training_samples
        noise_prior_cur = noise_prior

        loss_div_all = np.zeros((num_training_samples,int(args.epochs/5)))
        noise_prior_all = np.zeros((args.num_classes,args.epochs))
    else:
        warnings.warn('Sample rule is not listed')

    if 'coteaching' in args.loss:
        forget_rate = DT_noise_rate if args.train_rule == 'Dual_t' else args.noise_rate
        model2 = models.__dict__[args.arch](num_classes=args.num_classes)
        model2.cuda()
        optimizer2 = torch.optim.SGD(model2.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
        lr_scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, milestones=args.scheduler_steps)
        rate_schedule = np.ones(args.epochs)*forget_rate 
        rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate, args.num_gradual)

    # init log for training
    if not args.save_dir:
        save_dir = 'results'+ '/' + args.dataset + '/' + args.loss
        args.save_dir = save_dir
    else:
        save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.system('mkdir -p %s' % save_dir)


    txtfile=save_dir + '/' +  args.noise_type + str(args.noise_rate) + args.lt_type + str(args.lt_rate) + ('DT' if args.train_rule == 'Dual_t' else '') + ('WVN_RS' if args.WVN_RS else '') + '.txt' 
    if os.path.exists(txtfile):
        os.system('rm %s' % txtfile)
    with open(txtfile, "a") as myfile:
        myfile.write('epoch: train_acc last_acc best_acc\n')

    # training epoches
    for epoch in range(args.start_epoch, args.epochs):
        if args.loss == 'CE':
            criterion = cross_entropy
        elif args.loss == 'cores':
            criterion = cores
        elif args.loss == 'cores_no_select':
            criterion = cores_no_select
        elif args.loss == 'cores_logits_adjustment':
            criterion = cores_logits_adjustment
        elif args.loss == 'ELR':
            criterion = elr
        elif args.loss == 'CLS':
            criterion = NLLL
        #elif args.loss == 'DMI':
        #    criterion = DMI_loss
        elif args.loss == 'coteaching':
            criterion = co_teaching
        elif args.loss == 'coteaching_plus':
            if epoch < init_epoch:
                criterion = co_teaching
            else:
                criterion = co_teaching_plus
        else:
            warnings.warn('Loss type is not listed')
            return
        

        if 'coteach' in args.loss:
            train_acc, train_acc2 = run_coteaching(train_loader, criterion, epoch, args, model, optimizer, model2, optimizer2, rate_schedule)

            acc1 = validate(test_loader, model, args, t_m=np.eye(args.num_classes))
            acc2 = validate(test_loader, model2, args, t_m=np.eye(args.num_classes))
        elif 'cores' in args.loss:
            train_acc, noise_prior_delta = run_cores(train_loader, criterion, model, optimizer, epoch, args, loss_all, loss_div_all, noise_prior_cur)
            train_acc2 = 0
            acc1 = validate(test_loader, model, args, t_m=np.eye(args.num_classes))
            acc2 = 0

            noise_prior_cur = noise_prior*num_training_samples - noise_prior_delta
            noise_prior_cur = noise_prior_cur/sum(noise_prior_cur)
            noise_prior_all[:,epoch] = noise_prior_cur
        else:
            train_acc = train(train_loader, criterion, model, optimizer, epoch, args, loss_all, t_m=dual_T_estimation)
            train_acc2 = 0
            acc1 = validate(test_loader, model, args, t_m=np.eye(args.num_classes))
            acc2 = 0

        lr_scheduler.step()
        if 'coteaching' in args.loss:
            lr_scheduler2.step()

        # remember best acc@1 and save checkpoint
        is_best = max(acc1,acc2)>best_acc1
        best_acc1 = max(acc1,acc2,best_acc1)
        path = save_dir + '/' + args.noise_type + str(args.noise_rate) + args.lt_type + str(args.lt_rate) + ('DT' if args.train_rule == 'Dual_t' else '') + ('WVN_RS' if args.WVN_RS else '') + '.pth'
        if acc1>acc2:
            save_checkpoint(path, {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best)
        else:
            save_checkpoint(path, {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model2.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer2.state_dict(),
            }, is_best)
        print('train acc',train_acc)
        print('best acc', best_acc1)
        print('last acc', max(acc1,acc2))

        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ': '  + str(100.*train_acc) +' ' + str(max(acc1,acc2)) + ' ' + str(best_acc1) + "\n")


def train(train_loader, criterion, model, optimizer, epoch, args, loss_all, t_m=np.eye(100)):
    print('current epoch',epoch)
    # switch to train mode
    model.train()    
    train_loader.dataset.train_mode()

    img_num_per_cls = args.cls_num_list
    correct = 0
    total = 0
    for i, (images, labels, true_labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        output, _ = model(images)
        if args.train_rule == 'Dual_t':
            probs = F.softmax(output, dim=1)
            probs = torch.matmul(probs, t_m)
            output = torch.log(probs+1e-12)
        loss = criterion(epoch,output,labels,ind,img_num_per_cls,loss_all)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicted = output.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()    

    np.save(args.save_dir + '/' + args.noise_type + str(args.noise_rate) + args.lt_type + str(args.lt_rate) + ('DT' if args.train_rule == 'Dual_t' else '') + ('WVN_RS' if args.WVN_RS else '') +'_loss_all.npy',loss_all)
    return 100.*correct/total


def validate(val_loader, model, args, t_m=np.eye(100)):
    model.eval()
    correct = 0
    total = 0
    t_m = torch.Tensor(t_m).cuda()
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = Variable(images).cuda()
            # compute output
            logits, _ = model(images)
            outputs = F.softmax(logits, dim=1)
            if args.train_rule == 'Dual_t':
                probs = F.softmax(outputs, dim=1)
                probs = torch.matmul(probs, t_m)
                outputs = torch.log(probs+1e-12)
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