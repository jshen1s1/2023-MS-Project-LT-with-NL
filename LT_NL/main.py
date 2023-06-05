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
from sampler import *
from trainer import *
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
parser.add_argument('--num_gradual', type = int, default = 10,
                    help='how many epochs for linear drop rate. This parameter is equal to Ek for lambda(E) in the co-teaching_plus paper.')
parser.add_argument('--dataset', type = str, help='[cifar10,cifar100]', default='cifar10')
parser.add_argument('--lt_type', type = str, help='[None, exp, step]', default='exp')
parser.add_argument('--lt_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.02)
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('--loss', type = str, default='cross_entropy')
parser.add_argument('--random_state', type=int, default=0, help='random state')
parser.add_argument('--WVN_RS', action='store_true', help = 'whether to use WVN and re-scale weight vector or not')
parser.add_argument('--low_dim', action='store_true', help = 'whether to lower feature dim or not')
parser.add_argument('--data_aug', action='store_true', help = 'whether to use feature augmentation or not')
parser.add_argument('--model_dir', type=str, default=None, help = 'teacher model path')
parser.add_argument('--save_dir', type=str, default=None, help='save directory path')
parser.add_argument('--train_rule', default='None', type=str, help='model training strategy')
parser.add_argument('--train_opt', default='None', type=str, help='model training framework')
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
        warm_up = 10
    elif args.dataset == 'cifar100':
        args.scheduler_steps = [60,120]
        init_epoch = 5
        warm_up = 30

    # Data loading code
    train_dataset,test_dataset = input_dataset(args.dataset, args.noise_type, args.noise_rate,args.lt_type,args.lt_rate,args.random_state,aug=args.data_aug)
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

    est_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                  batch_size = args.batch_size, 
                                  num_workers=12,
                                  shuffle=False,
                                  pin_memory=True)
    
    warmup_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                  batch_size = args.batch_size*2, 
                                  num_workers=12,
                                  shuffle=True,
                                  pin_memory=True)
    
    img_num_per_cls = np.array(train_dataset.get_cls_num_list())
    args.cls_num_list = img_num_per_cls
    num_training_samples = sum(img_num_per_cls)
    loss_all = np.zeros((num_training_samples,int(args.epochs/5)))

    # create model
    print("=> creating model '{}'".format(args.arch))
    torch.cuda.set_device(args.gpu)
    model = models.__dict__[args.arch](num_classes=args.num_classes, low_dim=args.low_dim)
    model.to(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(),
                                    args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.scheduler_steps)

    if args.loss in ['coteaching', 'coteaching_plus', 'Semi', 'CNLCU_soft', 'CNLCU_hard']:
        model2 = models.__dict__[args.arch](num_classes=args.num_classes)
        model2.to(args.gpu)
        optimizer2 = torch.optim.SGD(model2.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
        lr_scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, milestones=args.scheduler_steps)
    else:
        model2 = None
        optimizer2 = None
        lr_scheduler2 = None

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

    # training init
    if 'coteaching' in args.loss:
        forget_rate = DT_noise_rate if args.train_opt == 'Dual_t' else args.noise_rate
        rate_schedule = np.ones(args.epochs)*forget_rate 
        rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate, args.num_gradual)
    if args.train_opt == 'Dual_t':
        train_sampler = None
        per_cls_weights = None
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.1)

        true_t_matrix = train_loader.dataset.t_matrix

        model_warmup = models.__dict__[args.arch](num_classes=args.num_classes)
        model_warmup = load_network(model_warmup, args)
        model_warmup.to(args.gpu)

        T_spadesuit, T_clubsuit = run_est_T_matrices(est_loader, model_warmup, args)
        # the emprical noisy class posterior is equal to the intermediate class posterior, therefore T_estimation and T_clubsuit are identical
        T_estimation = T_clubsuit
        T_estimator_err = l1_error_calculator(target = true_t_matrix, target_hat = T_clubsuit)
        print("T-estimator error", T_estimator_err)

        dual_T_estimation = compose_T_matrices(T_spadesuit=T_spadesuit, T_clubsuit = T_clubsuit)
        dual_T_estimator_err = l1_error_calculator(target = true_t_matrix, target_hat = dual_T_estimation)  
        print("DT-estimator error", dual_T_estimator_err)

        T_noise_rate = get_noise_rate(T_estimation)
        DT_noise_rate = get_noise_rate(dual_T_estimation)

        dual_T_estimation = torch.Tensor(dual_T_estimation).cuda(args.gpu)
    else:
        dual_T_estimation = np.eye(args.num_classes)
    if 'cores' in args.loss:
        noise_prior = img_num_per_cls/num_training_samples
        noise_prior_cur = noise_prior

        loss_div_all = np.zeros((num_training_samples,int(args.epochs/5)))
        noise_prior_all = np.zeros((args.num_classes,args.epochs))
    if args.train_opt == 'RoLT':
        args.noisy_labels = torch.LongTensor(est_loader.dataset.train_labels).cuda()
        #clean_labels = torch.LongTensor(train_loader.dataset.true_labels).cuda()
        args.current_labels = args.noisy_labels
        args.soft_targets = [args.noisy_labels]
        args.soft_weights = [1]

        ncm_classifier = models.__dict__['KNN'](feat_dim=512, num_classes=args.num_classes)
        lr_scheduler = WarmupMultiStepLR(optimizer, milestones=args.scheduler_steps)

        if args.train_rule == 'DRW':
            args.train_opt = 'RoLT-DRW'
    if 'CNLCU' in args.train_opt:
        forget_rate = DT_noise_rate if args.train_opt == 'Dual_t' else args.noise_rate
        rate_schedule = np.ones(args.epochs)*forget_rate 
        rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate, args.num_gradual)
        co_lambda_plan = args.weight_decay * np.linspace(1, 0, 80) 
    if args.train_opt == 'PCL':
        weights = None
        if args.dataset == 'cifar10':
            ramp_epoch = 0
            args.w_proto = 5
            args.knn_start_epoch = 5
            args.n_neighbors = 10
            args.low_th = 0.1
            args.high_th = -0.4
            args.gamma = 1.005
        elif args.dataset == 'cifar100':
            ramp_epoch = 40
            args.w_proto = 7
            args.knn_start_epoch = 15
            args.n_neighbors = 200
            args.low_th = 0.01
            args.high_th = 0.02
            args.gamma = 1.005

    # init log for training
    if not args.save_dir:
        save_dir = 'results'+ '/' + args.dataset + '/' + args.loss
        args.save_dir = save_dir
    else:
        save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.system('mkdir -p %s' % save_dir)


    txtfile=save_dir + '/' +  args.noise_type + str(args.noise_rate) + args.lt_type + str(args.lt_rate) + (args.train_opt if args.train_opt else '') + ('WVN_RS' if args.WVN_RS else '') + '.txt' 
    if os.path.exists(txtfile):
        os.system('rm %s' % txtfile)
    with open(txtfile, "a") as myfile:
        myfile.write('epoch: train_acc last_acc best_acc\n')

    # training epoches
    for epoch in range(args.start_epoch, args.epochs):
        print('=> current epoch',epoch)
        # train rule selection 
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
            if args.train_opt == 'RoLT':
                img_num_per_cls = np.array([(args.current_labels == i).sum().item() \
                                             for i in range(args.num_classes)])
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
        elif args.loss == 'coteaching':
            criterion = co_teaching
        elif args.loss == 'coteaching_plus':
            if epoch < init_epoch:
                criterion = co_teaching
            else:
                criterion = co_teaching_plus
        elif args.loss == 'Semi':
            if epoch < warm_up:
                criterion = cross_entropy
            else:
                criterion = semi_loss
        elif 'CNLCU' in args.loss:
            if 'soft' in args.loss:
                criterion = CNLCU_soft
            else:
                criterion = CNLCU_hard
            before_loss_1 = 0.0 * np.ones((len(train_dataset), 1))
            before_loss_2 = 0.0 * np.ones((len(train_dataset), 1))
            sn_1 = torch.from_numpy(np.ones((len(train_dataset), 1)))
            sn_2 = torch.from_numpy(np.ones((len(train_dataset), 1)))
        elif args.loss == 'PCL':
            if epoch < 1:
                criterion = cross_entropy
            else:
                criterion = PCL
        else:
            warnings.warn('Loss type is not listed')
            return
        
        # train models
        if 'coteach' in args.loss:
            train_acc, train_acc2 = run_coteaching(train_loader, criterion, epoch, args, model, optimizer, model2, optimizer2, rate_schedule)
        elif 'cores' in args.loss:
            train_acc, noise_prior_delta = run_cores(train_loader, criterion, model, optimizer, epoch, args, loss_all, loss_div_all, noise_prior_cur)
            train_acc2 = 0

            # update noise prior
            noise_prior_cur = noise_prior*num_training_samples - noise_prior_delta
            noise_prior_cur = noise_prior_cur/sum(noise_prior_cur)
            noise_prior_all[:,epoch] = noise_prior_cur
        elif args.loss == 'Semi' and epoch >= warm_up:
            prob1 = eval_train(args,model,est_loader)   
            prob2 = eval_train(args,model2,est_loader)

            pred1 = (prob1 > 0.5)      
            pred2 = (prob2 > 0.5)              

            # co-divide
            print('Train Net1')
            labeled_dataset, unlabeled_dataset = input_dataset(args.dataset, args.noise_type, args.noise_rate,args.lt_type,args.lt_rate,args.random_state,split=True,pred=pred2,prob=prob2) 
            labeled_trainloader = torch.utils.data.DataLoader(dataset=labeled_dataset,
                                  batch_size = args.batch_size, 
                                  num_workers=5,
                                  shuffle=True,
                                  pin_memory=True)
            unlabeled_trainloader = torch.utils.data.DataLoader(dataset=unlabeled_dataset,
                                  batch_size = args.batch_size, 
                                  num_workers=5,
                                  shuffle=True,
                                  pin_memory=True)
            train_acc = run_divideMix(epoch,args,model,model2,optimizer,criterion,labeled_trainloader,unlabeled_trainloader,loss_all)
            # co-divide
            print('\nTrain Net2')
            labeled_dataset, unlabeled_dataset = input_dataset(args.dataset, args.noise_type, args.noise_rate,args.lt_type,args.lt_rate,args.random_state,split=True,pred=pred1,prob=prob1)
            labeled_trainloader = torch.utils.data.DataLoader(dataset=labeled_dataset,
                                  batch_size = args.batch_size, 
                                  num_workers=5,
                                  shuffle=True,
                                  pin_memory=True)
            unlabeled_trainloader = torch.utils.data.DataLoader(dataset=unlabeled_dataset,
                                  batch_size = args.batch_size, 
                                  num_workers=5,
                                  shuffle=True,
                                  pin_memory=True)
            train_acc2 = run_divideMix(epoch,args,model2,model,optimizer2,criterion,labeled_trainloader,unlabeled_trainloader,loss_all)
        elif 'RoLT' in args.train_opt:
            label_cleaning_RoLT(est_loader, model, epoch, ncm_classifier, args)
            train_acc = run_RoLT(train_loader, criterion, per_cls_weights, model, optimizer, epoch, args, loss_all)
            train_acc2 = 0
        elif 'CNLCU' in args.train_opt:
            train_acc, train_acc2, before_loss_1_list, before_loss_2_list, ind_1_update_list, ind_2_update_list = run_CNLCU(train_loader, criterion, epoch, args, 
                                                                                                                            model, 
                                                                                                                            optimizer, 
                                                                                                                            model2, 
                                                                                                                            optimizer2, 
                                                                                                                            rate_schedule, 
                                                                                                                            before_loss_1, 
                                                                                                                            before_loss_2, 
                                                                                                                            sn_1, 
                                                                                                                            sn_2, 
                                                                                                                            co_lambda_plan)

            if 'soft' in args.train_opt:
                before_loss_1, before_loss_2 = np.array(before_loss_1_list).astype(float), np.array(before_loss_2_list).astype(float)
            else:
                before_loss_1_, before_loss_2_ = np.array(before_loss_1_list).astype(float), np.array(before_loss_2_list).astype(float)
                before_loss_1_numpy = np.zeros((len(train_dataset), 1))
                before_loss_2_numpy = np.zeros((len(train_dataset), 1))
                num = before_loss_1_.shape[0]
                before_loss_1_numpy[:num], before_loss_2_numpy[:num] = before_loss_1_[:, np.newaxis], before_loss_2_[:, np.newaxis]
                
                before_loss_1 = np.concatenate((before_loss_1, before_loss_1_numpy), axis=1)
                before_loss_2 = np.concatenate((before_loss_2, before_loss_2_numpy), axis=1)

            all_zero_array_1, all_zero_array_2 = np.zeros((len(train_dataset), 1)), np.zeros((len(train_dataset), 1))
            all_zero_array_1[np.array(ind_1_update_list)] = 1
            all_zero_array_2[np.array(ind_2_update_list)] = 1
            sn_1 += torch.from_numpy(all_zero_array_1)
            sn_2 += torch.from_numpy(all_zero_array_2)
        elif args.train_opt == 'PCL' and epoch >= 1:
            if ramp_epoch: 
                args.w_proto = min(1+epoch*(args.w_proto-1)/ramp_epoch, args.w_proto)
            train_acc, weights = run_PCL(train_loader, est_loader, criterion, per_cls_weights, weights, model, optimizer, epoch, args, loss_all)
            train_acc2 = 0
        else:
            loader = warmup_loader if args.loss == 'Semi' else train_loader
            train_acc = train(loader, criterion, per_cls_weights, model, optimizer, epoch, args, loss_all, t_m=dual_T_estimation)
            if model2:
                train_acc2 = train(loader, criterion, per_cls_weights, model2, optimizer2, epoch, args, loss_all, t_m=dual_T_estimation)
            else:
                train_acc2 = 0

        lr_scheduler.step()
        if lr_scheduler2:
            lr_scheduler2.step()

        # evaluate 
        if args.loss == 'Semi':
            acc1 = test(test_loader, model, model2, args, t_m=np.eye(args.num_classes))
            acc2 = 0
        else:
            acc1 = validate(test_loader, model, args, t_m=np.eye(args.num_classes))
            if model2:
                acc2 = validate(test_loader, model2, args, t_m=np.eye(args.num_classes))
            else:
                acc2 = 0

        # remember best acc@1 and save checkpoint
        is_best = max(acc1,acc2)>best_acc1
        best_acc1 = max(acc1,acc2,best_acc1)
        path = save_dir + '/' + args.noise_type + str(args.noise_rate) + args.lt_type + str(args.lt_rate) + (args.train_opt if args.train_opt else '') + ('WVN_RS' if args.WVN_RS else '') + '.pth'
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
        print('\ntrain acc', train_acc, 'train acc2', train_acc2)
        print('best acc', best_acc1)
        print('last acc', max(acc1,acc2))

        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ': '  + str(train_acc) +' ' + str(max(acc1,acc2)) + ' ' + str(best_acc1) + "\n")


def train(train_loader, criterion, per_cls_weights, model, optimizer, epoch, args, loss_all, t_m=np.eye(100)):
    # switch to train mode
    model.train()    
    train_loader.dataset.train_mode()

    img_num_per_cls = args.cls_num_list
    correct = 0
    total = 0
    num_iter = (len(train_loader.dataset)//train_loader.batch_size)+1
    for i, batch in enumerate(train_loader):
        ind=batch[3].cpu().numpy().transpose()
        images = Variable(batch[0]).cuda(args.gpu)
        labels = Variable(batch[1]).cuda(args.gpu)
        outputs, _ = model(images)
        if args.train_opt == 'Dual_t':
            probs = F.softmax(output, dim=1)
            probs = torch.matmul(probs, t_m)
            output = torch.log(probs+1e-12)
        loss = criterion(epoch,outputs,labels,ind,img_num_per_cls,per_cls_weights,loss_all)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()     

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s+%.3f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Loss: %.4f'
                %(args.dataset, args.noise_rate, args.noise_type, args.lt_rate, args.lt_type, epoch, args.epochs, i+1, num_iter, loss.item()))
        sys.stdout.flush()       

    acc = 100.*float(correct)/float(total) 
    np.save(args.save_dir + '/' + args.noise_type + str(args.noise_rate) + args.lt_type + str(args.lt_rate) + (args.train_opt if args.train_opt else '') + ('WVN_RS' if args.WVN_RS else '') +'_loss_all.npy',loss_all)
    return acc


def validate(val_loader, model, args, t_m=np.eye(100)):
    model.eval()
    correct = 0
    total = 0
    t_m = torch.Tensor(t_m).cuda(args.gpu)
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = Variable(images).cuda(args.gpu)
            # compute output
            logits, _ = model(images)
            outputs = F.softmax(logits, dim=1)
            if args.train_opt == 'Dual_t':
                probs = F.softmax(outputs, dim=1)
                probs = torch.matmul(probs, t_m)
                outputs = torch.log(probs+1e-12)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred.cpu() == labels).sum()
    acc = 100.*float(correct)/float(total) 
    return acc


def test(val_loader, model, model2, args, t_m=np.eye(100)):
    model.eval()
    model2.eval()
    correct = 0
    total = 0
    t_m = torch.Tensor(t_m).cuda(args.gpu)
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            # compute output
            outputs1, _ = model(images)
            outputs2, _ = model2(images)
            outputs = outputs1+outputs2
            if args.train_opt == 'Dual_t':
                probs = F.softmax(outputs, dim=1)
                probs = torch.matmul(probs, t_m)
                outputs = torch.log(probs+1e-12)
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += pred.eq(labels).cpu().sum().item()   
    acc = 100.*float(correct)/float(total) 
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