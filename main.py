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
import torchvision.models as models
import torch.optim as optim
from dataset import input_dataset
import numpy as np
from resnet import ResNet34
from loss import *
np.random.seed(0)

parser = argparse.ArgumentParser(description='Cross Entropy')
parser.add_argument('--model_path', type=str, default='results/128_0.5_200_512_1000_model.pth',
                    help='The pretrained model path')
parser.add_argument('--batch_size', type=int, default=64, help='Number of images in each mini-batch')
parser.add_argument('--lr', type = float, default=0.1)
parser.add_argument('--lr_decay', type=int, default=50, help='learning rate decay')
parser.add_argument('--epochs', type=int, default=150, help='Number of sweeps over the dataset to train')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.3)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric,instance]', default='symmetric')
parser.add_argument('--num_gradual', type = int, default = 10, help='how many epochs for linear drop rate. This parameter is equal to Ek for lambda(E) in the paper.')
parser.add_argument('--dataset', type = str, help='[cifar10,cifar100]', default='cifar10')
parser.add_argument('--lt_type', type = str, help='[None, exp, step]', default='exp')
parser.add_argument('--lt_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.02)
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--loss', type = str, default='cross_entropy')
parser.add_argument('--random_state', type=int, default=0, help='random state')

args = parser.parse_args()

if args.dataset == 'cifar10':
    args.scheduler_steps = [60,120]
    args.epochs = 200
    args.num_classes=10
    init_epoch = 20
elif args.dataset == 'cifar100':
    args.scheduler_steps = [60,120]
    args.epochs = 200
    args.num_classes=100
    init_epoch = 5

loss_dict = {'cross_entropy':cross_entropy,'focal_loss':focal_loss,'logits_adjustment':logits_adjustment,'cores':cores,'gce':gce,
            'cb_ce':cb_ce,'cb_focal':cb_focal,'cores_no_select':cores_no_select,'cores_logits_adjustment':cores_logits_adjustment,
            'erl':elr,'coteaching':co_teaching,'coteaching_plus':co_teaching_plus,'cls':NLLL}


model = ResNet34(args.num_classes)

train_dataset,test_dataset = input_dataset(args.dataset, args.noise_type, args.noise_rate,args.lt_type,args.lt_rate,args.random_state)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                  batch_size = args.batch_size, 
                                  num_workers=12,
                                  shuffle=True,pin_memory=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                  batch_size = args.batch_size, 
                                  num_workers=12,
                                  shuffle=False,pin_memory=True)


img_num_per_cls = np.array(train_dataset.get_cls_num_list())
num_training_samples = sum(img_num_per_cls)
noise_prior = img_num_per_cls/num_training_samples
noise_prior_cur = noise_prior

num_example = len(train_loader.dataset)

loss_all = np.zeros((num_training_samples,int(args.epochs/5)))
loss_div_all = np.zeros((num_training_samples,int(args.epochs/5)))
noise_prior_all = np.zeros((args.num_classes,args.epochs))



model.cuda()
criterion = loss_dict[args.loss]
optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.scheduler_steps)

### build second model for co_teaching 
if args.loss in ['coteaching','coteaching_plus']:
    forget_rate = args.noise_rate
    model2 = ResNet34(args.num_classes)
    model2.cuda()
    optimizer2 = torch.optim.SGD(model2.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    lr_scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, milestones=args.scheduler_steps)
    rate_schedule = np.ones(args.epochs)*forget_rate 
    rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate, args.num_gradual)

### saving path
save_dir = 'results'+'/' +args.dataset + '/' +args.loss
if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)


txtfile=save_dir + '/' +  args.noise_type + str(args.noise_rate) + args.lt_type + str(args.lt_rate) + '.txt' 
if os.path.exists(txtfile):
    os.system('rm %s' % txtfile)
with open(txtfile, "a") as myfile:
    myfile.write('epoch: train_acc last_acc best_acc\n')


'''
def adjust_learning_rate(optimizer, epoch,alpha_plan):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
'''
best_acc = [0]
def validate(val_loader, model, criterion):
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = Variable(images).cuda()
            # compute output
            logits = model(images)
            outputs = F.softmax(logits, dim=1)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred.cpu() == labels).sum()
            acc = 100*float(correct)/float(total) 
    return acc



for epoch in range(args.epochs):
    print('current epoch',epoch)
    model.train()
    if args.loss in ['coteaching','coteaching_plus']:
        model2.train()
    #adjust_learning_rate(optimizer, epoch, alpha_plan)
    correct = correct2 = 0
    total = total2 = 0
    acc2 = 0
    idx_each_class_noisy = [[] for i in range(args.num_classes)]
    for i, (images, labels,true_labels,indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
        batch_num = len(indexes)
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        output = model(images)
        if args.loss not in ['cores','cores_no_select','cores_logits_adjustment']:
            if args.loss in ['coteaching','coteaching_plus']:
                output2 = model2(images)
                if args.loss == 'coteaching_plus' and epoch < init_epoch:
                    loss, loss2 = co_teaching(epoch, output, output2, labels, rate_schedule[epoch], ind, epoch*i)
                else:
                    loss, loss2 = criterion(epoch, output, output2, labels, rate_schedule[epoch], ind, epoch*i)
            else:
                loss = criterion(epoch,output,labels,ind,img_num_per_cls,loss_all,num_example)
        else:
            loss,loss_v = criterion(epoch,output,labels,ind,img_num_per_cls,noise_prior_cur,loss_all,loss_div_all)
            for i in range(batch_num):
                if loss_v[i] == 0:
                    idx_each_class_noisy[labels[i]].append(indexes[i])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicted = output.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        if args.loss in ['coteaching','coteaching_plus']:
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()
            _, predicted2 = output2.max(1)
            total2 += labels.size(0)
            correct2 += predicted2.eq(labels).sum().item()        
    lr_scheduler.step()
    if args.loss in ['coteaching','coteaching_plus']:
        lr_scheduler2.step()
    noise_prior_delta = np.array([len(idx_each_class_noisy[i]) for i in range(args.num_classes)])
    noise_prior_cur = noise_prior*num_training_samples - noise_prior_delta
    noise_prior_cur = noise_prior_cur/sum(noise_prior_cur)
    noise_prior_all[:,epoch] = noise_prior_cur

    if args.loss in ['coteaching','coteaching_plus']:
        print('train acc',100.*correct/total,'train acc2',100.*correct2/total2)
        acc2 = validate(test_loader, model2, criterion)
    else:
        print('train acc',100.*correct/total)
    acc1 = validate(test_loader, model, criterion)
    if max(acc1,acc2)>best_acc[0]:
        best_acc[0] = max(acc1,acc2)
        torch.save({'state_dict': model.state_dict()},save_dir + '/' + args.noise_type + str(args.noise_rate) + args.lt_type + str(args.lt_rate)+'.pth')
    print('best acc',best_acc[0])
    print('last acc', max(acc1,acc2))

    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ': '  + str(100.*correct/total) +' ' + str(max(acc1,acc2)) + ' ' + str(best_acc[0]) + "\n")
    np.save(save_dir + '/' + args.noise_type + str(args.noise_rate) + args.lt_type + str(args.lt_rate)+'_loss_all.npy',loss_all)
    np.save(save_dir + '/' + args.noise_type + str(args.noise_rate) + args.lt_type + str(args.lt_rate)+'_loss_div_all.npy',loss_div_all)
    np.save(save_dir + '/' + args.noise_type + str(args.noise_rate) + args.lt_type + str(args.lt_rate)+'_noise_prior_all.npy',noise_prior_all)
    np.save(save_dir + '/' + args.noise_type + str(args.noise_rate) + args.lt_type + str(args.lt_rate)+'_noise_or_not.npy',train_dataset.noise_or_not)
    np.save(save_dir + '/' + args.noise_type + str(args.noise_rate) + args.lt_type + str(args.lt_rate)+'_train_labels.npy',train_dataset.train_labels)
    np.save(save_dir + '/' + args.noise_type + str(args.noise_rate) + args.lt_type + str(args.lt_rate)+'_true_labels.npy',train_dataset.true_labels)


