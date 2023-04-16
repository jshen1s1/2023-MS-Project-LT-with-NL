import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def run_coteaching(train_loader, criterion, epoch, args, model1, optimizer1, model2, optimizer2, rate_schedule):
    print('current epoch',epoch)
    model1.train()
    model2.train()
    train_total=0
    train_correct=0 
    train_total2=0
    train_correct2=0 

    for i, (images, labels, true_labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
      
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        
        # Forward + Backward + Optimize
        logits1, _ = model1(images)
        prec1, _ = accuracy(logits1, labels, topk=(1, 5))
        train_total+=1
        train_correct+=prec1

        logits2, _ = model2(images)
        prec2, _ = accuracy(logits2, labels, topk=(1, 5))
        train_total2+=1
        train_correct2+=prec2 
        loss_1, loss_2 = criterion(epoch, logits1, logits2, labels, rate_schedule[epoch], ind, epoch*i)

        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()

    train_acc1=float(train_correct)/float(train_total)
    train_acc2=float(train_correct2)/float(train_total2)
    return train_acc1, train_acc2


def run_cores(train_loader, criterion, model, optimizer, epoch, args, loss_all, loss_div_all, noise_prior_cur):
    print('current epoch',epoch)
    model.train()
    train_total=0
    train_correct=0 
    img_num_per_cls = args.cls_num_list
    idx_each_class_noisy = [[] for i in range(args.num_classes)]
    for i, (images, labels, true_labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
        batch_size = len(ind)

        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
       
        # Forward + Backward + Optimize
        logits, _ = model(images)
        prec, _ = accuracy(logits, labels, topk=(1, 5))
        train_total+=1
        train_correct+=prec
        loss, loss_v = criterion(epoch,logits,labels,ind,img_num_per_cls,noise_prior_cur,loss_all,loss_div_all)
        for i in range(batch_size):
            if loss_v[i] == 0:
                idx_each_class_noisy[labels[i]].append(ind[i])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    noise_prior_delta = np.array([len(idx_each_class_noisy[i]) for i in range(args.num_classes)])

    train_acc=float(train_correct)/float(train_total)
    np.save(args.save_dir + '/' + args.noise_type + str(args.noise_rate) + args.lt_type + str(args.lt_rate) + ('DT' if args.train_rule == 'Dual_t' else '') + ('WVN_RS' if args.WVN_RS else '') +'_loss_all.npy',loss_all)
    return train_acc, noise_prior_delta