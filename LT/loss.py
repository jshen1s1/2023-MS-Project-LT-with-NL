import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
# Loss functions

def get_one_hot(label, num_classes):
    batch_size = label.shape[0]
    onehot_label = torch.zeros((batch_size, num_classes))
    onehot_label = onehot_label.scatter_(1, label.unsqueeze(1).detach().cpu(), 1)
    onehot_label = (onehot_label.type(torch.FloatTensor)).cuda()
    return onehot_label


def f_beta(epoch):
    beta1 = np.linspace(0.0, 0.0, num=10)
    beta2 = np.linspace(0.0, 2, num=30)
    beta3 = np.linspace(2, 2, num=170)
 
    beta = np.concatenate((beta1,beta2,beta3),axis=0) 
    return beta[epoch]


def cross_entropy(epoch,logits,label,ind,img_num_per_cls,per_cls_weights,loss_all):
    num_batch = logits.shape[0]
    loss = F.cross_entropy(logits, label, weight=per_cls_weights, reduce = False)
    loss_numpy = loss.data.cpu().numpy()
    if epoch%5==0:
        loss_all[ind,int(epoch/5)] = loss_numpy
    return torch.sum(loss)/num_batch


def NLLL(epoch,logits,label,ind,img_num_per_cls,per_cls_weights,loss_all):
    num_batch = logits.shape[0]
    loss = F.nll_loss(logits, label, reduce = False)
    loss_numpy = loss.data.cpu().numpy()
    if epoch%5==0:
        loss_all[ind,int(epoch/5)] = loss_numpy
    return torch.sum(loss)/num_batch


def gce(epoch,logits,label,ind,img_num_per_cls,per_cls_weights,loss_all):
    num_batch = logits.shape[0]
    logits = F.softmax(logits,dim=1)
    q = 0.7
    loss = 0
    for i in range(num_batch):
        loss_per = (1.0 - (logits[i][label[i]]) ** q) / q
        if epoch%5==0:
            loss_all[ind[i],int(epoch/5)] = loss_per.data.cpu().numpy()
        loss += loss_per
    return loss


def focal_loss(epoch,logits,label,ind,img_num_per_cls,per_cls_weights,loss_all):
    num_classes = len(img_num_per_cls)
    num_batch = logits.shape[0]
    gamma = 2.0
    label = get_one_hot(label, num_classes)
    p = F.sigmoid(logits)
    focal_weights = torch.pow((1-p)*label + p * (1-label), gamma).cuda()
    loss = F.binary_cross_entropy_with_logits(logits, label, reduction = 'none') * focal_weights
    loss_numpy = loss.data.cpu().numpy()
    #if epoch%5==0:
    #    loss_all[ind,int(epoch/5)] = loss_numpy
    return torch.sum(loss)/num_batch


def cb_ce(epoch,logits,label,ind,img_num_per_cls,per_cls_weights,loss_all):
    num_batch = logits.shape[0]
    num_classes = len(img_num_per_cls)
    loss = F.cross_entropy(logits, label, weight=per_cls_weights, reduce = False)
    loss_numpy = loss.data.cpu().numpy()
    if epoch%5==0:
        loss_all[ind,int(epoch/5)] = loss_numpy
    return torch.sum(loss)/num_batch


def cb_focal(epoch,logits,label,ind,img_num_per_cls,per_cls_weights,loss_all):
    num_batch = logits.shape[0]
    num_classes = len(img_num_per_cls)
    gamma = 2.0

    weight = (per_cls_weights[label]).cuda()
    p = F.sigmoid(logits)
    focal_weights = torch.pow((1-p)*label + p * (1-label), gamma).cuda()
    loss_f = F.binary_cross_entropy_with_logits(logits, label, reduction = 'none') * focal_weights
    loss = loss_f * weight.view(-1, 1)
    loss_numpy = loss.data.cpu().numpy()
    #if epoch%5==0:
    #    loss_all[ind,int(epoch/5)] = loss_numpy
    return torch.sum(loss)/num_batch


def logits_adjustment(epoch,logits,label,ind,img_num_per_cls,per_cls_weights,loss_all):
    num_batch = logits.shape[0]
    tro = 1.0
    label_frequency = img_num_per_cls/sum(img_num_per_cls)
    adjustment = np.log(label_frequency ** tro + 1e-12)
    logits = logits + torch.FloatTensor(adjustment).cuda()

    loss = F.cross_entropy(logits, label, reduce = False)
    loss_numpy = loss.data.cpu().numpy()
    if epoch%5==0:
        loss_all[ind,int(epoch/5)] = loss_numpy
    return torch.sum(loss)/num_batch


def ldam(epoch,logits,label,ind,img_num_per_cls,per_cls_weights,loss_all):
    num_batch = logits.shape[0]
    num_classes = len(img_num_per_cls)
    max_m = 0.5
    s = 30.0

    m_list = 1.0 / np.sqrt(np.sqrt(img_num_per_cls))
    m_list = m_list * (max_m / np.max(m_list))
    m_list = torch.cuda.FloatTensor(m_list)

    index = torch.zeros_like(logits, dtype=torch.uint8)
    index.scatter_(1, label.data.view(-1, 1), 1) # one-hot index

    index_float = index.type(torch.cuda.FloatTensor)
    batch_m = torch.matmul(m_list[None, :], index_float.transpose(0,1))
    batch_m = batch_m.view((-1, 1))
    x_m = logits - batch_m

    output = torch.where(index, x_m, logits)
    loss =  F.cross_entropy(s*output, label, weight=per_cls_weights)
    loss_numpy = loss.data.cpu().numpy()
    if epoch%5==0:
        loss_all[ind,int(epoch/5)] = loss_numpy
    return  torch.sum(loss)/num_batch


def lade(epoch,logits,label,ind,img_num_per_cls,per_cls_weights,loss_all):
    num_batch = logits.shape[0]
    num_classes = len(img_num_per_cls)
    img_num_per_cls = torch.tensor(img_num_per_cls)
    remine_lambda=0.1
    prior = (img_num_per_cls / img_num_per_cls.sum()).cuda()
    balanced_prior = torch.tensor(1. / num_classes).float().cuda()
    cls_weight = (img_num_per_cls.float() / torch.sum(img_num_per_cls.float())).cuda()
    
    per_cls_pred_spread = logits.T * (label == torch.arange(0, num_classes).view(-1, 1).type_as(label))
    pred_spread = (logits - torch.log(prior + 1e-9) + torch.log(balanced_prior + 1e-9)).T

    num_samples_per_cls = torch.sum(label == torch.arange(0, num_classes).view(-1, 1).type_as(label), -1).float()

    N = per_cls_pred_spread.size(-1)
    first_term = torch.sum(per_cls_pred_spread, -1) / (num_samples_per_cls + 1e-8)
    second_term = torch.logsumexp(pred_spread, -1) - np.log(N)
    reg = (second_term ** 2) * remine_lambda

    loss = first_term - second_term - reg
    loss = -torch.sum(loss * cls_weight)
    loss_numpy = loss.data.cpu().numpy()
    if epoch%5==0:
        loss_all[ind,int(epoch/5)] = loss_numpy
    return torch.sum(loss)/num_batch


def KDLoss(epoch,logits,logits2,label,ind,img_num_per_cls,per_cls_weights,loss_all):
    num_batch = logits.shape[0]
    num_classes = len(img_num_per_cls)

    T = 2.0
    alpha = 1.0
    
    kd = F.kl_div(F.log_softmax(logits/T, dim=1),
                    F.softmax(logits2/T, dim=1),
                    reduction='none').mean(dim=0)
    kd_loss = F.kl_div(F.log_softmax(logits/T, dim=1),
                    F.softmax(logits2/T, dim=1),
                    reduction='batchmean') * T * T
    ce_loss = F.cross_entropy(logits, label, weight=per_cls_weights)
    loss = alpha * kd_loss + ce_loss
    loss_numpy = loss.data.cpu().numpy()
    if epoch%5==0:
        loss_all[ind,int(epoch/5)] = loss_numpy
    return torch.sum(loss)/num_batch


def BKDLoss(epoch,logits,logits2,label,ind,img_num_per_cls,per_cls_weights,loss_all):
    num_batch = logits.shape[0]
    num_classes = len(img_num_per_cls)

    T = 2.0
    alpha = 1.0
    pred_t = F.softmax(logits2/T, dim=1)
    if per_cls_weights is not None:
        pred_t = pred_t * per_cls_weights 
        pred_t = pred_t / pred_t.sum(1)[:, None]

    kd = F.kl_div(F.log_softmax(logits/T, dim=1),
                    pred_t,
                    reduction='none').mean(dim=0)
    kd_loss = kd.sum() * T * T
    ce_loss = F.cross_entropy(logits, label)
    loss = alpha * kd_loss + ce_loss
    loss_numpy = loss.data.cpu().numpy()
    if epoch%5==0:
        loss_all[ind,int(epoch/5)] = loss_numpy
    return torch.sum(loss)/num_batch


def CBS_RRS(epoch,logits,label,logits2,label2,ind,img_num_per_cls,per_cls_weights,loss_all):
    num_batch = logits.shape[0]
    loss_CBS = F.cross_entropy(logits, label)
    loss_RRS = F.cross_entropy(logits2, label2)
    loss = 0.5 * loss_CBS + loss_RRS
    loss_numpy = loss.data.cpu().numpy()
    if epoch%5==0:
        loss_all[ind,int(epoch/5)] = loss_numpy
    return torch.sum(loss)/num_batch


def IB_Loss(epoch,logits,label,features,ind,img_num_per_cls,per_cls_weights,loss_all):
    num_classes = len(img_num_per_cls)
    num_batch = logits.shape[0]
    alpha = 1000.0
    epsilon = 0.001

    grads = torch.sum(torch.abs(F.softmax(logits, dim=1) - F.one_hot(label, num_classes)), 1)
    ib = grads*features.reshape(-1)
    ib = (alpha / (ib + epsilon)).cuda()

    loss = F.cross_entropy(logits, label, reduction='none', weight=per_cls_weights) * ib
    loss_numpy = loss.data.cpu().numpy()
    if epoch%5==0:
        loss_all[ind,int(epoch/5)] = loss_numpy
    return torch.sum(loss)/num_batch


def IB_FocalLoss(epoch,logits,label,features,ind,img_num_per_cls,per_cls_weights,loss_all):
    num_classes = len(img_num_per_cls)
    num_batch = logits.shape[0]
    alpha = 1000.0
    epsilon = 0.001
    gamma = 1.0

    grads = torch.sum(torch.abs(F.softmax(logits, dim=1) - F.one_hot(label, num_classes)), 1)
    ib = grads*features.reshape(-1)
    ib = (alpha / (ib + epsilon)).cuda()

    ce_loss = F.cross_entropy(logits, label, reduction='none', weight=per_cls_weights)
    p = torch.exp(-ce_loss)
    loss = (1-p) ** gamma * ce_loss * ib
    loss_numpy = loss.data.cpu().numpy()
    if epoch%5==0:
        loss_all[ind,int(epoch/5)] = loss_numpy
    return torch.sum(loss)/num_batch


def vs_loss(epoch,logits,label,ind,img_num_per_cls,per_cls_weights,loss_all):
    num_classes = len(img_num_per_cls)
    num_batch = logits.shape[0]
    gamma = 0.15 if num_classes == 10 else 0.05
    tau = 1.25 if num_classes == 10 else 0.75

    cls_probs = img_num_per_cls / img_num_per_cls.sum()
    Delta_list = (1.0 / np.array(img_num_per_cls)) ** gamma
    Delta_list = Delta_list / np.min(Delta_list)
    iota_list = tau * np.log(cls_probs)
    iota_list = torch.cuda.FloatTensor(iota_list)
    Delta_list = torch.cuda.FloatTensor(Delta_list)

    output = logits / Delta_list + iota_list
    loss = F.cross_entropy(output, label, weight=per_cls_weights)
    loss_numpy = loss.data.cpu().numpy()
    if epoch%5==0:
        loss_all[ind,int(epoch/5)] = loss_numpy
    return torch.sum(loss)/num_batch