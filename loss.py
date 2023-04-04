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


def cross_entropy(epoch,logits,label,ind,img_num_per_cls,loss_all,num_example=None):
    num_batch = logits.shape[0]
    loss = F.cross_entropy(logits, label, reduce = False)
    loss_numpy = loss.data.cpu().numpy()
    if epoch%5==0:
        loss_all[ind,int(epoch/5)] = loss_numpy
    return torch.sum(loss)/num_batch

def NLLL(epoch,logits,label,ind,img_num_per_cls,loss_all,num_example=None):
    num_batch = logits.shape[0]
    loss = F.nll_loss(logits, label, reduce = False)
    loss_numpy = loss.data.cpu().numpy()
    if epoch%5==0:
        loss_all[ind,int(epoch/5)] = loss_numpy
    return torch.sum(loss)/num_batch

def gce(epoch,logits,label,ind,img_num_per_cls,loss_all,num_example=None):
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


def focal_loss(epoch,logits,label,ind,img_num_per_cls,loss_all,num_example=None):
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

def cb_ce(epoch,logits,label,ind,img_num_per_cls,loss_all,num_example=None):
    num_batch = logits.shape[0]
    num_classes = len(img_num_per_cls)
    beta = 0.9999
    class_balanced_weight = np.array([(1-beta)/(1- beta ** N) for N in img_num_per_cls])
    class_balanced_weight = torch.FloatTensor(class_balanced_weight / np.sum(class_balanced_weight) * num_classes).cuda()
    loss = F.cross_entropy(logits, label, weight=class_balanced_weight,reduce = False)
    loss_numpy = loss.data.cpu().numpy()
    if epoch%5==0:
        loss_all[ind,int(epoch/5)] = loss_numpy
    return torch.sum(loss)/num_batch


def cb_focal(epoch,logits,label,ind,img_num_per_cls,loss_all,num_example=None):
    num_batch = logits.shape[0]
    num_classes = len(img_num_per_cls)
    beta = 0.9999
    gamma = 2.0
    class_balanced_weight = np.array([(1-beta)/(1- beta ** N) for N in img_num_per_cls])
    class_balanced_weight = torch.FloatTensor(class_balanced_weight / np.sum(class_balanced_weight) * num_classes).cuda()
    weight = (class_balanced_weight[label]).cuda()
    p = F.sigmoid(logits)
    focal_weights = torch.pow((1-p)*label + p * (1-label), gamma).cuda()
    loss_f = F.binary_cross_entropy_with_logits(logits, label, reduction = 'none') * focal_weights
    loss = loss_f * weight.view(-1, 1)
    loss_numpy = loss.data.cpu().numpy()
    #if epoch%5==0:
    #    loss_all[ind,int(epoch/5)] = loss_numpy
    return torch.sum(loss)/num_batch


def logits_adjustment(epoch,logits,label,ind,img_num_per_cls,loss_all,num_example=None):
    tro = 1.0
    label_frequency = img_num_per_cls/sum(img_num_per_cls)
    adjustment = np.log(label_frequency ** tro + 1e-12)
    logits = logits + torch.FloatTensor(adjustment).cuda()

    num_batch = logits.shape[0]
    loss = F.cross_entropy(logits, label, reduce = False)
    loss_numpy = loss.data.cpu().numpy()
    if epoch%5==0:
        loss_all[ind,int(epoch/5)] = loss_numpy
    return torch.sum(loss)/num_batch



def cores_no_select(epoch,logits,label,ind,img_num_per_cls,noise_prior,loss_all,loss_div_all):
    noise_prior = torch.FloatTensor(noise_prior).cuda()
    num_batch = logits.shape[0]
    num_classes = len(img_num_per_cls)
    beta = f_beta(epoch)
    loss = F.cross_entropy(logits, label, reduce = False)
    loss_numpy = loss.data.cpu().numpy()
    if epoch%5==0:
        loss_all[ind,int(epoch/5)] = loss_numpy
    #loss_ = -torch.log(F.softmax(logits) + 1e-8)
    if noise_prior is None:
        loss =  loss - beta*torch.mean(loss_,1)
    else:
        loss =  loss - beta*torch.sum(torch.mul(noise_prior, loss_),1)
    loss_v = np.zeros(num_batch)
    for i in range(num_batch):
        loss_v[i] = 1.0
    loss_v = loss_v.astype(np.float32)
    loss_v_var = Variable(torch.from_numpy(loss_v)).cuda()
    loss_ = loss_v_var * loss
    return torch.sum(loss_)/sum(loss_v), loss_v.astype(int)



def cores(epoch,logits,label,ind,img_num_per_cls,noise_prior,loss_all,loss_div_all):
    noise_prior = torch.FloatTensor(noise_prior).cuda()
    num_batch = logits.shape[0]
    num_classes = len(img_num_per_cls)
    beta = f_beta(epoch)
    loss = F.cross_entropy(logits, label, reduce = False)
    loss_numpy = loss.data.cpu().numpy()
    loss_v = np.zeros(num_batch)
    loss_div_numpy = float(np.array(0))
    loss_ = -torch.log(F.softmax(logits) + 1e-8)
    # sel metric
    loss_sel =  loss - torch.mean(loss_,1) 
    if noise_prior is None:
        loss =  loss - beta*torch.mean(loss_,1)
    else:
        loss =  loss - beta*torch.sum(torch.mul(noise_prior, loss_),1)
    
    loss_div_numpy = loss_sel.data.cpu().numpy()
    if epoch%5==0:
        loss_all[ind,int(epoch/5)] = loss_numpy
        loss_div_all[ind,int(epoch/5)] = loss_div_numpy
    for i in range(num_batch):
        if epoch <=30:
            loss_v[i] = 1.0
        elif loss_div_numpy[i] <= 0:
            loss_v[i] = 1.0
    loss_v = loss_v.astype(np.float32)
    loss_v_var = Variable(torch.from_numpy(loss_v)).cuda()
    loss_ = loss_v_var * loss
    if sum(loss_v) == 0.0:
        return torch.mean(loss_)/100000000,loss_v.astype(int)
    else:
        return torch.sum(loss_)/sum(loss_v), loss_v.astype(int)



def cores_logits_adjustment(epoch,logits,label,ind,img_num_per_cls,noise_prior,loss_all,loss_div_all):
    noise_prior = torch.FloatTensor(noise_prior).cuda()
    num_batch = logits.shape[0]
    num_classes = len(img_num_per_cls)
    beta = f_beta(epoch)
    loss = F.cross_entropy(logits, label, reduce = False)
    loss_numpy = loss.data.cpu().numpy()
    loss_v = np.zeros(num_batch)
    loss_div_numpy = float(np.array(0))
    loss_ = -torch.log(F.softmax(logits) + 1e-8)
    # sel metric
    loss_sel =  loss - torch.mean(loss_,1) 
    
    loss_div_numpy = loss_sel.data.cpu().numpy()
    if epoch%5==0:
        loss_all[ind,int(epoch/5)] = loss_numpy
        loss_div_all[ind,int(epoch/5)] = loss_div_numpy
    for i in range(num_batch):
        if epoch <=30:
            loss_v[i] = 1.0
        elif loss_div_numpy[i] <= 0:
            loss_v[i] = 1.0
    loss_v = loss_v.astype(np.float32)
    loss_v_var = Variable(torch.from_numpy(loss_v)).cuda()

    # logits adjustment cores on selected data
    if epoch >= 55:
        tro = 0.3
        adjustment = torch.log(noise_prior ** tro + 1e-12)
        logits = logits + torch.FloatTensor(adjustment).cuda()
    else:
        pass
    loss_adjust = F.cross_entropy(logits, label, reduce = False)
    loss_ = -torch.log(F.softmax(logits) + 1e-8)
    if noise_prior is None:
        loss_adjust =  loss_adjust - beta*torch.mean(loss_,1)
    else:
        loss_adjust =  loss_adjust - beta*torch.sum(torch.mul(noise_prior, loss_),1)

    loss_ = loss_v_var * loss_adjust
    if sum(loss_v) == 0.0:
        return torch.mean(loss_)/100000000,loss_v.astype(int)
    else:
        return torch.sum(loss_)/sum(loss_v), loss_v.astype(int)



def co_teaching(epoch, logits, logits2, label, forget_rate, ind, step):
    loss_1 = F.cross_entropy(logits, label, reduce = False)
    ind_1_sorted = np.argsort(loss_1.cpu().data).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(logits2, label, reduce = False)
    ind_2_sorted = np.argsort(loss_2.cpu().data).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    #pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]])/float(num_remember)
    #pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]])/float(num_remember)

    ind_1_update=ind_1_sorted[:num_remember].cpu()
    ind_2_update=ind_2_sorted[:num_remember].cpu()
    if len(ind_1_update) == 0:
        ind_1_update = ind_1_sorted.cpu().numpy()
        ind_2_update = ind_2_sorted.cpu().numpy()
        num_remember = ind_1_update.shape[0]
    # exchange
    loss_1_update = F.cross_entropy(logits[ind_2_update], label[ind_2_update])
    loss_2_update = F.cross_entropy(logits2[ind_1_update], label[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember


def co_teaching_plus(epoch, logits, logits2, label, forget_rate, ind, step):
    outputs = F.softmax(logits, dim=1)
    outputs2 = F.softmax(logits2, dim=1)

    _, pred1 = torch.max(logits.data, 1)
    _, pred2 = torch.max(logits2.data, 1)

    pred1, pred2 = pred1.cpu().numpy(), pred2.cpu().numpy()

    logical_disagree_id=np.zeros(label.size(), dtype=bool)
    disagree_id = []
    for idx, p1 in enumerate(pred1): 
        if p1 != pred2[idx]:
            disagree_id.append(idx) 
            logical_disagree_id[idx] = True
    
    temp_disagree = ind*logical_disagree_id.astype(np.int64)
    ind_disagree = np.asarray([i for i in temp_disagree if i != 0]).transpose()
    try:
        assert ind_disagree.shape[0]==len(disagree_id)
    except:
        disagree_id = disagree_id[:ind_disagree.shape[0]]
     
    _update_step = np.logical_or(logical_disagree_id, step < 5000).astype(np.float32)
    update_step = Variable(torch.from_numpy(_update_step)).cuda()

    if len(disagree_id) > 0:
        update_labels = label[disagree_id]
        update_outputs = outputs[disagree_id] 
        update_outputs2 = outputs2[disagree_id] 
        
        loss_1, loss_2 = co_teaching(epoch, update_outputs, update_outputs2, update_labels, forget_rate, ind_disagree, step)
    else:
        update_labels = label
        update_outputs = outputs
        update_outputs2 = outputs2

        cross_entropy_1 = F.cross_entropy(update_outputs, update_labels)
        cross_entropy_2 = F.cross_entropy(update_outputs2, update_labels)

        loss_1 = torch.sum(update_step*cross_entropy_1)/label.size()[0]
        loss_2 = torch.sum(update_step*cross_entropy_2)/label.size()[0]
 
        #pure_ratio_1 = np.sum(noise_or_not[ind])/ind.shape[0]
        #pure_ratio_2 = np.sum(noise_or_not[ind])/ind.shape[0]
    return loss_1, loss_2  


def elr(epoch,logits,label,ind,img_num_per_cls,loss_all,num_example=5000):
    num_batch = logits.shape[0]
    num_classes = len(img_num_per_cls)
    beta = 0.7
    lambda_ = 3.0
    target = torch.zeros(num_example, num_classes).cuda()
    y_pred = F.softmax(logits,dim=1)
    y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)
    y_pred_ = y_pred.data.detach()
    target[ind] = beta * target[ind] + (1-beta) * ((y_pred_)/(y_pred_).sum(dim=1,keepdim=True))
    ce_loss = F.cross_entropy(logits, label)
    elr_reg = ((1-(target[ind] * y_pred).sum(dim=1)).log()).mean()
    final_loss = ce_loss + lambda_*elr_reg
    loss_numpy = final_loss.data.cpu().numpy()
    if epoch%5==0:
        loss_all[ind,int(epoch/5)] = loss_numpy
    return  torch.sum(final_loss)/num_batch



def semi_loss(epoch, logits, logits_x, logits_u, label_x, label_u, ind, img_num_per_cls, warm_up):
    num_classes = len(img_num_per_cls)
    probs_u = torch.softmax(logits_u, dim=1)
    Lx = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * label_x, dim=1))
    Lu = torch.mean((probs_u - label_u)**2)

    rampup_length = 16.0
    current = np.clip((epoch-warm_up) / rampup_length, 0.0, 1.0)
    lambda_u = 25
    lamb = lambda_u * float(current)

    prior = torch.ones(num_classes)/num_classes
    prior = prior.cuda()        
    pred_mean = torch.softmax(logits, dim=1).mean(0)
    penalty = torch.sum(prior*torch.log(prior/pred_mean))

    loss = Lx + lamb * Lu  + penalty
    return loss


def ldam(epoch,logits,label,ind,img_num_per_cls,loss_all,num_example=None):
    num_batch = logits.shape[0]
    num_classes = len(img_num_per_cls)
    max_m = 0.5
    s = 30.0
    ### DRW
    idx = epoch // 160
    beta = [0, 0.9999]
    per_cls_weights = np.array([(1-beta[idx])/(1- beta[idx] ** N) for N in img_num_per_cls])
    per_cls_weights = torch.FloatTensor(per_cls_weights / np.sum(per_cls_weights) * num_classes).cuda()

    m_list = 1.0 / np.sqrt(np.sqrt(img_num_per_cls))
    m_list = m_list * (max_m / np.max(m_list))
    m_list = torch.cuda.FloatTensor(m_list)

    index = torch.zeros_like(logits, dtype=torch.uint8)
    index.scatter_(1, label.data.view(-1, 1), 1) # one-hot index

    index_float = index.type(torch.cuda.FloatTensor)
    batch_m = torch.matmul(m_list[None, :], index_float.transpose(0,1))
    batch_m = batch_m.view((-1, 1))
    x_m = logits - batch_m * s

    output = torch.where(index, x_m, logits)
    loss =  F.cross_entropy(output, label, weight=per_cls_weights)
    loss_numpy = loss.data.cpu().numpy()
    if epoch%5==0:
        loss_all[ind,int(epoch/5)] = loss_numpy
    return  torch.sum(loss)/num_batch



def lade(epoch,logits,label,ind,img_num_per_cls,loss_all,num_example=None):
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




def KDLoss(epoch,logits,logits2,label,ind,img_num_per_cls,loss_all,num_example=None):
    num_batch = logits.shape[0]
    num_classes = len(img_num_per_cls)
    beta = 0.9999
    per_cls_weights  = np.array([(1-beta)/(1- beta ** N) for N in img_num_per_cls])
    per_cls_weights  = torch.FloatTensor(per_cls_weights  / np.sum(per_cls_weights ) * num_classes).cuda()

    T = 2.0
    alpha = 1.0
    
    kd = F.kl_div(F.log_softmax(logits/T, dim=1),
                    F.softmax(logits2/T, dim=1),
                    reduction='none').mean(dim=0)
    kd_loss = F.kl_div(F.log_softmax(logits/T, dim=1),
                    F.softmax(logits2/T, dim=1),
                    reduction='batchmean') * T * T
    ce_loss = F.cross_entropy(logits, label)
    loss = alpha * kd_loss + ce_loss
    loss_numpy = loss.data.cpu().numpy()
    if epoch%5==0:
        loss_all[ind,int(epoch/5)] = loss_numpy
    return torch.sum(loss)/num_batch



def BKDLoss(epoch,logits,logits2,label,ind,img_num_per_cls,loss_all,num_example=None):
    num_batch = logits.shape[0]
    num_classes = len(img_num_per_cls)
    beta = 0.9999
    per_cls_weights  = np.array([(1-beta)/(1- beta ** N) for N in img_num_per_cls])
    per_cls_weights  = torch.FloatTensor(per_cls_weights  / np.sum(per_cls_weights ) * num_classes).cuda()

    T = 2.0
    alpha = 1.0
    pred_t = F.softmax(logits2/T, dim=1)
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



def CBS_RRS(epoch,logits,label,logits2,label2,ind,img_num_per_cls,loss_all,num_example=None):
    num_batch = logits.shape[0]
    loss_CBS = F.cross_entropy(logits, label)
    loss_RRS = F.cross_entropy(logits2, label2)
    loss = 0.5 * loss_CBS + loss_RRS
    loss_numpy = loss.data.cpu().numpy()
    if epoch%5==0:
        loss_all[ind,int(epoch/5)] = loss_numpy
    return torch.sum(loss)/num_batch



def IB_Loss(epoch,logits,label,features,ind,img_num_per_cls,loss_all,num_example=None):
    num_classes = len(img_num_per_cls)
    num_batch = logits.shape[0]
    alpha = 1000.0
    epsilon = 0.001
    per_cls_weights  = np.array([1 / N for N in img_num_per_cls])
    per_cls_weights  = torch.FloatTensor(per_cls_weights  / np.sum(per_cls_weights ) * num_classes).cuda()
    grads = torch.sum(torch.abs(F.softmax(logits, dim=1) - F.one_hot(label, num_classes)), 1)
    ib = grads*features.reshape(-1)
    ib = (alpha / (ib + epsilon)).cuda()

    loss = F.cross_entropy(logits, label, reduction='none', weight=per_cls_weights) * ib
    loss_numpy = loss.data.cpu().numpy()
    if epoch%5==0:
        loss_all[ind,int(epoch/5)] = loss_numpy
    return torch.sum(loss)/num_batch



def IB_FocalLoss(epoch,logits,label,features,ind,img_num_per_cls,loss_all,num_example=None):
    num_classes = len(img_num_per_cls)
    num_batch = logits.shape[0]
    alpha = 1000.0
    epsilon = 0.001
    gamma = 1.0
    per_cls_weights  = np.array([1 / N for N in img_num_per_cls])
    per_cls_weights  = torch.FloatTensor(per_cls_weights  / np.sum(per_cls_weights ) * num_classes).cuda()
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



def vs_loss(epoch,logits,label,ind,img_num_per_cls,loss_all,num_example=None):
    num_classes = len(img_num_per_cls)
    num_batch = logits.shape[0]
    gamma = 0.2
    tau = 1.2
    cls_probs = img_num_per_cls / img_num_per_cls.sum()
    Delta_list = (1.0 / np.array(img_num_per_cls)) ** gamma
    Delta_list = Delta_list / np.min(Delta_list)
    iota_list = tau * np.log(cls_probs)
    iota_list = torch.cuda.FloatTensor(iota_list)
    Delta_list = torch.cuda.FloatTensor(Delta_list)

    output = logits / Delta_list + iota_list
    loss = F.cross_entropy(output, label)
    loss_numpy = loss.data.cpu().numpy()
    if epoch%5==0:
        loss_all[ind,int(epoch/5)] = loss_numpy
    return torch.sum(loss)/num_batch