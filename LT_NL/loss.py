import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import math
from scipy.special import lambertw
from sklearn.neighbors import LocalOutlierFactor


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


def cross_entropy(epoch,logits,label,ind,img_num_per_cls,per_cls_weights,loss_all,gamma=1):
    num_batch = logits.shape[0]
    loss = gamma * F.cross_entropy(logits, label, weight=per_cls_weights, reduction='none')
    loss_numpy = loss.data.cpu().numpy()
    if epoch%5==0:
        loss_all[ind,int(epoch/5)] = loss_numpy
    return torch.sum(loss)/num_batch

def NLLL(epoch,logits,label,ind,img_num_per_cls,loss_all):
    num_batch = logits.shape[0]
    loss = F.nll_loss(logits, label, reduction='none')
    loss_numpy = loss.data.cpu().numpy()
    if epoch%5==0:
        loss_all[ind,int(epoch/5)] = loss_numpy
    return torch.sum(loss)/num_batch

def gce(epoch,logits,label,ind,img_num_per_cls,loss_all):
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


def cores_no_select(epoch,logits,label,ind,img_num_per_cls,noise_prior,loss_all,loss_div_all):
    noise_prior = torch.FloatTensor(noise_prior).cuda()
    num_batch = logits.shape[0]
    num_classes = len(img_num_per_cls)
    beta = f_beta(epoch)
    loss = F.cross_entropy(logits, label, reduction='none')
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
    loss = F.cross_entropy(logits, label, reduction='none')
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
    loss = F.cross_entropy(logits, label, reduction='none')
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
        logits = logits + adjustment
    else:
        pass
    loss_adjust = F.cross_entropy(logits, label, reduction='none')
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
    loss_1 = F.cross_entropy(logits, label, reduction='none')
    ind_1_sorted = np.argsort(loss_1.cpu().data).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(logits2, label, reduction='none')
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


def elr(epoch,logits,label,ind,img_num_per_cls,loss_all):
    num_batch = logits.shape[0]
    num_classes = len(img_num_per_cls)
    num_example = sum(img_num_per_cls)
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



def semi_loss(epoch, logits_x, label_x, logits_u, label_u, img_num_per_cls):
    num_classes = len(img_num_per_cls)
    if num_classes == 10:
        warm_up = 10
    else:
        warm_up = 30
    probs_u = torch.softmax(logits_u, dim=1)
    Lx = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * label_x, dim=1))
    Lu = torch.mean((probs_u - label_u)**2)

    rampup_length = 16.0
    current = np.clip((epoch-warm_up) / rampup_length, 0.0, 1.0)
    lambda_u = 25
    lamb = lambda_u * float(current)

    return Lx, Lu, lamb



def CNLCU_soft(epoch, before_loss_1, before_loss_2, sn_1, sn_2, y_1, y_2, t, forget_rate, ind, co_lambda):
    # before_loss: the mean of soft_losses with size: batch_size * 1
    # co_lambda: sigma^2
    # sn_1, sn_2: selection number 
    before_loss_1, before_loss_2 = torch.from_numpy(before_loss_1).cuda().float(), torch.from_numpy(before_loss_2).cuda().float()
    
    s = torch.tensor(epoch + 1).float() # as the epoch starts from 0
    co_lambda = torch.tensor(co_lambda).float()
    
    loss_1 = F.cross_entropy(y_1, t, reduction='none')
    
    loss_1 = torch.log(1+loss_1+loss_1*loss_1/2)
    
    loss_1_mean = (before_loss_1 * s + loss_1) / (s + 1)
    confidence_bound_1 = co_lambda * (s + (co_lambda * torch.log(2 * s)) / (s * s)) / ((sn_1 + 1) - co_lambda)
    soft_criterion_1 = F.relu(loss_1_mean.float() - confidence_bound_1.cuda().float())
        
    ind_1_sorted = np.argsort(soft_criterion_1.cpu().data).cuda()
    soft_criterion_1_sorted = soft_criterion_1[ind_1_sorted]
    
   
    loss_2 = F.cross_entropy(y_2, t, reduction='none')
    loss_2 = torch.log(1+loss_2+loss_2*loss_2/2)
    
    loss_2_mean = (before_loss_2 * s + loss_2) / (s + 1)
    confidence_bound_2 = co_lambda * (s + (co_lambda * torch.log(2 * s)) / (s * s)) / ((sn_2 + 1) - co_lambda)
    soft_criterion_2 = F.relu(loss_2_mean.float() - confidence_bound_2.cuda().float())
    ind_2_sorted = np.argsort(soft_criterion_2.cpu().data).cuda() 
    
    soft_criterion_2_sorted = soft_criterion_2[ind_2_sorted]
                                      
                                      
    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(soft_criterion_1_sorted))
    
    # index for updates
    ind_1_update = ind_1_sorted[0][:num_remember].cpu()
    ind_2_update = ind_2_sorted[0][:num_remember].cpu()
    
    if len(ind_1_update) == 0:
        ind_1_update = ind_1_sorted.cpu().numpy()
        ind_2_update = ind_2_sorted.cpu().numpy()
        num_remember = ind_1_update.shape[0]
    

    #pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted.cpu()[0][:num_remember]]])/float(num_remember)
    #pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted.cpu()[0][:num_remember]]])/float(num_remember)
    
 
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, ind_1_update, ind_2_update, loss_1_mean, loss_2_mean



def CNLCU_hard(epoch, before_loss_1, before_loss_2, sn_1, sn_2, y_1, y_2, t, forget_rate, ind, co_lambda, loss_bound=2.0):
    # before_loss: the losses with size: batch_size * time_step type numpy
    # co_lambda: t_min
    # sn_1, sn_2: selection number 

    def hard_process(loss):
        # loss: numpy_matrix
        # return: loss_matrix
        loss = loss.detach().cpu().numpy()
        dim_1, dim_2 = loss.shape[0], loss.shape[1]
        if dim_2 >= 5:
            lof = LocalOutlierFactor(n_neighbors=2, algorithm='auto', contamination=0.1, n_jobs=-1)
            #lof = KNN(n_neighbors=2)
            t_o = []
            for i in range(dim_1):
                loss_single = loss[i].reshape((-1, 1))
                outlier_predict_bool = lof.fit_predict(loss_single)
                outlier_number = np.sum(outlier_predict_bool>0)
                loss_single[outlier_predict_bool==1] = 0.
                loss[i,:] = loss_single.transpose()
                t_o.append(outlier_number)
            t_o = np.array(t_o).reshape((dim_1, 1))
        else:
            t_o = np.zeros((dim_1, 1))
        loss = torch.from_numpy(loss).cuda().float()
        return loss, t_o
  
    s = torch.tensor(epoch + 1).float() # as the epoch starts from 0
    co_lambda = torch.tensor(co_lambda).float()
    loss_bound = torch.tensor(loss_bound).float()
    loss_1 = F.cross_entropy(y_1, t, reduction='none')
 
    before_and_loss_1 = torch.cat((torch.from_numpy(before_loss_1).cuda().float(), loss_1.unsqueeze(1).float()), 1)

    before_and_loss_1_hard, t_o_1 = hard_process(before_and_loss_1)
    loss_1_mean = torch.mean(before_and_loss_1_hard, dim=1)
    confidence_bound_1_list = []
    for i in range(loss_1_mean.shape[0]):
        confidence_bound_1 = 2 * torch.sqrt(2 * co_lambda) * loss_bound * (s + 1.414 * torch.tensor(t_o_1[i]).double()) * torch.sqrt(torch.log(4*s)/sn_1[i]) / ((s - torch.tensor(t_o_1[i]).double()) * torch.sqrt(s))
        confidence_bound_1_list.append(confidence_bound_1.item())
        
    confidence_bound_1_numpy = torch.from_numpy(np.array(confidence_bound_1_list)).float().cuda()
    
    hard_criterion_1 = F.relu(loss_1_mean - confidence_bound_1_numpy)
      
    ind_1_sorted = np.argsort(hard_criterion_1.cpu().data).cuda()
    hard_criterion_1_sorted = hard_criterion_1[ind_1_sorted]
    
   
    loss_2 = F.cross_entropy(y_2, t, reduction='none')
 
    before_and_loss_2 = torch.cat((torch.from_numpy(before_loss_2).cuda().float(), loss_2.unsqueeze(1).float()), 1)

    before_and_loss_2_hard, t_o_2 = hard_process(before_and_loss_2)
    loss_2_mean = torch.mean(before_and_loss_2_hard, dim=1)
    confidence_bound_2_list = []
    for i in range(loss_2_mean.shape[0]):
        confidence_bound_2 = 2 * torch.sqrt(2 * co_lambda) * loss_bound * (s + 1.414 * torch.tensor(t_o_2[i]).double()) * torch.sqrt(torch.log(4*s)/sn_2[i]) / ((s - torch.tensor(t_o_2[i]).double()) * torch.sqrt(s))
        confidence_bound_2_list.append(confidence_bound_2.item())
        
    confidence_bound_2_numpy = torch.from_numpy(np.array(confidence_bound_2_list)).float().cuda()
    
    hard_criterion_2 = F.relu(loss_2_mean - confidence_bound_2_numpy)
      
    ind_2_sorted = np.argsort(hard_criterion_2.cpu().data).cuda()
    hard_criterion_2_sorted = hard_criterion_2[ind_2_sorted]
                                      
                                      
    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(hard_criterion_1_sorted))
    
    # index for updates
    ind_1_update = ind_1_sorted[:num_remember].cpu().numpy()
    ind_2_update = ind_2_sorted[:num_remember].cpu().numpy()
    
    if len(ind_1_update) == 0:
        ind_1_update = ind_1_sorted.cpu().numpy()
        ind_2_update = ind_2_sorted.cpu().numpy()
        num_remember = ind_1_update.shape[0]
    

    #pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted.cpu()[:num_remember]]])/float(num_remember)
    #pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted.cpu()[:num_remember]]])/float(num_remember)
    
 
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])
    
    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, ind_1_update, ind_2_update, loss_1, loss_2


def PCL(epoch,logits,label,instance_logits,instance_labels,ind,img_num_per_cls,per_cls_weights,weights=None,clean_idx=None):
    num_classes = len(img_num_per_cls)
    num_batch = logits.shape[0]
    temperature = 0.3

    if weights is None:
        loss = F.cross_entropy(logits, label, weight=per_cls_weights, reduction='none')
    else:
        loss = torch.sum(weights[clean_idx] * F.cross_entropy(logits[clean_idx], label[clean_idx], weight=per_cls_weights, reduction='none')) / torch.sum(weights[clean_idx])

    ##**************Instance contrastive loss****************
    loss_instance =  F.cross_entropy(instance_logits/temperature, instance_labels, weight=per_cls_weights, reduction='none')               

    return loss.mean(), loss_instance.mean()

def cos_similarity(epoch,logits,label,logits2,label2):
    loss = -(F.cosine_similarity(logits, label2, dim=1).mean() + F.cosine_similarity(logits2, label, dim=1).mean()) * 0.5

    return loss

def super_logits_adjustment(epoch,logits,label,ind,img_num_per_cls,per_cls_weights,loss_all):
    num_batch = logits.shape[0]
    num_classes = len(img_num_per_cls)
    tau = math.log(num_classes)
    lam = 1 if num_classes == 10 else 0.25

    tro = 1.0
    label_frequency = img_num_per_cls/sum(img_num_per_cls)
    adjustment = np.log(label_frequency ** tro + 1e-12)
    logits = logits + torch.FloatTensor(adjustment).cuda()
    
    la_loss = F.cross_entropy(logits, label, reduction='none', weight=per_cls_weights).detach()

    x = torch.ones(la_loss.size())*(-2/math.exp(1.))
    x = x.cuda()
    y = 0.5*torch.max(x, (la_loss-tau)/lam)
    y = y.cpu().numpy()
    sigma = np.exp(-lambertw(y))
    sigma = sigma.real.astype(np.float32)
    sigma = torch.from_numpy(sigma).cuda()
    
    loss = (F.cross_entropy(logits, label, reduction='none', weight=per_cls_weights) - tau)*sigma + lam*(torch.log(sigma)**2)

    loss_numpy = loss.data.cpu().numpy()
    if epoch%5==0:
        loss_all[ind,int(epoch/5)] = loss_numpy
    return torch.sum(loss)/num_batch