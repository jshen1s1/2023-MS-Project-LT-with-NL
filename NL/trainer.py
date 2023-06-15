import os
import sys
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from sklearn.mixture import GaussianMixture


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
    model1.train()
    model2.train()
    train_total=0
    train_correct=0 
    train_total2=0
    train_correct2=0 
    num_iter = (len(train_loader.dataset)//train_loader.batch_size)+1
    for i, (images, labels, true_labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
      
        images = Variable(images).cuda(args.gpu)
        labels = Variable(labels).cuda(args.gpu)
        
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

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s+%.3f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Loss1: %.2f  Loss2: %.2f'
                %(args.dataset, args.noise_rate, args.noise_type, args.lt_rate, args.lt_type, epoch+1, args.epochs, i+1, num_iter, loss_1.item(), loss_2.item()))
        sys.stdout.flush()

    train_acc1=float(train_correct)/float(train_total)
    train_acc2=float(train_correct2)/float(train_total2)
    return train_acc1, train_acc2


def run_cores(train_loader, criterion, model, optimizer, epoch, args, loss_all, loss_div_all, noise_prior_cur):
    model.train()
    train_total=0
    train_correct=0 
    img_num_per_cls = args.cls_num_list
    idx_each_class_noisy = [[] for i in range(args.num_classes)]
    num_iter = (len(train_loader.dataset)//train_loader.batch_size)+1
    for i, (images, labels, true_labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
        batch_size = len(ind)

        images = Variable(images).cuda(args.gpu)
        labels = Variable(labels).cuda(args.gpu)
       
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

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s+%.3f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Loss: %.4f'
                %(args.dataset, args.noise_rate, args.noise_type, args.lt_rate, args.lt_type, epoch+1, args.epochs, i+1, num_iter, loss.item()))
        sys.stdout.flush()
    
    noise_prior_delta = np.array([len(idx_each_class_noisy[i]) for i in range(args.num_classes)])
    train_acc=float(train_correct)/float(train_total)
    np.save(args.save_dir + '/' + args.noise_type + str(args.noise_rate) + args.lt_type + str(args.lt_rate) + ('DT' if args.train_rule == 'Dual_t' else '') + ('WVN_RS' if args.WVN_RS else '') +'_loss_all.npy',loss_all)
    return train_acc, noise_prior_delta


def eval_train(args,model,eval_loader):    
    model.eval()
    img_num_per_cls = np.array(eval_loader.dataset.get_cls_num_list())
    num_training_samples = sum(img_num_per_cls)
    losses = torch.zeros(num_training_samples)    
    CE = nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        for batch_idx, (inputs, targets, _, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(args.gpu), targets.cuda(args.gpu) 
            outputs, _ = model(inputs) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]         
    losses = (losses-losses.min())/(losses.max()-losses.min())    

    input_loss = losses.reshape(-1,1)
    
    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]         
    return prob


def run_divideMix(epoch,args,net,net2,optimizer,criterion,labeled_trainloader,unlabeled_trainloader,loss_all):
    net.train()
    net2.eval() #fix one network and train the other
    train_total=0
    train_correct=0
    
    img_num_per_cls = np.array(labeled_trainloader.dataset.get_cls_num_list())
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x, indexes) in enumerate(labeled_trainloader):
        ind=indexes.cpu().numpy().transpose()      
        try:
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = next(unlabeled_train_iter)               
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_classes).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = Variable(inputs_x).cuda(args.gpu), Variable(inputs_x2).cuda(args.gpu), Variable(labels_x).cuda(args.gpu), Variable(w_x).cuda(args.gpu)
        inputs_u, inputs_u2 = Variable(inputs_u).cuda(args.gpu), Variable(inputs_u2).cuda(args.gpu)

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11, _ = net(inputs_u)
            outputs_u12, _ = net(inputs_u2)
            outputs_u21, _ = net2(inputs_u)
            outputs_u22, _ = net2(inputs_u2)            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/0.5) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            outputs_x, _ = net(inputs_x)
            outputs_x2, _ = net(inputs_x2)            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/0.5) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch
        l = np.random.beta(4, 4)        
        l = max(l, 1-l)
                
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
        # one-hot --> labels
        mixed_target_ = torch.argmax(mixed_target, dim=1)
                
        logits, _ = net(mixed_input)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]      

        _, pred = torch.max(logits.data, 1)
        train_total += mixed_target_.size(0)
        train_correct += pred.eq(mixed_target_).sum().item() 
  
        Lx, Lu, lamb = criterion(epoch+batch_idx/num_iter, logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], img_num_per_cls)

        # regularization
        prior = torch.ones(args.num_classes)/args.num_classes
        prior = prior.cuda(args.gpu)        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + lamb * Lu + penalty
        loss_numpy = loss.data.cpu().numpy()
        if epoch%5==0:
            loss_all[ind,int(epoch/5)] = loss_numpy
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s+%.3f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.4f'
                %(args.dataset, args.noise_rate, args.noise_type, args.lt_rate, args.lt_type, epoch+1, args.epochs, batch_idx+1, num_iter, Lx.item(), Lu.item()))
        sys.stdout.flush()

    train_acc=100.*float(train_correct)/float(train_total)
    np.save(args.save_dir + '/' + args.noise_type + str(args.noise_rate) + args.lt_type + str(args.lt_rate) + ('DT' if args.train_rule == 'Dual_t' else '') + ('WVN_RS' if args.WVN_RS else '') +'_loss_all.npy',loss_all)
    return train_acc
    
