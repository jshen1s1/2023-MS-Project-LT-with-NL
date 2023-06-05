import os
import sys
import torch 
import warnings
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from sklearn.mixture import GaussianMixture
from utils import accuracy, label_clean_PCL

# trainers

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
                %(args.dataset, args.noise_rate, args.noise_type, args.lt_rate, args.lt_type, epoch, args.epochs, i+1, num_iter, loss_1.item(), loss_2.item()))
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
                %(args.dataset, args.noise_rate, args.noise_type, args.lt_rate, args.lt_type, epoch, args.epochs, i+1, num_iter, loss.item()))
        sys.stdout.flush()
    
    noise_prior_delta = np.array([len(idx_each_class_noisy[i]) for i in range(args.num_classes)])
    train_acc=float(train_correct)/float(train_total)
    np.save(args.save_dir + '/' + args.noise_type + str(args.noise_rate) + args.lt_type + str(args.lt_rate) + (args.train_opt if args.train_opt else '') + ('WVN_RS' if args.WVN_RS else '') +'_loss_all.npy',loss_all)
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
        sys.stdout.write('%s:%.1f-%s+%.3f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.4f  Unlabeled loss: %.4f'
                %(args.dataset, args.noise_rate, args.noise_type, args.lt_rate, args.lt_type, epoch, args.epochs, batch_idx+1, num_iter, Lx.item(), Lu.item()))
        sys.stdout.flush()

    train_acc=100.*float(train_correct)/float(train_total)
    np.save(args.save_dir + '/' + args.noise_type + str(args.noise_rate) + args.lt_type + str(args.lt_rate) + (args.train_opt if args.train_opt else '') + ('WVN_RS' if args.WVN_RS else '') +'_loss_all.npy',loss_all)
    return train_acc
    
def run_RoLT(train_loader, criterion, per_cls_weights, model, optimizer, epoch, args, loss_all):
    # switch to train mode
    model.train()    
    train_loader.dataset.train_mode()

    img_num_per_cls = args.cls_num_list
    correct = 0
    total = 0
    num_iter = (len(train_loader.dataset)//train_loader.batch_size)+1
    for i, (images, labels, true_labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
        images = Variable(images).cuda(args.gpu)
        labels = Variable(labels).cuda(args.gpu)
        labels = args.current_labels[indexes]
        outputs, _ = model(images)
        loss = 0
        for soft_weight, soft_target in zip(args.soft_weights, args.soft_targets):
            targets = soft_target[indexes]
            loss_perf = criterion(epoch,outputs,targets,ind,img_num_per_cls,per_cls_weights,loss_all,gamma=soft_weight)
            loss += loss_perf
        loss_numpy = loss.data.cpu().numpy()
        if epoch%5==0:
            loss_all[ind,int(epoch/5)] = loss_numpy
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



def run_CNLCU(train_loader, criterion, epoch, args, model1, optimizer1, model2, optimizer2, rate_schedule, before_loss_1, before_loss_2, sn_1, sn_2, co_lambda_plan):
    model1.train()
    model2.train()
    train_total=0
    train_correct=0 
    train_total2=0
    train_correct2=0 
    
    num_iter = (len(train_loader.dataset)//args.batch_size)+1

    before_loss_1_list=[]
    before_loss_2_list=[]
    
    ind_1_update_list=[]
    ind_2_update_list=[]
    for i, (images, labels, true_labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
        batch_size = len(ind)
      
        images = Variable(images).cuda(args.gpu)
        labels = Variable(labels).cuda(args.gpu)
        
        start_point = int(i * batch_size)
        stop_point = int((i + 1) * batch_size)
        # Forward + Backward + Optimize
        logits1, _ = model1(images)
        prec1, _ = accuracy(logits1, labels, topk=(1, 5))
        train_total+=1
        train_correct+=prec1

        logits2, _ = model2(images)
        prec2, _ = accuracy(logits2, labels, topk=(1, 5))
        train_total2+=1
        train_correct2+=prec2 
        if epoch < 80:
            loss_1, loss_2, ind_1_update, ind_2_update, loss_1_mean, loss_2_mean = criterion(epoch, before_loss_1[start_point:stop_point], before_loss_2[start_point:stop_point], 
                                        sn_1[start_point:stop_point], sn_2[start_point:stop_point], logits1, logits2, labels, rate_schedule[epoch], ind, co_lambda_plan[epoch])
        else:
            loss_1, loss_2, ind_1_update, ind_2_update, loss_1_mean, loss_2_mean = criterion(epoch, before_loss_1[start_point:stop_point], before_loss_2[start_point:stop_point], 
                                        sn_1[start_point:stop_point], sn_2[start_point:stop_point], logits1, logits2, labels, rate_schedule[epoch], ind, 0.)

        before_loss_1_list += list(np.array(loss_1_mean.detach().cpu()))
        before_loss_2_list += list(np.array(loss_2_mean.detach().cpu()))
        
        ind_1_update_list += list(np.array(ind_1_update + i * batch_size))
        ind_2_update_list += list(np.array(ind_2_update + i * batch_size))

        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s+%.3f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Loss1: %.2f  Loss2: %.2f'
                %(args.dataset, args.noise_rate, args.noise_type, args.lt_rate, args.lt_type, epoch, args.epochs, i+1, num_iter, loss_1.item(), loss_2.item()))
        sys.stdout.flush()

    train_acc1=float(train_correct)/float(train_total)
    train_acc2=float(train_correct2)/float(train_total2)
    return train_acc1, train_acc2, before_loss_1_list, before_loss_2_list, ind_1_update_list, ind_2_update_list


def compute_features(dataloader, model, N, args):
    model.eval()
    batch_size = dataloader.batch_size
    with torch.no_grad():
        for i, (images, targets, true_labels, indexes, img_aug) in enumerate(dataloader):
            inputs = Variable(images).cuda(args.gpu)
            output, feat = model(inputs)             
            feat = feat.data.cpu().numpy()
            prob = F.softmax(output,dim=1)
            prob = prob.data.cpu()
            if i == 0:
                features = np.zeros((N, feat.shape[1]),dtype='float32')
                labels = torch.zeros(N,dtype=torch.long)                        
                probs = torch.zeros(N,args.num_classes) 
            if i < len(dataloader) - 1:
                features[i * batch_size: (i + 1) * batch_size] = feat
                labels[i * batch_size: (i + 1) * batch_size] = targets
                probs[i * batch_size: (i + 1) * batch_size] = prob
            else:
                # special treatment for final batch
                features[i * batch_size:] = feat
                labels[i * batch_size:] = targets
                probs[i * batch_size:] = prob
    return features,labels,probs 

def run_PCL(train_loader, eval_loader, criterion, per_cls_weights, weights, model, optimizer, epoch, args, loss_all):
    # Compute features
    prototypes = []
    features,labels,probs = compute_features(eval_loader, model, len(eval_loader.dataset), args)
    features = torch.Tensor(features)

    for c in range(args.num_classes):
        if weights is None:
            prototype = features[np.where(labels.numpy()==c)].mean(0)    #compute prototypes as mean embeddings
        else:
            class_idx = np.where(labels.numpy()==c)
            prototype = (weights[class_idx].view(-1, 1) * features[class_idx]).sum(0) / torch.sum(weights[class_idx])
        prototypes.append(torch.Tensor(prototype)) 
    prototypes = torch.stack(prototypes).cuda(args.gpu)
    prototypes = F.normalize(prototypes, p=2, dim=1)    #normalize the prototypes

    if epoch >= args.knn_start_epoch:
        '''
        if epoch == args.knn_start_epoch:
            gt_score = probs[labels>=0,labels]
            gt_clean = gt_score>args.low_th
            soft_labels = probs.clone() 
            soft_labels[gt_clean] = torch.zeros(gt_clean.sum(), args.num_classes).scatter_(1, labels[gt_clean].view(-1,1), 1)
        '''
        hard_labels, clean_idxs, weights = label_clean_PCL(args,epoch,features,labels,probs,prototypes,eval_loader.dataset.true_labels)

    # switch to train mode
    model.train()    
    train_loader.dataset.train_mode()

    img_num_per_cls = args.cls_num_list
    record = {}
    num_iter = (len(train_loader.dataset)//args.batch_size)+1
    for i, (images, labels, true_labels, indexes, img_aug) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
        batch_size = len(ind)
        images = Variable(images).cuda(args.gpu,non_blocking=True)
        labels = Variable(labels).cuda(args.gpu,non_blocking=True)
        outputs, feat = model(images)
        record['train_accuracy'] = accuracy(outputs,labels)[0].item() 

        img_aug = Variable(img_aug).cuda(args.gpu,non_blocking=True)
        shuffle_idx = torch.randperm(batch_size)
        mapping = {k:v for (v,k) in enumerate(shuffle_idx)}
        reverse_idx = torch.LongTensor([mapping[k] for k in sorted(mapping.keys())]) 
        feat_aug = model(img_aug[shuffle_idx], has_out=False)
        feat_aug = feat_aug[reverse_idx]

        if epoch >= args.knn_start_epoch:
            labels = Variable(hard_labels[indexes]).cuda(args.gpu,non_blocking=True)
            clean_idx = clean_idxs[indexes]
            weight = Variable(weights[indexes]).cuda(args.gpu,non_blocking=True)
        else:
            weight = None
            clean_idx = None
        ##**************Instance contrastive loss****************
        sim_clean = torch.mm(feat, feat.t())
        mask = (torch.ones_like(sim_clean) - torch.eye(batch_size, device=sim_clean.device)).bool()
        sim_clean = sim_clean.masked_select(mask).view(batch_size, -1)

        sim_aug = torch.mm(feat, feat_aug.t())
        sim_aug = sim_aug.masked_select(mask).view(batch_size, -1)

        logits_pos = torch.bmm(feat.view(batch_size,1,-1),feat_aug.view(batch_size,-1,1)).squeeze(-1)
        logits_neg = torch.cat([sim_clean,sim_aug],dim=1)

        instance_logits = torch.cat([logits_pos,logits_neg],dim=1)
        instance_labels = torch.zeros(batch_size).long().cuda(args.gpu)

        loss_CE, loss_instance = criterion(epoch,outputs,labels,instance_logits,instance_labels,ind,img_num_per_cls,per_cls_weights,weights=weight,clean_idx=clean_idx)
        loss = loss_CE + loss_instance
        record['acc_inst'] = accuracy(instance_logits,instance_labels)[0].item()
        ##**************Mixup Prototypical contrastive loss****************
        if epoch >= args.knn_start_epoch:
            if sum(clean_idx) > 0:
                L = np.random.beta(8, 8) 
                labels_ = torch.zeros(batch_size, args.num_classes).cuda(args.gpu).scatter_(1, labels.view(-1,1), 1)
                inputs = torch.cat([images[clean_idx],img_aug[clean_idx]],dim=0)
                idx = torch.randperm(clean_idx.sum()*2) 
                labels_ = torch.cat([labels_[clean_idx],labels_[clean_idx]],dim=0)
                weight = torch.cat([weight[clean_idx], weight[clean_idx]], dim=0)

                input_mix = L * inputs + (1 - L) * inputs[idx]  
                labels_mix = L * labels_ + (1 - L) * labels_[idx]
                weight = L * weight + (1 - L) * weight[idx]
                feat_mix = model(input_mix, has_out=False)

                logits_proto = torch.mm(feat_mix,prototypes.t())/0.3
                loss_proto = -torch.sum(weight * torch.sum(F.log_softmax(logits_proto, dim=1) * labels_mix, dim=1)) / torch.sum(weight)
                loss += args.w_proto*loss_proto
        else:
            target = torch.zeros(batch_size, args.num_classes).cuda(args.gpu).scatter_(1, labels.view(-1,1), 1)
            input_mix = torch.cat([images,img_aug],dim=0)
            labels_mix = torch.cat([target,target],dim=0)
            feat_mix = model(input_mix, has_out=False)
            logits_proto = torch.mm(feat_mix,prototypes.t())/0.3
            loss_proto = -torch.mean(torch.sum(F.log_softmax(logits_proto, dim=1) * labels_mix, dim=1))
            loss += args.w_proto*loss_proto

        loss_numpy = loss.data.cpu().numpy()
        if epoch%5==0:
            loss_all[ind,int(epoch/5)] = loss_numpy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s+%.3f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Loss: %.4f ce_loss: %.4f inst_loss: %.4f prot_loss: %.4f'
                %(args.dataset, args.noise_rate, args.noise_type, args.lt_rate, args.lt_type, epoch, args.epochs, i+1, num_iter, loss.item(), loss_CE.item(), loss_instance.item(), loss_proto.item()))
        sys.stdout.flush()       

    acc = record['train_accuracy']
    np.save(args.save_dir + '/' + args.noise_type + str(args.noise_rate) + args.lt_type + str(args.lt_rate) + (args.train_opt if args.train_opt else '') + ('WVN_RS' if args.WVN_RS else '') +'_loss_all.npy',loss_all)
    return acc, weights