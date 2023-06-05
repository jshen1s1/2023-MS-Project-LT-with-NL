from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10
import os
import os.path
import copy
import hashlib
import errno
import shutil
import warnings
import numpy as np
from numpy.testing import assert_array_almost_equal
import torch
import torch.nn.functional as F 
from torch.autograd import Variable
from bisect import bisect_right
from sklearn.mixture import GaussianMixture


class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])


test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])




# basic function#
def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    #print np.max(y), P.shape[0]
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    #print m
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        #print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    #print P

    return y_train, actual_noise, P

def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        P[nb_classes-1, nb_classes-1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        #print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    #print P

    return y_train, actual_noise, P

def noisify(dataset='mnist', nb_classes=10, train_labels=None, noise_type=None, noise_rate=0, random_state=0):
    if noise_type == 'pairflip':
        train_noisy_labels, actual_noise_rate, P = noisify_pairflip(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    if noise_type == 'symmetric':
        train_noisy_labels, actual_noise_rate, P = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    return train_noisy_labels, actual_noise_rate, P



def noisify_instance(train_data,train_labels,noise_rate,random_state=0):
    np.random.seed(random_state)
    if max(train_labels)>10:
        num_class = 100
    else:
        num_class = 10
    np.random.seed(0)

    q_ = np.random.normal(loc=noise_rate,scale=0.1,size=1000000)
    q = []
    for pro in q_:
        if 0 < pro < 1:
            q.append(pro)
        if len(q)==50000:
            break

    w = torch.tensor(np.random.normal(loc=0,scale=1,size=(32*32*3,num_class))).float().cuda()

    noisy_labels = []
    for i, sample in enumerate(train_data):
        sample = Image.fromarray(sample)
        sample = transforms.ToTensor()(sample).cuda()
        p_all = sample.reshape(1,-1).mm(w).squeeze(0)
        p_all[train_labels[i]] = -1000000
        p_all = q[i]*F.softmax(p_all,dim=0).cpu().numpy()
        p_all[train_labels[i]] = 1 - q[i]
        noisy_labels.append(np.random.choice(np.arange(num_class),p=p_all/sum(p_all)))
    over_all_noise_rate = 1 - float(torch.tensor(train_labels).eq(torch.tensor(noisy_labels)).sum())/50000
    return noisy_labels, over_all_noise_rate


def get_img_num_per_cls(data, cls_num, imb_type, imb_factor):
    img_max = len(data) / cls_num
    img_num_per_cls = []
    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
    elif imb_type == 'step':
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max * imb_factor))
    else:
        img_num_per_cls.extend([int(img_max)] * cls_num)
    return img_num_per_cls


def gen_imbalanced_data(data,targets, img_num_per_cls,random_state):
    np.random.seed(random_state)
    new_data = []
    new_targets = []
    targets_np = np.array(targets, dtype=np.int64)
    classes = np.unique(targets_np)
    # np.random.shuffle(classes)
    num_per_cls_dict = dict()
    indexes = np.array([],dtype="int64")
    for the_class, the_img_num in zip(classes, img_num_per_cls):
        num_per_cls_dict[the_class] = the_img_num
        idx = np.where(targets_np == the_class)[0]
        np.random.shuffle(idx)
        selec_idx = idx[:the_img_num]
        indexes = np.concatenate((indexes,selec_idx))
        new_data.append(data[selec_idx, ...])
        new_targets.extend([the_class, ] * the_img_num)
    new_data = np.vstack(new_data)
    return new_data, new_targets,indexes


def adjust_learning_rate(optimizer, epoch, base_lr, ajust_period=70):
    """Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
    lr = base_lr * (0.1 ** (epoch // ajust_period))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(path, state, is_best):
    
    filename = path
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth', 'best.pth'))


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
    
    
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.01,
        warmup_factor=1.0 / 3,
        warmup_epochs=5,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                #alpha = float(self.last_epoch) / self.warmup_epochs
                #warmup_factor = self.warmup_factor * (1 - alpha) + alpha
                warmup_factor = float(self.last_epoch + 1) / self.warmup_epochs
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
    
def get_knncentroids(feats=None, labels=None, mask=None):
    
    if feats is not None and labels is not None:
        feats = feats.cpu().numpy()
        labels = labels.cpu().numpy()
    else:
        warnings.warn('features and labels are not listed')

    featmean = feats.mean(axis=0)

    def get_centroids(feats_, labels_, mask_=None):
        if mask_ is None:
            mask_ = np.ones_like(labels_).astype('bool')
        elif isinstance(mask_, torch.Tensor):
            mask_ = mask_.cpu().numpy()
        
        centroids = []        
        for i in np.unique(labels_):
            centroids.append(np.mean(feats_[(labels_==i) & mask_], axis=0))
        return np.stack(centroids)

    # Get unnormalized centorids
    un_centers = get_centroids(feats, labels, mask)

    # Get l2n centorids
    l2n_feats = torch.Tensor(feats.copy())
    norm_l2n = torch.norm(l2n_feats, 2, 1, keepdim=True)
    l2n_feats = l2n_feats / norm_l2n
    l2n_centers = get_centroids(l2n_feats.numpy(), labels, mask)

    # Get cl2n centorids
    cl2n_feats = torch.Tensor(feats.copy())
    cl2n_feats = cl2n_feats - torch.Tensor(featmean)
    norm_cl2n = torch.norm(cl2n_feats, 2, 1, keepdim=True)
    cl2n_feats = cl2n_feats / norm_cl2n
    cl2n_centers = get_centroids(cl2n_feats.numpy(), labels, mask)

    return {'mean': featmean,
            'uncs': un_centers,
            'l2ncs': l2n_centers,   
            'cl2ncs': cl2n_centers}
    


def label_cleaning_RoLT(est_loader, model, epoch, ncm_classifier, args):
    def get_gmm_mask(ncm_logits):
        mask = torch.zeros_like(args.noisy_labels).bool()

        for i in range(args.num_classes):
            this_cls_idxs = (args.noisy_labels == i)
            this_cls_logits = ncm_logits[this_cls_idxs, i].view(-1, 1).cpu().numpy()
            
            # normalization, note that the logits are all negative
            this_cls_logits -= np.min(this_cls_logits)
            if np.max(this_cls_logits) != 0:
                this_cls_logits /= np.max(this_cls_logits)

            gmm = GaussianMixture(n_components=2, random_state=0).fit(this_cls_logits)
            gmm_preds = gmm.predict(this_cls_logits)
            inner_cluster = gmm.means_.argmax()

            this_cls_mask = mask[this_cls_idxs]
            this_cls_mask[gmm_preds == inner_cluster] = True

            if (gmm_preds != inner_cluster).all():
                this_cls_mask |= True  # not to exclude any instance

            mask[this_cls_idxs] = this_cls_mask
        return mask
    
    model.eval()

    total_features = torch.empty((0, 512)).cuda()
    total_logits = torch.empty((0, args.num_classes)).cuda()

    with torch.no_grad():
        for i, (inputs, labels, true_labels, indexes) in enumerate(est_loader):
            inputs = Variable(inputs).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)

            outputs, feature = model(inputs)
            total_features = torch.cat((total_features, feature))
            total_logits = torch.cat((total_logits, outputs))

    cfeats = get_knncentroids(feats=total_features, labels=args.noisy_labels)
    ncm_classifier.update(cfeats)
    ncm_logits = ncm_classifier(total_features, None)[0]

    refined_ncm_logits = ncm_logits
    mask = get_gmm_mask(refined_ncm_logits)
    refined_cfeats = get_knncentroids(feats=total_features, labels=args.noisy_labels, mask=mask)
    ncm_classifier.update(refined_cfeats)
    refined_ncm_logits = ncm_classifier(total_features, None)[0]

    alpha = 0.6
    epoch += 1
    if epoch == 1:
        args.ensemble_erm_logits_cur = total_logits
        args.ensemble_ncm_logits_cur = refined_ncm_logits
    else:
        args.ensemble_erm_logits_cur = alpha * args.ensemble_erm_logits_cur + (1. - alpha) * total_logits
        args.ensemble_ncm_logits_cur = alpha * args.ensemble_ncm_logits_cur + (1. - alpha) * refined_ncm_logits
    args.ensemble_erm_logits_cur = args.ensemble_erm_logits_cur * (1. / (1. - alpha ** epoch))
    args.ensemble_ncm_logits_cur = args.ensemble_ncm_logits_cur * (1. / (1. - alpha ** epoch))

    erm_outputs = args.ensemble_erm_logits_cur.softmax(dim=1)
    ncm_outputs = args.ensemble_ncm_logits_cur.softmax(dim=1)

    _, erm_preds = erm_outputs.max(dim=1)
    _, ncm_preds = ncm_outputs.max(dim=1)

    mask = get_gmm_mask(refined_ncm_logits)

    surrogate_labels = torch.where(mask, args.noisy_labels, erm_preds)

    if args.train_rule == 'DRW' and epoch > 160:
        if epoch == 161:
            args.current_labels = surrogate_labels

        args.soft_targets = [args.current_labels]
        args.soft_targets.extend( [torch.where(mask, args.noisy_labels, torch.ones_like(args.noisy_labels).long() * i)
                                    for i in range(args.num_classes) ])
        
        args.soft_weights = [0.5]
        args.soft_weights.extend([ 0.5 / args.num_classes
                                    for i in range(args.num_classes) ])
    elif epoch > 80:
        args.soft_targets = [torch.where(mask, args.noisy_labels, erm_preds),
                                torch.where(mask, args.noisy_labels, ncm_preds),
                                torch.where(mask, args.noisy_labels, args.noisy_labels)]
        args.soft_targets.extend( [torch.where(mask, args.noisy_labels, torch.ones_like(args.noisy_labels).long() * i)
                                    for i in range(args.num_classes) ])
        
        args.soft_weights = [0.4, 0.2, 0.2]
        args.soft_weights.extend([ 0.2 / args.num_classes
                                    for i in range(args.num_classes) ])

    return


def label_clean_PCL(args,epoch,features,labels,probs,prototypes,clean_label=None):
    gamma = 1.005
    temperature = 0.3
    logits_proto = torch.mm(features.cuda(args.gpu),prototypes.t()) / temperature
    logits_proto = F.softmax(logits_proto, dim=1)

    soft_labels = logits_proto.cpu()

    curr_thre = args.low_th * (gamma ** epoch)
    gt_score = soft_labels[labels>=0,labels]
    gt_score_erm = probs[labels>=0,labels]
    gt_clean = gt_score>curr_thre
    soft_labels[gt_clean] = torch.zeros(gt_clean.sum(), args.num_classes).scatter_(1, labels[gt_clean].view(-1,1), 1) 

    #get the hard pseudo label and the clean subset used to calculate supervised loss
    max_score, hard_labels = torch.max(soft_labels, 1)

    clean_idx = max_score>args.high_th
    weights = gt_score
    weights[~gt_clean] = 0.5 * soft_labels[~gt_clean, labels[~gt_clean]]

    N = features.shape[0]      
    k = args.n_neighbors
    condis = torch.mm(features.cuda(args.gpu), features.cuda(args.gpu).t()).cpu()
    _, sortDistance = condis.sort(1, descending=True)
    I = sortDistance[:, 1]
    for n in range(N):
        neighbor_labels = hard_labels[I[n]]
        if neighbor_labels != hard_labels[n]:
            weights[n] *= 0.5

    return hard_labels, clean_idx, weights
