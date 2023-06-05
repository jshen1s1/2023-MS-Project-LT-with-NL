import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def l1_error_calculator(target, target_hat, is_percentage=True):
    r"""
    input:
        target: e.g. true transition matrix
        target_hat: e.g. estimated transition matrix
    """
    target = np.array(target).flatten()
    target_hat = np.array(target_hat).flatten() 
    if is_percentage:
        return np.linalg.norm((target_hat-target), ord=1)/np.linalg.norm(target, ord=1)
    else:
        return np.linalg.norm((target_hat-target), ord=1)
    
def est_t_matrix(eta_corr, filter_outlier=False):
    # number of classes
    num_classes = eta_corr.shape[1]
    T = np.empty((num_classes, num_classes))

    # find a 'perfect example' for each class
    for i in np.arange(num_classes):
        if not filter_outlier:
            idx_best = np.argmax(eta_corr[:, i])
        else:
            eta_thresh = np.percentile(eta_corr[:, i], 97, interpolation='higher')
            robust_eta = eta_corr[:, i]
            robust_eta[robust_eta >= eta_thresh] = 0.0
            idx_best = np.argmax(robust_eta)

        for j in np.arange(num_classes):
            T[i, j] = eta_corr[idx_best, j]

    return T

def get_noise_rate(t):
    return 1-np.average(t.diagonal())

def get_transition_matrices(est_loader, model, args):
    model.eval()
    est_loader.dataset.eval()
    num_classes = args.num_classes
    p = []
    T_spadesuit = np.zeros((num_classes,num_classes))
    with torch.no_grad():
        for i, (images, labels, _, _) in enumerate(est_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            pred, _ = model(images)
            probs = F.softmax(pred, dim=1).cpu().data.numpy()
            _, pred = pred.topk(1, 1, True, True)           
            pred = pred.view(-1).cpu().data
            labels = labels.view(-1).cpu().data
            for i in range(len(labels)): 
                T_spadesuit[int(pred[i])][int(labels[i])]+=1
            p += probs[:].tolist()  
    T_spadesuit = np.array(T_spadesuit)
    sum_matrix = np.tile(T_spadesuit.sum(axis = 1),(num_classes,1)).transpose()
    T_spadesuit = T_spadesuit/sum_matrix
    p = np.array(p)
    T_clubsuit = est_t_matrix(p,filter_outlier=True)
    T_spadesuit = np.nan_to_num(T_spadesuit)
    return T_spadesuit, T_clubsuit

def run_est_T_matrices(est_loader, model, args):
    T_spadesuit, T_clubsuit = get_transition_matrices(est_loader, model, args)
    return T_spadesuit, T_clubsuit


def compose_T_matrices(T_spadesuit, T_clubsuit):
    dual_t_matrix = np.matmul(T_spadesuit, T_clubsuit)
    return dual_t_matrix


