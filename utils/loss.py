import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Function
import torch.nn as nn
from utils.utils import * 
from utils.return_dataset import * 
import sys

class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd
    @staticmethod
    def forward(self, x):
        return x.view_as(x)
    @staticmethod
    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) /
                    (1.0 + np.exp(-alpha * iter_num / max_iter)) -
                    (high - low) + low)


def entropy(F1, feat, lamda, eta=1.0):
    out_t1 = F1(feat, reverse=True, eta=-eta)
    out_t1 = F.softmax(out_t1)
    loss_ent = -lamda * torch.mean(torch.sum(out_t1 *
                                             (torch.log(out_t1 + 1e-5)), 1))
    return loss_ent


def adentropy(F1, feat, lamda, eta=1.0):
    out_t1 = F1(feat, reverse=True, eta=eta)
    out_t1 = F.softmax(out_t1,dim=1)
    loss_adent = lamda * torch.mean(torch.sum(out_t1 *
                                              (torch.log(out_t1 + 1e-5)), 1))
    return loss_adent


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction = 'none'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction=self.reduction) # important to add reduction='none' to keep per-batch-item loss
        pt = torch.exp(-ce_loss)
        if self.reduction == 'mean':
            focal_loss = (self.alpha * (1-pt)**self.gamma * ce_loss).mean() # mean over the batch
        elif self.reduction == 'none':
            focal_loss = (self.alpha * (1-pt)**self.gamma * ce_loss)
        return focal_loss

def focal_loss(input_values, gamma):
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class CBFocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0., reduction = 'none'):
        super(CBFocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction=self.reduction, weight=self.weight), self.gamma)


class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)


def update_loss_functions(args,label_bank, class_list, class_num_list_pseudo=None, class_num_list_source = None, beta=0.99,gamma=0):
    if class_num_list_pseudo is None:
        class_num_list_pseudo = get_per_class_examples(label_bank, class_list) + args.num
        print("Pred num ex per class (pseudo labels + labelled target examples): ", class_num_list_pseudo)
        
    if class_num_list_source is not None:
        class_num_list =  class_num_list_pseudo + np.array(class_num_list_source)
    else:
        class_num_list = class_num_list_pseudo

    effective_num = 1.0 - np.power(beta, class_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(class_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
    
    criterion = CBFocalLoss(weight=per_cls_weights, gamma=gamma, reduction='none').cuda()
    criterion_pseudo = CBFocalLoss(weight=per_cls_weights, gamma=gamma, reduction='none').cuda()
    criterion_lab_target = CBFocalLoss(weight=per_cls_weights, gamma=gamma,reduction='mean').cuda()
    criterion_strong_source = CBFocalLoss(weight=per_cls_weights, gamma=gamma,reduction='mean').cuda()
    print("CBFL per class weights:", per_cls_weights)
    return criterion, criterion_pseudo, criterion_lab_target, criterion_strong_source


class LDAMLoss_misclassification(nn.Module):

    def __init__(self, m_list, max_m=0.5, weight=None, s=1):
        super(LDAMLoss_misclassification, self).__init__()
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1) # makes a one hot vector from the target vector

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
        output = torch.where(index, x_m, x)
        loss_ldam = F.cross_entropy(self.s*output, target, weight=self.weight)
        loss_ce = F.cross_entropy(x, target, weight=self.weight)
        print("LDAM Loss is: ",loss_ldam.item())
        print("CE Loss is: ", loss_ce.item())
        return loss_ldam


def update_labeled_loss(criterion_labeled_target,confusion_matrix):
    n_class, _ = confusion_matrix.shape
    m_list = []
    for i in range(n_class): 
        m_list.append(sum(confusion_matrix[i,:]) - confusion_matrix[i,i])
    m_list = np.array(m_list)
    criterion_labeled_target = LDAMLoss_misclassification(m_list)
    print("m_list is: \n",m_list)
    return criterion_labeled_target