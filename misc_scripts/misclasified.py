from __future__ import print_function
import argparse
import os
import sys
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from model.resnet import resnet34
from model.basenet import AlexNetBase, VGGBase, Predictor, Predictor_deep
from utils.utils import weights_init
from utils.lr_schedule import inv_lr_scheduler, get_cosine_schedule_with_warmup
from utils.return_dataset import return_dataset, return_dataset_randaugment, return_dataset_ctaugment
from utils.loss import entropy, adentropy
from augmentations.augmentation_ours import *

# Training settings
parser = argparse.ArgumentParser(description='SSDA Classification')
parser.add_argument('--steps', type=int, default=50000, metavar='N',
                    help='maximum number of iterations '
                         'to train (default: 50000)')
parser.add_argument('--method', type=str, default='MME',
                    choices=['S+T', 'ENT', 'MME'],
                    help='MME is proposed method, ENT is entropy minimization,'
                         ' S+T is training only on labeled examples')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--multi', type=float, default=0.1, metavar='MLT',
                    help='learning rate multiplication')
parser.add_argument('--T', type=float, default=0.05, metavar='T',
                    help='temperature (default: 0.05)')
parser.add_argument('--lamda', type=float, default=0.1, metavar='LAM',
                    help='value of lamda')
parser.add_argument('--save_check', action='store_true', default=False,
                    help='save checkpoint or not')
parser.add_argument('--checkpath', type=str, default='./save_model_ssda',
                    help='dir to save checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging '
                         'training status')
parser.add_argument('--save_interval', type=int, default=500, metavar='N',
                    help='how many batches to wait before saving a model')
parser.add_argument('--net', type=str, default='alexnet',
                    help='which network to use')
parser.add_argument('--source', type=str, default='real',
                    help='source domain')
parser.add_argument('--target', type=str, default='sketch',
                    help='target domain')
parser.add_argument('--dataset', type=str, default='multi',
                    choices=['multi', 'office', 'office_home'],
                    help='the name of dataset')
parser.add_argument('--num', type=int, default=3,
                    help='number of labeled examples in the target')
parser.add_argument('--patience', type=int, default=5, metavar='S',
                    help='early stopping to wait for improvment '
                         'before terminating. (default: 5 (5000 iterations))')
parser.add_argument('--early', action='store_false', default=True,
                    help='early stopping on validation or not')
parser.add_argument('--pretrained_ckpt', type=str, default=None,
                    help='path to pretrained weights')
parser.add_argument('--augmentation_policy', type=str, default=None, choices=['ours', 'rand_augment','ct_augment'],
                    help='which augmentation starategy to use - essentially, which method to follow')
parser.add_argument('--LR_scheduler', type=str, default='standard', choices=['standard', 'cosine'],
                    help='Learning Rate scheduling policy')
parser.add_argument('--adentropy', action='store_true', default=True,
                    help='Use entropy maximization or not')
parser.add_argument('--use_ema', type=bool, default=False,
                    help='use ema for model eval or not')


args = parser.parse_args()
print('Dataset %s Source %s Target %s Labeled num perclass %s Network %s' %(args.dataset, args.source, args.target, args.num, args.net))
source_loader, target_loader, target_loader_unl, target_loader_val, target_loader_test, class_list = return_dataset(args)


use_gpu = torch.cuda.is_available()


torch.cuda.manual_seed(args.seed)
if args.net == 'resnet34':
    G = resnet34()
    inc = 512
elif args.net == "alexnet":
    G = AlexNetBase()
    inc = 4096
elif args.net == "vgg":
    G = VGGBase()
    inc = 4096
else:
    raise ValueError('Model cannot be recognized.')

if "resnet" in args.net:
    F1 = Predictor_deep(num_class=len(class_list),
                        inc=inc)
else:
    F1 = Predictor(num_class=len(class_list), inc=inc,
                   temp=args.T)
weights_init(F1)

ckpt = torch.load(args.pretrained_ckpt)
G.load_state_dict(ckpt["G"])
F1.load_state_dict(ckpt["F1"])
G.cuda()
F1.cuda()

G.eval()
F1.eval()

def test(loader):
    G.eval()
    F1.eval()
    test_loss = 0
    correct = 0
    size = 0
    num_class = len(class_list)
    output_all = np.zeros((0, num_class))
    criterion = nn.CrossEntropyLoss().cuda()
    confusion_matrix = torch.zeros(num_class, num_class)
    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            im_data_t = data_t[0].cuda()
            gt_labels_t = data_t[1].cuda()
            feat = G(im_data_t)
            output1 = F1(feat)
            output_all = np.r_[output_all, output1.data.cpu().numpy()]
            size += im_data_t.size(0)
            pred1 = output1.data.max(1)[1]
            for t, p in zip(gt_labels_t.view(-1), pred1.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += pred1.eq(gt_labels_t.data).cpu().sum()
            test_loss += criterion(output1, gt_labels_t) / len(loader)
    print('\nTest set: Average loss: {:.4f}, '
          'Accuracy: {}/{} F1 ({:.0f}%)\n'.
          format(test_loss, correct, size,
                 100. * correct / size))
    return test_loss.data, 100. * float(correct) / size



train()
# Inference/Model performance estimation by moving average of model parameters
class ModelEMA(object):
    def __init__(self, args, model, decay):
        self.ema = deepcopy(model)
        self.ema.to(args.device)
        self.ema.eval()
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, 'module')
        self.wd = args.wdecay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model, lr):
        wd = lr * self.wd
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                if needs_module:
                    k = 'module.' + k
                model_v = msd[k].detach()
                ema_v.copy_(ema_v * self.decay + (1. - self.decay) * model_v)
                # weight decay
                if 'bn' not in k and 'bias' not in k:
                    msd[k] = msd[k] * (1. - wd)

"""
for idx, weight in enumerate(mask_loss):
    weight = weight.cpu().detach().item()
    prob_strong_aug[idx] = (prob_strong_aug[idx] * torch.tensor(int(weight)))
    prob_weak_aug[idx] = (prob_weak_aug[idx].clone() * torch.tensor(int(weight)))
"""
