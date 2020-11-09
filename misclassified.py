from __future__ import print_function
import argparse
import os
import sys
sys.path.append("/home/megh/projects/domain-adaptation/SSAL/")
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
from utils.return_dataset import return_dataset, return_dataset_test
from utils.loss import entropy, adentropy
from augmentations.augmentation_ours import *

# Training settings
parser = argparse.ArgumentParser(description='misclassified examples')
parser.add_argument('--method', type=str, default='MME',
                    choices=['S+T', 'ENT', 'MME'],
                    help='MME is proposed method, ENT is entropy minimization,'
                         ' S+T is training only on labeled examples')
parser.add_argument('--T', type=float, default=0.05, metavar='T',
                    help='temperature (default: 0.05)')
parser.add_argument('--lamda', type=float, default=0.1, metavar='LAM',
                    help='value of lamda')
parser.add_argument('--save_check', action='store_true', default=False,
                    help='save checkpoint or not')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--source', type=str, default='real',
                    help='source domain')
parser.add_argument('--target', type=str, default='sketch',
                    help='target domain')
parser.add_argument('--dataset', type=str, default='multi',
                    choices=['multi', 'office', 'office_home'],
                    help='the name of dataset')
parser.add_argument('--num', type=int, default=3,
                    help='number of labeled examples in the target')
parser.add_argument('--net', type=str, default='resnet34',
                    help='which network to use')

args = parser.parse_args()
print('Dataset %s Source %s Target %s Labeled num perclass %s Network %s' %(args.dataset, args.source, args.target, args.num, args.net))
target_loader_unl,class_list = return_dataset_test(args)


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

ckpt = torch.load("./save_model_ssda/resnet34_real_sketch_6000")
G.load_state_dict(ckpt["G_state_dict"])
F1.load_state_dict(ckpt["F1_state_dict"])
G.cuda()
F1.cuda()
G.eval()
F1.eval()

count_mis = 0
count_thresh = 0
print(len(target_loader_unl.dataset))
for idx, batch in enumerate(target_loader_unl):
    output = F.softmax(F1(G(batch[0].cuda())),dim=1)
    max_prob = output.max().cpu().data.item()
    pred_label = output.argmax().cpu().data.item()
    gt_label = batch[1].cpu().data.item()
    if max_prob>0.9:
        count_thresh = count_thresh + 1
        print("count_thresh: ", count_thresh)
        if pred_label!=gt_label:
            print(pred_label,gt_label)
            count_mis = count_mis+1
            print("Count Mis: ", count_mis)



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
