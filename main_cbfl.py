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
from utils.utils import *
from utils.majority_voting import *
from utils.confidence_knn import *
from utils.lr_schedule import inv_lr_scheduler, get_cosine_schedule_with_warmup
from utils.return_dataset import return_dataset, return_dataset_randaugment
from utils.loss import *
from augmentations.augmentation_ours import *
import pickle
from easydict import EasyDict as edict

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
parser.add_argument('--LR_scheduler', type=str, default='standard', choices=['standard', 'cosine'], help='Learning Rate scheduling policy')
parser.add_argument('--adentropy', action='store_true', default=True,
                    help='Use entropy maximization or not')
parser.add_argument('--uda', type=int, default=0,
                    help='use uda training for model training')
parser.add_argument('--use_bank', type=int, default=1,
                    help='use feature bank method for experiments')


torch.autograd.set_detect_anomaly(True) # Gradient anomaly detection is set true for debugging purposes
args = parser.parse_args()
print('Dataset %s Source %s Target %s Labeled num perclass %s Network %s' %
      (args.dataset, args.source, args.target, args.num, args.net))

source_loader, target_loader, target_loader_unl, target_loader_val, target_loader_test, class_list = return_dataset_randaugment(args)    

use_gpu = torch.cuda.is_available()
# Seeding everything for removing non-deterministic components
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

params = []
for key, value in dict(G.named_parameters()).items():
    if value.requires_grad:
        if 'classifier' not in key:
            params += [{'params': [value], 'lr': args.multi,
                        'weight_decay': 0.0005}]
        else: 
            params += [{'params': [value], 'lr': args.multi * 10,
                        'weight_decay': 0.0005}]

if "resnet" in args.net:
    F1 = Predictor_deep(num_class=len(class_list), inc=inc)
else:
    F1 = Predictor(num_class=len(class_list), inc=inc, temp=args.T)
weights_init(F1)


if args.pretrained_ckpt is not None:
    ckpt = torch.load(args.pretrained_ckpt)
    G.load_state_dict(ckpt["G"])
    F1.load_state_dict(ckpt["F1"])

lr = args.lr
G.cuda()
F1.cuda()

G = nn.DataParallel(G, device_ids=[0, 1])
F1 = nn.DataParallel(F1, device_ids=[0, 1])

if os.path.exists(args.checkpath) == False:
    os.mkdir(args.checkpath)

def train():
    G.train()
    F1.train()
    optimizer_g = optim.SGD(params, momentum=0.9, weight_decay=0.0005, nesterov=True)
    optimizer_f = optim.SGD(list(F1.parameters()), lr=1.0, momentum=0.9, weight_decay=0.0005, nesterov=True)
    print("Unlabelled Target Dataset Size: ",len(target_loader_unl.dataset))
    print("Labelled Target Dataset Size: ",len(target_loader.dataset))
    # Loading the dictionary having the feature bank and corresponding metadata
    
    def zero_grad_all():
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
    param_lr_g = []
    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_f = []
    for param_group in optimizer_f.param_groups:
        param_lr_f.append(param_group["lr"])

    thresh = 0.9   # threshold for confident prediction to generate pseudo-labels
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    criterion_pseudo = nn.CrossEntropyLoss(reduction='none').cuda()
    criterion_lab_target = nn.CrossEntropyLoss(reduction='none').cuda()
    feat_dict_source, feat_dict_target, _ = load_bank(args)
    
    num_target = len(feat_dict_target.names)
    num_source = len(feat_dict_source.names)

    feat_dict_source.sample_weights = torch.tensor(np.ones(num_source)).cuda()
    

    label_bank = edict({"names": feat_dict_target.names, "labels": np.zeros(num_target,dtype=int)-1})

    all_step = args.steps
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    data_iter_t_unl = iter(target_loader_unl)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    len_train_target_semi = len(target_loader_unl)
    best_acc_val = 0
    counter = 0
    K = 3
    K_farthest_source = 5
    beta = 0.99
    for step in range(all_step):
        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step, init_lr=args.lr)
        optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step, init_lr=args.lr)

        lr = optimizer_f.param_groups[0]['lr']
        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)
        if step % len_train_target_semi == 0:
            data_iter_t_unl = iter(target_loader_unl)
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)
        
        # Extracting the batches from the iteratable dataloader
        data_t = next(data_iter_t)
        data_t_unl = next(data_iter_t_unl)
        data_s = next(data_iter_s)
        im_data_s = data_s[0].cuda()
        gt_labels_s = data_s[1].cuda()

        im_data_t = data_t[0][0].cuda()
        gt_labels_t = data_t[1].cuda()
        im_data_tu = data_t_unl[0][2].cuda()

        zero_grad_all()
        data = im_data_s
        target = gt_labels_s    
        #data = torch.cat((im_data_s, im_data_t), 0) #concatenating the labelled images
        #target = torch.cat((gt_labels_s, gt_labels_t), 0)
        im_data_tu_weak_aug, im_data_tu_strong_aug = data_t_unl[0][0].cuda(), data_t_unl[0][1].cuda()
        # Getting predictions of weak and strong augmented unlabled examples
        pred_strong_aug = F1(G(im_data_tu_strong_aug))
        with torch.no_grad():
            pred_weak_aug = F1(G(im_data_tu_weak_aug))
        prob_weak_aug = F.softmax(pred_weak_aug,dim=1)
        mask_loss = prob_weak_aug.max(1)[0]>thresh
        pseudo_labels = pred_weak_aug.max(axis=1)[1]
        loss_pseudo_unl = torch.mean(mask_loss.int() * criterion_pseudo(pred_strong_aug,pseudo_labels))
        loss_pseudo_unl.backward(retain_graph=True)
        # Updating the features in the bank for both source and target
        if args.use_bank == 1:
            f_batch_source, feat_dict_source = update_features(feat_dict_source, data_s, G, 0, source = True)
            if step >=3500 and step % 1500:
                feat_dict_source.sample_weights = torch.tensor(np.ones(num_source)).cuda()
                f_batch_source, sim_distribution = get_similarity_distribution(feat_dict_source,data_t,G)
                sim_distribution.cosines = 1 - sim_distribution.cosines
                k_neighbors, labels_k_neighbors = get_kNN(sim_distribution, feat_dict_source, K)    
                #weights_source  = source_weights_batch(feat_dict_source, data_s, K_farthest_source, k_neighbors)
                set_source_weights_all(target_loader,feat_dict_source,K_farthest_source,k_neighbors,G)

        update_label_bank(label_bank, data_t_unl, pseudo_labels, mask_loss)

        if step >= 3500:
            class_num_list = get_per_class_examples(label_bank, class_list)
            effective_num = 1.0 - np.power(beta, class_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(class_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
            criterion_lab_target = CBFocalLoss(weight=per_cls_weights, gamma=0.5).cuda()
            out_lab_target = F1(G(im_data_t))
            loss_lab_target = criterion_lab_target(out_lab_target,gt_labels_t)
            loss_lab_target.backward()

        #output = G(data)
        output = f_batch_source
        out1 = F1(output)

        if step >= 3500:
            names_batch = list(data_s[2])
            idx = [feat_dict_source.names.index(name) for name in names_batch] 
            weights_source = feat_dict_source.sample_weights[idx]
            loss = torch.mean(weights_source * criterion(out1, target))
            print(loss)
        else:
            loss = torch.mean(criterion(out1, target))
        
        loss.backward(retain_graph=True)
        optimizer_g.step()
        optimizer_f.step()

        zero_grad_all()
        if not args.method == 'S+T':
            output = G(im_data_tu)
            if args.method == 'ENT':
                loss_t = entropy(F1, output, args.lamda)
                loss_t.backward()
                optimizer_f.step()
                optimizer_g.step()
            elif args.method == 'MME':
                loss_t = adentropy(F1, output, args.lamda)
                loss_t.backward()
                optimizer_f.step()
                optimizer_g.step()
            else:
                raise ValueError('Method cannot be recognized.')
            
            log_train = 'S {} T {} Train Ep: {} lr{} \t Loss Classification: {:.6f} Loss T {:.6f} Method {}\n'.format(args.source, args.target, step, lr, loss.data, -loss_t.data, args.method)
        else:
            log_train = 'S {} T {} Train Ep: {} lr{} \t Loss Classification: {:.6f} Method {}\n'.format(args.source, args.target, step, lr, loss.data, args.method)

        G.zero_grad()
        F1.zero_grad()
        zero_grad_all()
        if step % args.log_interval == 0:
            print(log_train)
        if step % args.save_interval == 0:# and step > 0:
            if step % 2000 == 0:
                #save_stats(F1, G, target_loader_unl, step, feat_dict_combined, data_t_unl, K, mask_loss_uncertain)
                pass
            _, acc_labeled_target, _ = test(target_loader, mode = 'Labeled Target')
            _, acc_test,_ = test(target_loader_test, mode = 'Test')
            _, acc_val, _ = test(target_loader_val, mode = 'Val')

            G.train()
            F1.train()
            if acc_val >= best_acc_val:
                best_acc_val = acc_val
                best_acc_test = acc_test
                counter = 0
            else:
                counter += 1
               
            if args.early:
                if counter > args.patience:
                    break
            
            print('best acc test %f  acc val %f acc labeled target %f' % (best_acc_test, acc_val, acc_labeled_target))
            G.train()
            F1.train()
            if args.save_check:
                print('saving model...')
                torch.save({
                    'step': step,
                    'arch': args.net,
                    'G_state_dict': G.state_dict(),
                    'F1_state_dict': F1.state_dict(),
                    'best_acc_test': best_acc_test,
                    'optimizer_g' : optimizer_g.state_dict(),
                    'optimizer_f' : optimizer_f.state_dict(),
                    'feat_dict_target': feat_dict_target
                    },os.path.join(args.checkpath,"%s_%s_%s_%d.ckpt.pth.tar"%(args.net,args.source,args.target,step)))


def test(loader, mode='Test'):
    G.eval()
    F1.eval()
    test_loss = 0
    correct = 0
    size = 0
    num_class = len(class_list)
    criterion = nn.CrossEntropyLoss().cuda()
    confusion_matrix = torch.zeros(num_class, num_class)
    with torch.no_grad():
        for _ , data_t in enumerate(loader):
            if not mode == 'Labeled Target':
                im_data_t = data_t[0].cuda()
                gt_labels_t = data_t[1].cuda()
                feat = G(im_data_t)
                output1 = F1(feat)
                size += im_data_t.size(0)
                pred1 = output1.data.max(1)[1]
                for t, p in zip(gt_labels_t.view(-1), pred1.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                correct += pred1.eq(gt_labels_t.data).cpu().sum()
                test_loss += criterion(output1, gt_labels_t) / len(loader)

            elif mode == "Labeled Target":
                im_data_t_weak, im_data_t_strong, im_data_t_standard = data_t[0][0].cuda(), data_t[0][1].cuda(), data_t[0][2].cuda()
                gt_labels_t = data_t[1].cuda()
                feat_weak, feat_strong, feat_standard = G(im_data_t_weak), G(im_data_t_strong), G(im_data_t_standard)
                output1_weak, output1_strong, output1_standard = F1(feat_weak), F1(feat_strong), F1(feat_standard)

                size += im_data_t_weak.size(0) + im_data_t_strong.size(0) + im_data_t_standard.size(0)
                pred1_weak, pred1_strong, pred1_standard = output1_weak.data.max(1)[1], output1_strong.data.max(1)[1], output1_standard.data.max(1)[1]

                for t, p_weak, p_strong, p_standard in zip(gt_labels_t.view(-1), pred1_weak.view(-1), pred1_strong.view(-1), pred1_standard.view(-1)):
                    confusion_matrix[t.long(), p_weak.long()] += 1
                    confusion_matrix[t.long(), p_strong.long()] += 1
                    confusion_matrix[t.long(), p_standard.long()] += 1

                correct += pred1_weak.eq(gt_labels_t.data).cpu().sum() + pred1_strong.eq(gt_labels_t.data).cpu().sum() + pred1_standard.eq(gt_labels_t.data).cpu().sum()

                test_loss += criterion(output1_weak, gt_labels_t)/(3*len(loader))  + criterion(output1_strong, gt_labels_t)/(3*len(loader)) + criterion(output1_standard, gt_labels_t)/(3*len(loader)) 
                per_cls_acc = per_class_accuracy(confusion_matrix)


    if not mode == 'Labeled Target':
        np.save("cf_unlabeled_target.npy",confusion_matrix)
        weight = torch.ones([num_class,1]).cuda()
    elif mode =='Labeled Target':
        np.save("cf_labeled_target.npy",confusion_matrix)
        per_cls_acc = per_class_accuracy(confusion_matrix)
        weight = per_cls_acc 
        weight = (weight>0.5).int()*1.5 + (weight<0.5).int()*0.5
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} F1 ({:.4f}%)\n'.format(mode, test_loss,correct,size,100.*correct/size))
    return test_loss.data,100.*float(correct)/size, weight

def per_class_accuracy(confusion_matrix):
    num_class, _ = confusion_matrix.shape
    per_cls_acc = []
    for i in range(num_class):
        per_cls_acc.append(confusion_matrix[i,i]/sum(confusion_matrix[i,:]))
    per_cls_acc = torch.tensor(per_cls_acc).cuda()
    return per_cls_acc

train()