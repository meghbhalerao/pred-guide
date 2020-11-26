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
from utils.utils import get_confident, weights_init, update_features, get_similarity_distribution, get_kNN, k_means
from utils.lr_schedule import inv_lr_scheduler, get_cosine_schedule_with_warmup
from utils.return_dataset import return_dataset, return_dataset_randaugment
from utils.loss import entropy, adentropy
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
parser.add_argument('--LR_scheduler', type=str, default='standard', choices=['standard', 'cosine'],
                    help='Learning Rate scheduling policy')
parser.add_argument('--adentropy', action='store_true', default=True,
                    help='Use entropy maximization or not')
parser.add_argument('--use_ema', type=bool, default=False,
                    help='use ema for model eval or not')



torch.autograd.set_detect_anomaly(True) # Gradient anomaly detection is set true for debugging purposes
args = parser.parse_args()
print('Dataset %s Source %s Target %s Labeled num perclass %s Network %s' %
      (args.dataset, args.source, args.target, args.num, args.net))
if args.augmentation_policy == "ours" or args.augmentation_policy == None or args.augmentation_policy == "ct_augment":
    source_loader, target_loader, target_loader_unl, target_loader_val, target_loader_test, class_list = return_dataset(args)
elif args.augmentation_policy == "rand_augment":
    source_loader, target_loader, target_loader_unl, target_loader_val, target_loader_test, class_list = return_dataset_randaugment(args)    

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
    F1 = Predictor_deep(num_class=len(class_list),
                        inc=inc)
else:
    F1 = Predictor(num_class=len(class_list), inc=inc,
                   temp=args.T)
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
    matching_pseudo_labels = 0
    uncertain_counter = 0   
    G.train()
    F1.train()
    optimizer_g = optim.SGD(params, momentum=0.9, weight_decay=0.0005, nesterov=True)
    optimizer_f = optim.SGD(list(F1.parameters()), lr=1.0, momentum=0.9, weight_decay=0.0005, nesterov=True)
    print(len(target_loader_unl.dataset))
    # Define learning rate schedule here - to be updated at each iteration
    if args.LR_scheduler == "cosine":
        warmup = 0
        total_steps = int(args.steps/len(target_loader_unl))
        scheduler_g = get_cosine_schedule_with_warmup(optimizer_g, warmup, total_steps)
        scheduler_f = get_cosine_schedule_with_warmup(optimizer_f, warmup, total_steps)
    
    # Loading the dictionary having the feature bank and corresponsing metadata
    f = open("banks/unlabelled_target_%s.pkl"%(args.target), "rb")
    feat_dict_target = edict(pickle.load(f))
    feat_dict_target.feat_vec  = feat_dict_target.feat_vec.cuda() # Pushing computed features to cuda
    f = open("banks/labelled_source_%s.pkl"%(args.source), "rb") # Loading the feature bank for the source samples
    feat_dict_source = edict(pickle.load(f))
    feat_dict_source.feat_vec  = feat_dict_source.feat_vec.cuda() # Pushing computed features to cuda
    print("Bank keys - Target: ", feat_dict_target.keys(),"Source: ", feat_dict_source.keys())
    print("Num  - Target: ", len(feat_dict_target.names), "Source: ", len(feat_dict_source.names))

    def zero_grad_all():
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
    param_lr_g = []
    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_f = []
    for param_group in optimizer_f.param_groups:
        param_lr_f.append(param_group["lr"])

    # Instantiating the augmentation class with default params now
    if args.augmentation_policy == "ours":
        augmentation = Augmentation()
    if args.augmentation_policy == "ours" or args.augmentation_policy == "rand_augment" or args.augmentation_policy == "ct_augment":
        thresh = 0.9 # threshold for confident prediction to generate pseudo-labels
    
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_pseudo = nn.CrossEntropyLoss(reduction='none').cuda()
    all_step = args.steps
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    data_iter_t_unl = iter(target_loader_unl)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    len_train_target_semi = len(target_loader_unl)
    best_acc = 0
    counter = 0

    momentum = 0
    K = 5

    for step in range(all_step):
        if args.LR_scheduler == "standard": # choosing appropriate learning rate
            optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step, init_lr=args.lr)
            optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step, init_lr=args.lr)
        
        elif args.LR_scheduler == "cosine":
            scheduler_f.step()
            scheduler_g.step()

        lr = optimizer_f.param_groups[0]['lr']
        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)
        if step % len_train_target_semi == 0:
            data_iter_t_unl = iter(target_loader_unl)
            print("Matching are: ", matching_pseudo_labels/len(target_loader_unl.dataset))
            matching_pseudo_labels = 0
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)
        
        # Extracting the batches from the iteratable dataloader
        data_t = next(data_iter_t)
        data_t_unl = next(data_iter_t_unl)
        data_s = next(data_iter_s)
        im_data_s = data_s[0].cuda()
        gt_labels_s = data_s[1].cuda()
        im_data_t = data_t[0].cuda()
        gt_labels_t = data_t[1].cuda()

        if not args.augmentation_policy == "rand_augment":
            im_data_tu = data_t_unl[0].cuda()
        elif args.augmentation_policy == "rand_augment":
            im_data_tu = data_t_unl[0][2].cuda()

        zero_grad_all()
        data = torch.cat((im_data_s, im_data_t), 0) #concatenating the labelled images
        target = torch.cat((gt_labels_s, gt_labels_t), 0)
        
        if args.augmentation_policy == "ours": # can call a method here which does "ours" augmentation policy
            im_data_tu_strong, im_data_tu_weak = process_batch(im_data_tu, augmentation, label=False) #Augmentations happenning here - apply strong augmentation to labelled examples and (weak + strong) to unlablled examples
        elif args.augmentation_policy == "rand_augment":
            im_data_tu_weak_aug, im_data_tu_strong_aug = data_t_unl[0][0].cuda(), data_t_unl[0][1].cuda()
            gt_labels_tu = data_t_unl[1].cuda()
        elif args.augmentation_policy == "ct_augment":
            im_data_tu_weak, im_data_tu_strong = data_t_unl[0][0], data_t_unl[0][1]

        # Getting predictions of weak and strong augmented unlabled examples
        pred_strong_aug = F1(G(im_data_tu_strong_aug))
        with torch.no_grad():
            pred_weak_aug = F1(G(im_data_tu_weak_aug))
        
        prob_weak_aug = F.softmax(pred_weak_aug,dim=1)
        mask_loss = prob_weak_aug.max(1)[0]>thresh
        pseudo_labels = pred_weak_aug.max(axis=1)[1]        
        loss_pseudo_unl = torch.mean(mask_loss.int() * criterion_pseudo(pred_strong_aug,pseudo_labels))
        loss_pseudo_unl.backward(retain_graph=True)
        
        f_batch, feat_dict_target  = update_features(feat_dict_target, data_t_unl, G, momentum)        
        # Get max of similarity distribution to check which element or label is it closest to in these vectors
        if step > 0:
            sim_distribution = get_similarity_distribution(feat_dict_target,data_t_unl,G)
            k_neighbors, _ = get_kNN(sim_distribution, feat_dict_target, K)  
            #print(k_neighbors)  
            mask_loss_uncertain = prob_weak_aug.max(1)[0]<thresh # Setting criteria for uncertain examples
            knn_confident_pseudo_labels = get_confident(k_neighbors,feat_dict_target, K, F1, thresh, mask_loss_uncertain)
            if torch.sum(mask_loss_uncertain.int()):
                loss_pseudo_unl_knn = torch.sum(mask_loss_uncertain.int() * criterion_pseudo(pred_strong_aug,knn_confident_pseudo_labels))/(torch.sum(mask_loss_uncertain.int()))
                loss_pseudo_unl_knn.backward(retain_graph=True)
                #loss_pseudo_unl_knn = torch.mean(mask_loss_uncertain.int() * criterion_pseudo(pred_strong_aug,knn_confident_pseudo_labels))
                print("Uncertain Example Loss: ", loss_pseudo_unl_knn.cpu().data.item())
                #loss_pseudo_unl_knn.backward(retain_graph=True)
            matching_pseudo_labels = matching_pseudo_labels + torch.sum(mask_loss_uncertain.int()*(knn_confident_pseudo_labels==gt_labels_tu).int())
            print(matching_pseudo_labels)


        output = G(data)
        out1 = F1(output)
        loss = criterion(out1, target)
        
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
            
            log_train = 'S {} T {} Train Ep: {} lr{} \t ' \
                        'Loss Classification: {:.6f} Loss T {:.6f} ' \
                        'Method {}\n'.format(args.source, args.target,
                                             step, lr, loss.data,
                                             -loss_t.data, args.method)
        else:
            log_train = 'S {} T {} Train Ep: {} lr{} \t ' \
                        'Loss Classification: {:.6f} Method {}\n'.\
                format(args.source, args.target, step, lr, loss.data, args.method)

        G.zero_grad()
        F1.zero_grad()
        zero_grad_all()
        if step % args.log_interval == 0:
            print(log_train)
        if step % args.save_interval == 0 and step > 0:
            loss_test, acc_test = test(target_loader_test)
            loss_val, acc_val = test(target_loader_val)
            # Cluster the target features
            vectors = feat_dict_target.feat_vec
            #cluster_ids_x, cluster_centers = k_means(vectors, len(class_list))
            #print(cluster_ids_x, cluster_centers.shape)
            
            G.train()
            F1.train()
            if acc_val >= best_acc:
                best_acc = acc_val
                best_acc_test = acc_test
                counter = 0
            else:
                counter += 1
               
            if args.early:
                if counter > args.patience:
                    break
            
            print('best acc test %f best acc val %f' % (best_acc_test, acc_val))
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