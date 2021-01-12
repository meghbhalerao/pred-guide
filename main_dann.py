from __future__ import print_function
import argparse
import os
import sys
from utils.fixmatch import do_fixmatch
from utils.source_classwise_weighting import *
import numpy as np
import copy
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from model.resnet import resnet34
from model.basenet import AlexNetBase, Discriminator, Discriminator_classwise, VGGBase, Predictor, Predictor_deep
from utils.utils import *
from utils.majority_voting import *
from utils.confidence_knn import *
from utils.lr_schedule import inv_lr_scheduler, get_cosine_schedule_with_warmup
from utils.return_dataset import return_dataset, return_dataset_randaugment, TransformFix
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
parser.add_argument('--augmentation_policy', type=str, default='rand_augment', choices=['ours', 'rand_augment','ct_augment'],
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

source_loader, target_loader, target_loader_misc, target_loader_unl, target_loader_val, target_loader_test, class_list = return_dataset_randaugment(args)    

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
            params += [{'params': [value], 'lr': args.multi, 'weight_decay': 0.0005}]
        else: 
            params += [{'params': [value], 'lr': args.multi * 10, 'weight_decay': 0.0005}]

if "resnet" in args.net:
    F1 = Predictor_deep(num_class=len(class_list), inc=inc)
else:
    F1 = Predictor(num_class=len(class_list), inc=inc, temp=args.T)
weights_init(F1)

D = Discriminator(inc=inc)

D_all_class = Discriminator_classwise(inc=inc,num_class=len(class_list))

if args.pretrained_ckpt is not None:
    ckpt = torch.load(args.pretrained_ckpt)
    G.load_state_dict(ckpt["G"])
    F1.load_state_dict(ckpt["F1"])

lr = args.lr
G.cuda()
F1.cuda()
D.cuda()
D_all_class.cuda()

G = nn.DataParallel(G, device_ids=[0, 1])
F1 = nn.DataParallel(F1, device_ids=[0, 1])
D = nn.DataParallel(D, device_ids=[0, 1])

if os.path.exists(args.checkpath) == False:
    os.mkdir(args.checkpath)

def train():
    G.train()
    F1.train()

    optimizer_g = optim.SGD(params, momentum=0.9, weight_decay=0.0005, nesterov=True)
    optimizer_f = optim.SGD(list(F1.parameters()), lr=1.0, momentum=0.9, weight_decay=0.0005, nesterov=True)
    optimizer_d = optim.SGD(D.parameters(), lr=1.0, momentum=0.9,weight_decay=0.0005, nesterov=True)

    print("Unlabelled Target Dataset Size: ",len(target_loader_unl.dataset))
    print("Labelled Target Dataset Size: ",len(target_loader.dataset))
    print("Misc. Labelled Target Dataset Size: ",len(target_loader_misc.dataset))

    def zero_grad_all():
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
    
    param_lr_g = []
    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_f = []
    for param_group in optimizer_f.param_groups:
        param_lr_f.append(param_group["lr"])
    param_lr_d = []
    for param_group in optimizer_d.param_groups:
        param_lr_d.append(param_group["lr"])

    thresh = 0.9
    root_folder = "./data/%s"%(args.dataset)
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    criterion_pseudo = nn.CrossEntropyLoss(reduction='none').cuda()
    criterion_lab_target = nn.CrossEntropyLoss(reduction='mean').cuda()
    criterion_strong_source = nn.CrossEntropyLoss(reduction='mean').cuda()
    feat_dict_source, feat_dict_target, _ = load_bank(args)
    criterion_discriminator = nn.CrossEntropyLoss(reduction='mean')

    """
    if args.augmentation_policy == 'rand_augment':
        augmentation  = TransformFix(args.augmentation_policy,args.net,mean =[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    """

    num_target = len(feat_dict_target.names)
    num_source = len(feat_dict_source.names)

    feat_dict_source.sample_weights = torch.tensor(np.ones(num_source))
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
    per_cls_acc = np.array([1 for _ in range(len(class_list))]) # Just defining for sake of clarity and debugging
    source_strong_near_loader = None
    for step in range(all_step):

        source_strong_near_loader = None

        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step, init_lr=args.lr)
        optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step, init_lr=args.lr)
        optimizer_d = inv_lr_scheduler(param_lr_d, optimizer_d, step, init_lr=args.lr)

        lr = optimizer_f.param_groups[0]['lr']
        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)
        if step % len_train_target_semi == 0:
            data_iter_t_unl = iter(target_loader_unl)
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)
        if source_strong_near_loader is not None:
            if step % len(source_strong_near_loader) == 0:
                data_iter_s_near_strong = iter(source_strong_near_loader)
            
        # Extracting the batches from the iteratable dataloader
        data_t = next(data_iter_t)
        data_t_unl = next(data_iter_t_unl)
        data_s = next(data_iter_s)
        im_data_s = data_s[0].cuda()
        gt_labels_s = data_s[1].cuda()
        im_data_t = data_t[0][0].cuda()
        gt_labels_t = data_t[1].cuda()
        im_data_tu = data_t_unl[0][2].cuda()
        gt_labels_tu = data_t_unl[1].cuda()

        if source_strong_near_loader is not None:
            try:
                im_near_source_strong = next(source_strong_near_loader)[0][1].cuda()
                gt_near_source_strong = next(source_strong_near_loader)[1].cuda()
            except:
                source_strong_near_loader = iter(source_strong_near_loader)
            strong_logits = F1(G(im_near_source_strong))
            loss_source_strong = criterion_strong_source(strong_logits,gt_near_source_strong)
            loss_source_strong.backward(retain_graph=True)
        
        zero_grad_all()
        data = im_data_s
        target = gt_labels_s
        
        pseudo_labels, mask_loss = do_fixmatch(data_t_unl,F1,G,thresh,criterion_pseudo)
        f_batch_source, feat_dict_source = update_features(feat_dict_source, data_s, G, 0, source = True)
        update_label_bank(label_bank, data_t_unl, pseudo_labels, mask_loss)

        #if step >=0 and step % 250 == 0 and step<=3500:
        if step>0:
            if step % 1500 == 0:
                print("here")
                poor_class_list = list(np.argsort(per_cls_acc))[0:125]
                print(per_cls_acc)
                print(poor_class_list)

                #classwise_near = do_source_weighting(target_loader_misc,feat_dict_source,G,K_farthest_source,weight=1, aug = 2, only_for_poor=True, poor_class_list=poor_class_list,weighing_mode='N')

                #do_source_weighting(target_loader_misc,feat_dict_source,G,K_farthest_source,weight=1, aug = 2, only_for_poor=True, poor_class_list=poor_class_list,weighing_mode='F')

                do_probability_weighing(G,D,source_loader,feat_dict_source)


                print("Assigned weights to source")
                #print(len(classwise_near.names))
                #print(len(classwise_near.labels))

                #source_strong_near_loader = make_st_aug_loader(args,classwise_near)

        feat_disc_t = do_lab_target_loss(label_bank,class_list,G,F1,data_t,im_data_t, gt_labels_t, criterion_lab_target,beta=0.99,mode='ce')

        #output = G(data)
        output = f_batch_source
        output_tu = G(im_data_tu)
        feat_disc_source = output.clone().detach()
        feat_disc_tu = output_tu.clone().detach()

        do_domain_classification(D_all_class,feat_disc_source, feat_disc_tu, feat_disc_t, gt_labels_s, gt_labels_t, gt_labels_tu, pseudo_labels, criterion_discriminator,optimizer_d,mode='classwise')

        out1 = F1(output)

        if step>0:
            names_batch = list(data_s[2])
            idx = [feat_dict_source.names.index(name) for name in names_batch] 
            weights_source = feat_dict_source.sample_weights[idx]
            loss = torch.mean(weights_source.cuda() * criterion(out1, target))
        else:
            loss = torch.mean(criterion(out1, target))
        
        if not args.method == 'S+T':
            if args.method == 'ENT':
                loss_t = entropy(F1, output_tu, args.lamda)
                loss_t.backward(retain_graph=True)
                #optimizer_f.step()
                #optimizer_g.step()
            elif args.method == 'MME':
                loss_t = adentropy(F1, output_tu, args.lamda)
                loss_t.backward(retain_graph=True)
                #optimizer_f.step()
                #optimizer_g.step()
            else:
                raise ValueError('Method cannot be recognized.')

            loss.backward()
            optimizer_g.step()
            optimizer_f.step()
            zero_grad_all()

            log_train = 'S {} T {} Train Ep: {} lr{} \t Loss Classification: {:.6f} Loss T {:.6f} Method {}\n'.format(args.source, args.target, step, lr, loss.data, -loss_t.data, args.method)
        else:
            log_train = 'S {} T {} Train Ep: {} lr{} \t Loss Classification: {:.6f} Method {}\n'.format(args.source, args.target, step, lr, loss.data, args.method)

        G.zero_grad()
        F1.zero_grad()
        zero_grad_all()
        
        if step % args.log_interval == 0:
            print(log_train)
        if step % args.save_interval == 0 and step > 0:
            if step % 2000 == 0:
                #save_stats(F1, G, target_loader_unl, step, feat_dict_combined, data_t_unl, K, mask_loss_uncertain)
                pass
            _, acc_labeled_target, _, per_cls_acc = test(target_loader, mode = 'Labeled Target')
            _, acc_test,_,_ = test(target_loader_test, mode = 'Test')
            _, acc_val, _, _ = test(target_loader_val, mode = 'Val')

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
        per_cls_acc = torch.tensor(np.ones(num_class)).cuda()
    elif mode =='Labeled Target':
        np.save("cf_labeled_target.npy",confusion_matrix)
        per_cls_acc = per_class_accuracy(confusion_matrix)
        weight = per_cls_acc 
        weight = (weight>0.5).int()*1.5 + (weight<0.5).int()*0.5
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} F1 ({:.4f}%)\n'.format(mode, test_loss,correct,size,100.*correct/size))
    return test_loss.data,100.*float(correct)/size, weight, per_cls_acc.cpu().numpy()

def per_class_accuracy(confusion_matrix):
    num_class, _ = confusion_matrix.shape
    per_cls_acc = []
    for i in range(num_class):
        per_cls_acc.append(confusion_matrix[i,i]/sum(confusion_matrix[i,:]))
    per_cls_acc = torch.tensor(per_cls_acc).cuda()
    return per_cls_acc

train()