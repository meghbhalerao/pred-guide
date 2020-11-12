from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model.resnet import resnet34
from model.basenet import AlexNetBase, VGGBase, Predictor, Predictor_deep
from utils.lr_schedule import inv_lr_scheduler
from utils.utils import weights_init
from utils.return_dataset import return_dataset_rot, return_dataset
from utils.loss import entropy, adentropy

# Training settings
parser = argparse.ArgumentParser(description='SSDA Classification')
parser.add_argument('--step', type=int, default=50000, metavar='N',
                    help='maximum number of iterations '
                         'to train (default: 50000)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--multi', type=float, default=0.1, metavar='MLT',
                    help='learning rate multiplication')
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
parser.add_argument('--net', type=str, default='alexnet',
                    help='which network to use')
parser.add_argument('--target', type=str, default='real',
                    help='source domain')
parser.add_argument('--dataset', type=str, default='multi',
                    choices=['multi', 'office', 'office_home'],
                    help='the name of dataset')
parser.add_argument('--patience', type=int, default=5, metavar='S',
                    help='early stopping to wait for improvment '
                         'before terminating. (default: 5 (5000 iterations))')
parser.add_argument('--early', action='store_false', default=True,
                    help='early stopping on validation or not')
parser.add_argument('--num', type=int, default=1,
                    help='number of labeled samples per class')
parser.add_argument('--save_interval', type=int, default=500, metavar='N',
                    help='how many batches to wait before saving a model')


args = parser.parse_args()
all_step = args.step
print('Dataset %s Target %s Network %s Num per class %s' % (args.dataset, args.target, args.net, str(args.num)))
target_loader, target_loader_unl, class_list = return_dataset_rot(args)

len_target = len(target_loader)
len_target_unl = len(target_loader_unl)

print("%s classes in this dataset"%(len(class_list)))
use_gpu = torch.cuda.is_available()
record_dir = 'record/%s/' % (args.dataset)
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
record_file = os.path.join(record_dir, 'net_%s_%s' %(args.net, args.target))

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

# Defining the rotation classification matrix
F_rot = nn.Linear(inc,4)

# Defining the class classification network
if "resnet" in args.net:
    F1 = Predictor_deep(num_class=len(class_list),inc=inc)
else:
    F1 = Predictor(num_class=len(class_list), inc=inc, temp=args.T)

weights_init(F1)

G.cuda()
F_rot.cuda()
#F1.cuda()

params = []
for key, value in dict(G.named_parameters()).items():
    if value.requires_grad:
        if 'classifier' not in key:
            params += [{'params': [value], 'lr': args.multi,
                        'weight_decay': 0.0005}]
        else:
            params += [{'params': [value], 'lr': args.multi * 10,
                        'weight_decay': 0.0005}]

if os.path.exists(args.checkpath) == False:
    os.mkdir(args.checkpath)


def train():
    G.train()
    F1.train()
    F_rot.train()

    optimizer_g = optim.SGD(params, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)
    optimizer_f = optim.SGD(list(F1.parameters()), lr=1.0, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)
    optimizer_f_rot = optim.SGD(list(F_rot.parameters()), lr=1, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)

    def zero_grad_all():
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        optimizer_f_rot.zero_grad()

    param_lr_g = []
    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_f = []
    for param_group in optimizer_f.param_groups:
        param_lr_f.append(param_group["lr"])
    param_lr_f_rot =  []
    for param_group in optimizer_f_rot.param_groups:
        param_lr_f_rot.append(param_group["lr"])

    criterion = nn.CrossEntropyLoss().cuda()
    best_acc_class = 0
    best_acc_rot = 0
    counter = 0 

    data_iter_t = iter(target_loader)
    data_iter_t_unl = iter(target_loader_unl)

    for step in range(all_step):
        #lr = optimizer_f.param_groups[0]['lr']
        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step,
                                       init_lr=args.lr)
        optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step,
                                       init_lr=args.lr)
        optimizer_f_rot = inv_lr_scheduler(param_lr_f_rot, optimizer_f_rot, step,
                                       init_lr=args.lr)
        
        lr = optimizer_f.param_groups[0]['lr']

        if step % len_target == 0:
            data_iter_t = iter(target_loader)
        if step % len_target_unl == 0:
            data_iter_t_unl = iter(target_loader_unl)
        data_t = next(data_iter_t)
        data_t_unl = next(data_iter_t_unl)

        # Extracting data from the dataloader
        im_data_t = data_t[0].cuda()
        gt_labels_rot_t = data_t[1].cuda()
        gt_labels_class_t = data_t[2].cuda()
        
        im_data_t_unl = data_t_unl[0].cuda()
        gt_labels_rot_t_unl = data_t_unl[1].cuda()
        

        zero_grad_all()
        # Loss labeled classification
        output = G(im_data_t)
        out_class = F1(output)
        loss_class = criterion(out_class, gt_labels_class_t)
        loss_class.backward(retain_graph=True)
        # Loss labeled rotation
        out_rot = F_rot(output)
        loss_rot = criterion(out_rot, gt_labels_rot_t)
        loss_rot.backward(retain_graph=True)
        # Loss unlabeled adentropy
        output = G(im_data_t_unl)
        loss_ent = adentropy(F1, output, args.lamda)
        loss_ent.backward(retain_graph=True)
        # Loss unlabeled rot
        out_rot = F_rot(output)
        loss_rot_unl = criterion(out_rot, gt_labels_rot_t_unl)
        loss_rot_unl.backward()


        optimizer_g.step()
        optimizer_f.step()
        optimizer_f_rot.step()
        zero_grad_all()

        log_train = 'T {} Ep: {} lr{} \t Loss Class: {:.6f} RotLab: {} Ent: {} RotUnl: {} \n'.format(args.target, step, lr, loss_class.data, loss_rot.data, -loss_ent.data, loss_rot_unl.data)

        G.zero_grad()
        F1.zero_grad()
        F_rot.zero_grad()
        zero_grad_all()
        
        if step % args.log_interval == 0:
            print(log_train)
        if step % args.save_interval == 0 and step > 0:
            loss_train_class, acc_train_class = test(target_loader_unl, rot=False)
            loss_train_rot, acc_train_rot  = test(target_loader_unl, rot=True)

            G.train()
            F1.train()
            F_rot.train()

            if acc_train_class >= best_acc_class:
                best_acc_class = acc_train_class
                counter = 0
            else:
                counter += 1
            if args.early:
                if counter > args.patience:
                    break
            print('best acc class %f' % (best_acc_class))
            print('record %s' % record_file)
            with open(record_file, 'a') as f:
                f.write('step %d best %f  \n' % (step, best_acc_class))

            G.train()
            F1.train()
            F_rot.train()
            if args.save_check:
                print('saving model')
                torch.save({"G": G.state_dict(), "F1": F1.state_dict(), "F_rot": F_rot.state_dict()}, os.path.join(args.checkpath, "model_{}_step_{}.pth.tar".format(args.target, step)))



def test(loader, rot = False):
    G.eval()
    F1.eval()
    F_rot.eval()
    test_loss = 0
    correct = 0
    size = 0
    if rot:
        num_class = 4
    else:
        num_class = len(class_list)
    output_all = np.zeros((0, num_class))
    criterion = nn.CrossEntropyLoss().cuda()
    confusion_matrix = torch.zeros(num_class, num_class)
    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            #im_data_t.data.resize_(data_t[0].size()).copy_(data_t[0])
            #gt_labels_t.data.resize_(data_t[1].size()).copy_(data_t[1])
            
            im_data_t = data_t[0].cuda()
            if rot:
                gt_labels_t = data_t[1].cuda()
            else:
                gt_labels_t = data_t[2].cuda()
            feat = G(im_data_t)
            if rot:
                output1 = F_rot(feat)
            else:
                output1 = F1(feat)
            output_all = np.r_[output_all, output1.data.cpu().numpy()]
            size += im_data_t.size(0)
            pred1 = output1.data.max(1)[1]
            for t, p in zip(gt_labels_t.view(-1), pred1.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += pred1.eq(gt_labels_t.data).cpu().sum()
            test_loss += criterion(output1, gt_labels_t) / len(loader)
    if rot:
        print('\Rotation: Average loss: {:.4f}, '
              'Accuracy: {}/{} F1 ({:.0f}%)\n'.
            format(test_loss, correct, size,
                     100. * correct / size))
    else:
        print('\Target: Average loss: {:.4f}, '
              'Accuracy: {}/{} F1 ({:.0f}%)\n'.
            format(test_loss, correct, size,
                     100. * correct / size))
    return test_loss.data, 100. * float(correct) / size


train()









"""
    for epoch in range(1, args.epochs + 1):
        train_epoch(epoch, args, G, F1, F_rot, target_loader, target_loader_unl, optimizer_g, optimizer_f, optimizer_f_rot, criterion, zero_grad_all, param_lr_g, param_lr_f, param_lr_f_rot)
        loss_train, acc_train = test(target_loader)
        G.train()
        F1.train()
        if acc_train >= best_acc:
            best_acc = acc_train
            counter = 0
        else:
            counter += 1
        if args.early:
            if counter > args.patience:
                break
        print('best acc  %f' % (best_acc))
        print('record %s' % record_file)
        with open(record_file, 'a') as f:
            f.write('epoch %d best %f  \n' % (epoch, best_acc))
        if args.save_check:
            print('saving model')
            torch.save({"G": G.state_dict(), "F1": F1.state_dict(), "F_rot": F_rot.state_dict()}, os.path.join(args.checkpath, "model_{}_epoch_{}.pth.tar".format(args.target, epoch)))



# Training function            
def train_epoch(epoch, args, G, F1, F_rot, target_loader, target_loader_unl, optimizer_g, optimizer_f, optimizer_f_rot, criterion, zero_grad_all, param_lr_g, param_lr_f, param_lr_f_rot):

    for batch_idx, data_t in enumerate(target_loader):
        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, batch_idx+epoch*len(data_loader), init_lr=args.lr)
        optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, batch_idx+epoch*len(data_loader),init_lr=args.lr)
        optimizer_f_rot = inv_lr_scheduler(param_lr_f_rot, optimizer_f_rot, batch_idx+epoch*len(data_loader), init_lr=args.lr)
       
        zero_grad_all()
        
#        im_data_t.data.resize_(data_t[0].size()).copy_(data_t[0])
#        gt_labels_t.data.resize_(data_t[1].size()).copy_(data_t[1])

        print(data_t.keys())

        out1 = F1(G(im_data_t))
        loss = criterion(out1, gt_labels_t)
        loss.backward(retain_graph=True)
        optimizer_g.step()
        optimizer_f.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(im_data_t), len(data_loader.dataset),100. * batch_idx / len(data_loader), loss.item()))
"""
