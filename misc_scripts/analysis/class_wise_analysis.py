import torch
import numpy as np
import sys
sys.path.append("/home/megh/projects/domain-adaptation/SSAL/")
from loaders.data_list import Imagelists_VISDA, return_classlist
from model.basenet import *
from model.resnet import *
from torchvision import transforms
from utils.return_dataset import ResizeImage
from utils.utils import k_means
import time
import os
from torch.autograd import Variable
import pickle
from easydict import EasyDict as edict
from utils.return_dataset import *
from easydict import EasyDict as edict
#from misc_scripts.plot_class_wise import *

def main():
    
    net = "resnet34"
    root = '../data/multi/'
    target = "sketch"
    image_list_target_unl = "../data/txt/multi/unlabeled_target_images_%s_3.txt"%(target)
    num = 3
    f = open(image_list_target_unl,"r")
    print(len([line for line in f]))

    args = edict({"net":net,"source":"real","target":target,"dataset":"multi","num":num,"uda":1})
    source_loader, target_loader, target_loader_unl, target_loader_val, target_loader_test, class_list = return_dataset_randaugment(args,txt_path='../data/txt/',root_path='../data/', bs_alex=1, bs_resnet=1)        
    n_class = len(class_list)
    print(len(target_loader_unl.dataset))

    # Deinfining the pytorch networks
    if net == 'resnet34':
        G = resnet34()
        inc = 512
        print("Using: ", net)
    elif net == 'resnet50':
        G = resnet50()
        inc = 2048
        print("Using: ", net)
    elif net == "alexnet":
        G = AlexNetBase()
        inc = 4096
        print("Using: ", net)
    elif net == "vgg":
        G = VGGBase()
        inc = 4096
        print("Using: ", net)
    else:
        raise ValueError('Model cannot be recognized.')

    if net == 'resnet34':
        F1 = Predictor_deep(num_class=n_class,inc=inc)
        print("Using: Predictor_deep_attributes")
    else:
        F1 = Predictor(num_class=n_class,inc=inc)
        print("Using: Predictor_attributes")

    G.eval().cuda()
    F1.eval().cuda()


    # Loading the weights from the checkpoint
    ckpt = torch.load("../save_model_ssda/resnet34_real_sketch_8000.ckpt.pth.tar")
    G_dict = ckpt["G_state_dict"]

    G_dict_backup = G_dict.copy()
    for key in G_dict_backup.keys():
        new_key = key.replace("module.","")
        G_dict[new_key] = G_dict.pop(key)
    G.load_state_dict(G_dict)
    F1_dict = ckpt["F1_state_dict"]
    F1_dict_backup = F1_dict.copy()
    for key in F1_dict_backup.keys():
        new_key = key.replace("module.","")
        F1_dict[new_key] = F1_dict.pop(key)
    G.load_state_dict(G_dict)
    F1.load_state_dict(F1_dict)

    pred_matrix = np.zeros((6,126),dtype=int)
    confidence_matrix = np.zeros((4,126),dtype=int)

    print(dir(target_loader_unl))
    test(target_loader_test, G, F1, len(class_list), pred_matrix, confidence_matrix, mode = 'Test')
    test(target_loader, G, F1, len(class_list), pred_matrix, confidence_matrix,mode='Labeled Target')
    np.save("pred_matrix.npy", pred_matrix)
    np.save("confidence_matrix.npy", confidence_matrix)

def test(loader, G, F1, num_class, pred_matrix, confidence_matrix, mode='Test'):
    G.eval()
    F1.eval()
    test_loss = 0
    correct = 0
    size = 0
    num_class = num_class
    criterion = nn.CrossEntropyLoss().cuda()
    confusion_matrix = torch.zeros(num_class, num_class)
    with torch.no_grad():
        for idx , data_t in enumerate(loader):
            if  mode == 'Test':
                im_data_t = data_t[0].cuda()
                gt_labels_t = data_t[1].cuda()
                feat = G(im_data_t)
                output1 = F.softmax(F1(feat),dim=1)
                size += im_data_t.size(0)
                pred1 = output1.data.max(1)[1]
                prob = output1.data.max(1)[0]
                print(prob)
                if pred1.eq(gt_labels_t.data).cpu().sum():
                    if prob > 0.9:
                        pred_matrix[0, gt_labels_t.data] +=1
                        confidence_matrix[0, gt_labels_t.data] +=1
                    elif prob > 0.8:
                        pred_matrix[1, gt_labels_t.data] +=1
                    elif prob > 0.7:
                        pred_matrix[2,gt_labels_t.data] +=1
                    else:
                        pass
                else:
                    if prob > 0.9:
                        confidence_matrix[1, gt_labels_t.data] +=1

                for t, p in zip(gt_labels_t.view(-1), pred1.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                correct += pred1.eq(gt_labels_t.data).cpu().sum()
                test_loss += criterion(output1, gt_labels_t) / len(loader)
                np.save("cf_unlabelled_target.npy",confusion_matrix)
                print(idx)
            elif mode == "Labeled Target":
                im_data_t_weak, im_data_t_strong, im_data_t_standard = data_t[0][0].cuda(), data_t[0][1].cuda(), data_t[0][2].cuda()
                gt_labels_t = data_t[1].cuda()
                feat_weak, feat_strong, feat_standard = G(im_data_t_weak), G(im_data_t_strong),G(im_data_t_standard)
                output1_weak, output1_strong, output1_standard = F.softmax(F1(feat_weak),dim=1), F.softmax(F1(feat_strong)), F.softmax(F1(feat_standard))

                size += im_data_t_weak.size(0) + im_data_t_strong.size(0) + im_data_t_standard.size(0)
                pred1_weak, pred1_strong, pred1_standard = output1_weak.data.max(1)[1], output1_strong.data.max(1)[1], output1_standard.data.max(1)[1]
                prob_weak, prob_strong, prob_standard = output1_weak.data.max(1)[0],  output1_strong.data.max(1)[0], output1_standard.data.max(1)[0]      
                print(prob_standard,prob_weak,prob_strong)   

                if pred1_weak.eq(gt_labels_t.data).cpu().sum() or pred1_strong.eq(gt_labels_t).cpu().sum() or pred1_standard.eq(gt_labels_t).cpu().sum():
                    if prob_weak > 0.9 or prob_strong > 0.9 or prob_standard > 0.9:
                        pred_matrix[3, gt_labels_t.data] += int(pred1_weak.eq(gt_labels_t.data).cpu().sum()) + int(pred1_strong.eq(gt_labels_t.data).cpu().sum()) + int(pred1_standard.eq(gt_labels_t).cpu().sum())
                        confidence_matrix[2,gt_labels_t.data] +=int(pred1_weak.eq(gt_labels_t.data).cpu().sum()) + int(pred1_strong.eq(gt_labels_t.data).cpu().sum()) + int(pred1_standard.eq(gt_labels_t).cpu().sum())
                    elif prob_weak > 0.8 or prob_strong > 0.8 or prob_standard > 0.8:
                        pred_matrix[4, gt_labels_t.data] += int(prob_weak > 0.8) + int(prob_strong > 0.8) + int(prob_standard > 0.8)
                    elif prob_weak > 0.7 or prob_strong>0.7 or prob_standard>0.7:
                        pred_matrix[5,gt_labels_t.data] += int(prob_weak > 0.7) + int(prob_strong > 0.7) + int(prob_standard > 0.7)
                    else:
                        pass
                else:
                    if prob_weak > 0.9 or prob_strong > 0.9 or prob_standard > 0.9:
                        confidence_matrix[3,gt_labels_t.data] += int(prob_weak > 0.9) + int(prob_strong > 0.9) + int(prob_standard > 0.9)


                for t, p_weak, p_strong, p_standard in zip(gt_labels_t.view(-1), pred1_weak.view(-1), pred1_strong.view(-1), pred1_standard.view(-1)):
                    confusion_matrix[t.long(), p_weak.long()] += 1
                    confusion_matrix[t.long(), p_strong.long()] += 1
                    confusion_matrix[t.long(), p_standard.long()] += 1

                correct += pred1_weak.eq(gt_labels_t.data).cpu().sum() + pred1_strong.eq(gt_labels_t.data).cpu().sum() + pred1_standard.eq(gt_labels_t.data).cpu().sum()

                test_loss += criterion(output1_weak, gt_labels_t)/(3*len(loader))  + criterion(output1_strong, gt_labels_t)/(3*len(loader)) + criterion(output1_standard, gt_labels_t)/(3*len(loader)) 
                np.save("cf_labelled_target.npy",confusion_matrix)
                print(idx)
    if not mode == 'Labeled Target':
        np.save("cf_unlabelled_target.npy",confusion_matrix)
    else:
        np.save("cf_labelled_target.npy",confusion_matrix)
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} F1 ({:.4f}%)\n'.format(mode, test_loss,correct,size,100.*correct/size))
    return test_loss.data,100.*float(correct)/size, pred_matrix

def get_dataset(net,root,image_set_file_test):
    if net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    target_dataset_unl = Imagelists_VISDA(image_set_file_test, root=root, transform=data_transforms['test'], test=True)
    class_list = return_classlist(image_set_file_test)

    if net == 'alexnet':
        bs = 1
    else:
        bs = 1

    target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl, batch_size=bs, num_workers=3, shuffle=False, drop_last=False)
    return target_loader_unl,class_list

if __name__ == '__main__':
    main()

"""
G_dict = ckpt["G_state_dict"]
G_dict_backup = G_dict.copy()
for key in G_dict_backup.keys():
    new_key = key.replace("module.","")
    G_dict[new_key] = G_dict.pop(key)
"""
