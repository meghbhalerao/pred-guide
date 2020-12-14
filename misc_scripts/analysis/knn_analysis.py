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
from utils.utils import *

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

    # Defining the pytorch networks
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
    G.load_state_dict(G_dict)
    F1_dict = ckpt["F1_state_dict"]
    G.load_state_dict(G_dict)
    F1.load_state_dict(F1_dict)
    # Loading the feature bank from pkl file
    f = open("../banks/resnet34_sketch_8000.pkl","rb")
    feat_dict = edict(pickle.load(f))
    print(feat_dict.keys())
    print(feat_dict.feat_vec.shape)
    feat_dict.feat_vec = feat_dict.feat_vec.cuda()
    top1 = 0
    top2 = 0
    top3 = 0
    total  = 0
    worst_classes = get_worst_classes(np.load("cf_labelled_target.npy"))
    print(worst_classes)
    print(len(worst_classes) * 9)
    with torch.no_grad():
        for idx, batch in enumerate(target_loader):
            gt_label = batch[1].cpu().data.item()
            if gt_label in worst_classes:
                sim_distribution_weak = get_similarity_distribution(feat_dict,batch,G, source = False, i = 0, mode = 'euclid')
                sim_distribution_strong = get_similarity_distribution(feat_dict,batch,G, source = False, i = 1, mode  = 'euclid')
                sim_distribution_standard = get_similarity_distribution(feat_dict,batch,G, source = False, i = 2, mode = 'euclid')       

                k_neighbors, labels_k_neighbors_weak = get_kNN(sim_distribution_weak, feat_dict, 3)                
                k_neighbors, labels_k_neighbors_strong = get_kNN(sim_distribution_weak, feat_dict, 3)                
                k_neighbors, labels_k_neighbors_standard = get_kNN(sim_distribution_weak, feat_dict, 3)                
                labels_k_neighbors_weak, labels_k_neighbors_strong, labels_k_neighbors_standard = labels_k_neighbors_weak[0], labels_k_neighbors_strong[0], labels_k_neighbors_standard[0]

                top1 = top1 + int(labels_k_neighbors_weak[0]==gt_label) + int(labels_k_neighbors_standard[0]==gt_label) + int(labels_k_neighbors_strong[0]==gt_label)
                top2 =  top2 + int(labels_k_neighbors_weak[1]==gt_label) + int(labels_k_neighbors_standard[1]==gt_label) + int(labels_k_neighbors_strong[1]==gt_label)
                top3 = top3  + int(labels_k_neighbors_weak[2]==gt_label) + int(labels_k_neighbors_standard[2]==gt_label) + int(labels_k_neighbors_strong[2]==gt_label)   
                total = total + 9
                print(top1,top2,top3)

def get_worst_classes(confusion_matrix, top_k = 30):
    num_classes, num_classes = confusion_matrix.shape
    acc_list = []
    for i in range(num_classes):
        acc_list.append(confusion_matrix[i,i]/sum(confusion_matrix[i,:]))
    arr_acc_list = np.array(acc_list)
    idx = np.argsort(arr_acc_list)
    idx = idx[0:top_k]
    print(arr_acc_list[idx])
    return idx

if __name__ == "__main__":
    main()

