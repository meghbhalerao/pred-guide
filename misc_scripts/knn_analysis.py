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
from misc_scripts.plot_class_wise import *

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
    ckpt = torch.load("../save_model_ssda/resnet34_real_sketch_5000.ckpt.pth.tar")
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
    for idx, batch in enumerate(target)


