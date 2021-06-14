#from tsnecuda import TSNE
from __future__ import barry_as_FLUFL
import torch
import numpy as np
import sys

from torch._C import Value
from loaders.data_list import Imagelists_VISDA, return_classlist
from model.basenet import *
from model.resnet import *
from torchvision import transforms
from utils.return_dataset import ResizeImage
import time
import os
from torch.autograd import Variable
import pickle
import copy 

# Defining return dataset function here
net = "alexnet"
dataset_name = "office_home"
root = '../data/%s/'%(dataset_name)
domain = "Art"
domain_identifier = "target"
n_class = 65
num = 1
load_pretrained = False

if domain_identifier == "target":
    image_list_target_unl = "../data/txt/%s/unlabeled_target_images_%s_%s.txt"%(dataset_name,domain,num)
elif domain_identifier == "source":
    image_list_target_unl = "../data/txt/%s/labeled_source_images_%s.txt"%(dataset_name,domain)
else:
    raise ValueError("Please Enter Valid Domain Identifier!")
f = open(image_list_target_unl,"r")
print(len([line for line in f]))
ours = False

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
    num_images = len(target_dataset_unl)
    if net == 'alexnet':
        bs = 1
    else:
        bs = 1

    target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl, batch_size=bs, num_workers=3,shuffle=False, drop_last=False)
    return target_loader_unl,class_list

target_loader_unl, _ = get_dataset(net,root,image_list_target_unl)
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

G.cuda()
F1.cuda()
G.eval()
F1.eval()

# Loading the weights from the checkpoint
if load_pretrained:
    ckpt = torch.load("../save_model_ssda/resnet34_real_sketch_8000.ckpt.pth.tar")
    G_dict = ckpt["G_state_dict"]
    G_dict_new = {}
    G_dict_new_keys = G_dict_new.keys()
    for key in G_dict.keys():
        new_key = copy.copy(key)
        new_key = new_key.replace("module.","")
        G_dict_new[new_key] = G_dict[key]
        print(new_key)
    print(G_dict_new.keys())
    #print(G_dict.keys())
    #sys.exit()
    G.load_state_dict(G_dict_new)
    #G.load_state_dict(ckpt["G_state_dict"])

features = []   
labels = []
name_list = []
start = time.time()
# Features for easy examples
print(len(target_loader_unl.dataset))

with torch.no_grad():
    for idx, image_obj in enumerate(target_loader_unl):
        image = image_obj[0].cuda()
        label = image_obj[1].cpu().data.item()
        name  = image_obj[2]
        img_path = image_obj[2][0]
        output = G(image)
        output = torch.flatten(output)
        output = output.cpu().numpy()
        features.append(output)
        labels.append(label)
        name_list.append(name[0])
        print(label)

feat_dict = {}
end = time.time()
print((end-start)/60)
features = torch.tensor(np.array(features))
feat_dict['feat_vec'] = features
feat_dict['labels'] = labels
feat_dict['names'] = name_list
print(len(labels))
print(features.shape)
print(len(name_list))
print("Saving dictionary as pickle")
if domain_identifier == "source":
    filehandler = open("./%s_labelled_source_%s.pkl"%(net, domain),'wb')
elif domain_identifier == "target":
    filehandler = open("./%s_unlabelled_target_%s_%s.pkl"%(net, domain,str(num)), 'wb')

pickle.dump(feat_dict, filehandler)


def test(loader):
    G.eval()
    F1.eval()
    test_loss = 0
    correct = 0
    size = 0
    num_class = n_class
    output_all = np.zeros((0, num_class))

    im_data_t = torch.FloatTensor(1)
    im_data_t = im_data_t.cuda()
    im_data_t = Variable(im_data_t)

    gt_labels_t = torch.LongTensor(1)
    gt_labels_t = gt_labels_t.cuda()
    gt_labels_t = Variable(gt_labels_t)

    confusion_matrix = torch.zeros(num_class, num_class)
    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
            gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])
            feat = G(im_data_t)
            output1 = F1(feat)
            output_all = np.r_[output_all, output1.data.cpu().numpy()]
            size += im_data_t.size(0)
            pred1 = output1.data.max(1)[1]
            for t, p in zip(gt_labels_t.view(-1), pred1.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += pred1.eq(gt_labels_t.data).cpu().sum()
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} F1 ({:.0f}%)\n'.format(test_loss, correct, size, 100. * correct / size))
    return confusion_matrix
