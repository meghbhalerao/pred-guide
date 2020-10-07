import torch
import numpy as np
import sys
sys.path.append('/cbica/home/bhaleram/comp_space/random/personal/iisc_project/MME/')
from loaders.data_list import Imagelists_VISDA, return_classlist, return_number_of_label_per_class
from model.basenet import *
from model.resnet import *
from torchvision import transforms
from utils.return_dataset import ResizeImage
import time
import os

# Defining return dataset function here
net = "alexnet"
root = '../data/multi/'
source = "painting"
target = "real"
n_class = 126
num = 1 
method = "mme"
image_set_file_test = "/cbica/home/bhaleram/comp_space/random/personal/iisc_project/MME/data/txt/multi/unlabeled_target_images_%s_%s.txt"%(target,str(num))
model_path = "../freezed_models/alexnet_mme_p2r.ckpt.best.pth.tar"
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
    target_dataset_unl = Imagelists_VISDA(image_set_file_test, root=root,
                                            transform=data_transforms['test'],
                                            test=True)
    class_list = return_classlist(image_set_file_test)
    num_images = len(target_dataset_unl)
    if net == 'alexnet':
        bs = 1
    else:
        bs = 1

    target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl, batch_size=bs, num_workers=1,shuffle=False, drop_last=False)
    return target_loader_unl,class_list


target_loader_unl,class_list = get_dataset(net,root,image_set_file_test)
print(len(target_loader_unl.dataset))
# Deinfining the pytorch networks
if net == 'resnet34':
    G = resnet34()
    inc = 512
elif net == 'resnet50':
    G = resnet50()
    inc = 2048
elif net == "alexnet":
    G = AlexNetBase()
    inc = 4096
elif net == "vgg":
    G = VGGBase()
    inc = 4096
else:
    raise ValueError('Model cannot be recognized.')


if ours: 
    if net == 'resnet34':
        F1 = Predictor_deep_2(num_class=n_class,inc=inc,feat_dim = 50)
        print("Using: Predictor_deep_attributes")
    else:
        F1 = Predictor_attributes(num_class=n_class,inc=inc,feat_dim = 50)
        print("Using: Predictor_attributes")

else:
    if net == 'resnet34':
        F1 = Predictor_deep(num_class=n_class,inc=inc)
        print("Using: Predictor_deep")
    else:
        F1 = Predictor(num_class=n_class, inc=inc, temp=0.05)
        print("Using: Predictor")


G.cuda()
F1.cuda()

checkpoint = torch.load(model_path)

# Loading the weights from the checkpoint
G.load_state_dict(checkpoint["G_state_dict"])
F1.load_state_dict(checkpoint["F1_state_dict"])

f_hard_samples = open("%s_hard_samples_unlabeled_target_images_%s_%s.txt"%(method, target, str(num)),"w")
f_easy_samples = open("%s_easy_samples_unlabeled_target_images_%s_%s.txt"%(method, target, str(num)),"w")
h = 0
e = 0
with torch.no_grad():
    for idx, image_list in enumerate(target_loader_unl):
        image = image_list[0].cuda()
        label = image_list[1].cpu().data.item()
        img_path = image_list[2][0]
        output = G(image)
        output = F1(output)
        pred1 = output.data.max(1)[1]
        pred1 = pred1.cpu().data.item()
        if(pred1==label):
            f_easy_samples.write(str(img_path) + " " + str(label) + "\n")
            e = e + 1
        else:
            f_hard_samples.write(str(img_path) + " " + str(label) + "\n")
            h = h + 1 
        print(idx)

f_hard_samples.close()
f_easy_samples.close()

print(h,e)
print(e/(e+h))
