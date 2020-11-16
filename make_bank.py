#from tsnecuda import TSNE
import torch
import numpy as np
import sys
sys.path.append('/cbica/home/bhaleram/comp_space/random/personal/others/SSAL/')
from loaders.data_list import Imagelists_VISDA, return_classlist
from model.basenet import *
from model.resnet import *
from torchvision import transforms
from utils.return_dataset import ResizeImage
import time
import os
from torch.autograd import Variable

# Defining return dataset function here
net = "resnet34"
root = './data/multi/'
target = "sketch"
n_class = 126
n_class_plot = 30
k = 10
image_list_target_unl = "./data/txt/multi/unlabeled_target_images_%s_1.txt"%(target)
ours = False


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

# Loading the weights from the checkpoint
"""
ckpt = torch.load("./save_model_ssda/resnet34_real_sketch_9500")
G_dict = ckpt["G_state_dict"]
G_dict_backup = G_dict.copy()
for key in G_dict_backup.keys():
    new_key = key.replace("module.","")
    G_dict[new_key] = G_dict.pop(key)
#print(G_dict.keys())
G.load_state_dict(ckpt["G_state_dict"])
"""
features = []   
labels = []
weights = []

start = time.time()
# Features for easy examples
with torch.no_grad():
    for idx, image_obj in enumerate(target_loader_unl):
        image = image_obj[0].cuda()
        label = image_obj[1].cpu().data.item()
        img_path = image_obj[2][0]
        output = G(image)
        output = torch.flatten(output)
        output = output.cpu().numpy()
        features.append(output)
        labels.append(label)
        weights.append(1)
        print(label)

end = time.time()
print((end-start)/60)


features = np.array(features)
print(features.shape)
np.save('features.npy', features)
print(len(labels))
labels = np.array(labels)
np.save('labels.npy',labels)
print(len(weights))
weights = np.array(weights)
np.save('weights.npy', weights)
