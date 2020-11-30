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
# Defining return dataset function here
net = "resnet34"
root = '../data/multi/'
target = "sketch"
image_list_target_unl = "../data/txt/multi/unlabeled_target_images_%s_3.txt"%(target)

f = open(image_list_target_unl,"r")
print(len([line for line in f]))

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

    target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl, batch_size=bs, num_workers=3,shuffle=False, drop_last=False)
    return target_loader_unl,class_list

target_loader_unl, class_list = get_dataset(net,root,image_list_target_unl)
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

G.eval()
F1.eval()


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

# Loading the feature banks
#f = open("../banks/unlabelled_target_sketch_8500.pkl","rb")
#feat_dict_target = edict(pickle.load(f))
print(ckpt.keys())
feat_dict_target = ckpt["target_feat_dict"]
feat_vectors_target  = feat_dict_target.feat_vec.cuda() # Pushing computed features to cuda

# Conditionally performing k means clustering
if not os.path.exists("../banks/k_means_clusters.pth.tar"):
    print("Computing K means clustering on the learnt target features")
    cluster_ids_x, cluster_centers = k_means(feat_vectors_target, len(class_list))
    print("Saving the k means cluster centers and cluster ids")
    torch.save({"cluster_ids_x": cluster_ids_x, "cluster_centers": cluster_centers},"../banks/k_means_clusters.pth.tar")
else:
    print("Found the clustered vectors, loading them, not performing k means")
    main_dict = torch.load("../banks/k_means_clusters.pth.tar")
    cluster_ids_x  = main_dict["cluster_ids_x"]
    cluster_centers = main_dict["cluster_centers"]

G.cuda()
F1.cuda()
cluster_centers = cluster_centers.cuda()
print("Cluster Center shape (n_class x feat_dim)", cluster_centers.shape)
cluster_center_labels = F1(cluster_centers) # Getting labels for cluster centroids
cluster_center_labels = cluster_center_labels.max(1)[1]
print("cluster_center_labels:", cluster_center_labels)
features = []   
labels = []
name_list = []
start = time.time()

# Pairwise distance between uncertain feature and cluster centers
def d_matrix_vector(matrix,vector):
    num_class = matrix.shape[0]
    d_vector = torch.ones(num_class)
    vector = vector[0]
    for i in range(num_class):
        d = torch.pow(vector - matrix[i],2.0).sum()
        d_vector[i] = d
    return d_vector

uncertain_count = 0
correct_uncertain = 0
correct_pseudo = 0
with torch.no_grad():
    for idx, image_obj in enumerate(target_loader_unl):
        image = image_obj[0].cuda()
        label = image_obj[1].cpu().data.item()
        name  = image_obj[2]
        output = F1(G(image))
        output = F.softmax(output,dim=1)
        is_uncertain = (output.max(1)[0]<0.9)
        if is_uncertain:
            uncertain_count  = uncertain_count + 1
            pseudo_label = output.max(1)[1].cpu().data.item()
            uncertain_feature = G(image)
            dist_from_centroid = d_matrix_vector(cluster_centers,uncertain_feature.detach())
            which_centroid = dist_from_centroid.argmin().cpu().data.item()
            class_label_centroid = cluster_center_labels[which_centroid].cpu().data.item()
            print("k_means label: ",class_label_centroid)
            print("pseudo label: ", pseudo_label)
            correct_uncertain += int(class_label_centroid==label)
            correct_pseudo += int(pseudo_label==label)
        features.append(output)
        labels.append(label)
        name_list.append(name[0])
        print(label)

print("Correctly Labelled Uncertain using K-Means: ", correct_uncertain/uncertain_count)
print("Correctly Labelled Uncertain using Pseudo Label: ", correct_pseudo/uncertain_count)
feat_dict = {}
end = time.time()
print((end-start)/60, " minutes")


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


"""
G_dict = ckpt["G_state_dict"]
G_dict_backup = G_dict.copy()
for key in G_dict_backup.keys():
    new_key = key.replace("module.","")
    G_dict[new_key] = G_dict.pop(key)
"""