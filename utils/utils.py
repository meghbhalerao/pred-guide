import os
from os import name
import torch
import torch.nn as nn
import shutil
import torch.nn.functional as F
from easydict import EasyDict as edict
import numpy as np
from kmeans_pytorch import kmeans

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.1)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.1)
        m.bias.data.fill_(0)


def save_checkpoint(state, is_best, checkpoint='checkpoint',
                    filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def update_features(feat_dict, data, G, momentum, source  = False):
    names_batch = data[2]
    if not source:
        img_batch = data[0][0].cuda()
    else:
        img_batch = data[0].cuda()
    names_batch = list(names_batch)
    idx = [feat_dict.names.index(name) for name in names_batch]
    f_batch = G(img_batch)
    feat_dict.feat_vec[idx] = (momentum * feat_dict.feat_vec[idx] + (1 - momentum) * f_batch).detach()
    return f_batch, feat_dict

def get_similarity_distribution(feat_dict,data_t_unl, G):
    img_batch = data_t_unl[0][0].cuda()
    f_batch = G(img_batch)
    sim_distribution  = torch.mm(F.normalize(feat_dict.feat_vec, dim=1),F.normalize(torch.transpose(f_batch,0,1),dim = 0))
    sim_distribution = edict({"cosines": sim_distribution, "names": data_t_unl[2], "labels": data_t_unl[1]})
    return sim_distribution

def get_kNN(sim_distribution, feat_dict, k = 1):
    k_neighbors = torch.topk(torch.transpose(sim_distribution.cosines,0,1), k, dim = 1)
    idxs = k_neighbors[1]
    b_size = idxs.shape[0]
    k_neighbors = []
    labels_k_neighbors = []
    for i in range(b_size):
        l = idxs[i].cpu().data.numpy()
        k_neighbors.append(list(l))
        labels_k_neighbors.append(list(np.array(feat_dict.labels)[l]))
    print("True Labels: ", sim_distribution.labels)
    print("KNN Labels:", labels_k_neighbors)
    return k_neighbors, labels_k_neighbors

def k_means(vectors, num_clusters):
    cluster_ids_x, cluster_centers = kmeans( X=vectors, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0'))
    return cluster_ids_x, cluster_centers
