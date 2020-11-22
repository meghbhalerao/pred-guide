import os
from os import name
from numpy.core.fromnumeric import shape
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
    #print("True Labels: ", sim_distribution.labels)
    #print("KNN Labels:", labels_k_neighbors)
    return k_neighbors, labels_k_neighbors

def k_means(vectors, num_clusters):
    cluster_ids_x, cluster_centers = kmeans( X=vectors, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0'))
    return cluster_ids_x, cluster_centers

def get_confident_label(list_predictions, thresh):
    for prediction in list_predictions:
        prediction = F.softmax(prediction,dim=1)
        if prediction.max(1)[0] > thresh:
            return prediction.max(1)[1].cpu().data.item()
    return -1

def get_confident(k_neighbors,feat_dict, K, F1, thresh, mask_loss_uncertain):
    feat_vec = feat_dict.feat_vec
    k_feats = []
    for img in k_neighbors:
        img_feats = []
        for neighbor in range(K):
            img_feats.append(feat_vec[img[neighbor]])
        k_feats.append(img_feats)
    pseudo_labels = []
    for idx, img_nearest in enumerate(k_feats):
        pred_list = []
        for feat in img_nearest:
            pred_label = F1(feat.unsqueeze(0))
            pred_list.append(pred_label)
        confident_label = get_confident_label(pred_list, thresh)
        if confident_label == -1: # Disregard example when not confident
            mask_loss_uncertain[idx] = False
            confident_label = 0 # Making it compatible with CE loss - anyways this is not considered for loss calculation
        pseudo_labels.append(confident_label) 
    return torch.tensor(pseudo_labels).cuda()

 ####### functions for majority voting #####
def unique(list1): 
    unique_list = [] 
    for x in list1: 
        if x not in unique_list: 
            unique_list.append(x) 
    return unique_list

def get_majority_from_list(x):
    x.sort()
    unique_x = unique(x)
    element_count = []
    count = 0
    for unique_number in unique_x:
        for all_num in x:
            if all_num == unique_number:
                count = count + 1
        element_count.append(count)
        count = 0
    element_count = np.array(element_count)
    pos = np.argmax(element_count)
    majority_label = unique_x[pos]
    return majority_label, element_count[pos]
        
def get_majority_vote_label(list_predictions,K):
    label_list = []
    for prediction in list_predictions:
        prediction = F.softmax(prediction,dim=1)
        label_list.append(prediction.max(1)[1].cpu().data.item())
    majority_label, num_maj = get_majority_from_list(label_list)
    return majority_label, num_maj

def get_majority_vote(k_neighbors,feat_dict, K, F1, mask_loss_uncertain):
    feat_vec = feat_dict.feat_vec
    k_feats = []
    for img in k_neighbors:
        img_feats = []
        for neighbor in range(K):
            img_feats.append(feat_vec[img[neighbor]])
        k_feats.append(img_feats)
    pseudo_labels = []
    for idx, img_nearest in enumerate(k_feats):
        pred_list = []
        for feat in img_nearest:
            pred_label = F1(feat.unsqueeze(0))
            pred_list.append(pred_label)
        majority_vote_label, num_maj = get_majority_vote_label(pred_list,K)
        if num_maj < int(K/2): # Disregard example when it's not absolute majority
            mask_loss_uncertain[idx] = False
            majority_vote_label = 0 # Making it compatible with CE loss - anyways this is not considered for loss calculation
        pseudo_labels.append(majority_vote_label)
    return torch.tensor(pseudo_labels).cuda()