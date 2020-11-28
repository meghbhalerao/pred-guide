import torch
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
import copy

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
"""       
def get_majority_vote_label(list_predictions,K):
    label_list = []
    for prediction in list_predictions:
        prediction = F.softmax(prediction,dim=1)
        label_list.append(prediction.max(1)[1].cpu().data.item())
    majority_label, num_maj = get_majority_from_list(label_list)
    return majority_label, num_maj
"""

def get_majority_vote(k_neighbors,feat_dict, K, F1, mask_loss_uncertain, len_target, source = False):
    feat_vec = feat_dict.feat_vec
    labels = feat_dict.labels
    if source:
        k_feats = []
        for img in k_neighbors:
            pseudo_labels = []
            for neighbor in range(K):
                pseudo_labels.append(labels[img[neighbor]])
            k_feats.append(pseudo_labels)
        pseudo_labels_final = []
        for img in k_feats:
            majority_label, _ = get_majority_from_list(img)
            pseudo_labels_final.append(majority_label)
        return torch.tensor(pseudo_labels_final).cuda()

    k_feats = []
    for img in k_neighbors:
        img_feats = []
        for neighbor in range(K):
            img_feats.append(feat_vec[img[neighbor]])
        k_feats.append(img_feats)
    pseudo_labels = []
    for idx, img_nearest in enumerate(k_feats):
        pred_list = []
        prob_list = []
        for idx_feat, feat in enumerate(img_nearest):
            if feat_dict.domain_identifier[k_neighbors[idx][idx_feat]] == "S":
            #if k_neighbors[idx][idx_feat] > len_target:
                #print(feat_dict.domain_identifier[k_neighbors[idx][idx_feat]])
                pred_list.append(feat_dict.labels[k_neighbors[idx][idx_feat]])
                prob_list.append(1)
            else:
                prediction = F1(feat.unsqueeze(0))
                pred_list.append(F.softmax(prediction, dim=1).max(1)[1].cpu().data.item())
                prob_list.append(F.softmax(prediction, dim=1).max(1)[0].cpu().data.item())
        #prob_list = list((np.array(prob_list)>0.9).astype(np.int8))
        #idxs_keep = [i for i in range(len(prob_list)) if prob_list[i] == 1]
        #pred_list = [pred_list[keep] for keep in idxs_keep] 
        #print(prob_list)
        #print(len(pred_list))
        if pred_list:
            majority_vote_label, num_maj = get_majority_from_list(pred_list)
            if num_maj < int(K/2): # Disregard example when it's not absolute majority
                mask_loss_uncertain[idx] = False
                majority_vote_label = 0 # Making it compatible with CE loss - anyways this is not considered for loss calculation
        else:
            majority_vote_label = 0
            mask_loss_uncertain[idx] = False
        pseudo_labels.append(majority_vote_label)
    return torch.tensor(pseudo_labels).cuda()