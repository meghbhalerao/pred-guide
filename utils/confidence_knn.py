import torch
import numpy as np
import torch.nn.functional as F

def get_confident_label(list_predictions, thresh):
    for prediction in list_predictions:
        prediction = F.softmax(prediction,dim=1)
        if prediction.max(1)[0] > thresh:
            return prediction.max(1)[1].cpu().data.item()
    return -1

def get_confident(k_neighbors,feat_dict, K, F1, thresh, mask_loss_uncertain,source = False):
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


def get_most_confident_label(list_predictions):
    max_prob = 0
    for prediction in list_predictions:
        prediction = F.softmax(prediction,dim=1)
        if prediction.max(1)[0].cpu().data.item() > max_prob:
            max_prob = prediction.max(1)[0].cpu().data.item()
            curr_label = prediction.max(1)[1].cpu().data.item()
    return curr_label

def get_most_confident(k_neighbors,feat_dict, K, F1):
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
        confident_label = get_most_confident_label(pred_list)
        pseudo_labels.append(confident_label) 
    return torch.tensor(pseudo_labels).cuda()
