import numpy as np
import torch 
import matplotlib.pyplot as plt
import torch.nn.functional as F
from easydict import EasyDict as edict
import sys

def per_class_accuracy(confusion_matrix):
    num_class, _ = confusion_matrix.shape
    per_cls_acc = []
    for i in range(num_class):
        per_cls_acc.append(confusion_matrix[i,i]/sum(confusion_matrix[i,:]))
    per_cls_acc = torch.tensor(per_cls_acc).cuda()
    return per_cls_acc

def plot_tsne_figure(features,feat_dict):
    x = features[:,0]
    y = features[:,1]
    data_length = len(feat_dict.gt_labels)
    #print("Total datapoints to plot are:", data_length)
    set_near_far(feat_dict) # finds the near and far samples from the labeled target to the source and changes the tag indicator accordingly
    print(feat_dict.tag_global)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    #ax1.axis("off")
    #ax2.axis("off")
    
    ax1.set_yticklabels([]); ax2.set_yticklabels([]);ax1.set_yticks([]);ax2.set_yticks([])
    ax1.set_xticklabels([]); ax2.set_yticklabels([]);ax1.set_xticks([]);ax2.set_xticks([])

    for idx in range(data_length):
        if feat_dict.tag_global[idx] == 's':
            ax1.scatter(x[idx],y[idx],color='blue',marker = "x")
        elif feat_dict.tag_global[idx] == 'u' and int(feat_dict.is_correct_label[idx]) == 1:
            ax2.scatter(x[idx],y[idx],color='green', marker = 'x')
            print("Correct unl:", x[idx],y[idx])
        elif feat_dict.tag_global[idx] == 'u' and int(feat_dict.is_correct_label[idx]) == 0:
            ax2.scatter(x[idx],y[idx],color='red', marker='x')  
            print("Wrong unl:", x[idx],y[idx])
        elif feat_dict.tag_global[idx] == 'f':
            ax1.scatter(x[idx],y[idx], color='black', marker = 'x')
            print("Far source examples")
        elif feat_dict.tag_global[idx] == 'n':
            ax1.scatter(x[idx],y[idx], color = 'lightblue', marker='x')
            print("Near source examples")
        elif feat_dict.tag_global[idx] == 'lt':
            ax1.scatter(x[idx],y[idx], color = 'yellow', marker='x')
            print("Labeled Target examples")
    plt.savefig('fig1.png')
    plt.show()
    

def set_near_far(feat_dict):
    data_length  = len(feat_dict.gt_labels)
    #print("Type of datapoint identifiers are: ", feat_dict.tag_global)
    #print("Correctness of Predicted Labels are:  ", feat_dict.is_correct_label)
    idx_lt = [i for i in range(data_length) if feat_dict.tag_global[i] ==  'lt']
    idx_s = [i for i in range(data_length) if feat_dict.tag_global[i] ==  's']
    #print(idx_lt,idx_s)
    feat_dict_lt = make_feat_dict_from_idx(feat_dict,idx_lt)
    feat_dict_s = make_feat_dict_from_idx(feat_dict,idx_s)
    #print("Labeled Target Example  Matrix Shape: ", feat_dict_lt.feat_global.shape)
    #print("Labeled Source Example Matrix Shape: ", feat_dict_s.feat_global.shape)
    sim_distribution = get_similarity_distribution_2(feat_dict_s,feat_dict_lt,mode='cosine')
    print("Shape of cosine similarity is:", sim_distribution.cosines)
    k_neighbors, names_k_near = get_k_points(sim_distribution, feat_dict_s, k =5, mode = "near")
    k_far, names_k_far = get_k_points(sim_distribution, feat_dict_s, k =5, mode = "far")
    print("Far examples are:", names_k_far)
    print("Near Examples are:", names_k_near)
    for name in names_k_near:
        idxx =  feat_dict.name_list.index(name)
        feat_dict.tag_global[idxx] = 'n'

    for name in names_k_far:
        idxx =  feat_dict.name_list.index(name)
        feat_dict.tag_global[idxx] = 'f'
    
    return feat_dict_lt, feat_dict_s


def get_similarity_distribution_2(feat_dict_1,feat_dict_2,mode='cosine'):
    if mode == 'cosine':
        sim_distribution  = torch.mm(F.normalize(torch.tensor(feat_dict_1.feat_global).cuda(), dim=1),F.normalize(torch.transpose(torch.tensor(feat_dict_2.feat_global).cuda(),0,1),dim = 0))        
    elif mode == 'euclid':
        #sim_distribution = pairwise_distance(feat_dict.feat_vec, f_batch)
        pass
    sim_distribution = edict({"cosines": sim_distribution, "names": feat_dict_2.name_list})
    return sim_distribution


def get_k_points(sim_distribution, feat_dict, k = 1, mode = 'near'):
    if mode == 'near':
        k_neighbors = torch.topk(torch.transpose(sim_distribution.cosines,0,1), k, dim = 1)
    elif mode == 'far':
        k_neighbors = torch.topk(torch.transpose(1-sim_distribution.cosines,0,1), k, dim = 1)
    idxs = k_neighbors[1]
    b_size = idxs.shape[0]
    k_neighbors = []
    labels_k_neighbors = []
    names_k = []
    for i in range(b_size):
        l = idxs[i].cpu().data.numpy()
        k_neighbors.append(list(l))
        labels_k_neighbors.append(list(np.array(feat_dict.gt_labels)[l]))
    for l in k_neighbors:
        for index in l:
            names_k.append(feat_dict.name_list
            [index])
    return k_neighbors, names_k


def make_feat_dict_from_idx(feat_dict,idxs):
    feat_dict_idx = edict({})
    feat_dict_idx.feat_global = feat_dict.feat_global[idxs]
    feat_dict_idx.gt_labels = [feat_dict.gt_labels[idx] for idx in idxs]
    feat_dict_idx.name_list = [feat_dict.name_list[idx] for idx in idxs]
    feat_dict_idx.tag_global = [feat_dict.tag_global[idx] for idx in idxs]
    feat_dict_idx.is_correct_label = [feat_dict.is_correct_label[idx] for idx in idxs]
    return feat_dict_idx


def per_class_accuracy(confusion_matrix):
    num_class, _ = confusion_matrix.shape
    per_cls_acc = []
    for i in range(num_class):
        per_cls_acc.append(confusion_matrix[i,i]/sum(confusion_matrix[i,:]))
    per_cls_acc = torch.tensor(per_cls_acc).cuda()
    return per_cls_acc
