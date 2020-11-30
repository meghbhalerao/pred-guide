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
from utils.majority_voting import *
import pickle 
from utils.majority_voting import *
from utils.confidence_knn import *

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

def get_similarity_distribution(feat_dict,data_batch, G, source = False):
    if source:
        img_batch  = data_batch[0].cuda()
    else:
        img_batch = data_batch[0][0].cuda()
    f_batch = G(img_batch)
    sim_distribution  = torch.mm(F.normalize(feat_dict.feat_vec, dim=1),F.normalize(torch.transpose(f_batch,0,1),dim = 0))
    sim_distribution = edict({"cosines": sim_distribution, "names": data_batch[2], "labels": data_batch[1]})
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

def combine_dicts(feat_dict_target, feat_dict_source): # expects the easydict object
    feat_dict_combined = edict({})
    feat_dict_combined.feat_vec = torch.cat([feat_dict_target.feat_vec, feat_dict_source.feat_vec], dim=0)
    feat_dict_combined.labels = feat_dict_target.labels + feat_dict_source.labels
    feat_dict_combined.names = feat_dict_target.names + feat_dict_source.names
    feat_dict_combined.domain_identifier = feat_dict_target.domain_identifier + feat_dict_source.domain_identifier
    return feat_dict_combined

def save_stats(F1, G, loader, step, feat_dict_combined, batch, K, mask_loss_uncertain):
    fixmatch_label_list, gt_label_list, names_list, knn_labels_list = [], [], [], []
    weak_prob= []
    for idx, batch in enumerate(loader):
        im_weak = batch[0][0].cuda()
        pred_label = F1(G(im_weak))
        pred_label_list = list(pred_label.max(1)[1].cpu().detach().numpy())
        fixmatch_label_list.extend(pred_label_list)
        weak_prob.extend(list(pred_label.max(1)[0].cpu().detach().numpy()))
        gt_labels = batch[1]
        gt_label_list.extend(gt_labels)
        names = batch[2]
        names_list.extend(names)
        sim_distribution = get_similarity_distribution(feat_dict_combined,batch,G)
        k_neighbors, _ = get_kNN(sim_distribution, feat_dict_combined, K)    
        knn_majvot_pseudo_labels = get_majority_vote(k_neighbors,feat_dict_combined, K, F1, mask_loss_uncertain, len(loader.dataset)).cpu().detach().numpy()
        knn_labels_list.extend(knn_majvot_pseudo_labels)
    # Writing all these lists to a csv file
    f = open(os.path.join("logs","analysis_" + str(step) + ".csv"),"w")
    f.write("Name,GT Label, Fixmatch, KNN Label\n")
    num_examples = len(names_list)
    for i in range(num_examples):
        f.write(str(names_list[i]) + "," + str(gt_label_list[i].data.item()) + "," + str(fixmatch_label_list[i]) + "," + str(knn_labels_list[i]))
        f.write("\n")  
    return 0

def load_bank(args):
    f = open("./banks/unlabelled_target_%s.pkl"%(args.target), "rb")
    feat_dict_target = edict(pickle.load(f))
    feat_dict_target.feat_vec  = feat_dict_target.feat_vec.cuda()
    num_target = len(feat_dict_target.names)
    domain = ["T" for i in range(num_target)]
    feat_dict_target.domain_identifier = domain

    f = open("./banks/labelled_source_%s.pkl"%(args.source), "rb") # Loading the feature bank for the source samples
    feat_dict_source = edict(pickle.load(f))
    feat_dict_source.feat_vec  = feat_dict_source.feat_vec.cuda() 
    num_source = len(feat_dict_source.names)
    domain = ["S" for i in range(num_source)]
    feat_dict_source.domain_identifier = domain
    # Concat the corresponsing components of the 2 dictionaries
    feat_dict_combined = edict({})
    feat_dict_combined  = combine_dicts(feat_dict_source, feat_dict_target)

    print("Bank keys - Target: ", feat_dict_target.keys(),"Source: ", feat_dict_source.keys())
    print("Num  - Target: ", len(feat_dict_target.names), "Source: ", len(feat_dict_source.names))

    return feat_dict_source, feat_dict_target, feat_dict_combined
    
def do_method_bank(feat_dict_source, feat_dict_target, feat_dict_combined, momentum, data_t_unl, data_s, prob_weak_aug, thresh, K, pred_strong_aug, criterion_pseudo, target_loader_unl, G, F1):

    f_batch_target, feat_dict_target  = update_features(feat_dict_target, data_t_unl, G, momentum)
    f_batch_target = f_batch_target.detach()

    f_batch_source, feat_dict_source  = update_features(feat_dict_source, data_s, G, momentum, source = True)
    f_batch_source = f_batch_source.detach()
    # Get max of similarity distribution to check which element or label is it closest to in these vectors
    feat_dict_combined = combine_dicts(feat_dict_target, feat_dict_source)
    sim_distribution = get_similarity_distribution(feat_dict_combined,data_t_unl,G)
    k_neighbors, _ = get_kNN(sim_distribution, feat_dict_combined, K)    
    #mask_loss_uncertain = (prob_weak_aug.max(1)[0]<thresh) & (prob_weak_aug.max(1)[0]>0.7)
    mask_loss_uncertain = prob_weak_aug.max(1)[0]>thresh
    knn_majvot_pseudo_labels = get_majority_vote(k_neighbors,feat_dict_combined, K, F1, mask_loss_uncertain, len(target_loader_unl.dataset))
    #loss_pseudo_unl_knn = torch.mean(mask_loss_uncertain.int() * criterion_pseudo(pred_strong_aug, knn_majvot_pseudo_labels))
    
    if  not torch.sum(mask_loss_uncertain.int()) == 0:
        loss_pseudo_unl_knn = torch.sum(mask_loss_uncertain.int() * criterion_pseudo(pred_strong_aug, knn_majvot_pseudo_labels))/(torch.sum(mask_loss_uncertain.int()))
        loss_pseudo_unl_knn.backward(retain_graph=True)