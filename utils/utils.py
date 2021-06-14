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
from torch.optim import optimizer
from utils.majority_voting import *
import pickle 
from utils.majority_voting import *
from utils.confidence_knn import *
import sys

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

def save_mymodel(args, state, is_best):
    filename = '%s/%s_%s_%s.ckpt.pth.tar' % (args.checkpath, args.method, args.source, args.target)
    torch.save(state, filename)
    bestfilename = filename.replace('pth.tar', 'best.pth.tar')
    if is_best:
        if os.path.exists(bestfilename):
            existing_bestfile = torch.load(bestfilename)
            if state['best_acc_test'] > existing_bestfile['best_acc_test']:
                shutil.copyfile(filename, bestfilename)
                return
        else:
            shutil.copyfile(filename, bestfilename) 

def save_model_iteration(G, F1, step, args, optimizer_g, optimizer_f):
    checkpoint_name = str(args.net + "_" + args.source + "_" + args.target + "_" + args.which_method + "_" + str(step) + ".ckpt.pth.tar")
    checkpoint_path = os.path.join(args.checkpath,checkpoint_name)
    checkpoint_dict = {"G_state_dict": G.state_dict(),
                      "F1_state_dict": F1.state_dict(),
                      "optimizer_g": optimizer_g.state_dict(),
                      "optimizer_f": optimizer_f.state_dict()}
    torch.save(checkpoint_dict, checkpoint_path)



def update_features(feat_dict, data, G, momentum, source  = False):
    '''
    Description:
    1. This function updates the feature bank using the reprsentations of the current batch
    Inputs:
    1. feat_dict - the bank of representations taken after the image has passed through G
    2. data - the current batch
    3. G - the feature extraction vector
    4. momentum - the momentum with which we could use to update the feature bank
    5. source [bool] - whether the batch passed is of the source or not the source
    Outputs:
    1. f_batch - the reprsentations of the batch, to save further computation
    2. feat_dict - the updated feature dictionary
    '''
    names_batch = data[2]
    if not source:
        img_batch = data[0][0].cuda()
    else:
        img_batch = data[0].cuda() 
    names_batch = list(names_batch)
    idx = [feat_dict.names.index(name) for name in names_batch]
    f_batch = G(img_batch)
    #print(f_batch.shape)
    #print(len(idx))
    #print(feat_dict.feat_vec[idx])
    feat_dict.feat_vec[idx] = (momentum * feat_dict.feat_vec[idx] + (1 - momentum) * f_batch).detach()
    return f_batch, feat_dict

def update_label_bank(label_bank, data, pseudo_labels, mask_loss):
    '''
    Description:
    This function continuously updates the label bank using the predicted labels of the current batch
    Inputs: 
    1. Label bank - which is 1D tensor of length as the number of target examples
    2. data - which is the batch of the unlabeled target data
    3. pseudo_labels - this is the pseudo label which is calculated for that batch of data
    4. mask_loss - this gives the examples in the batch which are confident 
    Output: None as of now, can be customized according to requirements
    '''
    names_batch = data[2]
    names_batch = list(names_batch)
    names_batch_confident = []
    #names_batch_unconfident = []
    pseudo_labels_confident = []
    # Use only names with confidence greater than 0.9
    mask_loss_list = list(mask_loss.cpu().detach().numpy().astype(int))
    for i, item in enumerate(mask_loss_list):
        if not item == 0:
            names_batch_confident.append(names_batch[i])
            pseudo_labels_confident.append(pseudo_labels[i])
        #else:
            #names_batch_unconfident.append(names_batch[i])

    idx = [label_bank.names.index(name) for name in names_batch_confident]
    #idx_unconfident = [label_bank.names.index(name) for name in names_batch_unconfident]
    label_bank.labels[idx] =  np.array(pseudo_labels_confident)
    #label_bank.labels[idx_unconfident] = np.ones(len(idx_unconfident))* -1
    return 0

def pairwise_distance(feat_bank, feat_batch):
    '''
    Description: 
    Calculates the pairwise euclidian distance between a batch of incoming features and a global feature bank. 
    Inputs:
    feat_batch - batch_size x feat_dim
    feat_bank - total_examples_num x feat_dim
    Outputs:
    output - total_examples x batch_size
    '''
    batch_size = feat_batch.shape[0]
    total_examples_num = feat_bank.shape[0]
    output = torch.zeros(total_examples_num, batch_size)
    print(output.shape)
    for idx_im_batch in range(batch_size):
        for idx_im_bank in range(total_examples_num):
            eu_dist = torch.sum(torch.pow(feat_batch[idx_im_batch] - feat_bank[idx_im_bank],2),dim=0)
            #sys.exit()
            output[idx_im_bank][idx_im_batch] = eu_dist
    return output

def get_similarity_distribution(feat_dict,data_batch, G, source = False, i=0, mode='cosine'):
    if source:
        img_batch  = data_batch[0].cuda()
    else:
        img_batch = data_batch[0][i].cuda()
    f_batch = G(img_batch)
    if mode == 'cosine':
        #print(feat_dict.feat_vec.shape,f_batch.shape)
        sim_distribution  = torch.mm(F.normalize(feat_dict.feat_vec, dim=1),F.normalize(torch.transpose(f_batch,0,1),dim = 0))        
    elif mode == 'euclid':
        sim_distribution = pairwise_distance(feat_dict.feat_vec, f_batch)
    sim_distribution = edict({"cosines": sim_distribution, "names": data_batch[2], "labels": data_batch[1]})
    return f_batch, sim_distribution


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
    cluster_ids_x, cluster_centers = kmeans(X=vectors, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0'))
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
    for _, batch in enumerate(loader):
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

def load_bank(args,mode = 'pkl'):
    if mode == 'pkl':
        f = open("./banks/%s_unlabelled_target_%s_%s.pkl"%(args.net,args.target,args.num), "rb")
        feat_dict_target = edict(pickle.load(f))
        feat_dict_target.feat_vec = feat_dict_target.feat_vec.cuda()
        num_target = len(feat_dict_target.names)
        domain = ["T" for i in range(num_target)]
        feat_dict_target.domain_identifier = domain

        f = open("./banks/%s_labelled_source_%s.pkl"%(args.net,args.source), "rb") # Loading the feature bank for the source samples
        feat_dict_source = edict(pickle.load(f))
        feat_dict_source.feat_vec  = feat_dict_source.feat_vec.cuda() 
        num_source = len(feat_dict_source.names)
        domain = ["S" for i in range(num_source)]
        feat_dict_source.domain_identifier = domain
        # Concat the corresponsing components of the 2 dictionaries
        print(feat_dict_source.feat_vec.shape)
        print(feat_dict_target.feat_vec.shape)
        feat_dict_combined = edict({})
        feat_dict_combined  = combine_dicts(feat_dict_source, feat_dict_target)

    elif mode == 'random':
        feat_dict_target = edict({})
        f_target = open(os.path.join("./data/txt/%s"%(args.dataset),"unlabeled_target_images_%s_%s.txt"%(args.target,str(args.num))),"r")
        names_target, labels_target = [],[]
        for line in f_target:
            line = line.replace("\n","")
            names_target.append(line.split()[0])
            labels_target.append(int(line.split()[1]))

        feat_dict_target.names = names_target
        feat_dict_target.labels = labels_target
        feat_dim = 512 if args.net == 'resnet34' else 4096
        feat_dict_target.feat_vec = torch.randn(len(names_target),feat_dim).cuda()
        num_target = len(feat_dict_target.names)
        domain = ["T" for i in range(num_target)]
        feat_dict_target.domain_identifier = domain

        feat_dict_source = edict({})
        f_source = open(os.path.join("./data/txt/%s"%(args.dataset),"labeled_source_images_%s.txt"%(args.source)),"r")
        
        names_source,labels_source = [],[]
        for line in f_source:
            line = line.replace("\n","")
            names_source.append(line.split()[0])
            labels_source.append(int(line.split()[1]))
        feat_dict_source.feat_vec = torch.randn(len(names_source),feat_dim).cuda()
        feat_dict_source.names = names_source
        feat_dict_source.labels = labels_source
        num_source = len(feat_dict_source.names)
        domain = ["S" for i in range(num_source)]
        feat_dict_source.domain_identifier = domain
        # Concat the corresponsing components of the 2 dictionaries
        feat_dict_combined = edict({})
        feat_dict_combined  = combine_dicts(feat_dict_source, feat_dict_target)
        
    print("Bank keys - Target: ", feat_dict_target.keys(),"Source: ", feat_dict_source.keys())
    print("Num  - Target: ", len(feat_dict_target.names), "Source: ", len(feat_dict_source.names))

    return feat_dict_source, feat_dict_target, feat_dict_combined
    
def do_method_bank(feat_dict_source, feat_dict_target, feat_dict_combined, momentum, data_t_unl, data_s, prob_weak_aug, thresh, K, pred_strong_aug, criterion_pseudo, target_loader_unl, G, F1, backprop = True):
    f_batch_target, feat_dict_target  = update_features(feat_dict_target, data_t_unl, G, momentum)
    f_batch_target = f_batch_target.detach()
    
    f_batch_source, feat_dict_source  = update_features(feat_dict_source, data_s, G, momentum, source = True)
    f_batch_source = f_batch_source.detach()
    # Get max of similarity distribution to check which element or label is it closest to in these vectors
    feat_dict_combined = combine_dicts(feat_dict_target, feat_dict_source)
    sim_distribution = get_similarity_distribution(feat_dict_combined,data_t_unl,G)
    k_neighbors, labels_k_neighbors = get_kNN(sim_distribution, feat_dict_combined, K)    
    #mask_loss_uncertain = (prob_weak_aug.max(1)[0]<thresh) & (prob_weak_aug.max(1)[0]>0.7)
    mask_loss_uncertain = prob_weak_aug.max(1)[0]>thresh
    knn_majvot_pseudo_labels = get_majority_vote(k_neighbors,feat_dict_combined, K, F1, mask_loss_uncertain, len(target_loader_unl.dataset))
    #loss_pseudo_unl_knn = torch.mean(mask_loss
    # _uncertain.int() * criterion_pseudo(pred_strong_aug, knn_majvot_pseudo_labels))
    if backprop:
        if  not torch.sum(mask_loss_uncertain.int()) == 0:
            loss_pseudo_unl_knn = torch.sum(mask_loss_uncertain.int() * criterion_pseudo(pred_strong_aug, knn_majvot_pseudo_labels))/(torch.sum(mask_loss_uncertain.int()))
            loss_pseudo_unl_knn.backward(retain_graph=True)
    return mask_loss_uncertain

def get_per_class_examples(label_bank, class_list):
    classes_idx = np.arange(len(class_list))
    class_num_list = np.zeros(len(class_list),dtype=np.int32)
    for i, class_ in enumerate(classes_idx):
        for l in label_bank.labels:
            if l == class_:
                class_num_list[i] +=1
    return class_num_list

def set_source_weights_batch(feat_dict_source, K_farthest_source, k_neighbors):
    b_size = len(k_neighbors)
    for example in range(b_size):
        for neighbor in range(K_farthest_source):
            feat_dict_source.sample_weights[neighbor] = 0.7
    #names_batch = list(data[2])
    #idx = [feat_dict_source.names.index(name) for name in names_batch] 
    #weights_source = feat_dict_source.sample_weights[idx]
    #return weights_source
    pass

def set_source_weights_all(target_loader,feat_dict_source,K_farthest_source,k_neighbors,G):
    for data_t in target_loader:
        f_batch_source, sim_distribution = get_similarity_distribution(feat_dict_source,data_t,G)
        sim_distribution.cosines = 1 - sim_distribution.cosines
        k_neighbors, labels_k_neighbors = get_kNN(sim_distribution, feat_dict_source, K_farthest_source)    
        set_source_weights_batch(feat_dict_source,K_farthest_source,k_neighbors)
    pass

