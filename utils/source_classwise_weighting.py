from numpy.lib.utils import source
from numpy.testing._private.utils import break_cycles
import torch 
import numpy as np
import os
import sys 
from utils.utils import *
from easydict import EasyDict as edict
from utils.loss import *
from utils.return_dataset import *
from PIL import Image
from loaders.data_list import Imagelists_VISDA
import torch.nn.functional as F
import math 

def get_kNN(sim_distribution, feat_dict, k = 1):
    '''
    Description: 
    Gets the K nearest examples for every example in the incoming batch. 
    Inputs:
    1. sim_distribution - this is the list of cosine similarities for every example in the batch which is pre-calculated
    2. feat_dict - dictionary which consists of the feature bank and corresponding metadata
    3. k - the number of k nearest neighbors
    Outputs:
    1. k_neighbors - the repersentations of the k nearest neighbors
    2. labels_k_neighbors - the predicted labels of the nearest neighbors
    3. names_k_neighbors -  the names/paths of the images which correspond to the nearset neighbors
    '''
    k_neighbors = torch.topk(torch.transpose(sim_distribution.cosines,0,1), k, dim = 1)
    idxs = k_neighbors[1]
    b_size = idxs.shape[0]
    k_neighbors = []
    labels_k_neighbors = []
    names_k_neighbors = []
    for i in range(b_size):
        l = idxs[i].cpu().data.numpy()
        k_neighbors.append(list(l))
        labels_k_neighbors.append(list(np.array(feat_dict.labels)[l]))
        names_k_neighbors.append([feat_dict.names[idx] for idx in list(l)])
    return k_neighbors, labels_k_neighbors, names_k_neighbors

def make_feat_dict_from_idx(feat_dict,idxs):
    feat_dict_idx = edict({})
    feat_dict_idx.feat_vec = feat_dict.feat_vec[idxs]
    feat_dict_idx.labels = [feat_dict.labels[idx] for idx in idxs]
    feat_dict_idx.names = [feat_dict.names[idx] for idx in idxs]
    feat_dict_idx.domain_identifier = [feat_dict.domain_identifier[idx] for idx in idxs]
    return feat_dict_idx

def get_k_farthest_neighbors(sim_distribution,feat_dict,K_farthest):
        '''
        Description: 
        Gets the K farthest examples for every example in the incoming batch. 
        Inputs:
        1. sim_distribution - this is the list of cosine similarities for every example in the batch which is pre-calculated
        2. feat_dict - dictionary which consists of the feature bank and corresponding metadata
        3. K_farthest - the number of k nearest neighbors
        Outputs:
        1. k_farthest - the repersentations of the k farthest neighbors
        2. labels_k_farthest - the predicted labels of the farthest neighbors
        3. names_k_farthest -  the names/paths of the images which correspond to the farthest neighbors
        '''   
        sim_distribution.cosines = 1 - sim_distribution.cosines
        k_farthest, labels_k_farthest, names_k_farthest = get_kNN(sim_distribution, feat_dict, K_farthest)
        return k_farthest, labels_k_farthest, names_k_farthest

def do_source_weighting(loader, feat_dict, G, K_farthest,per_class_raw = None, weight=0.8, aug = 0, phi = 0.2, only_for_poor = False, poor_class_list = None, weighing_mode='F', weigh_using = 'pseudo_labels'):
    if weigh_using == 'pseudo_labels':
        min_raw = np.min(per_class_raw)
        max_raw = np.max(per_class_raw)
        per_class_raw = (per_class_raw - min_raw)/(max_raw - min_raw + 10**-5)
    elif weigh_using == 'target_acc':
        pass

    if per_class_raw is not None:
        if weighing_mode == 'N':
            per_class_weights = 1 * (1 + phi/np.exp(per_class_raw))
        elif weighing_mode == 'F':
            per_class_weights = 1 * (1 - phi/np.exp(per_class_raw))
    
        per_class_weights = torch.tensor(per_class_weights)
        print("Per cls weights according to the accuracy are: ", per_class_weights)
    class_wise_examples = edict({"names":[],"labels":[]})
    n_examples = len(feat_dict.domain_identifier)
    for idx, batch in enumerate(loader):
        #img_vec = G(batch[0][aug])
        print(idx)
        img_label = batch[1]
        idxs_label = [i for i, x in enumerate(feat_dict.labels) if x == img_label]
        feat_dict_label = make_feat_dict_from_idx(feat_dict,idxs_label)
        f_batch, sim_distribution = get_similarity_distribution(feat_dict_label,batch,G,i=aug)

        if weighing_mode == 'F':
            k, labels_k, names_k = get_k_farthest_neighbors(sim_distribution,feat_dict_label,K_farthest)
        elif weighing_mode == 'N':
            k_nearest, labels_k_nearest, names_k = get_kNN(sim_distribution,feat_dict_label,K_farthest)   
            class_wise_examples.names.extend(names_k[0])
            class_wise_examples.labels.extend(labels_k_nearest[0])

        #print(names_k)
        names_k = names_k[0] # 0 - since batch_size is 1 for 
        for name in names_k:
            idx_to_weigh = feat_dict.names.index(name)
            if only_for_poor:
                if img_label in poor_class_list:
                    if per_class_raw is None:
                        feat_dict.sample_weights[idx_to_weigh] = weight
                    else:
                        feat_dict.sample_weights[idx_to_weigh] = per_class_weights[img_label]

            else:
                if per_class_raw is None:
                    feat_dict.sample_weights[idx_to_weigh] = weight  
                else:
                    feat_dict.sample_weights[idx_to_weigh] = per_class_weights[img_label]
        #break
    return class_wise_examples        

def do_lab_target_loss(G,F1,data_t,im_data_t, gt_labels_t, criterion_lab_target):
    #for i in range(len(data_t[0])):
    #im_data_t = data_t[0][i]
    feat_lab = G(im_data_t)
    out_lab_target = F1(feat_lab)
    try:
        loss_lab_target = criterion_lab_target(out_lab_target,gt_labels_t)
        loss_lab_target.backward()
    except:
        pass
    return feat_lab.clone().detach()

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def make_st_aug_loader(args,classwise,root_folder="./data/multi/"):
    file_path = "source_near_images_%s.txt"%(args.source)
    
    if args.augmentation_policy == 'rand_augment':
        augmentation  = TransformFix("randaugment",args.net,mean =[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    f = open(file_path,"w")
    for idx, image in enumerate(classwise.names):
        f.write(str(image) + " " + str(classwise.labels[idx]) + "\n")
    source_dataset = Imagelists_VISDA(os.path.join(file_path), root=root_folder, transform=augmentation,test=False)
    source_strong_near_loader = torch.utils.data.DataLoader(source_dataset,batch_size=1,num_workers=3,shuffle=False,drop_last=False)
    print(len(source_strong_near_loader))
    return iter(source_strong_near_loader)
        
def update_loss_functions(args,label_bank, class_list, class_num_list_pseudo=None, class_num_list_source = None, beta=0.99,gamma=0):
    if class_num_list_pseudo is None:
        class_num_list_pseudo = get_per_class_examples(label_bank, class_list) + args.num
        print("Pred num ex per class (pseudo labels + labelled target examples): ", class_num_list_pseudo)
        
    if class_num_list_source is not None:
        class_num_list =  class_num_list_pseudo + np.array(class_num_list_source)
    else:
        class_num_list = class_num_list_pseudo

    effective_num = 1.0 - np.power(beta, class_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(class_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
    
    criterion = CBFocalLoss(weight=per_cls_weights, gamma=gamma, reduction='none').cuda()
    criterion_pseudo = CBFocalLoss(weight=per_cls_weights, gamma=gamma, reduction='none').cuda()
    criterion_lab_target = CBFocalLoss(weight=per_cls_weights, gamma=gamma,reduction='mean').cuda()
    criterion_strong_source = CBFocalLoss(weight=per_cls_weights, gamma=gamma,reduction='mean').cuda()
    print("CBFL per zclass weights:", per_cls_weights)
    return criterion, criterion_pseudo, criterion_lab_target, criterion_strong_source
    
def get_per_class_weight_matrix(confusion_matrix):
    num_class, _ = confusion_matrix.shape
    per_class_weight_matrix = []
    for i in range(num_class):
        per_class_weight_list = []
        for j in range(num_class):
            per_class_weight_list.append(confusion_matrix[i,j]/sum(confusion_matrix[i,:]))
        per_class_weight_matrix.append(per_class_weight_list)
    per_class_weight_matrix = np.array(per_class_weight_matrix)
    return per_class_weight_matrix

def prototype_reg(args,G,F1,confusion_matrix,mode="euclid"):
    if args.net == "resnet34":
        P = F1.fc2.cpu()
    elif args.net == "alexnet":
        P = F1.fc.cpu()
    P = F.normalize(P,dim=1)
    per_class_weight_matrix = get_per_class_weight_matrix(confusion_matrix)
    loss_reg = 0
    n_class, n_class = confusion_matrix.shape
    for anchor_class in range(n_class):
        class_weight = per_class_accuracy[anchor_class]
        for class_ in range(n_class):
            if mode == "euclid":
                loss_reg  = loss_reg + torch.sum(torch.pow(P[anchor_class,:] - P[class_,:],2),dim=1)
            elif mode == "cosine":
                loss_reg = loss_reg + torch.dot(P[anchor_class,:],P[class_,:],dim = 0)        
    loss_reg.backward()
    pass
"""
print(os.path.join(root_folder,image))
img = pil_loader(os.path.join(root_folder,image))
gt = classwise.labels[idx]
_,strong,_ = augmentation(img)
strong = strong.cuda()
"""
