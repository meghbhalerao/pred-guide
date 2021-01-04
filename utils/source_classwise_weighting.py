from numpy.lib.utils import source
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


def get_kNN(sim_distribution, feat_dict, k = 1):
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
        sim_distribution.cosines = 1 - sim_distribution.cosines
        k_farthest, labels_k_farthest, names_k_farthest = get_kNN(sim_distribution, feat_dict, K_farthest)
        return k_farthest, labels_k_farthest, names_k_farthest

def do_source_weighting(loader, feat_dict,G,K_farthest,weight=0.8,aug = 0, only_for_poor = False, poor_class_list = None, weighing_mode='F'):
    class_wise_examples = edict({"names":[],"labels":[]})
    n_examples = len(feat_dict.domain_identifier)
    feat_dict.sample_weights = torch.tensor(np.ones(n_examples)).cuda()
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
        names_k = names_k[0] # 0 - since batch_size is 1 for 
        for name in names_k:
            idx_to_weigh = feat_dict.names.index(name)
            if only_for_poor:
                if img_label in poor_class_list:
                    feat_dict.sample_weights[idx_to_weigh] = weight
            else:
                feat_dict.sample_weights[idx_to_weigh] = weight  
    return class_wise_examples        

def do_lab_target_loss(label_bank,class_list,G,F1,data_t,im_data_t, gt_labels_t, criterion_lab_target,beta=0.99,mode='cbfl'):
    if mode == 'cbfl':
        class_num_list = get_per_class_examples(label_bank, class_list)
        effective_num = 1.0 - np.power(beta, class_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(class_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
        criterion_lab_target = CBFocalLoss(weight=per_cls_weights, gamma=0.5).cuda()
    elif mode == 'ce':
        pass
    for i in range(len(data_t[0])):
        im_data_t = data_t[0][i]
        out_lab_target = F1(G(im_data_t))
        loss_lab_target = criterion_lab_target(out_lab_target,gt_labels_t)
        loss_lab_target.backward()

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

"""
print(os.path.join(root_folder,image))
img = pil_loader(os.path.join(root_folder,image))
gt = classwise.labels[idx]
_,strong,_ = augmentation(img)
strong = strong.cuda()
"""









