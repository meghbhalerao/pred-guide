import numpy as np
import torch 
import matplotlib.pyplot as plt
import torch.nn.functional as F


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
    print("Total datapoints to plot are:", data_length)
    set_near_fathest(feat_dict) # finds the near and far samples from the labeled target to the source and changes the tag indicator accordingly
    for idx in range(data_length):
        print(idx)
        if feat_dict.tag_global[idx] == 's':
            plt.scatter(x[idx],y[idx],color='blue')
        elif feat_dict.tag_global[idx] == 'u' and feat_dict.is_correct_label == 1:
            plt.scatter(x[idx],y[idx],color='green')
        elif feat_dict.tag_global[idx] == 'u' and feat_dict.is_correct_label == 0:
            plt.scatter(x[idx],y[idx],color='red')  
        elif feat_dict.tag_global[idx] == 'n':
            plt.scatter(x[idx],y[idx], color='navy')
        elif feat_dict.tatag_global[idx] == 'f':
            plt.scatter(x[idx],y[idx], color = 'lightblue')
        
    plt.show()

def set_near_fathest(feat_dict):
    data_length  = len(feat_dict.gt_labels)
    idx_lt = [i for i in range(data_length) if feat_dict.tag_global ==  'lt']
    idx_s = [i for i in range(data_length) if feat_dict.tag_global ==  's']
    feat_labeled_target = torch.tensor(feat_dict.feat_global[idx_lt]).cuda()
    feat_source = torch.tensor(feat_dict.feat_global[idx_s]).cuda()


def get_similarity_distribution_2(feat_dict,data_batch, G, source = False, i=0, mode='cosine'):
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


