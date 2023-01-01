import torch
import numpy as np

def do_fixmatch(data_t_unl,F1,G,thresh,criterion_pseudo, class_acc):
    im_data_tu_weak_aug, im_data_tu_strong_aug = data_t_unl[0][0].cuda(), data_t_unl[0][1].cuda()
    # Getting predictions of weak and strong augmented unlabled examples
    pred_strong_aug = F1(G(im_data_tu_strong_aug))
    with torch.no_grad():
        pred_weak_aug = F1(G(im_data_tu_weak_aug))
    prob_weak_aug = torch.softmax(logits_w, dim=-1)
    max_probs, pseudo_labels = torch.max(prob_weak_aug, dim=-1)
    mask_loss = max_probs.ge(thresh * (class_acc[pseudo_labels] / (2. - class_acc[pseudo_labels]))).float()  # convex
    #try:
    loss_pseudo_unl = torch.mean(mask_loss.int() * criterion_pseudo(pred_strong_aug,pseudo_labels))
    loss_pseudo_unl.backward(retain_graph=True)
    #except:
    #pass
    return pseudo_labels, mask_loss