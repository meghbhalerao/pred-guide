import torch
from utils.utils import *

def do_domain_classification(D,feat_disc_source, feat_disc_tu, feat_disc_t, gt_labels_s,gt_labels_t,gt_labels_tu, pseudo_labels, criterion_discriminator,optimizer_d, mode='all'):
    if mode == 'all':
        prob_domain_source = D(feat_disc_source)
        prob_domain_target = D(feat_disc_tu)
        prob_domain_lab_target = D(feat_disc_t)

        gt_source = gt_labels_s.clone().detach() * 0
        gt_target_lab = gt_labels_t.clone().detach() * 0 + 1
        gt_target_unl = gt_labels_tu.clone().detach() * 0 + 1

        loss_domain_source = criterion_discriminator(prob_domain_source, gt_source)
        loss_domain_target = criterion_discriminator(prob_domain_target, gt_target_unl)
        loss_domain_lab_target = criterion_discriminator(prob_domain_lab_target,gt_target_lab)

        loss_total = loss_domain_source + loss_domain_target + loss_domain_lab_target

        loss_total.backward()
        optimizer_d.step()
        optimizer_d.zero_grad()
        D.zero_grad()
    elif mode == 'classwise':
        gt_source = gt_labels_s.clone().detach() * 0
        gt_target_lab = gt_labels_t.clone().detach() * 0 + 1
        gt_target_unl = gt_labels_tu.clone().detach() * 0 + 1 
        class_sources = gt_source
        class_lab_targets = gt_target_lab
        class_targets = pseudo_labels

        prob_domain_source = D(feat_disc_source,reverse=False,eta=1.0,choose_class=4)

def do_probability_weighing(G,D,source_loader,feat_dict):
    for idx, batch in enumerate(source_loader):
        names_batch = list(batch[2])
        indexes = [feat_dict.names.index(name) for name in names_batch]
        probablities_weight = F.softmax(D(G(batch[0])),dim=1)
        probability_target = probablities_weight[:,1]
        feat_dict.sample_weights[indexes] = probability_target.detach().double().cpu()
    print("Done Probablity Weighing")