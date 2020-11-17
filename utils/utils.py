import os
from os import name
import torch
import torch.nn as nn
import shutil


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

def update_features(feat_dict, data_t_unl, G, momentum):
    names_batch = data_t_unl[2]
    img_batch = data_t_unl[0][0]
    names_batch = list(names_batch)
    idx = [feat_dict.names.index(name) for name in names_batch]
    feat_dict.feat_vec[idx] = momentum * feat_dict.feat_vec[idx] + (1 - momentum) * G(img_batch)
    return feat_dict