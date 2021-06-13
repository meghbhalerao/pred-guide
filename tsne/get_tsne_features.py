#from tsnecuda import TSNE
import torch
import numpy as np
import sys
sys.path.append('/home/megh/projects/domain-adaptation/SSAL/')
from tsne.tsne_utils import * 
from torchvision.models import resnet
from loaders.data_list import Imagelists_VISDA, return_classlist
from model.basenet import *
from model.resnet import *
from torchvision import transforms
from utils.return_dataset import ResizeImage
#from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.manifold import TSNE
import time
import os
from torch.autograd import Variable
from easydict import EasyDict as edict
import pickle 
from utils.return_dataset import *

def main():
    # Defining return dataset function here
    args  = edict({})
    args.net = "alexnet"
    args.root = '../data/multi/'
    args.source = "painting"
    args.target = "clipart"
    args.dataset = "multi"
    args.num  = 3
    args.uda = 1
    it = 7500                                       
    source_loader, target_loader, target_loader_misc, target_loader_unl, target_loader_val, target_loader_test, class_num_list_source, class_list = return_dataset_randaugment(args,txt_path = "../data/txt/",root_path="../data/",bs_alex=1,bs_resnet=1,set_shuffle=False)
    is_dict_saved = False
    cf_saved = False

    print("Unlabeled source dataset size:", len(source_loader.dataset))
    print("Labeled target dataset size:",len(target_loader_unl.dataset))

    # Defining the pytorch networks
    if args.net == 'resnet34':
        G = resnet34()
        inc = 512
        F1 = Predictor_deep(num_class=len(class_list), inc=inc)

    elif args.net == "alexnet":
        G = AlexNetBase()
        inc = 4096
        F1 = Predictor(num_class=len(class_list), inc=inc)
    else:
        raise ValueError('Model cannot be recognized.')

    # Loading the weights from the checkpoint
    model_path = "../save_model_ssda/%s_%s_%s_SEW_%s.ckpt.pth.tar"%(args.net,args.source, args.target, str(it))
    main_dict = torch.load(model_path)
    print(main_dict.keys())
    G.load_state_dict(main_dict["G_state_dict"])
    F1.load_state_dict(main_dict["F1_state_dict"])

    G.cuda();F1.cuda()

    tag_unl_target =  'u'
    tag_lab_target  =  'lt'
    tag_source =  's'

    features_dict = edict({"feat_global":[],"tag_global":[], "is_correct_label":[],"name_list":[],"gt_labels":[]})

    if not cf_saved:
        cf = test(target_loader_misc,G,F1,class_list)
        np.save("confusion_matrix_tsne_%s.npy"%(str(it)),cf)
    elif cf_saved:
        cf = np.load("confusion_matrix_tsne_%s.npy"%(str(it)))
    pc_acc = per_class_accuracy(cf)
    print("Per Class Accuracy on the unlabled target data is: ", pc_acc)
    sys.exit()
    classes_to_plot = [1]#range(1,len(class_list))
    if not is_dict_saved:
        with torch.no_grad():
            G.eval()
            F1.eval()
            # getting the features for the source domain
            for idx, image_obj in enumerate(source_loader):
                image = image_obj[0].cuda()
                gt_label = image_obj[1].cpu().data.item()
                if gt_label in classes_to_plot:
                    output = G(image)
                    output = torch.flatten(output)
                    output = output.cpu().numpy()
                    features_dict.feat_global.append(output)
                    features_dict.tag_global.append(tag_source)
                    features_dict.is_correct_label.append(1)
                    features_dict.name_list.append(image_obj[2][0])
                    features_dict.gt_labels.append(gt_label)
                    pred_label = F1(G(image)).data.max(1)[1].cpu().data.item()
                    print("GT Label Source:", gt_label)
                    print("Pred Label Source:", pred_label)
                if gt_label == classes_to_plot[0] + 1:
                    break

            print("Finished source features")               
            # getting the features for the unlabeled target data
            for idx, image_obj in enumerate(target_loader_test):
                image = image_obj[0].cuda()
                gt_label  = image_obj[1].cpu().data.item()
                if gt_label in classes_to_plot:
                    features_dict.gt_labels.append(gt_label)
                    output = G(image)
                    output = torch.flatten(output)
                    output = output.cpu().numpy()
                    features_dict.feat_global.append(output)
                    features_dict.tag_global.append(tag_unl_target)
                    pred_label = F1(G(image)).max(1)[1].cpu().data.item()
                    print("GT Label Unl Tar:", gt_label)
                    print("Pred Label Unl Tar:", pred_label)
                    features_dict.is_correct_label.append(int(pred_label == gt_label))
                    features_dict.name_list.append(image_obj[2][0])
                if gt_label == classes_to_plot[0] + 1:
                    break
            print("Finished Unlabeled Target Features")
            
            # getting the features for the labeled target data
            for idx, image_obj in enumerate(target_loader_misc):
                image = image_obj[0][0].cuda()
                gt_label  = image_obj[1].cpu().data.item()
                if gt_label in classes_to_plot:
                    features_dict.gt_labels.append(gt_label)
                    output = G(image)
                    output = torch.flatten(output)
                    output = output.cpu().numpy()
                    features_dict.feat_global.append(output)
                    features_dict.tag_global.append(tag_lab_target)
                    pred_label = F1(G(image)).max(1)[1].cpu().data.item()
                    features_dict.is_correct_label.append(1)
                    features_dict.name_list.append(image_obj[2][0])    
                    print(gt_label)
                if gt_label == classes_to_plot[0] + 1:
                    break
            
            print("Finished Labeled Target Features")

        with open('features_dict_%s_%s_%s.pickle'%(args.source,args.target,str(it)), 'wb') as handle:
            pickle.dump(features_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    elif is_dict_saved:
        print("Loading feature dict from saved file ....")
        with open('features_dict_%s_%s_%s.pickle'%(args.source,args.target,str(it)), 'rb') as handle:
            features_dict = pickle.load(handle)
    
    features_dict.feat_global = np.array(features_dict.feat_global)
    print("Shape of features to be plotted on tSNE is : ", features_dict.feat_global.shape)
    print("Saving features to plot ...")
    np.save('features.npy', features_dict.feat_global)
    print("Length of the label dict is, i.e. the total S + T size",len(features_dict.gt_labels))
    labels = np.array(features_dict.gt_labels)
    print("Saving the labels ...")
    np.save('labels.npy',labels)

    tsne = TSNE(perplexity = 5, n_iter = 3000, verbose=1)
    X_embedded = tsne.fit_transform(features_dict.feat_global)
    np.save('tsne_embeddings.npy', X_embedded)

    plot_tsne_figure(X_embedded,features_dict)


def test(loader,G,F1,class_list):
    G.eval()
    F1.eval()
    test_loss = 0
    correct = 0
    size = 0
    num_class = len(class_list)
    output_all = np.zeros((0, num_class))

    im_data_t = torch.FloatTensor(1)
    im_data_t = im_data_t.cuda()
    im_data_t = Variable(im_data_t)

    gt_labels_t = torch.LongTensor(1)
    gt_labels_t = gt_labels_t.cuda()
    gt_labels_t = Variable(gt_labels_t)

    confusion_matrix = torch.zeros(num_class, num_class)
    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            #im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
            #gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])
            im_data_t = data_t[0][0].cuda()
            gt_labels_t = data_t[1].cuda()
            feat = G(im_data_t)
            output1 = F1(feat)
            output_all = np.r_[output_all, output1.data.cpu().numpy()]
            size += im_data_t.size(0)
            pred1 = output1.data.max(1)[1]
            print("Predicted Label:", pred1)
            print("Ground Truth Label:", gt_labels_t)
            for t, p in zip(gt_labels_t.view(-1), pred1.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += pred1.eq(gt_labels_t.data).cpu().sum()
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} F1 ({:.0f}%)\n'.format(test_loss, correct, size, 100. * correct / size))
    np.save("confusion_matrix_tsne.npy", confusion_matrix)
    return confusion_matrix


if __name__ == "__main__":
    main()
