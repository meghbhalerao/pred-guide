#from tsnecuda import TSNE
import torch
import numpy as np
import sys
sys.path.append('/cbica/home/bhaleram/comp_space/random/personal/others/SSAL/')
from loaders.data_list import Imagelists_VISDA, return_classlist
from model.basenet import *
from model.resnet import *
from torchvision import transforms
from utils.return_dataset import ResizeImage
from MulticoreTSNE import MulticoreTSNE as TSNE
import time
import os
from torch.autograd import Variable
#from sklearn.decomposition import PCA


# Defining return dataset function here
net = "alexnet"
root = '../data/multi/'
source = "real"
target = "clipart"
n_class = 126
n_class_plot = 30
k = 10
image_list_easy = "./mme_easy_samples_unlabeled_target_images_%s_1.txt"%(target)
image_list_hard = "./mme_hard_samples_unlabeled_target_images_%s_1.txt"%(target)
image_list_source = "../data/txt/multi/labeled_source_images_%s.txt"%(source)
image_list_target_unl = "../data/txt/multi/unlabeled_target_images_%s_1.txt"%(target)
model_path = "../freezed_models/alexnet_mme_p2r.ckpt.best.pth.tar"
ours = False


def test(loader):
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
            im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
            gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])
            feat = G(im_data_t)
            output1 = F1(feat)
            output_all = np.r_[output_all, output1.data.cpu().numpy()]
            size += im_data_t.size(0)
            pred1 = output1.data.max(1)[1]
            for t, p in zip(gt_labels_t.view(-1), pred1.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += pred1.eq(gt_labels_t.data).cpu().sum()
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} F1 ({:.0f}%)\n'.format(test_loss, correct, size, 100. * correct / size))
    return confusion_matrix


def get_dataset(net,root,image_set_file_test):
    if net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    target_dataset_unl = Imagelists_VISDA(image_set_file_test, root=root,
                                            transform=data_transforms['test'],
                                            test=True)
    class_list = return_classlist(image_set_file_test)
    num_images = len(target_dataset_unl)
    if net == 'alexnet':
        bs = 1
    else:
        bs = 1

    target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl, batch_size=bs, num_workers=3,shuffle=False, drop_last=False)
    return target_loader_unl,class_list

target_loader_unl, _ = get_dataset(net,root,image_list_target_unl)
target_loader_unl_easy,class_list = get_dataset(net,root,image_list_easy)
target_loader_unl_hard,class_list = get_dataset(net,root,image_list_hard)
source_loader, _ = get_dataset(net,root,image_list_source)
print(len(target_loader_unl_hard.dataset))
print(len(target_loader_unl_easy.dataset))
print(len(source_loader.dataset))
print(len(target_loader_unl.dataset))
print(len(target_loader_unl_hard.dataset) + len(target_loader_unl_easy.dataset) - len(target_loader_unl.dataset))


# Deinfining the pytorch networks
if net == 'resnet34':
    G = resnet34()
    inc = 512
elif net == 'resnet50':
    G = resnet50()
    inc = 2048
elif net == "alexnet":
    G = AlexNetBase()
    inc = 4096
elif net == "vgg":
    G = VGGBase()
    inc = 4096
else:
    raise ValueError('Model cannot be recognized.')


if ours: 
    if net == 'resnet34':
        F1 = Predictor_deep_2(num_class=n_class,inc=inc,feat_dim = 50)
        print("Using: Predictor_deep_attributes")
    else:
        F1 = Predictor_attributes(num_class=n_class,inc=inc,feat_dim = 50)
        print("Using: Predictor_attributes")

else:
    F1 = nn.Linear(inc,4)


G.cuda()
F1.cuda()

#checkpoint = torch.load(model_path)
#f = open("/cbica/home/bhaleram/comp_space/random/personal/iisc_project/MME/entropy_weight_files/MME_alexnet_real_painting_0.txt")


# Loading the weights from the checkpoint
G.load_state_dict(torch.load("../save_model_ssda/G_iter_model_clipart_step_2000.pth.tar"))
F1.load_state_dict(torch.load("../save_model_ssda/F1_iter_model_clipart_step_2000.pth.tar"))

features = []   
labels = []
weights = []

# Read the weight file and converting it to a dictionary
#weight_dict = {}
#for line in f:
#    line_list = line.split()
#    weight_dict[line_list[1]] = float(line_list[0])


# Get classwise accuracies

"""
cf = test(target_loader_unl)
per_cls_acc = []
for i in range(n_class):
    acc = (cf[i,i]/sum(cf[i,:])).data.item()
    per_cls_acc.append(acc)
np.save("cf.npy",cf)
print(per_cls_acc)
argsorted_list = list(np.argsort(per_cls_acc))
to_plot = argsorted_list[0:10]
print(to_plot)
"""
start = time.time()
# Features for easy examples
print(G)
L1 = torch.nn.Sequential(G.features[0], G.features[1],G.features[2], G.features[3],G.features[4], G.features[5],G.features[6], G.features[7], G.features[8])

print(L1)
with torch.no_grad():
    for idx, image_obj in enumerate(target_loader_unl):
        image = image_obj[0].cuda()
        label = image_obj[1].cpu().data.item()
        img_path = image_obj[2][0]
        #if label in to_plot:
        output = L1(image)
        output = torch.flatten(output)
        #print(output)
        #output = output.view(output.size(0), 256 * 6 * 6)
        #output = L2(output)
        output = output.cpu().numpy()
        #output = output[0]
        features.append(output)
        labels.append(label)
        weights.append(1)
        print(label)
        if label == n_class_plot:   
           break


end = time.time()
print((end-start)/60)

#prototype = 20 * F1.fc.weight.cpu().detach().numpy()
#print(prototype.shape)

"""
for i in range(n_class_plot):
    features.append(prototype[i,:])

for i in range(n_class_plot):
    weights.append(-50)
"""
"""
for idx, class_ in enumerate(to_plot):
    features.append(prototype[class_,:])
for i in range(len(to_plot)):
    weights.append(-50)
"""

features = np.array(features)
print(features.shape)
np.save('features.npy', features)
print(len(labels))
labels = np.array(labels)
np.save('labels.npy',labels)
print(len(weights))
weights = np.array(weights)
np.save('weights.npy', weights)

tsne = TSNE(perplexity = 25, n_jobs=1, n_iter = 3000, verbose=1)
X_embedded = tsne.fit_transform(features)
np.save('tsne_embeddings.npy', X_embedded)
print((end-start)/60)










# Getting which classes have most and least number of samples in both the source and target combined
"""
class_list.sort()
class_num_list = []
for class_ in class_list:
    class_num_list.append(len(os.listdir(os.path.join(root,source,class_))) + len(os.listdir(os.path.join(root,target,class_))))

sorted_list = [x for _,x in sorted(zip(class_num_list,class_list))]
print(sorted_list)
minority_k = sorted_list[0:k]
majority_k = sorted_list[n_class-k:n_class]

f = open(image_set_file_test,"r")
f_majority = open("unlabeled_target_images_%s_3_majority.txt"%(target), "w")
for line in f:
    for class_ in majority_k:
        if str(class_) in str(line):
            f_majority.write(str(line))

f_majority.close()

f = open(image_set_file_test,"r")
f_minority = open("unlabeled_target_images_%s_3_minority.txt"%(target), "w")
for line in f:
    for class_ in minority_k:
        if str(class_) in str(line):
            f_minority.write(str(line))
f_minority.close()

f.close()

target_loader_unl_minority,class_list = get_dataset(net,root,"./unlabeled_target_images_%s_3_minority.txt"%(target))
target_loader_unl_majority,class_list = get_dataset(net,root,"./unlabeled_target_images_%s_3_majority.txt"%(target))
"""




"""
with torch.no_grad():
    for idx, image_obj in enumerate(target_loader_unl_easy):
        image = image_obj[0].cuda()
        label = image_obj[1].cpu().data.item()
        img_path = image_obj[2][0]
        if label in to_plot:
            output = G(image)
            output = output.cpu().numpy()
            output = output[0]
            features.append(output)
            labels.append(label)
            weights.append(0)
            print(label)
        #if label==n_class_plot:
        #    break


print("Easy done")

# Features for hard examples
with torch.no_grad():
    for idx, image_obj in enumerate(target_loader_unl_hard):
        image = image_obj[0].cuda()
        label = image_obj[1].cpu().data.item()
        img_path = image_obj[2][0]
        if label in to_plot:
            output = G(image)
            output = output.cpu().numpy()
            output = output[0]
            features.append(output)
            labels.append(label)
            weights.append(1)
            print(label)
        #if label == n_class_plot:
        #   break


print("Starting source")
with torch.no_grad():
    for idx, image_obj in enumerate(source_loader):
        image = image_obj[0].cuda()
        label = image_obj[1].cpu().data.item()
        img_path = image_obj[2][0]
        output = G(image)
        output = output.cpu().numpy()
        output = output[0]
        features.append(output)
        labels.append(label)
        weights.append(10)
        print(label)
        if label == n_class_plot:
            break
"""