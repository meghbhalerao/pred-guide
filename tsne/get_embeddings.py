#from tsnecuda import TSNE
import torch
import numpy as np
import sys
sys.path.append('/cbica/home/bhaleram/comp_space/random/personal/iisc_project/MME/')
from loaders.data_list import Imagelists_VISDA, return_classlist, return_number_of_label_per_class
from model.basenet import *
from model.resnet import *
from torchvision import transforms
from utils.return_dataset import ResizeImage
from sklearn.manifold import TSNE
import time


# Defining return dataset function here
root = '../data/multi/' 
net = "alexnet"
image_set_file_test = "/cbica/home/bhaleram/comp_space/random/personal/iisc_project/MME/data/txt/multi_2/unlabeled_target_images_painting_3.txt"
 

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

"""
if net == 'resnet34':
    F1 = Predictor_deep_attributes(num_class=len(class_list),inc=inc,feat_dim = 50)
    print("Using: Predictor_deep_attributes")
else:
    F1 = Predictor_attributes(num_class=len(class_list),inc=inc,feat_dim = 50)
    print("Using: Predictor_attributes")
"""

if net == 'resnet34':
    F1 = Predictor_deep(num_class=len(class_list),inc=inc)
    print("Using: Predictor_deep")
else:
    F1 = Predictor(num_class=len(class_list), inc=inc, temp=0.05)
    print("Using: Predictor")


G.cuda()
F1.cuda()

checkpoint = torch.load("../freezed_models/mme.ckpt.best.pth.tar")

f = open("/cbica/home/bhaleram/comp_space/random/personal/iisc_project/MME/entropy_weight_files/MME_alexnet_real_painting_0.txt")


# Loading the weights from the checkpoint
G.load_state_dict(checkpoint["G_state_dict"])
F1.load_state_dict(checkpoint["F1_state_dict"])








features = []
labels = []
weights = []

# Read the weight file and converting it to a dictionary
weight_dict = {}
for line in f:
    line_list = line.split()
    weight_dict[line_list[1]] = float(line_list[0])



with torch.no_grad():
    for idx, image_list in enumerate(target_loader_unl):
        #print(image_list)
        image = image_list[0].cuda()
        label = image_list[1].cpu().data.item()
        img_path = image_list[2][0]
        output = G(image)
        #output = F1.fc1(output)
        output = output.cpu().numpy()
        output = output[0]
        features.append(output)
        labels.append(label)
        weights.append(weight_dict[img_path])




#prototype = F1.fc2.weight.cpu().detach().numpy()

#for i in range(45):
#    features.append(prototype[i,:])

#for i in range(45):
#    labels.append(-100)


features = np.array(features)
print(features.shape)
np.save('features.npy', features)
print(len(labels))
labels = np.array(labels)
np.save('labels.npy',labels)
print(len(weights))
weights = np.array(weights)
np.save('weights.npy', weights)



start = time.time()
X_embedded = TSNE(n_components=2,perplexity=50).fit_transform(features)
np.save('tsne_embeddings.npy', X_embedded)
end = time.time()
print((end-start)/60)












