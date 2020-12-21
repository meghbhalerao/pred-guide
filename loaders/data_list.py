import numpy as np
import os
import os.path
from PIL import Image
import random
import torch

random.seed(1)
np.random.seed(1)

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def make_dataset_fromlist(image_list):
    with open(image_list) as f:
        image_index = [x.split(' ')[0] for x in f.readlines()]
    with open(image_list) as f:
        label_list = []
        selected_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip()
            label_list.append(int(label))
            selected_list.append(ind)
        image_index = np.array(image_index)
        label_list = np.array(label_list)
    image_index = image_index[selected_list]
    return image_index, label_list

def return_number_of_label_per_class(image_list,number_of_classes):
    with open(image_list) as f:
        label_list = [0] * number_of_classes
        for ind, x in enumerate(f.readlines()):
            label = int(x.split(' ')[1])
            label_list[label]+=1
    return label_list


def return_classlist(image_list):
    with open(image_list) as f:
        label_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[0].split('/')[-2]
            if label not in label_list:
                label_list.append(str(label))
    return label_list


class Imagelists_VISDA(object):
    def __init__(self, image_list, root="./data/multi/",
                 transform=None, target_transform=None, test=False):
        imgs, labels = make_dataset_fromlist(image_list)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.test = test

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.test:
            return img, target, self.imgs[index]
        else:
            return img, target, self.imgs[index]

    def __len__(self):
        return len(self.imgs)

# Rotation function for PIL images
def rotate_img(image, rot):
    rotated_image = image.rotate(rot)
    return rotated_image

class Imagelists_VISDA_rot(object):
    def __init__(self, image_list, root="./data/multi/",
                 transform=None):
        imgs, labels = make_dataset_fromlist(image_list)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.loader = pil_loader
        self.rotate  = rotate_img
        self.root = root

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):

        path = os.path.join(self.root, self.imgs[index])
        img = self.loader(path)
        # Randomly select rotation angle and rotating the image
        angles = [0,90,180,270]
        rot_angle = random.choice(angles)
        img = self.rotate(img,rot_angle)
        img = self.transform(img)
        target = torch.tensor(int(rot_angle/90))
        return img, target, self.labels[index]
