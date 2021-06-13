import os
import torch
from torchvision import transforms
from loaders.data_list import Imagelists_VISDA, Imagelists_VISDA_rot, return_classlist, return_number_of_label_per_class
from augmentations.randaugment import RandAugmentMC, RandAugmentPC
from augmentations.ctaugment import CTAugment
import numpy as np

class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


def return_dataset(args):
    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset
    image_set_file_s = \
        os.path.join(base_path,
                     'labeled_source_images_' +
                     args.source + '.txt')
    image_set_file_t = \
        os.path.join(base_path,
                     'labeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))
    image_set_file_t_val = \
        os.path.join(base_path,
                     'validation_target_images_' +
                     args.target + '_3.txt')
    image_set_file_unl = \
        os.path.join(base_path,
                     'unlabeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    source_dataset = Imagelists_VISDA(image_set_file_s, root=root,
                                      transform=data_transforms['train'])
    target_dataset = Imagelists_VISDA(image_set_file_t, root=root,
                                      transform=data_transforms['val'])
    target_dataset_val = Imagelists_VISDA(image_set_file_t_val, root=root,
                                          transform=data_transforms['val'])
    target_dataset_unl = Imagelists_VISDA(image_set_file_unl, root=root,
                                          transform=data_transforms['val'])
    target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=root,
                                           transform=data_transforms['test'])
    
    class_list = return_classlist(image_set_file_s)

    print("%d classes in this dataset" % len(class_list))
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=bs,
                                                num_workers=3, shuffle=True,
                                                drop_last=True)
    target_loader = \
        torch.utils.data.DataLoader(target_dataset,
                                    batch_size=min(bs, len(target_dataset)),
                                    num_workers=3,
                                    shuffle=True, drop_last=True)
    target_loader_val = \
        torch.utils.data.DataLoader(target_dataset_val,
                                    batch_size=min(bs,
                                                   len(target_dataset_val)),
                                    num_workers=3,
                                    shuffle=True, drop_last=True)
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=bs * 2, num_workers=3,
                                    shuffle=True, drop_last=True)
    target_loader_test = \
        torch.utils.data.DataLoader(target_dataset_test,
                                    batch_size=bs * 2, num_workers=3,
                                    shuffle=True, drop_last=True)
    return source_loader, target_loader, target_loader_unl, target_loader_val, target_loader_test, class_list


def return_dataset_test(args):
    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset
    image_set_file_s = os.path.join(base_path, "labeled_source_images_%s.txt"%(args.source))
    image_set_file_test = os.path.join(base_path,
                                       'unlabeled_target_images_' +
                                       args.target + '_%d.txt' % (args.num))
    if args.net == 'alexnet':
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
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=1, num_workers=3,
                                    shuffle=False, drop_last=False)
    return target_loader_unl, class_list

def return_dataset_rot(args):
    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset

    image_set_file_t = os.path.join(base_path, "labeled_target_images_%s_%s.txt"%(args.target,args.num))
    image_set_file_t_unl = os.path.join(base_path, "unlabeled_target_images_%s_%s.txt"%(args.target,args.num))

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224

    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            #transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    target_dataset = Imagelists_VISDA_rot(image_set_file_t, root=root, transform=data_transforms['train'])
    target_dataset_unl = Imagelists_VISDA_rot(image_set_file_t_unl, root=root, transform=data_transforms['train'])

    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24

    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=min(bs, len(target_dataset)), num_workers=3, shuffle=True, drop_last=True)
    target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl, batch_size=bs*2, num_workers=3, shuffle=True, drop_last=True)
    class_list = return_classlist(image_set_file_t_unl)
    
    return target_loader, target_loader_unl, class_list

# Defining dataloaders needed for fixmatch integration with SSDA
class TransformFix(object):
    def __init__(self, aug_policy, net, mean, std):
        self.net = net
        self.aug_policy = aug_policy
        if self.net == 'alexnet':
            self.crop_size = 227
        else:
            self.crop_size = 224
        
        # Might want to add resize image as done in other transforms
        self.weak = transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=self.crop_size, padding=int(self.crop_size*0.125),padding_mode='reflect')])

        if self.aug_policy == "randaugment":
            self.strong = transforms.Compose([
                ResizeImage(256),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=self.crop_size, padding=int(self.crop_size*0.125),padding_mode='reflect'), RandAugmentPC(n=2, m=10)])

        self.standard = transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(self.crop_size)])

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        standard = self.standard(x)
        return self.normalize(weak), self.normalize(strong), self.normalize(standard)


def return_dataset_randaugment(args,txt_path='./data/txt/',root_path='./data/',bs_alex = 32, bs_resnet = 24,set_shuffle = True):
    base_path = os.path.join(txt_path,args.dataset)
    root = os.path.join(root_path,args.dataset)
    image_set_file_s = os.path.join(base_path, 'labeled_source_images_' + args.source + '.txt')
    image_set_file_t = os.path.join(base_path, 'labeled_target_images_' + args.target + '_%d.txt' % (args.num))
    image_set_file_t_val = os.path.join(base_path, 'validation_target_images_' + args.target + '_3.txt')
    image_set_file_unl = os.path.join(base_path, 'unlabeled_target_images_' + args.target + '_%d.txt' % (args.num))

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224

    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    source_dataset = Imagelists_VISDA(image_set_file_s, root=root, transform=data_transforms['train'])

    if args.uda:
        target_dataset = Imagelists_VISDA(image_set_file_t, root=root, transform=TransformFix("randaugment", args.net, mean =[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    else:
        target_dataset = Imagelists_VISDA(image_set_file_t, root=root, transform=data_transforms['val'])       

    target_dataset_val = Imagelists_VISDA(image_set_file_t_val, root=root, transform=data_transforms['val'])
    target_dataset_unl = Imagelists_VISDA(image_set_file_unl, root=root, transform=TransformFix("randaugment", args.net, mean =[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=root, transform=data_transforms['test'])
    class_list = return_classlist(image_set_file_s)
    
    class_num_list_source = return_number_of_label_per_class(image_set_file_s, len(class_list))

    print("%d classes in this dataset" % len(class_list))
    if args.net == 'alexnet':
        bs = bs_alex
    else:
        bs = bs_resnet
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=bs,
                                                num_workers=3, shuffle=set_shuffle,
                                                drop_last=True)
    target_loader = \
        torch.utils.data.DataLoader(target_dataset,
                                    batch_size=min(bs, len(target_dataset)),
                                    num_workers=3,
                                    shuffle=set_shuffle, drop_last=False)
                                    
    target_loader_misc = torch.utils.data.DataLoader(target_dataset, batch_size=1, num_workers=3, shuffle=False)

    target_loader_val = \
        torch.utils.data.DataLoader(target_dataset_val,
                                    batch_size=min(bs, len(target_dataset_val)),
                                    num_workers=3,
                                    shuffle=set_shuffle, drop_last=True)
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=bs * 2, num_workers=3,
                                    shuffle=set_shuffle, drop_last=True)
    target_loader_test = \
        torch.utils.data.DataLoader(target_dataset_test,
                                    batch_size=bs, num_workers=3,
                                    shuffle=set_shuffle, drop_last=True)
    return source_loader, target_loader, target_loader_misc, target_loader_unl, target_loader_val, target_loader_test, class_num_list_source, class_list

"""    
elif self.aug_policy == "ct_augment":
    self.strong = transforms.Compose([
        ResizeImage(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=self.crop_size,
                            padding=int(self.crop_size*0.125),
                            padding_mode='reflect'), CTAugment()])    

"""