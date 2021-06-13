import os
import sys
from numpy.lib.type_check import imag
import random
domain = "CK+"
mode = "target"
num_labeled = 3
num_classes = 7
domain_path = os.path.join("/home/megh/projects/domain-adaptation/SSAL/data/FER",domain)

if mode == "source":
    f_domain = open("/home/megh/projects/domain-adaptation/SSAL/data/txt/FER/labeled_%s_images_%s.txt"%(mode, domain),"w")
elif mode == "target":
    f_fewshot = open(os.path.join("/home/megh/projects/domain-adaptation/SSAL/data/txt/FER/labeled_%s_images_%s_%s.txt"%(mode,domain,str(num_labeled))),"w")
    f_val = open(os.path.join("/home/megh/projects/domain-adaptation/SSAL/data/txt/FER/validation_%s_images_%s_%s.txt"%(mode,domain,str(num_labeled))),"w")
    f_unlabeled = open(os.path.join("/home/megh/projects/domain-adaptation/SSAL/data/txt/FER/unlabeled_%s_images_%s_%s.txt"%(mode,domain,str(num_labeled))),"w")


if os.path.exists(os.path.join("/home/megh/projects/domain-adaptation/SSAL/data/txt/FER/labeled_source_images_%s.txt"%(domain))):
    lt_flag = []
    for i in range(num_classes):
        for _ in range(num_labeled):
            lt_flag.append(i+1)

    all_examples = open(os.path.join("/home/megh/projects/domain-adaptation/SSAL/data/txt/FER/labeled_source_images_%s.txt"%(domain)))
    all_ex = [ex.rstrip("\n") for ex in all_examples]
    all_ex_copy = all_ex.copy()
    lt_flag_copy = lt_flag.copy()
    for example in all_ex:
        for label_ in lt_flag:
            if label_  == int(example.strip()[1]):
                f_fewshot.write(example + "\n")
                all_ex_copy.remove(example)
                lt_flag.remove(label_)
        if not lt_flag:
            break
    
    for example in all_ex_copy:
        

    for example in all_ex_copy:
        f_unlabeled


    #print(all_ex)

sys.exit()

for class_idx, emotion_class in enumerate(os.listdir(domain_path)):
    class_path = os.path.join(domain_path,emotion_class)
    for image in os.listdir(os.path.join(class_path)):
        image_path_relative = os.path.join(domain,emotion_class,image)
        f_domain.write(image_path_relative + " " + str(class_idx) + "\n")
