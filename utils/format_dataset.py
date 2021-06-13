import os
import shutil

label_file = open(os.path.join("/home/megh/projects/domain-adaptation/SSAL/data/FER/RAF/list_patition_label.txt"),"r")
emotion_list = ['Surprised','Fear','Disgust','Happy','Sad','Anger','Neutral']

for line in label_file:
    line = line.rstrip("\n")
    image_name, image_label = line.split(" ")
    print(image_name,image_label)
    image_name = image_name.replace(".jpg","")+ "_aligned.jpg"
    copy_from = os.path.join("/home/megh/projects/domain-adaptation/SSAL/data/FER/RAF/images",str(image_name))
    copy_to = os.path.join("/home/megh/projects/domain-adaptation/SSAL/data/FER/RAF/", str(image_label))
    shutil.copy(copy_from,copy_to)