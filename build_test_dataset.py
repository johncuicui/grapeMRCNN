import os
import numpy as np
import random
import shutil

data_folder = 'D:/DLCode/wgisd/data/'
train_masked_path = 'D:/DLCode/wgisd/train_masked.txt'

ROOT_DIR = os.path.abspath(".")
print(ROOT_DIR)

#create dataset folder
dataset_folder= os.path.sep.join([ROOT_DIR,"dataset"])
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

#build train dataset
dataset_folder_test= os.path.sep.join([dataset_folder,"test"])
if not os.path.exists(dataset_folder_test):
    os.makedirs(dataset_folder_test)


# load the names of the images
with open(train_masked_path, 'r') as fp:
    data_list = fp.readlines()

data_list = set([i[:-1] for i in data_list])
print(data_list)

i=0

files = os.listdir(data_folder)
for file in files:
    # extract image names
    filename, filename_ext = os.path.splitext(file)
    if filename_ext == ".jpg":
        if filename not in data_list:

            i=i+1
            image_src_path = data_folder + filename + '.jpg'
            image_dst_path = dataset_folder_test + os.sep + filename + '.jpg'

            print("copy: ",image_src_path)
            shutil.copy2(image_src_path,image_dst_path)

print("\n\ntest images:",i)


