import os
import numpy as np
import random
import shutil

def create_dataset(dataset_folder,data_folder,image_list):

	for image in image_list:

		image_src_path=data_folder+image+'.jpg'
		image_dst_path=dataset_folder+os.sep+image+'.jpg'

		shutil.copy2(image_src_path,image_dst_path)

		bbox_src_path = data_folder + image + '.txt'
		bbox_dst_path = dataset_folder +os.sep+image+'.txt'
		shutil.copy2(bbox_src_path, bbox_dst_path)

		mask_src_path = data_folder + image + '.npz'
		mask_dst_path = dataset_folder + os.sep + image + '.npz'
		shutil.copy2(mask_src_path, mask_dst_path)


data_folder='D:/DLCode/wgisd/data/'
train_masked_path ='D:/DLCode/wgisd/train_masked.txt'

ROOT_DIR = os.path.abspath(".")
print(ROOT_DIR)

# load the names of the images
with open(train_masked_path, 'r') as fp:
    data_list = fp.readlines()

data_list = set([i[:-1] for i in data_list])

# split
data_list=sorted(data_list)
random.shuffle(data_list)

i = int(len(data_list) * 0.8)
data_list_train = data_list[:i]
data_list_val = data_list[i:]

#create dataset folder
dataset_folder= os.path.sep.join([ROOT_DIR,"dataset"])
if not os.path.exists(dataset_folder):
	os.makedirs(dataset_folder)

#build train dataset
dataset_folder_train= os.path.sep.join([dataset_folder,"train"])
if not os.path.exists(dataset_folder_train):
	os.makedirs(dataset_folder_train)
create_dataset(dataset_folder_train,data_folder,data_list_train)

# build Validation dataset
dataset_folder_val= os.path.sep.join([dataset_folder,"val"])
if not os.path.exists(dataset_folder_val):
	os.makedirs(dataset_folder_val)
create_dataset(dataset_folder_val,data_folder,data_list_val)

#for i in data_list:
#   print(i)

print("\ntrain:{},val:{}".format(len(data_list_train),len(data_list_val)))

