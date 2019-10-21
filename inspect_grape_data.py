
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

from mrcnn.visualize import display_images
from mrcnn import visualize

path ="D:\DLCode\wgisd\data\CDY_2019"

image_path=path+'.jpg'
image = skimage.io.imread(image_path)
list=[]
list.append(image)
#display_images(images=list,cols=1)

mask_path=path+'.npz'
masks = np.load(mask_path)['arr_0']

bbox_path=path+'.txt'
bboxes = np.loadtxt(bbox_path)
bboxes=bboxes[:,1:]

bboxes2=np.zeros(bboxes.shape,dtype=np.int)

bboxes2[:,0]=bboxes[:,1]*1365-bboxes[:,3]*1365*0.5
bboxes2[:,1]=bboxes[:,0]*2048-bboxes[:,2]*2048*0.5
bboxes2[:,2]=bboxes[:,1]*1365+bboxes[:,3]*1365*0.5
bboxes2[:,3]=bboxes[:,0]*2048+bboxes[:,2]*2048*0.5

'''
bboxes2[:,0]=bboxes[:,0]*2048
bboxes2[:,1]=bboxes[:,1]*1365
bboxes2[:,2]=bboxes[:,2]*2048
bboxes2[:,3]=bboxes[:,3]*1365
'''


print(bboxes2)



print(masks.shape)

print(bboxes.shape)
#print(bboxes)

#display_images([image]+[masks[:,:,i] for i in range(masks.shape[-1])])
#class_ids
#class_ids=[0,0,]

class_ids=np.ones(masks.shape[-1],dtype=np.int)
class_names=["BG","grape"]
print(class_ids.shape)

visualize.display_instances(image, bboxes2, masks,class_ids,class_names)


print("show image..")