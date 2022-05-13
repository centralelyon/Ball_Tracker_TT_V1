import csv
from msvcrt import kbhit
from time import strftime
import cv2
import numpy as np
import os
import torch
import torchvision.transforms as transforms

data_set = open('train_set.csv','r')
reader = csv.reader(data_set, delimiter =',')

mean =[0,0,0]
std = [0,0,0]
nbre_image=0

for i in reader:
    if i!=[] and i[0]!='Nom':
        nbre_image+=1
        image = cv2.imread(os.path.join('imager',i[0]))
        image = np.array(image)
        transform = transforms.ToTensor()
        img_tensor=transform(image)
        for i in range (len(mean)):
            mean[i] += torch.mean(img_tensor, dim=[1,2])[i]
            std[i] += torch.std(img_tensor, dim=[1,2])[i]
print([mean_c/nbre_image for mean_c in mean])
print([std_c/nbre_image for std_c in std])