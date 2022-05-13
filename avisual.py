from asyncio import base_tasks
import torch
import os
import csv
import cv2
import pandas as pd
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
from matplotlib import pyplot as plt

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
 
        #self.Sequence = nn.Sequential(

            # first convolution layer
        self.conv1=nn.Conv2d(3, 32 , (3,3), padding='same')
        self.batch1=nn.BatchNorm2d(32)
        self.relu=nn.ReLU()
        self.conv2=nn.Conv2d(32, 64, (3,3))
        self.batch2=nn.BatchNorm2d(64)
            #nn.ReLU(),
        self.pool=nn.MaxPool2d((2,2))
        self.drop=nn.Dropout2d()

            # second convolution layer
        self.conv3=nn.Conv2d(64, 128, (3,3), padding='same')
        self.batch3=nn.BatchNorm2d(128)
            #nn.ReLU(),
        self.conv4=nn.Conv2d(128, 128, (3,3), padding='same')
            #nn.BatchNorm2d(128),
            #nn.ReLU(),
            #nn.MaxPool2d((2,2)),
            #nn.Dropout2d(),

            # third convolution layer
        self.conv5=nn.Conv2d(128, 256, (3,3), padding='same')
        self.batch4=nn.BatchNorm2d(256)
            #nn.ReLU(),
        self.conv6=nn.Conv2d(256, 256, (3,3), padding='same')
            #nn.BatchNorm2d(256),
            #nn.MaxPool2d((2,2)),

            # fourth convolution layer
        self.conv7=nn.Conv2d(256, 512, (3,3), padding='same')
        self.batch5=nn.BatchNorm2d(512)
            #nn.ReLU(),
        self.conv8=nn.Conv2d(512, 512, (3,3), padding='same')
            #nn.BatchNorm2d(512),

            # 1D tensor with flatten
        """nn.Flatten(),

            # depth layer
            nn.Linear(269824, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,2),"""
    def forward(self, x):
        logits = self.drop(self.pool(self.relu(self.batch2(self.conv2(self.relu(self.batch1(self.conv1(x))))))))
        logits = self.drop(self.pool(self.relu(self.batch3(self.conv4(self.relu(self.batch3(self.conv3(logits))))))))
        logits = self.pool(self.batch4(self.conv6(self.relu(self.batch4(self.conv5(logits))))))
        logits = self.batch5(self.conv8(self.relu(self.batch5(self.conv7(logits)))))
        return logits

class data_set(Dataset):
    def __init__(self,dir_path,transform=None):

        self.landmark = pd.read_csv(f'{dir_path}.csv')
        self.transform = transform
    
    def __len__(self):
        return len(self.landmark)

    def __getitem__(self, idx, device='cuda'):
        if torch.is_tensor(idx):
           idx = idx.tolist()
        img_name = self.landmark.iloc[idx,0]
        image = cv2.imread(os.path.join('imager',img_name))
        image = np.array(image)
        landmark = np.array([float(self.landmark.iloc[idx,1].replace(",","."))/7.5,float(self.landmark.iloc[idx,2].replace(",","."))/7.5])
        sample = {'image':image, 'coord':landmark}
        if self.transform:
            sample = self.transform(sample)
        sample['image'] = sample['image'].to(device)
        sample['coord'] = sample['coord'].to(device)
        return sample

class ToTensor(object):
    def __call__(self, sample):
        image = sample['image']
        coord = sample['coord']
        transform = transforms.ToTensor()
        image = transform(image)
        return {'image': image, 'coord': torch.from_numpy(coord).float()}

model = NeuralNetwork()
batch = DataLoader(data_set('train_tracker',ToTensor()),batch_size = 32, shuffle = False)
with torch.no_grad():
    img = next(iter(batch))
    img = model(img['image'].detach().cpu())
final = torch.sum(img[1],0)
final = final / img[1].shape[0]
plt.imshow(final, interpolation='nearest')
plt.show()