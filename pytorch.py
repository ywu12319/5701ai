#For accessing files
import os
import glob

#For Images
import cv2
from skimage import io
from skimage.transform import resize
#import matplotlib.pyplot as plt
import numpy as np

#For checking progress
#from tqdm import tqdm_notebook

import datetime

#PyTorch Packages
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(16,8,kernel_size=3,padding=1)
        self.fc1 = nn.Linear(8*8*8,32)
        self.fc2 = nn.Linear(32,2)
    def forward(self,x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)),2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)),2)
        out = out.view(-1,8*8*8)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out

def classify(img, model):
    s = nn.Softmax(dim=1)
    device = torch.device('cpu');
    out = s(model(img.unsqueeze(0).to(device)))
    return out
    print('Prediction: {}'.format(out))



def imgRead(img_path, transformations) :
    #img = io.imread(img_path)
    #img = resize(img, (32,32,3))
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(32,32))
    if transformations is not None:
        img_as_tensor = transformations(img)     
        return img_as_tensor
    return img
    

if __name__ == "__main__":
    model = Net()

    model.load_state_dict(torch.load("./fruit.pt",map_location=torch.device('cpu')))
    print(model.eval())
    transformations = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.7369, 0.6360, 0.5318),
                                                           (0.3281, 0.3417, 0.3704))
                                      ])
    res = classify(imgRead('./test_1.png',transformations), model)
    print(res);



