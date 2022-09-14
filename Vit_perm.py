"""
Created on Fri Sep  9 15:05:25 2022

@author: Lei Liu (UT Austin, leiliu@utexas.edu)

ViT for Permeability of 3D image
"""

# -----------------------------------
# INPUT DATA (3 for train, 2 for val)
# -----------------------------------

import matplotlib.pyplot as plt
from hdf5storage import loadmat
import time
import torch
from torch import nn
from ViT import ViT
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import wandb

P_1_MPa = loadmat('P_1_MPa.mat')
P_2_MPa = loadmat('P_2_MPa.mat')
P_5_MPa = loadmat('P_5_MPa.mat')
P_10_MPa = loadmat('P_10_MPa.mat')
P_20_MPa = loadmat('P_20_MPa.mat')

sample1 = P_1_MPa['uz']  #3D array of size 256**3
x1 = ~(sample1==0) #1 for pore space and 0 for solid space
x1 = x1 * 1.0
y1 = sample1.mean()/5e-9 # perm, a floating point number

sample2 = P_2_MPa['uz']  
x2 = ~(sample2==0) 
x2 = x2 * 1.0
y2 = sample2.mean()/5e-9 

sample3 = P_5_MPa['uz']  
x3 = ~(sample3==0) 
x3 = x3 * 1.0
y3 = sample3.mean()/5e-9 

sample4 = P_10_MPa['uz']  
x4 = ~(sample4==0) 
x4 = x4 * 1.0
y4 = sample4.mean()/5e-9 

sample5 = P_20_MPa['uz']  
x5 = ~(sample5==0) 
x5 = x5 * 1.0
y5 = sample5.mean()/5e-9 

#plt.imshow(x[:,:,100])

# -----------------------------------
# DATA PRE-PROCESS
# -----------------------------------

x = np.stack((x1,x2,x3,x4,x5),axis=-1)
y = np.stack((y1,y2,y3,y4,y5),axis=-1)
x = torch.from_numpy(x.reshape(5,256,256,256)).float()
y = torch.from_numpy(y.reshape(-1,1)).float()

X_ss = (x-x.mean())/(x.std())
Y_ss = (y-y.min())/(y.max()-y.min());

# below is for training monitoring in W&B, feel free to edit this
'''
wandb.init(name='256x256x256',
           project = 'ViT_Perm',
           entity = 'leiliu',
           config = {'image_size':256,
                           'patch_size':16,
                           'depth':8,
                           'heads':16,
                           'mlp_dim':4096,
                           'learning_rate':1e-6,
                           'epochs':200,
                           'batch_size':1,
                           })
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

## play with different parameters; make sure the patch size should be divisable by image_size
model = ViT(
    image_size = 256,
    patch_size = 16,
    num_classes = 1,
    dim= 8192,
    depth = 8,
    heads = 16,
    mlp_dim = 4096,
    dropout = 0,
    emb_dropout = 0
    )
#model.to(device)  ##train with gpu if you have; otherwise cpu

wandb.watch(model)

class Mydata(Dataset):
    
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        
    def __getitem__(self, index):
        image = self.data[index]
        label = self.targets[index]
        return image, label
    
    def __len__(self):
        return len(self.data)

dataset = Mydata(X_ss,Y_ss)
train_size = int(0.7*len(dataset))
test_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, test_size]) #train_set:2100; val_set:900

train_loader = DataLoader(train_set, batch_size=1, shuffle=True) 
val_loader = DataLoader(val_set, batch_size=1, shuffle=True)     


##define loss function
loss_fn = nn.MSELoss()
loss_fn = loss_fn.cuda()

##define optimizer
lr = 1e-6
optimizer = torch.optim.Adam(model.parameters(),lr=lr, eps=1e-8, weight_decay=1e-4)

##traiing parameters#
total_train_step = 0
total_val_step = 0
epoch = 200
start_time = time.time()
##training
for i in range(epoch):
    print('----------This is the {} epoch training---------'.format(i+1))
    train_loss = 0
    
    model.train()
    for imgs, labels in train_loader:  
        #imgs, labels = imgs.to(device), labels.to(device)           
        outputs = model(imgs)
        loss = loss_fn(outputs, labels)

        ##optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() #update weights       
        total_train_step = total_train_step + 1 #step means iteration, 1 iter means train the number of (batch size) samples         
        train_loss += loss.item()
    wandb.log({'epoch':i,
               'train_loss':train_loss/len(train_loader),
               })
    print('total train loss:{}'.format(train_loss))
    print('time elapsed: {:.4f}s'.format(time.time()-start_time))
    
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            #imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            total_val_loss = total_val_loss + loss.item()             
            total_val_step += 1 
    wandb.log({'epoch':i,
               'val_loss':total_val_loss/len(val_loader),
               })  
    print('total val loss:{}'.format(total_val_loss))
    
