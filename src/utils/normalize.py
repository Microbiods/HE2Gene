
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import pickle
import copy
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.stats import pearsonr
import torchvision.transforms as transforms
from torchvision.models import resnet50
# from datasets.dataloader import BreastCancerSTDataset
import random


# def Normalize(root, patient, image_size, batch_size, target_genes,aux_task, num_workers, window):

#     transform = transforms.Compose([
#         transforms.Resize((image_size, image_size)), # resize image to 224x224 pixels
#         transforms.ToTensor()          # convert image to a PyTorch tensor
#     ])

#     train_dataset = BreastCancerSTDataset(root,
#                                         patient=patient, # patient id needs to be specified
#                                         target_genes=target_genes,
#                                         transform=transform,
#                                         gene_norm = "log1p",
#                                         aux_task=aux_task,
#                                         window = window,
#                                         )
    
#     # create a PyTorch dataloader for the training set
#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#     # Calculate the mean and standard deviation of the training set
#     train_images = []
#     train_counts = []
#     for (images, counts, *_) in train_dataloader:
#         train_images.append(images)
#         train_counts.append(counts)

#     train_images = torch.cat(train_images, dim=0)
#     train_counts = torch.cat(train_counts, dim=0)

#     image_mean = train_images.mean(dim=(0, 2, 3)) 
#     image_std = train_images.std(dim=(0, 2, 3)) 
    
#     count_mean = torch.mean(train_counts, axis=0)
#     count_std = torch.std(train_counts, axis=0)
   
#     return image_mean, image_std, count_mean, count_std

def SetSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
