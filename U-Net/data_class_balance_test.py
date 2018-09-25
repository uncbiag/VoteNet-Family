import SimpleITK as sitk
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils
import math
import matplotlib.pyplot as plt
import time
from Dataset import *
from Transform import *

patch_size = (72, 72, 72)

label_list = [21, 22, 23, 24]
sample_threshold = (0.01, 0.01)
balanced_random_crop = MyBalancedRandomCrop(patch_size, threshold=sample_threshold, label_list=label_list)

random_crop = MyRandomCrop((72, 72, 72), 0.1)
sitk_to_tensor = SitkToTensor()

training_list_file = os.path.realpath('../Data/LPBA40.txt')
training_data_dir = os.path.realpath('../Data/LPBA40/')

# transform = transforms.Compose([balanced_random_crop, sitk_to_tensor])
transform = transforms.Compose([sitk_to_tensor])
training_dataset = NiftiDataset(training_list_file, training_data_dir, mode="training", preload=False, transform=transform)
training_data_loader = DataLoader(training_dataset, batch_size=4, shuffle=True, num_workers=4)
perc_1_list = []
perc_2_list = []
n_classes=56

start = time.time()
for i in range(5):
    for i_batch, (images, labels, name) in enumerate(training_data_loader):
        print(name)
        perc_1 = torch.sum(labels==1) / labels.view(-1).size()[0]
        perc_2 = torch.sum(labels==2) / labels.view(-1).size()[0]
        perc_1_list.append(perc_1)
        perc_2_list.append(perc_2)
        print("Label 1: {}, Label 2: {}".format(perc_1, perc_2))
    # print(np.mean(perc_1_list) / np.mean(perc_2_list))
print(np.mean(perc_1_list)/np.mean(perc_2_list))
print(time.time() - start)