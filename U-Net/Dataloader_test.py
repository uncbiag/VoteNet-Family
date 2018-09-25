import SimpleITK as sitk
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import math
import matplotlib.pyplot as plt
import time
from Dataset import *
import Transform as bio_transform

patch_size = (72, 72, 72)
random_seed = 1234
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

sitk_to_tensor = bio_transform.SitkToTensor()

random_state = np.random.RandomState(seed=random_seed)
rand_crop = bio_transform.RandomCrop(patch_size, random_state=random_state)
train_transforms = [rand_crop, sitk_to_tensor]
train_transform = transforms.Compose(train_transforms)
training_list_file = os.path.realpath('../Data/LPBA40.txt')
training_data_dir = os.path.realpath('../Data/LPBA40/')
training_dataset = NiftiDataset(training_list_file, training_data_dir, mode="training", preload=True, transform=train_transform)
training_data_loader = DataLoader(training_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=False)


n_classes = 56
for epoch in range(1):
    data_iter = iter(training_data_loader)
    start_0 = time.time()
    for i in range(len(training_data_loader)):
        start = time.time()
        print("batch: {}".format(i))
        images, labels, name = data_iter.__next__()

        print(time.time() - start)
        print("Name: {}".format(name))

    print("Epoch", epoch, time.time() - start_0)

