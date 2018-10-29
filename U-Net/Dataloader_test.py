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
import utils as helper

random_seed = 1234
torch.manual_seed(random_seed)
# torch.cuda.manual_seed(random_seed)

random_state = random.RandomState(seed=random_seed)
patch_size = (72, 72, 72)
sitk_to_tensor = bio_transform.SitkToTensor()
rand_crop = bio_transform.RandomCrop(patch_size, random_state=random_state)
balanced_random_crop = bio_transform.BalancedRandomCrop(patch_size, threshold=0.005)
train_transforms = [balanced_random_crop, sitk_to_tensor]
train_transform = transforms.Compose(train_transforms)
training_list_file = os.path.realpath('../Data/LPBA40.txt')
training_data_dir = os.path.realpath('../Data/LPBA40/')
validation_list_file = os.path.realpath("../Data/LPBA40_data_sep/LPBA40_fold_1_validate.txt")
valid_data_dir=os.path.realpath("../Data/LPBA40/")

seg_label_density = helper.get_label_density('../Data/LPBA40/corr_label/s1.nii')
# seg_label = helper.get_label('../Data/LPBA40/label_affine_icbm/s1.nii')
seg_label = [i for i in range(57)]
img_size = helper.get_img_size('../Data/LPBA40/corr_label/s1.nii')

partition = bio_transform.Partition(patch_size, overlap_size=(16, 16, 16))
valid_transforms = [partition]
valid_transform = transforms.Compose(valid_transforms)
# training_dataset = NiftiDataset(training_list_file, training_data_dir, mode="training", scale_ratio=0.01, bg_th_ratio=0, label_list=seg_label, label_density=seg_label_density, img_size=img_size, max_crop_num=-1, patch_size=patch_size, preload=True, transform=train_transform)
# training_data_loader = DataLoader(training_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=False)

# def worker_init_fn(worker_id):
#     np.random.seed(np.random.get_state()[1][0] + worker_id)

training_dataset = NiftiDataset(training_list_file, training_data_dir, mode="training", preload=True, transform=train_transform)
training_data_loader = DataLoader(training_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=False)
validation_data = NiftiDataset(validation_list_file, valid_data_dir, mode="training", preload=True, transform=valid_transform)

# for epoch in range(2):
#     # np.random.seed()
#     data_iter = iter(training_data_loader)
#     start_0 = time.time()
#     for i in range(len(training_data_loader)):
#         # print(len(training_data_loader))
#         start = time.time()
#         # print("batch: {}".format(i))
#         images, labels, name = data_iter.__next__()
#
#         # print(time.time() - start)
#         # print("Name: {}".format(name))
#
#     # print("Epoch", epoch, time.time() - start_0)

for j in range(len(validation_data)):
    image_tiles, segs, names = validation_data[j]