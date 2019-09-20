"""
    Code referenced from Zhenlin Xu on OAI segmentation project @ https://github.com/uncbiag/OAI_segmentation/blob/master/utils/datasets.py

"""

import SimpleITK as sitk
import os
from torch.utils.data import Dataset
# from Transform import my_balanced_random_crop
from numpy import random
import numpy as np


class NiftiDataset(Dataset):
    """
    a dataset class to load medical image data in Nifti format using simpleITK
    """

    def __init__(self, txt_file, data_dir, mode, preload=False, transform=None):
        """
        :param txt_file: txt file with lines of "image_file, segmentation_file"
        :param root_dir: the data root dir
        :param transform: transformations on a sample for data augmentation
        """
        self.data_dir = data_dir
        self.mode = mode
        self.preload = preload
        self.transform = transform

        if preload:
            print("Preloading data:")
            self.image_list, self.segmentation_list, self.prior_list, self.name_list = self.read_image_segmentation(txt_file)
            print('Preloaded {} training samples'.format(len(self.image_list)))
        else:
            self.image_list, self.segmentation_list, self.prior_list, self.name_list = self.read_image_segmentation_list(txt_file)

        if len(self.image_list) != len(self.segmentation_list):
            raise ValueError("The numbers of images and segmentations are different")

    def __len__(self):
        return self.image_list.__len__()


    def __getitem__(self, id):
        id = id % len(self.image_list)
        if self.preload:
            image = self.image_list[id]
            segmentation = self.segmentation_list[id]
            prior = self.prior_list[id]
        else:
            # img_folder = '/brain_affine_icbm_hist_oasis/'
            # seg_folder = '/corr_label/'
            # prior_folder = '/corr_label/'
            img_folder = ''
            seg_folder = ''
            prior_folder = '/brain_affine_icbm_hist_oasis/'
            # prior_folder = ''
            image_file_name = os.path.join(self.data_dir+img_folder, self.image_list[id])
            segmentation_file_name = os.path.join(self.data_dir+seg_folder, self.segmentation_list[id])
            prior_file_name = os.path.join(self.data_dir+prior_folder, self.prior_list[id])

            if not os.path.exists(image_file_name):
                print(image_file_name + ' not exist!')
                return
            if not os.path.exists(segmentation_file_name):
                print(segmentation_file_name + ' not exist!')
                return
            if not os.path.exists(prior_file_name):
                print(prior_file_name + ' not exist!')
                return
            image = sitk.ReadImage(image_file_name)
            segmentation = sitk.ReadImage(segmentation_file_name)
            prior = sitk.ReadImage(prior_file_name)

        image_name = self.name_list[id]
        sample = {'image': image, 'segmentation': segmentation, 'prior': prior, 'name': image_name}

        if self.transform:
            sample = self.transform(sample)

        return sample['image'], sample['segmentation'], sample['prior'], sample['name']


    def read_image_segmentation_list(self, text_file):
        image_list = []
        segmentation_list = []
        prior_list = []
        name_list = []
        with open(text_file) as file:
            for line in file:
                try:
                    image, seg, prior = line.strip("\n").split(', ')
                except ValueError:
                    image = seg = prior = line.strip("\n")
                image_list.append(image)
                segmentation_list.append(seg)
                prior_list.append(prior)
                name_list.append(image[:-7])
        return image_list, segmentation_list, prior_list, name_list


    def read_image_segmentation(self, text_file):
        image_list = []
        segmentation_list = []
        prior_list = []
        name_list = []
        # img_folder = '/brain_affine_icbm_hist_oasis/'
        # seg_folder = '/corr_label/'
        # prior_folder = '/corr_label/'
        img_folder = ''
        seg_folder = ''
        prior_folder = '/brain_affine_icbm_hist_oasis/'
        # prior_folder = ''
        with open(text_file) as file:
            for line in file:
                try:
                    image_file_name, segmentation_file_name, prior_file_name = line.strip("\n").split(', ')
                except ValueError:
                    image_file_name = segmentation_file_name = prior_file_name = line.strip("\n")

                name_list.append(image_file_name[:-7])

                image_file_name = os.path.join(self.data_dir+img_folder, image_file_name)
                segmentation_file_name = os.path.join(self.data_dir+seg_folder, segmentation_file_name)
                prior_file_name = os.path.join(self.data_dir+prior_folder, prior_file_name)

                if not os.path.exists(image_file_name):
                    print(image_file_name + ' not exist!')
                    continue
                if not os.path.exists(segmentation_file_name):
                    print(segmentation_file_name + ' not exist!')
                    continue
                if not os.path.exists(prior_file_name):
                    print(prior_file_name + ' not exist!')
                    continue

                image_list.append(sitk.ReadImage(image_file_name))
                segmentation_list.append(sitk.ReadImage(segmentation_file_name))
                prior_list.append(sitk.ReadImage(prior_file_name))

        return image_list, segmentation_list, prior_list, name_list

