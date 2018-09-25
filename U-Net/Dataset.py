"""
    Code referenced from Zhenlin Xu on OAI segmentation project @ https://github.com/uncbiag/OAI_segmentation/blob/master/utils/datasets.py

"""

import SimpleITK as sitk
import os
from torch.utils.data import Dataset
from Transform import my_balanced_random_crop
from numpy import random
import numpy as np


class NiftiDataset(Dataset):
    """
    a dataset class to load medical image data in Nifti format using simpleITK
    """

    def __init__(self, txt_file, data_dir, mode, scale_ratio, bg_th_ratio, label_list, label_density, img_size, max_crop_num, patch_size, preload=False, transform=None):
        """
        :param txt_file: txt file with lines of "image_file, segmentation_file"
        :param root_dir: the data root dir
        :param transform: transformations on a sample for data augmentation
        """
        self.data_dir = data_dir
        self.mode = mode
        self.preload = preload
        self.transform = transform
        self.transform_crop = True
        self.scale_ratio = scale_ratio
        self.bg_th_ratio = bg_th_ratio
        self.label_list = label_list
        self.label_density = label_density
        self.img_size = img_size
        self.max_crop_num = max_crop_num
        self.patch_size = patch_size

        if preload:
            print("Preloading data:")
            self.image_list, self.segmentation_list, self.name_list = self.read_image_segmentation(txt_file)
            print('Preloaded {} training samples'.format(len(self.image_list)))
        else:
            self.image_list, self.segmentation_list, self.name_list = self.read_image_segmentation_list(txt_file)

        if len(self.image_list) != len(self.segmentation_list):
            raise ValueError("The numbers of images and segmentations are different")
        self.__init_transform_list()

    def __len__(self):
        return self.image_list.__len__()*1000


    def __init_transform_list(self):
        self.transform_list = [my_balanced_random_crop(self.scale_ratio, self.bg_th_ratio, self.label_list, self.label_density, self.img_size, self.max_crop_num, self.patch_size) for _ in range(len(self.image_list))]

    def __getitem__(self, id):
        rand_label_id = random.randint(0, 1000)
        id = id % len(self.image_list)
        if self.preload:
            image = self.image_list[id]
            segmentation = self.segmentation_list[id]
        else:
            img_floder = '/brain_affine_icbm_hist_oasis/'
            seg_folder = '/label_affine_icbm/'
            image_file_name = os.path.join(self.data_dir+img_floder, self.image_list[id])
            segmentation_file_name = os.path.join(self.data_dir+seg_folder, self.segmentation_list[id])

            # check file existence
            if not os.path.exists(image_file_name):
                print(image_file_name + ' not exist!')
                return
            if not os.path.exists(segmentation_file_name):
                print(segmentation_file_name + ' not exist!')
                return
            image = sitk.ReadImage(image_file_name)
            segmentation = sitk.ReadImage(segmentation_file_name)

        image_name = self.name_list[id]
        sample = {'image': image, 'segmentation': segmentation, 'name': image_name}

        if self.transform_crop:
            sample = self.transform_list[id](sample,rand_label_id)

        if self.transform:
            sample = self.transform(sample)

        # sitk_to_tensor = bio_transform.SitkToTensor()
        # tensor_sample = sitk_to_tensor(sample)
        return sample['image'], sample['segmentation'], sample['name']

    def read_image_segmentation_list(self, text_file):
        image_list = []
        segmentation_list = []
        name_list = []
        with open(text_file) as file:
            for line in file:
                try:
                    image, seg = line.strip("\n").split(', ')
                except ValueError:  # Adhoc for test.
                    image = seg = line.strip("\n")
                image_list.append(image)
                segmentation_list.append(seg)
                name_list.append(image[:-4])
        return image_list, segmentation_list, name_list

    def read_image_segmentation(self, text_file):
        image_list = []
        segmentation_list = []
        name_list = []
        img_floder = '/brain_affine_icbm_hist_oasis/'
        seg_folder = '/label_affine_icbm/'
        with open(text_file) as file:
            for line in file:
                try:
                    image_file_name, segmentation_file_name = line.strip("\n").split(', ')
                except ValueError:  # Adhoc for test.
                    image_file_name = segmentation_file_name = line.strip("\n")

                name_list.append(image_file_name[:-4])

                image_file_name = os.path.join(self.data_dir+img_floder, image_file_name)
                segmentation_file_name = os.path.join(self.data_dir+seg_folder, segmentation_file_name)
                # check file existence
                if not os.path.exists(image_file_name):
                    print(image_file_name + ' not exist!')
                    continue
                if not os.path.exists(segmentation_file_name):
                    print(segmentation_file_name + ' not exist!')
                    continue

                image_list.append(sitk.ReadImage(image_file_name))
                segmentation_list.append(sitk.ReadImage(segmentation_file_name))

        return image_list, segmentation_list, name_list

