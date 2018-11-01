from __future__ import print_function
import numpy as np
import SimpleITK as sitk
import sys
sys.path.append('../Evaluate_Analysis')
from evaluate_results import *

atlases_dir = '../Data/NiftyReg/target_s33/'
target_label = '../U-Net/results/UNet3D_bias_LPBA40_LPBA40_fold_1_train_patch_72_72_72_batch_4_sampleThreshold_0.005_lr_0.001_scheduler_multiStep_10102018_005414/s33_prediction_16_reflect.nii.gz'

def all_atlases_majority_voting(atlases_dir, img_num, save_name):
    """
    Majority voting to generate new label for target image
    :param atlases_dir: the directory of warped atlases into target image space for forging target label
    :param img_num: number of atlases
    :param save_name: saved target label name
    :return: saved target label from atlases majority voting
    """

    tmp_img = sitk.ReadImage(atlases_dir + 's1/s1_warp.nii')
    tmp_label = sitk.GetArrayFromImage(tmp_img)
    N = len(np.unique(tmp_label))

    warped_atlases_array = np.zeros((N, tmp_label.shape[0], tmp_label.shape[1], tmp_label.shape[2]))
    for i in range(img_num):
        img_label_array = sitk.GetArrayFromImage(sitk.ReadImage(atlases_dir + 's' + str(i+1) + '/s' + str(i+1) + '_warp.nii'))
        for j in range(N):
            warped_atlases_array[j,...] = warped_atlases_array[j,...] + (img_label_array == j).astype('int')

    voted_result = np.argmax(warped_atlases_array, axis=0)
    voted_img = sitk.GetImageFromArray(voted_result.astype('float32'))
    voted_img.CopyInformation(tmp_img)

    sitk.WriteImage(voted_img, save_name)
    print('save furged image {}'.format(save_name))



def select_k_highest_dice_score_global(atlases_dir, target_label, img_num, save_name, k=6):
    tmp_img = sitk.ReadImage(atlases_dir + 's1/s1_warp.nii')
    tmp_label = sitk.GetArrayFromImage(tmp_img)
    N = len(np.unique(tmp_label))

    warped_atlases_array = np.zeros((N, tmp_label.shape[0], tmp_label.shape[1], tmp_label.shape[2]))
    for i in range(img_num):
        img_label_array = sitk.GetArrayFromImage(sitk.ReadImage(atlases_dir + 's' + str(i+1) + '/s' + str(i+1) + '_warp.nii'))
        for j in range(N):
            warped_atlases_array[j,...] = warped_atlases_array[j,...] + (img_label_array == j).astype('int')

    voted_result = np.argmax(warped_atlases_array, axis=0)
    voted_img = sitk.GetImageFromArray(voted_result.astype('float32'))
    voted_img.CopyInformation(tmp_img)

    sitk.WriteImage(voted_img, save_name)
    print('save furged image {}'.format(save_name))

def select_k_highest_dice_score_local(atlases_dir, k=6):
    pass


def top_k_highest_dice_score_global(atlases_dir, img_num, target_label, k=6):

    res = np.zeros(img_num)
    for i in range(img_num):
        atlas_name = atlases_dir + 's' + str(i + 1) + '/s' + str(i + 1) + '_warp.nii'
        res[i] = np.mean(Get_Dice_of_each_label(atlas_name, target_label))




    tmp_img = sitk.ReadImage(atlases_dir + 's1/s1_warp.nii')
    N = len(np.unique(tmp_label))





"""
pred_lab = '../U-Net/results/UNet3D_bias_LPBA40_LPBA40_fold_1_train_patch_72_72_72_batch_4_sampleThreshold_0.005_lr_0.001_scheduler_multiStep_10102018_005414/s33_prediction_16_reflect.nii.gz'
gt_lab = '../Data/LPBA40/corr_label/s33.nii'
res = Get_Dice_of_each_label(pred_lab, gt_lab)
print(res)
"""

if __name__ == '__main__':
    pass
    # all atlases majority voting
    # for i in range(33, 41):
    #     all_atlases_majority_voting('../Data/NiftyReg/target_s' + str(i) + '/', img_num=28, save_name='./all_atlases_majority_voting_s' + str(i) + '.nii.gz')