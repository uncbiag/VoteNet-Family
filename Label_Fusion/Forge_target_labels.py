from __future__ import print_function
import numpy as np
import SimpleITK as sitk
import sys
sys.path.append('../Evaluate_Analysis')
from evaluate_results import *
from collections import Counter

atlases_dir = '../Data/NiftyReg/target_s33/'
target_label = '../U-Net/results/UNet3D_bias_LPBA40_LPBA40_fold_1_train_patch_72_72_72_batch_4_sampleThreshold_0.005_lr_0.001_scheduler_multiStep_10102018_005414/s33_prediction_16_reflect.nii.gz'

def all_atlases_majority_voting(atlases_dir):
    tmp_label = sitk.GetArrayFromImage(sitk.ReadImage(atlases_dir + 's1/s1_warp.nii'))
    N = len(np.unique(tmp_label))
    warped_atlases_array = np.zeros((28, N, tmp_label.shape[0], tmp_label.shape[1], tmp_label.shape[2]))
    for i in range(28):
        img_label_array = sitk.GetArrayFromImage(sitk.ReadImage(atlases_dir + 's' + str(i+1) + '/s' + str(i+1) + '_warp.nii'))
        for j in range(N):
            print(i, j)
            warped_atlases_array[i,j,...] = (img_label_array == j).astype('int')

    warped_atlases_class_sum = np.sum(warped_atlases_array, axis=0)
    voted_result = np.max(warped_atlases_class_sum, axis=0)
    voted_img = sitk.GetImageFromArray(voted_result)

    sitk.WriteImage(voted_img, './all_atlases_majority_voting_s33.nii.gz')



def select_k_highest_dice_score_global(atlases_dir, k=6):
    pass

def select_k_highest_dice_score_local(atlases_dir, k=6):
    pass





"""
pred_lab = '../U-Net/results/UNet3D_bias_LPBA40_LPBA40_fold_1_train_patch_72_72_72_batch_4_sampleThreshold_0.005_lr_0.001_scheduler_multiStep_10102018_005414/s33_prediction_16_reflect.nii.gz'
gt_lab = '../Data/LPBA40/corr_label/s33.nii'
res = Get_Dice_of_each_label(pred_lab, gt_lab)
print(res)
"""

if __name__ == '__main__':
    all_atlases_majority_voting('../Data/NiftyReg/target_s33/')