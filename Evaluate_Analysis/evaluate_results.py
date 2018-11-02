#!/usr/bin/env python2

import numpy as np
import sys
import scipy.spatial
import SimpleITK as sitk
import matplotlib.pyplot as plt

def Get_Dice_of_each_label(pred_label, gt_label):
    """
    Get dice score of two labels
    :param pred_label: predicted label
    :param gt_label: ground true label
    :return: a list of dice score, i.e., each label has a dice score
    """
    src_label = sitk.ReadImage(pred_label)
    tar_label = sitk.ReadImage(gt_label)
    src_label_array = sitk.GetArrayFromImage(src_label)
    tar_label_array = sitk.GetArrayFromImage(tar_label)

    N = len(np.unique(src_label_array))
    print(pred_label + ' has {} of labels'.format(N))
    dice_all = np.zeros(N)
    for i in range(N):
        pred = src_label_array == i
        gt = tar_label_array == i
        dice_all[i] = 1.0 - scipy.spatial.distance.dice(pred.reshape(-1), gt.reshape(-1))

    return dice_all

def store_dice_score_data(pred_dir, gt_dir, img_num, label_num, save_name):
    """
    Save dice score of testing data for plot
    :param pred_dir: prediction label directory
    :param gt_dir: ground truth label directory
    :param img_num: number of atlases
    :param label_num: number of labels
    :param save_name: save file name
    :return: Dice score saved in a file
    """

    res = np.zeros(img_num, label_num)

    for i in range(33, 41):
        pred_lab = pred_dir + 's' + str(i) + '_prediction_16_reflect.nii.gz'
        gt_lab = gt_dir + 's' + str(i) + '.nii'

        res[i-33, :] = Get_Dice_of_each_label(pred_lab, gt_lab)

    np.save(save_name, res)


if __name__ == '__main__':
    res_unet_seg = np.zeros(8)
    res_unet_seg_std = np.zeros(8)
    res_all_majority_voting = np.zeros(8)
    res_all_majority_voting_std = np.zeros(8)
    res_top_6_majority_voting = np.zeros(8)
    res_top_6_majority_voting_std = np.zeros(8)

    for i in range(33, 41):
        pred_lab = '../U-Net/results/UNet3D_bias_LPBA40_LPBA40_fold_1_train_patch_72_72_72_batch_4_sampleThreshold_0.005_lr_0.001_scheduler_multiStep_10102018_005414/s' + str(i) + '_prediction_16_reflect.nii.gz'
        gt_lab = '../Data/LPBA40/corr_label/s' + str(i) + '.nii'
        res_unet_seg[i-33] = np.mean(Get_Dice_of_each_label(pred_lab, gt_lab))
        res_unet_seg_std[i - 33] = np.std(Get_Dice_of_each_label(pred_lab, gt_lab))

    for j in range(33, 41):
        pred_lab = '../Label_Fusion/all_atlases_majority_voting/all_atlases_majority_voting_s' + str(j) + '.nii.gz'
        gt_lab = '../Data/LPBA40/corr_label/s' + str(j) + '.nii'
        res_all_majority_voting[j - 33] = np.mean(Get_Dice_of_each_label(pred_lab, gt_lab))
        res_all_majority_voting_std[j - 33] = np.std(Get_Dice_of_each_label(pred_lab, gt_lab))

    for k in range(33, 41):
        pred_lab = '../Label_Fusion/top_6_atlases_majority_voting/top_6_atlases_majority_voting_s' + str(j) + '.nii.gz'
        gt_lab = '../Data/LPBA40/corr_label/s' + str(j) + '.nii'
        res_top_6_majority_voting[k - 33] = np.mean(Get_Dice_of_each_label(pred_lab, gt_lab))
        res_top_6_majority_voting_std[k - 33] = np.std(Get_Dice_of_each_label(pred_lab, gt_lab))

    ind = np.arange(len(res_unet_seg))  # the x locations for the groups
    width = 0.3  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width, res_unet_seg, width, yerr=res_unet_seg_std, color='SkyBlue', label='Unet')
    rects2 = ax.bar(ind, res_all_majority_voting, width, yerr=res_all_majority_voting_std, color='IndianRed', label='All Majority Voting')
    rects3 = ax.bar(ind + width, res_top_6_majority_voting, width, yerr=res_top_6_majority_voting_std, color='Blue', label='Top 6 Majority Voting')

    ax.set_ylabel('Dice Score')
    ax.set_title('Scores by Methods')
    ax.set_xticks(ind)
    ax.set_xticklabels(('S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40'))
    ax.legend()
    fig.savefig('test.png', dpi=1000)
    # plt.show()