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

def store_dice_score_data(pred_dir, gt_dir, img_num, label_num, save_name, res_type):
    """
    Save dice score of testing data for plot
    :param pred_dir: prediction label directory
    :param gt_dir: ground truth label directory
    :param img_num: number of atlases
    :param label_num: number of labels
    :param save_name: save file name
    :return: Dice score saved in a file
    """

    res = np.zeros((img_num, label_num))

    for i in range(33, 41):
        if res_type == 'unet':
            pred_lab = pred_dir + 's' + str(i) + '_prediction_16_reflect.nii.gz'
        elif res_type == 'all_atlases':
            pred_lab = pred_dir + 'all_atlases_majority_voting_s' + str(i) + '.nii.gz'
        elif res_type == 'top_k_global':
            pred_lab = pred_dir + 'top_15_atlases_majority_voting_s' + str(i) + '.nii.gz'
        elif res_type == 'top_k_local':
            pred_lab = pred_dir + 'top_15_atlases_local_majority_voting_s' + str(i) + '.nii.gz'
        elif res_type == 'voted_global':
            pred_lab = pred_dir + 'voted_common_atlas.nii.gz'
        else:
            raise TypeError('Undefined experiment type!')
        gt_lab = gt_dir + 's' + str(i) + '.nii'

        res[i-33, :] = Get_Dice_of_each_label(pred_lab, gt_lab)

    np.save(save_name, res)


if __name__ == '__main__':
    ## save results as npy file
    # store_dice_score_data(pred_dir='../U-Net/results/UNet3D_bias_LPBA40_LPBA40_fold_1_train_patch_72_72_72_batch_4_sampleThreshold_0.005_lr_0.001_scheduler_multiStep_10102018_005414/', gt_dir='../Data/LPBA40/corr_label/', img_num=8, label_num=57, save_name='./unet_seg_res.npy', res_type='unet')
    # store_dice_score_data(pred_dir='../Label_Fusion/all_atlases_majority_voting/', gt_dir='../Data/LPBA40/corr_label/', img_num=8, label_num=57, save_name='./all_atlases_seg_res.npy', res_type='all_atlases')
    # store_dice_score_data(pred_dir='../Label_Fusion/top_6_atlases_majority_voting/', gt_dir='../Data/LPBA40/corr_label/', img_num=8, label_num=57, save_name='./top_k_global_seg_res.npy', res_type='top_k_global')
    # store_dice_score_data(pred_dir='../Label_Fusion/top_6_atlases_local_majority_voting/', gt_dir='../Data/LPBA40/corr_label/', img_num=8, label_num=57, save_name='./top_k_local_seg_res.npy', res_type='top_k_local')
    # store_dice_score_data(pred_dir='../Label_Fusion/top_15_atlases_majority_voting/',
    #                       gt_dir='../Data/LPBA40/corr_label/', img_num=8, label_num=57,
    #                       save_name='./top_15_global_seg_res.npy', res_type='top_k_global')
    # store_dice_score_data(pred_dir='../Label_Fusion/top_15_atlases_local_majority_voting/',
    #                       gt_dir='../Data/LPBA40/corr_label/', img_num=8, label_num=57,
    #                       save_name='./top_15_local_seg_res.npy', res_type='top_k_local')
    # store_dice_score_data(pred_dir='../Label_Fusion/',
    #                       gt_dir='../Data/LPBA40/corr_label/', img_num=8, label_num=57,
    #                       save_name='./voted_global_seg_res.npy', res_type='voted_global')

    ## draw bar graph
    # res_unet_seg = np.load('./results_folder/unet_seg_res.npy')
    # res_unet_seg_mean = np.mean(res_unet_seg, axis=1)
    # res_unet_seg_std = np.std(res_unet_seg, axis=1)
    #
    # res_all_majority_voting = np.load('./results_folder/all_atlases_seg_res.npy')
    # res_all_majority_voting_mean = np.mean(res_all_majority_voting, axis=1)
    # res_all_majority_voting_std = np.std(res_all_majority_voting, axis=1)
    #
    # res_top_6_majority_voting = np.load('./results_folder/voted_global_seg_res.npy')
    # res_top_6_majority_voting_mean = np.mean(res_top_6_majority_voting, axis=1)
    # res_top_6_majority_voting_std = np.std(res_top_6_majority_voting, axis=1)
    #
    # res_top_6_local_majority_voting = np.load('./results_folder/top_15_local_seg_res.npy')
    # res_top_6_local_majority_voting_mean = np.mean(res_top_6_local_majority_voting, axis=1)
    # res_top_6_local_majority_voting_std = np.std(res_top_6_local_majority_voting, axis=1)
    #
    # ind = np.arange(len(res_unet_seg))  # the x locations for the groups
    # width = 0.2  # the width of the bars
    #
    # fig, ax = plt.subplots()
    # rects1 = ax.bar(ind - 2*width, res_unet_seg_mean, width, yerr=res_unet_seg_std, color='SkyBlue', label='Unet')
    # rects2 = ax.bar(ind - width, res_all_majority_voting_mean, width, yerr=res_all_majority_voting_std, color='IndianRed', label='All Majority Voting')
    # rects3 = ax.bar(ind, res_top_6_majority_voting_mean, width, yerr=res_top_6_majority_voting_std, color='Blue', label='Top 6 Majority Voting')
    # rects4 = ax.bar(ind + width, res_top_6_local_majority_voting_mean, width, yerr=res_top_6_local_majority_voting_std, color='Red', label='Top 6 Local Majority Voting')
    #
    #
    # ax.set_ylabel('Dice Score')
    # ax.set_title('Scores by Methods')
    # ax.set_xticks(ind)
    # ax.set_xticklabels(('S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40'))
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # fig.savefig('test_3.png', dpi=1000)

    #
    # for i in range(33, 41):
    #     pred_lab = '../U-Net/results/UNet3D_bias_LPBA40_LPBA40_fold_1_train_patch_72_72_72_batch_4_sampleThreshold_0.005_lr_0.001_scheduler_multiStep_10102018_005414/s' + str(i) + '_prediction_16_reflect.nii.gz'
    #     gt_lab = '../Data/LPBA40/corr_label/s' + str(i) + '.nii'
    #     res_unet_seg[i-33] = np.mean(Get_Dice_of_each_label(pred_lab, gt_lab))
    #     res_unet_seg_std[i - 33] = np.std(Get_Dice_of_each_label(pred_lab, gt_lab))
    #
    # for j in range(33, 41):
    #     pred_lab = '../Label_Fusion/all_atlases_majority_voting/all_atlases_majority_voting_s' + str(j) + '.nii.gz'
    #     gt_lab = '../Data/LPBA40/corr_label/s' + str(j) + '.nii'
    #     res_all_majority_voting[j - 33] = np.mean(Get_Dice_of_each_label(pred_lab, gt_lab))
    #     res_all_majority_voting_std[j - 33] = np.std(Get_Dice_of_each_label(pred_lab, gt_lab))
    #
    # for k in range(33, 41):
    #     pred_lab = '../Label_Fusion/top_6_atlases_majority_voting/top_6_atlases_majority_voting_s' + str(j) + '.nii.gz'
    #     gt_lab = '../Data/LPBA40/corr_label/s' + str(j) + '.nii'
    #     res_top_6_majority_voting[k - 33] = np.mean(Get_Dice_of_each_label(pred_lab, gt_lab))
    #     res_top_6_majority_voting_std[k - 33] = np.std(Get_Dice_of_each_label(pred_lab, gt_lab))
    #
    # ind = np.arange(len(res_unet_seg))  # the x locations for the groups
    # width = 0.3  # the width of the bars
    #
    # fig, ax = plt.subplots()
    # rects1 = ax.bar(ind - width, res_unet_seg, width, yerr=res_unet_seg_std, color='SkyBlue', label='Unet')
    # rects2 = ax.bar(ind, res_all_majority_voting, width, yerr=res_all_majority_voting_std, color='IndianRed', label='All Majority Voting')
    # rects3 = ax.bar(ind + width, res_top_6_majority_voting, width, yerr=res_top_6_majority_voting_std, color='Blue', label='Top 6 Majority Voting')
    #
    # ax.set_ylabel('Dice Score')
    # ax.set_title('Scores by Methods')
    # ax.set_xticks(ind)
    # ax.set_xticklabels(('S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40'))
    # ax.legend()
    # fig.savefig('test.png', dpi=1000)
    # # plt.show()