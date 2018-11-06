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
    """
    Use the atlases of highest k dice scores to furge the target label
    :param atlases_dir: the directory of warped atlases into target image space for forging target label
    :param target_label: Unet segmentated target image label
    :param img_num: number of atlases
    :param save_name: saved target label name
    :param k: number of atlases chosen
    :return: saved target label from top k atlases majority voting
    """

    tmp_img = sitk.ReadImage(atlases_dir + 's1/s1_warp.nii')
    tmp_label = sitk.GetArrayFromImage(tmp_img)
    N = len(np.unique(tmp_label))

    top_k = top_k_highest_dice_score_global(atlases_dir=atlases_dir, img_num=img_num, target_label=target_label, k=k)
    warped_atlases_array = np.zeros((N, tmp_label.shape[0], tmp_label.shape[1], tmp_label.shape[2]))
    for ind in top_k:
        img_label_array = sitk.GetArrayFromImage(sitk.ReadImage(atlases_dir + 's' + str(ind+1) + '/s' + str(ind+1) + '_warp.nii'))
        for j in range(N):
            warped_atlases_array[j,...] = warped_atlases_array[j,...] + (img_label_array == j).astype('int')

    voted_result = np.argmax(warped_atlases_array, axis=0)
    voted_img = sitk.GetImageFromArray(voted_result.astype('float32'))
    voted_img.CopyInformation(tmp_img)

    sitk.WriteImage(voted_img, save_name)
    print('save furged image {}'.format(save_name))


def select_k_highest_dice_score_local(atlases_dir, target_label, img_num, save_name, k=6):
    """
    Use the atlases of highest k dice scores of each individual label to furge the target label
    :param atlases_dir: the directory of warped atlases into target image space for forging target label
    :param target_label: Unet segmentated target image label
    :param img_num: number of atlases
    :param save_name: saved target label name
    :param k: number of atlases chosen
    :return: saved target label from top k atlases majority voting locally
    """

    tmp_img = sitk.ReadImage(atlases_dir + 's1/s1_warp.nii')
    tmp_label = sitk.GetArrayFromImage(tmp_img)
    N = len(np.unique(tmp_label))

    top_k = top_k_highest_dice_score_local(atlases_dir=atlases_dir, img_num=img_num, target_label=target_label, k=k)
    warped_atlases_array = np.zeros((N, tmp_label.shape[0], tmp_label.shape[1], tmp_label.shape[2]))
    for label_ind in range(len(top_k)):
        for ind in top_k[label_ind]:
            img_label_array = sitk.GetArrayFromImage(sitk.ReadImage(atlases_dir + 's' + str(ind+1) + '/s' + str(ind+1) + '_warp.nii'))
            warped_atlases_array[label_ind,...] = warped_atlases_array[label_ind,...] + (img_label_array == label_ind).astype('int')

    voted_result = np.argmax(warped_atlases_array, axis=0)
    voted_img = sitk.GetImageFromArray(voted_result.astype('float32'))
    voted_img.CopyInformation(tmp_img)

    sitk.WriteImage(voted_img, save_name)
    print('save furged image {}'.format(save_name))


def select_k_highest_ssd_intensity_global(atlases_dir, target_label, img_num, save_name, k=6):
    """
    Use the atlases of highest k dice scores to furge the target label
    :param atlases_dir: the directory of warped atlases into target image space for forging target label
    :param target_label: Unet segmentated target image label
    :param img_num: number of atlases
    :param save_name: saved target label name
    :param k: number of atlases chosen
    :return: saved target label from top k atlases majority voting
    """

    tmp_img = sitk.ReadImage(atlases_dir + 's1/s1_warp.nii')
    tmp_label = sitk.GetArrayFromImage(tmp_img)
    N = len(np.unique(tmp_label))

    top_k = top_k_highest_ssd_intensity_global(gt_dir=atlases_dir, img_num=img_num, target_label=target_label, k=k)
    warped_atlases_array = np.zeros((N, tmp_label.shape[0], tmp_label.shape[1], tmp_label.shape[2]))
    for ind in top_k:
        img_label_array = sitk.GetArrayFromImage(sitk.ReadImage(atlases_dir + 's' + str(ind+1) + '/s' + str(ind+1) + '_warp.nii'))
        for j in range(N):
            warped_atlases_array[j,...] = warped_atlases_array[j,...] + (img_label_array == j).astype('int')

    voted_result = np.argmax(warped_atlases_array, axis=0)
    voted_img = sitk.GetImageFromArray(voted_result.astype('float32'))
    voted_img.CopyInformation(tmp_img)

    sitk.WriteImage(voted_img, save_name)
    print('save furged image {}'.format(save_name))


def top_k_highest_dice_score_global(atlases_dir, img_num, target_label, k=6):
    """
    The index of the top k highest dice score between warped atlases and Unet segmented target label
    :param atlases_dir: the directory of warped atlases into target image space for forging target label
    :param img_num: number of atlases
    :param target_label: Unet segmentated target image label
    :param k: number of atlases chosen
    :return: the index of top k highest dice score between warped atlases and Unet segmented target label
    """

    res = np.zeros(img_num)
    for i in range(img_num):
        atlas_name = atlases_dir + 's' + str(i + 1) + '/s' + str(i + 1) + '_warp.nii'
        res[i] = np.mean(Get_Dice_of_each_label(atlas_name, target_label))

    res_k = np.argsort(res)[-k:]

    return res_k


def top_k_highest_dice_score_local(atlases_dir, img_num, target_label, k=6):
    """
    The index of the top k highest dice score of each label between warped atlases and Unet segmented target label
    :param atlases_dir: the directory of warped atlases into target image space for forging target label
    :param img_num: number of atlases
    :param target_label: Unet segmentated target image label
    :param k: number of atlases chosen
    :return: the index list of top k highest dice score of each label between warped atlases and Unet segmented target label
    """

    tmp_img = sitk.ReadImage(target_label)
    tmp_label = sitk.GetArrayFromImage(tmp_img)
    N = len(np.unique(tmp_label))
    res = np.zeros((N, img_num))
    for i in range(img_num):
        atlas_name = atlases_dir + 's' + str(i + 1) + '/s' + str(i + 1) + '_warp.nii'
        res[:,i] = np.transpose(Get_Dice_of_each_label(atlas_name, target_label))

    res_k = np.argsort(res, axis=1)[:, -k:]

    return res_k


def top_k_highest_ssd_intensity_global_nonwarp(gt_dir, img_num, target_img, k=6):
    """
    The index of the top k lowest intensity difference between the ground truth (nonwarped) atlases and target image
    :param gt_dir: the directory of ground truth atlases
    :param img_num: number of atlases
    :param target_img: the traget image
    :param k: number of atlases chosen
    :return: the index list of top k lowest intensity difference between ground truth (nonwarped) atlases and the target image
    """

    res = np.zeros(img_num)
    target_img = sitk.ReadImage(target_img)
    target_img_array = sitk.GetArrayFromImage(target_img)
    for i in range(img_num):
        atlas_name = gt_dir + 's' + str(i + 1) + '.nii'
        atlas_img = sitk.ReadImage(atlas_name)
        atlas_img_array = sitk.GetArrayFromImage(atlas_img)
        res[i] = np.sum((atlas_img_array - target_img_array)**2)

    res_k = np.argsort(res)[-k:]

    return res_k


def top_k_highest_ssd_intensity_global_warp(atlases_dir, img_num, target_img, k=6):
    """
    The index of the top k lowest intensity difference between the warped atlases and target image
    :param atlases_dir: the directory of warped atlases
    :param img_num: number of atlases
    :param target_img: the traget image
    :param k: number of atlases chosen
    :return: the index list of top k lowest intensity difference between warped atlases and the target image
    """

    res = np.zeros(img_num)
    target_img = sitk.ReadImage(target_img)
    target_img_array = sitk.GetArrayFromImage(target_img)
    for i in range(img_num):
        atlas_name = atlases_dir + 's' + str(i + 1) + '/s' + str(i + 1) + '_res.nii'
        atlas_img = sitk.ReadImage(atlas_name)
        atlas_img_array = sitk.GetArrayFromImage(atlas_img)
        res[i] = np.sum((atlas_img_array - target_img_array)**2)

    res_k = np.argsort(res)[-k:]

    return res_k

def weighted_voting_local_intensity_gaussian(atlases_dir, img_num, target_img):
    pass


def weighted_voting_local_intensity_inverse_distance(atlases_dir, img_num, target_img):
    pass


def weighted_voting_local_intensity_joint_fusion(atlases_dir, img_num, atrget_img):
    pass

def weighted_voting_local_dice_gaussian(atlases_dir, img_num, target_img):
    pass


def weighted_voting_local_dice_inverse_distance(atlases_dir, img_num, target_img):
    pass


def weighted_voting_local_dice_joint_fusion(atlases_dir, img_num, atrget_img):
    pass


def voted_gloabl_label(gt_dir, img_num, save_name):
    """
    [Baseline] Use all atlases to vote a common segmentation to account for the segmentation result for each target
    :param gt_dir: the directory of the ground truth atlases
    :param img_num: number of atlases
    :param save_name: saved target label name
    :return: The common segmentation generated from atlases voting
    """

    tmp_img = sitk.ReadImage(gt_dir + 's1.nii')
    tmp_label = sitk.GetArrayFromImage(tmp_img)
    N = len(np.unique(tmp_label))

    warped_atlases_array = np.zeros((N, tmp_label.shape[0], tmp_label.shape[1], tmp_label.shape[2]))
    for i in range(img_num):
        img_label_array = sitk.GetArrayFromImage(sitk.ReadImage(gt_dir + 's' + str(i+1) + '.nii'))
        for j in range(N):
            warped_atlases_array[j,...] = warped_atlases_array[j,...] + (img_label_array == j).astype('int')

    voted_result = np.argmax(warped_atlases_array, axis=0)
    voted_img = sitk.GetImageFromArray(voted_result.astype('float32'))
    voted_img.CopyInformation(tmp_img)

    sitk.WriteImage(voted_img, save_name)
    print('save furged image {}'.format(save_name))



if __name__ == '__main__':
    pass
    ## all atlases majority voting
    # for i in range(33, 41):
    #     all_atlases_majority_voting('../Data/NiftyReg/target_s' + str(i) + '/', img_num=28, save_name='./all_atlases_majority_voting_s' + str(i) + '.nii.gz')

    ## top k highest dice score global
    # print(top_k_highest_dice_score_global(atlases_dir='../Data/NiftyReg/target_s33/', img_num=28, target_label='../U-Net/results/UNet3D_bias_LPBA40_LPBA40_fold_1_train_patch_72_72_72_batch_4_sampleThreshold_0.005_lr_0.001_scheduler_multiStep_10102018_005414/s33_prediction_16_reflect.nii.gz'))

    ## top k highest dice score global label fusion
    # for i in range(33, 41):
    #     select_k_highest_dice_score_global(atlases_dir='../Data/NiftyReg/target_s' + str(i) + '/', target_label='../U-Net/results/UNet3D_bias_LPBA40_LPBA40_fold_1_train_patch_72_72_72_batch_4_sampleThreshold_0.005_lr_0.001_scheduler_multiStep_10102018_005414/s' + str(i) + '_prediction_16_reflect.nii.gz', img_num=28, save_name='./top_15_atlases_majority_voting_s' + str(i) + '.nii.gz', k=15)

    ## top k highest dice score local
    # print(top_k_highest_dice_score_local(atlases_dir='../Data/NiftyReg/target_s33/', img_num=28, target_label='../U-Net/results/UNet3D_bias_LPBA40_LPBA40_fold_1_train_patch_72_72_72_batch_4_sampleThreshold_0.005_lr_0.001_scheduler_multiStep_10102018_005414/s33_prediction_16_reflect.nii.gz', k=6))

    ## top k highest dice score local label fusion
    # for i in range(33, 41):
    #     select_k_highest_dice_score_local(atlases_dir='../Data/NiftyReg/target_s' + str(i) + '/', target_label='../U-Net/results/UNet3D_bias_LPBA40_LPBA40_fold_1_train_patch_72_72_72_batch_4_sampleThreshold_0.005_lr_0.001_scheduler_multiStep_10102018_005414/s' + str(i) + '_prediction_16_reflect.nii.gz', img_num=28, save_name='./top_15_atlases_local_majority_voting_s' + str(i) + '.nii.gz', k=15)

    ## global voted atlas
    # voted_gloabl_label(gt_dir='../Data/LPBA40/corr_label/', img_num=28, save_name='voted_common_atlas.nii.gz')