import numpy as np
import sys
import scipy.spatial
import SimpleITK as sitk


def Get_Dice_of_each_label(pred_label, gt_label):
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


if __name__ == '__main__':
    pred_lab = '../U-Net/results/UNet3D_bias_LPBA40_LPBA40_fold_1_train_patch_72_72_72_batch_4_sampleThreshold_0.005_lr_0.001_scheduler_multiStep_10102018_005414/s33_prediction_16_reflect.nii.gz'
    gt_lab = '../Data/LPBA40/corr_label/s33.nii'
    res = Get_Dice_of_each_label(pred_lab, gt_lab)
    print(res)