from evaluate_results import *



## save results as npy file
# Unet
store_dice_score_data(pred_dir='../U-Net/results/UNet3D_bias_LPBA40_LPBA40_fold_1_train_patch_72_72_72_batch_4_sampleThreshold_0.005_lr_0.001_scheduler_multiStep_10102018_005414/',
                      gt_dir='../Data/LPBA40/corr_label/', img_num=8, label_num=57,
                      save_name='./unet_seg_res.npy', res_type='unet')

# All atlases majority voting
store_dice_score_data(pred_dir='../Label_Fusion/all_atlases_majority_voting/',
                      gt_dir='../Data/LPBA40/corr_label/', img_num=8, label_num=57,
                      save_name='./all_atlases_seg_res.npy', res_type='all_atlases')

# Top 6 global majority voting
store_dice_score_data(pred_dir='../Label_Fusion/top_6_atlases_majority_voting/',
                      gt_dir='../Data/LPBA40/corr_label/', img_num=8, label_num=57,
                      save_name='./top_k_global_seg_res.npy', res_type='top_k_global')

# Top 6 local majority voting
store_dice_score_data(pred_dir='../Label_Fusion/top_6_atlases_local_majority_voting/',
                      gt_dir='../Data/LPBA40/corr_label/', img_num=8, label_num=57,
                      save_name='./top_k_local_seg_res.npy', res_type='top_k_local')

# Top 15 global majority voting
store_dice_score_data(pred_dir='../Label_Fusion/top_15_atlases_majority_voting/',
                      gt_dir='../Data/LPBA40/corr_label/', img_num=8, label_num=57,
                      save_name='./top_15_global_seg_res.npy', res_type='top_k_global')

# Top 15 local majority voting
store_dice_score_data(pred_dir='../Label_Fusion/top_15_atlases_local_majority_voting/',
                      gt_dir='../Data/LPBA40/corr_label/', img_num=8, label_num=57,
                      save_name='./top_15_local_seg_res.npy', res_type='top_k_local')

# Baseline: voted globally (no registration involved, i.e., one common segmentation result for all targets)
store_dice_score_data(pred_dir='../Label_Fusion/',
                      gt_dir='../Data/LPBA40/corr_label/', img_num=8, label_num=57,
                      save_name='./voted_global_seg_res.npy', res_type='voted_global')