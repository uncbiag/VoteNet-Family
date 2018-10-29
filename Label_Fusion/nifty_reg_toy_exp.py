import os


# LPBA40
data_dir = '../Data/LPBA40/brain_affine_icbm_hist_oasis/'
label_dir = '../Data/LPBA40/corr_label/'

f = open('bash_script_niftyreg_lpba40_MAS_toy_exp.sh', 'w')

save_registration_path = '../Data/NiftyReg/'
if not os.path.isdir(save_registration_path):
    os.mkdir(save_registration_path)


for tar in range(33, 41):

    save_each_registration_path = save_registration_path + 'target_s' + str(tar) + '/'
    if not os.path.isdir(save_each_registration_path):
        os.mkdir(save_each_registration_path)

    for k in range(1, 29):
        save_each_registration_path_src = save_each_registration_path + 's' + str(k) + '/'
        if not os.path.isdir(save_each_registration_path_src):
            os.mkdir(save_each_registration_path_src)

        f.write('reg_f3d -flo ' + data_dir + 's' + str(k) + '.nii'
                + ' -ref ' + data_dir + 's' + str(tar) + '.nii'
                + ' -res ' + save_each_registration_path_src + 's' + str(k) + '_res.nii'
                + ' -cpp ' + save_each_registration_path_src + 's' + str(k) + '_ctrlpts.nii' + '\n')

        f.write('reg_resample -flo ' + label_dir + 's' + str(k) + '.nii'
                + ' -ref ' + label_dir + 's' + str(tar) + '.nii'
                + ' -trans ' + save_each_registration_path_src + 's' + str(k) + '_ctrlpts.nii'
                + ' -res ' + save_each_registration_path_src + 's' + str(k) + '_warp.nii'
                + ' -inter 0' '\n')

        f.write('\n')

f.close()