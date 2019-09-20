import SimpleITK as sitk
import numpy as np
import os
import glob


# check the number of labels in each image is the same
def check_label(label_dir):
    label_path = os.path.realpath(label_dir)
    label_files_list = glob.glob(label_path+'/*.nii')

    for file in label_files_list:
        I = sitk.ReadImage(file)
        L = sitk.GetArrayFromImage(I)
        L = np.reshape(L, (1, -1))
        L_unique = np.unique(L)
        print(L_unique)
        print("{} labels in file {}".format(len(L_unique), file))


# find the additional label in s37.nii
# Label 116 is only contained in s37.nii, will be set to 0 in later experiment
def find_diff_label(label_dir):
    label_path = os.path.realpath(label_dir)
    label_files_list = glob.glob(label_path+'/*.img')
    tmp_file = label_path + '/21_seg_003.img'
    tmp_I = sitk.ReadImage(tmp_file)
    tmp_L = sitk.GetArrayFromImage(tmp_I)
    tmp_L = np.reshape(tmp_L, (1, -1))
    tmp_L_unique = np.unique(tmp_L)

    for file in label_files_list:
        I = sitk.ReadImage(file)
        L = sitk.GetArrayFromImage(I)
        L = np.reshape(L, (1, -1))
        L_unique = np.unique(L)
        L_diff = np.setdiff1d(L_unique, tmp_L_unique)
        print("label {} in file {} is unique!".format(L_diff, file))


# calculate the label intensity
def get_label_density(label_file_path):
    label_density = []
    I = sitk.ReadImage(label_file_path)
    L = sitk.GetArrayFromImage(I)
    L = np.reshape(L, (1, -1))
    L_unique = np.unique(L)
    for label_ind in L_unique:
        label_density.append(len(L[L==label_ind])*1.0/L.shape[1])

    return label_density


# get the label
def get_label(label_file_path):
    I = sitk.ReadImage(label_file_path)
    L = sitk.GetArrayFromImage(I)
    L = np.reshape(L, (1, -1))

    return np.unique(L)


# get the image size
def get_img_size(image_file_path):
    I = sitk.ReadImage(image_file_path)
    I_array = sitk.GetArrayFromImage(I)

    # note that sitk return image in [z, y, x]
    return [I_array.shape[2], I_array.shape[1], I_array.shape[0]]


# transform label ids into [0, 1, ..., 56]
def label_preprocessing(label_dir):
    tmp_file = label_dir + 's1.nii'
    tmp_I = sitk.ReadImage(tmp_file)
    tmp_L = sitk.GetArrayFromImage(tmp_I)
    tmp_L = np.reshape(tmp_L, (1, -1))
    tmp_L_unique = np.unique(tmp_L)

    save_path = '../Data/LPBA40/corr_label/'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    for i in range(1, 41):
        I = sitk.ReadImage(label_dir + 's' + str(i) + '.nii')
        L = sitk.GetArrayFromImage(I)

        for ind in range(len(tmp_L_unique)):
            L[L==tmp_L_unique[ind]] = ind
        if i == 37:
            L[L==116] = 0

        new_image = sitk.GetImageFromArray(L)
        new_image.SetOrigin((-1, -1, 1))
        new_image.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
        sitk.WriteImage(new_image, save_path + 's' + str(i) + '.nii')


def label_preprocessing_MAMBIS(label_dir):
    tmp_file = label_dir + 's1_seg_003.img'
    tmp_I = sitk.ReadImage(tmp_file)
    tmp_L = sitk.GetArrayFromImage(tmp_I)
    tmp_L = np.reshape(tmp_L, (1, -1))
    tmp_L_unique = np.unique(tmp_L)

    save_path = '/playpen/zpd/remote/MAS/Data/MAMBIS_seg_res/folder_2/MAMBIS_folder_2_result/corr_res/'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    for i in range(9, 10):
        I = sitk.ReadImage(label_dir + 's' + str(i) + '_seg_003.img')
        L = sitk.GetArrayFromImage(I)

        for ind in range(len(tmp_L_unique)):
            L[L==tmp_L_unique[ind]] = ind

        new_image = sitk.GetImageFromArray(L)
        new_image.SetOrigin((-1, -1, 1))
        new_image.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
        sitk.WriteImage(new_image, save_path + 's' + str(i) + '.nii.gz')


# transform label ids into [0, 1, ..., 96]
def label_preprocessing_ibsr(label_dir):
    tmp_file = label_dir + 'c1.nii'
    tmp_I = sitk.ReadImage(tmp_file)
    tmp_L = sitk.GetArrayFromImage(tmp_I)
    tmp_L = np.reshape(tmp_L, (1, -1))
    tmp_L_unique = np.unique(tmp_L)
    print(tmp_L_unique)

    save_path = '../Data/IBSR18/corr_label/'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    for i in range(1, 19):
        I = sitk.ReadImage(label_dir + 'c' + str(i) + '.nii')
        L = sitk.GetArrayFromImage(I)

        for ind in range(len(tmp_L_unique)):
            L[L==tmp_L_unique[ind]] = ind

        new_image = sitk.GetImageFromArray(L)
        # new_image.SetOrigin((-1, -1, 1))
        # new_image.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
        new_image.CopyInformation(I)
        sitk.WriteImage(new_image, save_path + 'c' + str(i) + '.nii')

if __name__ == '__main__':
    pass
    # check_label('../Data/LPBA40/label_affine_icbm/')
    # find_diff_label('../Data/LPBA40/corr_label/')
    # A = get_label_density('../Data/LPBA40/label_affine_icbm/s1.nii')
    # for i in range(len(A)):
    #     if A[i]*229.0*229.0*193.0/(72.0*72.0*72.0) > 0.01:
    #         print(i, A[i]*229.0*229.0*193.0/(72.0*72.0*72.0))
    # B = get_label('../Data/LPBA40/label_affine_icbm/s1.nii')
    # B = get_img_size('../Data/LPBA40/label_affine_icbm/s1.nii')
    # label_preprocessing('../Data/LPBA40/label_affine_icbm/')
    # check_label('../Data/IBSR18/label_affine_icbm/')
    # check_label('../Data/IBSR18/corr_label/')
    # label_preprocessing_ibsr('../Data/IBSR18/label_affine_icbm/')
    # find_diff_label('/playpen/zpd/remote/MAS/Data/MAMBIS_seg_res/folder_1/MAMBIS_folder_1_result/')
    label_preprocessing_MAMBIS('/playpen/zpd/remote/MAS/Data/MAMBIS_seg_res/folder_2/MAMBIS_folder_2_result/')