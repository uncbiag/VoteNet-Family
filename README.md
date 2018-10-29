# multi-atlas-segmentation

## 0. Data augmentation and preprocessing

MGH10, CUMC12, IBSR18, LPBA40, OASIS, ADNI
Currently, data augmentation step is hold on. 

- LPBA40 dataset is used for training a U-net both as a segmentation net and as a baseline for segmentation prediction.

- ~~OASIS dataset is used to train a fast registration prediction net.~~
    Update: Since OASIS dataset only has 384 images, that is not enough for training a well-behaviored registration network, 
    hence we switch to ADNI dataset, which has over 6000 images of MRI brain images. 

Preprocessing:

- All brain images are affine transformed, histogram equalized and registrated to a common atlas space.
- LPBA40 labels are reorganized into continuous integer from 0 (background) to 56.
- There is an additional label (taged 116) in s37.nii. Currently set it to 0 (background).
 
## 1. U-Net for segmentation (baseline)
5-fold cross validation is used. 28 images for training, 4 images for validation, 8 images for testing. 5 folds sum up to cover all images in LPBA40 dataset. 

Note:

1. There is a problem with random cropping of the images in dataloader with multiple processors due to np.random generates the same random numbers for each data batch. 
This is because numpy doesn't properly handle RNG states when fork subprocesses. It's numpy's issue with multiprocessing tracked at numpy/numpy#9248). 
Thus, currently, using one worker in dataloader to extract patches.
Update: this issue is fixed by remove the random state generate in random_crop_3d function. Now, can use multiple workers to load training data.

## 2. Registration Net for fast registration prediction

## 3. Atlas Selection (Dice Score)
### 3.1 Nifty Reg toy example to examine the atlas selection procedure
- Use Nifty Reg to register 28 atlases images to the 8 testing images, which involves 28*8=224 pairwise image registrations 
- Warp the labels of the 28 atlases into the testing image space
- Select template atlases based on global Dice or local Dice

## 4. Multi-atlas segmentation